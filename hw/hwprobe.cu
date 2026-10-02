// LM-00 hardware gate: device facts read from the driver, DRAM roofline (copy + read-only), launch overheads.
// Prints one JSON object on stdout. Every timing is 20 runs after warmup: median, p10, p90, spread.
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

constexpr int RUNS = 20;

struct Stat { double med, p10, p90; };
static Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double f) { return v[(size_t)(f * (v.size() - 1) + 0.5)]; };
  return {q(0.5), q(0.1), q(0.9)};
}

__global__ void copy_kernel(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void read_kernel(const uint4* __restrict__ a, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldg(a + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
__global__ void empty_kernel() {}

template <class F> static std::vector<double> time_runs(F f, int inner) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i) f();
  cudaDeviceSynchronize();
  std::vector<double> out;
  for (int r = 0; r < RUNS; ++r) {
    cudaEventRecord(e0);
    for (int i = 0; i < inner; ++i) f();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    out.push_back(ms / inner);
  }
  return out;
}

static void emit(const char* key, Stat s, const char* unit, bool last = false) {
  printf("  \"%s\": {\"median\": %.3f, \"p10\": %.3f, \"p90\": %.3f, \"spread_pct\": %.2f, \"unit\": \"%s\", \"runs\": %d}%s\n",
         key, s.med, s.p10, s.p90, 100.0 * (s.p90 - s.p10) / s.med, unit, RUNS, last ? "" : ",");
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int drv = 0, rt = 0, memclk = 0, smclk = 0, busw = 0, optin = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&smclk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, 0);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const double spec_gbps = 2.0 * memclk * 1e3 * (busw / 8) / 1e9;  // DDR: 2 transfers per clock as reported

  const size_t bytes = 512ull << 20;  // 128x the 4 MB L2
  uint4 *a, *b;
  uint32_t* sink;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(a, 1, bytes));
  const size_t n16 = bytes / 16;
  const int blocks = p.multiProcessorCount * 8;

  auto to_gbps = [&](std::vector<double> ms, double moved) {
    for (auto& m : ms) m = moved / (m * 1e-3) / 1e9;
    return stat(ms);
  };
  Stat copy = to_gbps(time_runs([&] { copy_kernel<<<blocks, 256>>>(a, b, n16); }, 5), 2.0 * bytes);
  Stat read = to_gbps(time_runs([&] { read_kernel<<<blocks, 256>>>(a, n16, sink); }, 5), (double)bytes);

  // Launch overhead: back-to-back empty kernels on one stream (per launch, us).
  auto us = [](std::vector<double> ms) { for (auto& m : ms) m *= 1e3; return stat(ms); };
  Stat launch = us(time_runs([&] { empty_kernel<<<1, 32>>>(); }, 1000));
  // CUDA-graph replay: a graph of 100 empty kernels, per node (us).
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  for (int i = 0; i < 100; ++i) empty_kernel<<<1, 32, 0, s>>>();
  cudaStreamEndCapture(s, &g);
  CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<double> gms;
  {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
    for (int r = 0; r < RUNS; ++r) {
      cudaEventRecord(e0, s);
      for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, s);
      cudaEventRecord(e1, s);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      gms.push_back(ms * 1e3 / (20 * 100));
    }
  }
  Stat graph_node = stat(gms);
  CK(cudaGetLastError());

  printf("{\n");
  printf("  \"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"sm_count\": %d,\n", p.name, p.major, p.minor, p.multiProcessorCount);
  printf("  \"smem_per_block_default\": %zu, \"smem_per_block_optin\": %d, \"smem_per_sm\": %zu,\n", p.sharedMemPerBlock, optin,
         p.sharedMemPerMultiprocessor);
  printf("  \"regs_per_sm\": %d, \"regs_per_block\": %d, \"max_threads_per_sm\": %d, \"l2_bytes\": %d,\n", p.regsPerMultiprocessor,
         p.regsPerBlock, p.maxThreadsPerMultiProcessor, p.l2CacheSize);
  printf("  \"global_mem_bytes\": %zu, \"mem_bus_bits\": %d, \"mem_clock_khz\": %d, \"sm_clock_max_khz\": %d,\n", p.totalGlobalMem,
         busw, memclk, smclk);
  printf("  \"spec_bandwidth_gbps\": %.1f, \"driver_cuda_version\": %d, \"runtime_cuda_version\": %d,\n", spec_gbps, drv, rt);
  printf("  \"buffer_bytes\": %zu,\n", bytes);
  emit("copy_bandwidth_gbps", copy, "GB/s (read+write bytes)");
  emit("read_bandwidth_gbps", read, "GB/s");
  emit("launch_overhead_us", launch, "us per back-to-back empty launch");
  emit("graph_node_overhead_us", graph_node, "us per empty node in a 100-node graph", true);
  printf("}\n");
  return 0;
}
