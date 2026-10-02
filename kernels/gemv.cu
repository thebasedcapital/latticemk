// Batch-1 GEMV kernels for Turing (sm_75): fp16, INT4-g128 (asymmetric), and LUT-decoded VQ codes.
// y[out] = W[out, in] . x[in], x/y fp16. Group = 128 input columns; chunk = 32 weights handled by one lane.
//
// Shared structure (identical across formats so only the weight decode differs):
//   * block stages x in SMEM once (optionally applying the fused randomized Hadamard transform),
//     then each lane keeps its NC = in/1024 chunks of x in registers as half2 for every row it touches;
//   * a warp owns a row; lanes stride over chunks; two rows in flight per warp; warp-shuffle reduction;
//   * per chunk: half2 FMAs into a half2 partial, then fp32 accumulate with the group scale.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum Fmt : int { F16 = 0, I4 = 1, V4E0 = 2, V4E1 = 3, V4E2 = 4, V2E0 = 5, V2E1 = 6 };

struct Desc {
  int fmt, out, in, rht;
  const void* codes;  // F16: halves; I4: uint4 per chunk; VQ: base plane (uint2 per chunk)
  const void* hi;     // VQ extra-bit plane (E*32/D bits per chunk) or null
  const void* meta;   // I4: half2 (scale, offset) per group; VQ: half scale per group
  const void* lut;    // VQ: 2^K entries of D halves
  const __half* x;
  __half* y;
  const float* signs;  // RHT sign vector (length in) when rht != 0
};

// RPW = rows a warp keeps in flight. Sweep on RTX 4000 (2/4/8): 2 is best for every format; the VQ kernels are
// decode bound, not latency bound, so more rows only cost registers/occupancy.
// VQ decode cost is SMEM bank conflicts on the random LUT reads: a 16/32-entry half2 LUT (A2) maps one entry per
// bank and is conflict-free (~70% of roofline); 512-2048-entry uint2 LUTs (A4) conflict (~38%). Expanding A2-k4 to a
// 256-entry pair table halves the LDS count but was 1.7x slower end-to-end (conflicts), so it was dropped.
#ifndef VQ_RPW
#define VQ_RPW 2
#endif
#ifndef I4_RPW
#define I4_RPW 2
#endif
template <int FMT> struct FmtInfo;
template <> struct FmtInfo<F16> { static constexpr int D = 0, E = 0, LUT_U32 = 1, RPW = 1; };  // 64 B/lane/chunk
template <> struct FmtInfo<I4> { static constexpr int D = 0, E = 0, LUT_U32 = 1, RPW = I4_RPW; };
template <> struct FmtInfo<V4E0> { static constexpr int D = 4, E = 0, LUT_U32 = 512, RPW = VQ_RPW; };
template <> struct FmtInfo<V4E1> { static constexpr int D = 4, E = 1, LUT_U32 = 1024, RPW = VQ_RPW; };
template <> struct FmtInfo<V4E2> { static constexpr int D = 4, E = 2, LUT_U32 = 2048, RPW = VQ_RPW; };
template <> struct FmtInfo<V2E0> { static constexpr int D = 2, E = 0, LUT_U32 = 16, RPW = VQ_RPW; };
template <> struct FmtInfo<V2E1> { static constexpr int D = 2, E = 1, LUT_U32 = 32, RPW = VQ_RPW; };

struct Chunk {
  uint4 q[4];     // F16 uses all four, I4 uses q[0], VQ uses q[0].x/.y as the base plane
  uint32_t hi;    // VQ extra bits
  __half2 meta;   // I4 (scale, offset); VQ (scale, -)
};

__device__ __forceinline__ __half2 as_h2(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }

template <int FMT>
__device__ __forceinline__ Chunk load_chunk(const Desc& d, int row, int c, int cpr) {
  Chunk k;
  const int64_t rc = (int64_t)row * cpr + c;
  const int64_t g = (int64_t)row * (cpr >> 2) + (c >> 2);
  if constexpr (FMT == F16) {
    const uint4* p = reinterpret_cast<const uint4*>(d.codes) + rc * 4;
#pragma unroll
    for (int u = 0; u < 4; ++u) k.q[u] = __ldg(p + u);
  } else if constexpr (FMT == I4) {
    k.q[0] = __ldg(reinterpret_cast<const uint4*>(d.codes) + rc);
    k.meta = __ldg(reinterpret_cast<const __half2*>(d.meta) + g);
  } else {
    constexpr int D = FmtInfo<FMT>::D, E = FmtInfo<FMT>::E;
    uint2 lo = __ldg(reinterpret_cast<const uint2*>(d.codes) + rc);
    k.q[0].x = lo.x;
    k.q[0].y = lo.y;
    constexpr int HB = E * 32 / D;  // extra bits per chunk: 0, 8, 16
    if constexpr (HB == 8) k.hi = __ldg(reinterpret_cast<const uint8_t*>(d.hi) + rc);
    else if constexpr (HB == 16) k.hi = __ldg(reinterpret_cast<const uint16_t*>(d.hi) + rc);
    else k.hi = 0;
    k.meta = __half2half2(__ldg(reinterpret_cast<const __half*>(d.meta) + g));
  }
  return k;
}

template <int FMT>
__device__ __forceinline__ float dot_chunk(const Chunk& k, const __half2* xr, float xsum, const uint32_t* slut) {
  // Four independent HFMA2 chains: one serial chain of 16 caps a warp at ~32 weights per ~100 cycles.
  __half2 acc[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) acc[a] = __float2half2_rn(0.f);
  if constexpr (FMT == F16) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      acc[0] = __hfma2(as_h2(k.q[u].x), xr[4 * u + 0], acc[0]);
      acc[1] = __hfma2(as_h2(k.q[u].y), xr[4 * u + 1], acc[1]);
      acc[2] = __hfma2(as_h2(k.q[u].z), xr[4 * u + 2], acc[2]);
      acc[3] = __hfma2(as_h2(k.q[u].w), xr[4 * u + 3], acc[3]);
    }
  } else if constexpr (FMT == I4) {
    const uint32_t w[4] = {k.q[0].x, k.q[0].y, k.q[0].z, k.q[0].w};
    const __half2 magic = as_h2(0x64006400u);  // 1024.0
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 v = __hsub2(as_h2(((w[i] >> (4 * j)) & 0x000F000Fu) | 0x64006400u), magic);
        acc[j] = __hfma2(v, xr[4 * i + j], acc[j]);
      }
  } else {
    constexpr int D = FmtInfo<FMT>::D, E = FmtInfo<FMT>::E;
    constexpr uint32_t EM = (1u << E) - 1;
    if constexpr (D == 4) {
      const uint2* lut = reinterpret_cast<const uint2*>(slut);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        uint32_t idx = ((t < 4 ? k.q[0].x : k.q[0].y) >> (8 * (t & 3))) & 0xFFu;
        if constexpr (E > 0) idx |= ((k.hi >> (E * t)) & EM) << 8;
        uint2 e = lut[idx];
        acc[2 * (t & 1)] = __hfma2(as_h2(e.x), xr[2 * t], acc[2 * (t & 1)]);
        acc[2 * (t & 1) + 1] = __hfma2(as_h2(e.y), xr[2 * t + 1], acc[2 * (t & 1) + 1]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        uint32_t idx = ((t < 8 ? k.q[0].x : k.q[0].y) >> (4 * (t & 7))) & 0xFu;
        if constexpr (E > 0) idx |= ((k.hi >> (E * t)) & EM) << 4;
        acc[t & 3] = __hfma2(as_h2(slut[idx]), xr[t], acc[t & 3]);
      }
    }
  }
  const __half2 s2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
  const float s = __low2float(s2) + __high2float(s2);
  if constexpr (FMT == F16) return s;
  else if constexpr (FMT == I4) return s * __low2float(k.meta) + __high2float(k.meta) * xsum;
  else return s * __low2float(k.meta);
}

constexpr int THREADS = 256, WARPS = THREADS / 32, MAX_IN = 3072;

template <int FMT, int NC, bool RHT>
__global__ void __launch_bounds__(THREADS, NC == 1 ? 4 : 2) gemv_kernel(Desc d) {
  __shared__ __align__(16) __half sx[MAX_IN];
  __shared__ __align__(16) uint32_t slut[FmtInfo<FMT>::LUT_U32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, n = d.in;

  if constexpr (FmtInfo<FMT>::D > 0)
    for (int i = tid; i < FmtInfo<FMT>::LUT_U32; i += THREADS) slut[i] = reinterpret_cast<const uint32_t*>(d.lut)[i];
  if constexpr (RHT) {
    // x' = H (s * x) per 1024-block, H = Sylvester Hadamard / 32 (matches lmk.quant.rht_apply).
    // One warp per block; lane holds elements lane*32 + j: stages h < 32 in registers, h >= 32 via shfl_xor.
    for (int blk = warp; blk < n / 1024; blk += WARPS) {
      const int base = blk * 1024 + lane * 32;
      float v[32];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint4 hx = *reinterpret_cast<const uint4*>(d.x + base + 8 * u);
        const float4 s0 = *reinterpret_cast<const float4*>(d.signs + base + 8 * u);
        const float4 s1 = *reinterpret_cast<const float4*>(d.signs + base + 8 * u + 4);
        const float2 a = __half22float2(as_h2(hx.x)), b = __half22float2(as_h2(hx.y));
        const float2 c = __half22float2(as_h2(hx.z)), e = __half22float2(as_h2(hx.w));
        v[8 * u + 0] = a.x * s0.x; v[8 * u + 1] = a.y * s0.y; v[8 * u + 2] = b.x * s0.z; v[8 * u + 3] = b.y * s0.w;
        v[8 * u + 4] = c.x * s1.x; v[8 * u + 5] = c.y * s1.y; v[8 * u + 6] = e.x * s1.z; v[8 * u + 7] = e.y * s1.w;
      }
#pragma unroll
      for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int j = 0; j < 32; ++j)
          if (!(j & h)) {
            const float a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
          }
#pragma unroll
      for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
          v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
#pragma unroll
      for (int j = 0; j < 32; j += 2)
        *reinterpret_cast<__half2*>(sx + base + j) = __floats2half2_rn(v[j] * (1.f / 32.f), v[j + 1] * (1.f / 32.f));
    }
  } else {
    for (int i = tid * 8; i < n; i += THREADS * 8)
      *reinterpret_cast<uint4*>(sx + i) = *reinterpret_cast<const uint4*>(d.x + i);
  }
  __syncthreads();

  __half2 xr[NC][16];
  float xsum[NC];
#pragma unroll
  for (int t = 0; t < NC; ++t) {
    const uint4* p = reinterpret_cast<const uint4*>(sx + (lane + 32 * t) * 32);
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      uint4 v = p[u];
      xr[t][4 * u + 0] = as_h2(v.x);
      xr[t][4 * u + 1] = as_h2(v.y);
      xr[t][4 * u + 2] = as_h2(v.z);
      xr[t][4 * u + 3] = as_h2(v.w);
    }
    if constexpr (FMT == I4)
#pragma unroll
      for (int u = 0; u < 16; ++u) s += __low2float(xr[t][u]) + __high2float(xr[t][u]);
    xsum[t] = s;
  }

  constexpr int R = FmtInfo<FMT>::RPW;
  const int cpr = n / 32, stride = gridDim.x * WARPS;
  for (int row0 = blockIdx.x * WARPS + warp; row0 < d.out; row0 += R * stride) {
    Chunk c[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r * stride < d.out)
#pragma unroll
        for (int t = 0; t < NC; ++t) c[r][t] = load_chunk<FMT>(d, row0 + r * stride, lane + 32 * t, cpr);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = 0.f;
      if (row0 + r * stride < d.out)
#pragma unroll
        for (int t = 0; t < NC; ++t) acc[r] += dot_chunk<FMT>(c[r][t], xr[t], xsum[t], slut);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    if (lane < R && row0 + lane * stride < d.out) {
      float v = acc[0];
#pragma unroll
      for (int r = 1; r < R; ++r) v = lane == r ? acc[r] : v;
      d.y[row0 + lane * stride] = __float2half(v);
    }
  }
}

static int g_sms = 0;

template <int FMT, int NC, bool RHT>
static void launch_t(const Desc& d, cudaStream_t s) {
  static int occ = 0;
  if (!occ) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gemv_kernel<FMT, NC, RHT>, THREADS, 0);
  int want = (d.out + FmtInfo<FMT>::RPW * WARPS - 1) / (FmtInfo<FMT>::RPW * WARPS);  // every warp fills its rows
  int blocks = want < occ * g_sms ? want : occ * g_sms;
  gemv_kernel<FMT, NC, RHT><<<blocks, THREADS, 0, s>>>(d);
}

template <int FMT>
static int launch_f(const Desc& d, cudaStream_t s) {
  const int nc = d.in / 1024;
  if (d.in % 1024 || nc < 1 || nc > 3) return -10;
#define L(N)                                        \
  if (nc == N) {                                    \
    if (d.rht) launch_t<FMT, N, true>(d, s);        \
    else launch_t<FMT, N, false>(d, s);             \
    return 0;                                       \
  }
  L(1) L(2) L(3)
#undef L
  return -11;
}

static int launch(const Desc& d, cudaStream_t s) {
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
  switch (d.fmt) {
    case F16: return launch_f<F16>(d, s);
    case I4: return launch_f<I4>(d, s);
    case V4E0: return launch_f<V4E0>(d, s);
    case V4E1: return launch_f<V4E1>(d, s);
    case V4E2: return launch_f<V4E2>(d, s);
    case V2E0: return launch_f<V2E0>(d, s);
    case V2E1: return launch_f<V2E1>(d, s);
  }
  return -12;
}

extern "C" int gemv_run(const Desc* ds, int n) {
  for (int i = 0; i < n; ++i)
    if (int rc = launch(ds[i], 0)) return rc;
  cudaError_t e = cudaDeviceSynchronize();
  return e == cudaSuccess ? 0 : (int)e;
}

// Average milliseconds per replay of the whole descriptor sequence, captured as one CUDA graph
// (removes CPU launch overhead; per-kernel GPU launch/drain gaps remain, as in any non-persistent engine).
extern "C" float gemv_time(const Desc* ds, int n, int iters) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  for (int i = 0; i < n; ++i) launch(ds[i], s);  // resolve occupancy caches outside capture
  cudaStreamSynchronize(s);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  for (int i = 0; i < n; ++i) launch(ds[i], s);
  cudaStreamEndCapture(s, &g);
  if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) return -1.f;
  for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, s);
  for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, s);
  cudaEventRecord(e1, s);
  cudaEventSynchronize(e1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaStreamDestroy(s);
  return cudaGetLastError() == cudaSuccess ? ms / iters : -2.f;
}

// Read-only streaming roofline: every byte of buf read once per iteration with 16-byte loads.
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldg(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // defeat DCE without a store per thread
}

extern "C" float read_gbps(const void* buf, size_t bytes, int iters) {
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
  uint32_t* sink;
  cudaMalloc(&sink, 4);
  int blocks = g_sms * 8;
  for (int i = 0; i < 3; ++i) read_kernel<<<blocks, 256>>>((const uint4*)buf, bytes / 16, sink);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) read_kernel<<<blocks, 256>>>((const uint4*)buf, bytes / 16, sink);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaFree(sink);
  return (float)((double)bytes * iters / (ms * 1e-3) / 1e9);
}
