// Exact nearest-codeword search for offline weight encoding (any LUT codebook).
// Codebook lives in SMEM; every thread scans all K codewords (broadcast reads, no bank conflicts).
// Built as a standalone .so and called from Python via ctypes on torch device pointers.
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>

template <int D>
__global__ void nearest_kernel(const float* __restrict__ x, int64_t n, const float* __restrict__ cb, int K,
                               int32_t* __restrict__ out) {
  extern __shared__ float s_cb[];
  for (int i = threadIdx.x; i < K * D; i += blockDim.x) s_cb[i] = cb[i];
  __syncthreads();
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v < n; v += (int64_t)gridDim.x * blockDim.x) {
    float xv[D];
#pragma unroll
    for (int j = 0; j < D; ++j) xv[j] = x[v * D + j];
    float best = FLT_MAX;
    int bi = 0;
    for (int k = 0; k < K; ++k) {
      float d = 0.f;
#pragma unroll
      for (int j = 0; j < D; ++j) {
        float t = xv[j] - s_cb[k * D + j];
        d = fmaf(t, t, d);
      }
      if (d < best) { best = d; bi = k; }
    }
    out[v] = bi;
  }
}

extern "C" int vq_nearest(const float* x, int64_t n, const float* cb, int K, int D, int32_t* out) {
  size_t smem = (size_t)K * D * sizeof(float);
  if (smem > 48 * 1024) return -1;
  int blocks = 36 * 16;
  switch (D) {
    case 1: nearest_kernel<1><<<blocks, 256, smem>>>(x, n, cb, K, out); break;
    case 2: nearest_kernel<2><<<blocks, 256, smem>>>(x, n, cb, K, out); break;
    case 4: nearest_kernel<4><<<blocks, 256, smem>>>(x, n, cb, K, out); break;
    case 8: nearest_kernel<8><<<blocks, 256, smem>>>(x, n, cb, K, out); break;
    default: return -2;
  }
  cudaError_t e = cudaDeviceSynchronize();
  return e == cudaSuccess ? 0 : (int)e;
}
