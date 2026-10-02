#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef MT
#define MT 1
#endif
#ifndef THREADS
#define THREADS 512
#endif
#define WARPS (THREADS/32)
#define NCTA 36
struct W2 {const uint4* codes;const __half2* meta;const __half* x;__half* y;float* yf;int n_in,n_out;};
__shared__ __align__(16) union {uint4 sx4[MT][384];} u;
__device__ __forceinline__ __half2 as_h2(uint32_t v){return *reinterpret_cast<__half2*>(&v);}
#ifdef MMA
#include "gemm_mma.cuh"
#else
#include "gemm_preload.cuh"
#endif
__global__ void __launch_bounds__(THREADS,1) shape(W2 w) {
  int cpr=w.n_in/32;
  for(int m=0;m<MT;++m)
    for(int i=threadIdx.x;i<cpr*16;i+=THREADS){
      int pair=i/cpr,c=i%cpr;
      reinterpret_cast<uint32_t*>(u.sx4[m])[i]=reinterpret_cast<const uint32_t*>(w.x+m*w.n_in)[c*16+pair];
    }
  __syncthreads();gemv_run(w,blockIdx.x);
}
extern "C" int shape_time(int64_t codes,int64_t meta,int64_t x,int64_t y,int ni,int no,int reps,float* ms){
  W2 w={(const uint4*)codes,(const __half2*)meta,(const __half*)x,nullptr,(float*)y,ni,no};
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
  for(int i=0;i<reps;++i)shape<<<NCTA,THREADS>>>(w);
  cudaEventRecord(b);cudaError_t e=cudaEventSynchronize(b);cudaEventElapsedTime(ms,a,b);*ms/=reps;
  cudaEventDestroy(a);cudaEventDestroy(b);return e!=cudaSuccess?(int)e:(int)cudaGetLastError();
}
