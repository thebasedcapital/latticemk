#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef MT
#define MT 1
#endif
#ifndef THREADS
#define THREADS 1024
#endif
#define WARPS (THREADS / 32)
#define NCTA 36
struct W2 { const uint4* codes; const __half2* meta; const __half* x; __half* y; float* yf; int n_in,n_out; };
__shared__ __align__(16) union {uint4 sx4[MT][384];} u;
__device__ __forceinline__ __half2 as_h2(uint32_t v){return *reinterpret_cast<__half2*>(&v);}
#ifdef ORIGINAL
#include "gemm_original.cuh"
#elif defined(NONVOLATILE)
#include "gemm_nv.cuh"
#elif defined(PRELOAD)
#include "gemm_preload.cuh"
#else
#include "gemm.cuh"
#endif
// Each clock boundary consumes the measured value before reading clock64.
// This measures serialized latency, not hardware retired-stall counters.
__device__ __forceinline__ unsigned long long stamp(uint32_t v) {
  unsigned long long t;
  asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p bra zero; mov.u64 %0, %%clock64; bra done; zero: mov.u64 %0, %%clock64; done: }" : "=l"(t) : "r"(v) : "memory");
  return t;
}
__device__ __forceinline__ uint32_t bits(__half2 h){return *reinterpret_cast<uint32_t*>(&h);}
#include "shape_profile_best.cuh"
__global__ void __launch_bounds__(THREADS,1) shape(W2 w) {
  int cpr=w.n_in/32;
  for(int m=0;m<MT;++m)
    for(int i=threadIdx.x;i<cpr*16;i+=THREADS){
      int pair=i/cpr,c=i%cpr;
      __half2 h=__halves2half2(w.x[m*w.n_in+c*32+(pair/4)*8+pair%4],w.x[m*w.n_in+c*32+(pair/4)*8+pair%4+4]);
      reinterpret_cast<__half2*>(u.sx4[m])[i]=h;
    }
  __syncthreads();gemv_run(w,blockIdx.x);
}
__global__ void __launch_bounds__(512,1) profile(W2 w,unsigned long long* out) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32,cpr=w.n_in/32;
  for(int m=0;m<MT;++m)
    for(int i=threadIdx.x;i<cpr*16;i+=blockDim.x){
      int pair=i/cpr,c=i%cpr;
      reinterpret_cast<__half2*>(u.sx4[m])[i]=__halves2half2(w.x[m*w.n_in+c*32+(pair/4)*8+pair%4],w.x[m*w.n_in+c*32+(pair/4)*8+pair%4+4]);
    }
  __syncthreads();
  unsigned long long cycles[6]={};float sums[MT]={};
  // Profile every warp's first output row, all its chunks, without barriers.
  int row=blockIdx.x*(blockDim.x/32)+warp;
  if(row>=w.n_out)return;
  for(int c=lane;c<cpr;c+=32){
    auto start=stamp(0);uint4 q=__ldg(w.codes+(int64_t)row*cpr+c);
    __half2 meta=__ldg(w.meta+(int64_t)row*(cpr/4)+c/4);
    auto loaded=stamp(q.x^q.y^q.z^q.w^bits(meta));cycles[0]+=loaded-start;
    uint32_t qw[4]={q.x,q.y,q.z,q.w};__half2 decoded[16];
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
      for(int j=0;j<4;++j){
        __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000f000fu)|0x64006400u),as_h2(0x64006400u));
        decoded[4*i+j]=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
      }
    auto dec=stamp(bits(decoded[0])^bits(decoded[15]));cycles[1]+=dec-loaded;
#pragma unroll
    for(int m=0;m<MT;++m){
      __half2 acc[4]={};
#pragma unroll
      for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<4;++j){
          auto a0=stamp(bits(acc[j]));
          __half2 a=as_h2(*reinterpret_cast<const volatile uint32_t*>(reinterpret_cast<const uint32_t*>(u.sx4[m])+(4*i+j)*cpr+c));
          auto a1=stamp(bits(a));cycles[2]+=a1-a0;
          acc[j]=__hfma2(decoded[4*i+j],a,acc[j]);
          auto a2=stamp(bits(acc[j]));cycles[3]+=a2-a1;
        }
      auto r0=stamp(bits(acc[0])^bits(acc[1])^bits(acc[2])^bits(acc[3]));
      __half2 a=__hadd2(__hadd2(acc[0],acc[1]),__hadd2(acc[2],acc[3]));
      sums[m]+=__low2float(a)+__high2float(a);
      auto r1=stamp(__float_as_uint(sums[m]));cycles[4]+=r1-r0;
    }
  }
  auto r0=stamp(__float_as_uint(sums[0]));
#pragma unroll
  for(int o=16;o>0;o>>=1)
#pragma unroll
    for(int m=0;m<MT;++m)sums[m]+=__shfl_xor_sync(0xffffffff,sums[m],o);
  auto r1=stamp(__float_as_uint(sums[MT-1]));cycles[5]=r1-r0;
  if(lane==0){for(int i=0;i<6;++i)out[((blockIdx.x*WARPS+warp)*6)+i]=cycles[i];}
}
extern "C" int shape_time(int64_t codes,int64_t meta,int64_t x,int64_t y,int ni,int no,int reps,float* ms){
  W2 w={(const uint4*)codes,(const __half2*)meta,(const __half*)x,nullptr,(float*)y,ni,no};
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  cudaEventRecord(a);
  for(int i=0;i<reps;++i)shape<<<NCTA,THREADS>>>(w);
  cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(ms,a,b);*ms/=reps;
  cudaEventDestroy(a);cudaEventDestroy(b);return (int)cudaGetLastError();
}
extern "C" int shape_profile(int64_t codes,int64_t meta,int64_t x,int64_t y,int ni,int no,int64_t out){
  W2 w={(const uint4*)codes,(const __half2*)meta,(const __half*)x,nullptr,(float*)y,ni,no};
  profile<<<NCTA,512>>>(w,(unsigned long long*)out);return (int)cudaDeviceSynchronize();
}
