// Causal latency test: move NC=1 activation reads out of the output-row loop.
// Wider NC uses the nonvolatile control to avoid an unbounded register footprint.
#define gemm_mt gemm_fallback
#define gemv_run gemv_fallback
#include "gemm_nv.cuh"
#undef gemm_mt
#undef gemv_run
__device__ __forceinline__ void gemm_preload(const W2& w,int r0,int r1) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  __half2 x[MT][16];
#pragma unroll
  for(int m=0;m<MT;++m)
#pragma unroll
    for(int k=0;k<16;++k)x[m][k]=as_h2(reinterpret_cast<const uint32_t*>(u.sx4[m])[k*32+lane]);
  for(int row=r0+warp;row<r1;row+=WARPS) {
    const uint4 q=__ldg(w.codes+(int64_t)row*32+lane);
    const __half2 meta=__ldg(w.meta+(int64_t)row*8+lane/4);
    const uint32_t qw[4]={q.x,q.y,q.z,q.w};
    __half2 acc[MT][4]={};
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
      for(int j=0;j<4;++j){
        __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000f000fu)|0x64006400u),as_h2(0x64006400u));
        __half2 weight=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
#pragma unroll
        for(int m=0;m<MT;++m)acc[m][j]=__hfma2(weight,x[m][4*i+j],acc[m][j]);
      }
    float sums[MT];
#pragma unroll
    for(int m=0;m<MT;++m){
      __half2 a=__hadd2(__hadd2(acc[m][0],acc[m][1]),__hadd2(acc[m][2],acc[m][3]));
      sums[m]=__low2float(a)+__high2float(a);
    }
#pragma unroll
    for(int o=16;o>0;o>>=1)
#pragma unroll
      for(int m=0;m<MT;++m)sums[m]+=__shfl_xor_sync(0xffffffff,sums[m],o);
    if(lane==0)
#pragma unroll
      for(int m=0;m<MT;++m){if(w.yf)w.yf[m*w.n_out+row]=sums[m];else w.y[m*w.n_out+row]=__float2half(sums[m]);}
  }
}
__device__ __forceinline__ void gemv_run(const W2& w,int blk){
  int chunk=(w.n_out+NCTA-1)/NCTA,r0=blk*chunk,r1=min(w.n_out,r0+chunk);
  if(r0>=r1)return;
  if(w.n_in==1024)gemm_preload(w,r0,r1);
  else if(w.n_in==2048)gemm_fallback<2>(w,r0,r1);
  else gemm_fallback<3>(w,r0,r1);
}
