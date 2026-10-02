// One fully dequantized 32-weight chunk stays in registers. Activations
// remain in scalar-transposed SMEM, one current-column half2 at a time.
// M FP32 row sums combine v2's four FP16 partial chains per column/chunk.
template<int NC>
__device__ __forceinline__ void gemm_mt(const W2& w,int r0,int r1) {
  const int cpr=w.n_in>>5,ngrp=cpr>>2;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int row=r0+warp;row<r1;row+=WARPS) {
    float sums[MT]={};
#pragma unroll
    for(int t=0;t<NC;++t) {
      const int c=lane+32*t;
      const uint4 q=__ldg(w.codes+(int64_t)row*cpr+c);
      const __half2 meta=__ldg(w.meta+(int64_t)row*ngrp+(c>>2));
      const uint32_t qw[4]={q.x,q.y,q.z,q.w};
      __half2 decoded[16];
#pragma unroll
      for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<4;++j) {
          const __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000F000Fu)|0x64006400u),as_h2(0x64006400u));
          decoded[4*i+j]=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
        }
#pragma unroll
      for(int m=0;m<MT;++m) {
        __half2 acc[4]={__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f)};
#pragma unroll
        for(int i=0;i<4;++i)
#pragma unroll
          for(int j=0;j<4;++j) {
            const uint32_t* words=reinterpret_cast<const uint32_t*>(u.sx4[m]);
            // Prevent the compiler from hoisting all columns' operands.
            const __half2 a=as_h2(*reinterpret_cast<const volatile uint32_t*>(words+(4*i+j)*cpr+c));
            acc[j]=__hfma2(decoded[4*i+j],a,acc[j]);
          }
        const __half2 a=__hadd2(__hadd2(acc[0],acc[1]),__hadd2(acc[2],acc[3]));
        sums[m]+=__low2float(a)+__high2float(a);
      }
    }
#pragma unroll
    for(int o=16;o>0;o>>=1)
#pragma unroll
      for(int m=0;m<MT;++m) sums[m]+=__shfl_xor_sync(0xffffffffu,sums[m],o);
    if(lane==0)
#pragma unroll
      for(int m=0;m<MT;++m) {
        if(w.yf) w.yf[m*w.n_out+row]=sums[m];
        else w.y[m*w.n_out+row]=__float2half(sums[m]);
      }
  }
}
__device__ __forceinline__ void gemv_run(const W2& w,int blk) {
  const int chunk=(w.n_out+NCTA-1)/NCTA;
  const int r0=blk*chunk,r1=min(w.n_out,r0+chunk);
  if(r0>=r1) return;
  if(w.n_in==1024) gemm_mt<1>(w,r0,r1);
  else if(w.n_in==2048) gemm_mt<2>(w,r0,r1);
  else gemm_mt<3>(w,r0,r1);
}
