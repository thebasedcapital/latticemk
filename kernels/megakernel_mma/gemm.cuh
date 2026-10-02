// Interleave independent columns while one weight pair is live. Preserve
// each column's four FP16 chains and the original FP32/reduction order.
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
      __half2 acc[MT][4]={};
#pragma unroll
      for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<4;++j) {
          const __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000F000Fu)|0x64006400u),as_h2(0x64006400u));
          const __half2 weight=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
#pragma unroll
          for(int m=0;m<MT;++m) {
            const uint32_t* words=reinterpret_cast<const uint32_t*>(u.sx4[m]);
            const __half2 a=as_h2(words[(4*i+j)*cpr+c]);
            acc[m][j]=__hfma2(weight,a,acc[m][j]);
          }
        }
#pragma unroll
      for(int m=0;m<MT;++m) {
        const __half2 a=__hadd2(__hadd2(acc[m][0],acc[m][1]),__hadd2(acc[m][2],acc[m][3]));
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
