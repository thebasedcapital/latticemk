// Each INT4 chunk loads once. Dequantize four pairs into registers, reuse
// them across token columns, then advance the quarter. Only M FP32 row
// accumulators and one SMEM activation quarter are live across each tile.
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
      const float scale=__low2float(meta),offset=__high2float(meta);
      const uint32_t qw[4]={q.x,q.y,q.z,q.w};
#pragma unroll
      for(int i=0;i<4;++i) {
        float2 decoded[4];
#pragma unroll
        for(int j=0;j<4;++j) {
          const __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000F000Fu)|0x64006400u),as_h2(0x64006400u));
          decoded[j]=make_float2(__low2float(code)*scale+offset,__high2float(code)*scale+offset);
        }
#pragma unroll
        for(int m=0;m<MT;++m) {
          const uint4 x=u.sx4[m][i*cpr+c];
          const uint32_t xw[4]={x.x,x.y,x.z,x.w};
#pragma unroll
          for(int j=0;j<4;++j) {
            const float2 a=__half22float2(as_h2(xw[j]));
            sums[m]=fmaf(decoded[j].x,a.x,sums[m]);
            sums[m]=fmaf(decoded[j].y,a.y,sums[m]);
          }
        }
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
