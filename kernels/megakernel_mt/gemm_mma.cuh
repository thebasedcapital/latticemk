// Turing m16n8k8. Leaders load complete INT4 uint4 chunks once, then share
// the four packed words with each quartet. Dequantization goes directly to
// the two A-fragment registers. B contains M token columns, padded to eight.
__device__ __forceinline__ uint4 mma_codes(const W2& w,int row,int c) {
  const int lane=threadIdx.x&31;
  uint4 q={};
  if((lane&3)==0 && row<w.n_out) q=__ldg(w.codes+(int64_t)row*(w.n_in/32)+c);
  return q;
}
__device__ __forceinline__ uint32_t mma_pair(uint32_t packed,__half2 meta,int j) {
  const __half2 code=__hsub2(as_h2(((packed>>(4*j))&0x000F000Fu)|0x64006400u),as_h2(0x64006400u));
  const __half2 value=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
  return *reinterpret_cast<const uint32_t*>(&value);
}
__device__ __forceinline__ void gemv_run(const W2& w,int blk) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,j=lane&3;
  const int tiles=(w.n_out+15)/16,per=(tiles+NCTA-1)/NCTA;
  const int lo=blk*per,hi=min(tiles,lo+per),cpr=w.n_in/32;
  for(int tile=lo+warp;tile<hi;tile+=WARPS) {
    const int row=tile*16+lane/4;
    float d0=0,d1=0,d2=0,d3=0;
    for(int c=0;c<cpr;++c) {
      const uint4 q0=mma_codes(w,row,c),q1=mma_codes(w,row+8,c);
      const uint32_t words0[4]={q0.x,q0.y,q0.z,q0.w};
      const uint32_t words1[4]={q1.x,q1.y,q1.z,q1.w};
      const __half2 mt0=row<w.n_out?__ldg(w.meta+(int64_t)row*(w.n_in/128)+(c/4)):__float2half2_rn(0);
      const __half2 mt1=row+8<w.n_out?__ldg(w.meta+(int64_t)(row+8)*(w.n_in/128)+(c/4)):__float2half2_rn(0);
#pragma unroll
      for(int i=0;i<4;++i) {
        const uint32_t a0=mma_pair(__shfl_sync(0xffffffffu,words0[i],lane&~3),mt0,j);
        const uint32_t a1=mma_pair(__shfl_sync(0xffffffffu,words1[i],lane&~3),mt1,j);
        uint32_t b0=0;
        const int col=lane/4;
        if(col<MT) {
          const uint4 x=u.sx4[col][i*cpr+c];
          b0=j==0?x.x:j==1?x.y:j==2?x.z:x.w;
        }
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(a0),"r"(a1),"r"(b0));
      }
    }
    const int col=(lane%4)*2;const float ds[4]={d0,d1,d2,d3};
#pragma unroll
    for(int i=0;i<4;++i) {
      const int rr=row+(i>=2?8:0),cc=col+(i&1);
      if(rr<w.n_out && cc<MT) {
        if(w.yf) w.yf[cc*w.n_out+rr]=ds[i];
        else w.y[cc*w.n_out+rr]=__float2half(ds[i]);
      }
    }
  }
}
