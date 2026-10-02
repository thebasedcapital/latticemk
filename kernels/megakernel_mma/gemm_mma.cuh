// PTX m16n8k8: lane/4 owns rows r,r+8; lane%4 owns K pairs 2*j,2*j+1.
// Four lanes coalesce the complete 128-weight group of each row. Shuffles
// select its four 32-weight chunks. Weight files and sx4 staging are unchanged.
#ifndef MMA_SPLIT
#define MMA_SPLIT 1
#endif
__shared__ float mma_partial[WARPS][32][4];
__device__ __forceinline__ uint32_t mma_pair(uint32_t q,__half2 meta,int j) {
  __half2 code=__hsub2(as_h2(((q>>(4*j))&0x000f000fu)|0x64006400u),as_h2(0x64006400u));
  __half2 value=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));
  return *reinterpret_cast<uint32_t*>(&value);
}
#ifdef MMA_SWIZZLE
template<int NI>
__device__ __forceinline__ void mma_stage() {
  constexpr int cpr=NI/32,count=(MT*NI/2+THREADS-1)/THREADS;
  uint32_t values[count];
#pragma unroll
  for(int t=0;t<count;++t) {
    const int idx=threadIdx.x+t*THREADS,col=idx/(cpr*16),off=idx%(cpr*16);
    if(col<MT)values[t]=reinterpret_cast<const uint32_t*>(u.sx4[col])[off];
  }
  __syncthreads();
#pragma unroll
  for(int t=0;t<count;++t) {
    const int idx=threadIdx.x+t*THREADS,col=idx/(cpr*16),off=idx%(cpr*16);
    if(col<MT)reinterpret_cast<uint32_t*>(u.sx4[col])[((off%cpr)*16+off/cpr)^(col*4)]=values[t];
  }
  __syncthreads();
}
#endif
template<int NI>
__device__ __forceinline__ void gemm_mma(const W2& w,int blk) {
  static_assert(MMA_SPLIT==1 || MMA_SPLIT==2 || MMA_SPLIT==4 || MMA_SPLIT==8,"split K power of two");
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,j=lane&3;
  const int groups=WARPS/MMA_SPLIT,kg=warp%MMA_SPLIT,wg=warp/MMA_SPLIT;
  const int tiles=(w.n_out+15)/16,per=(tiles+NCTA-1)/NCTA;
  const int lo=blk*per,hi=min(tiles,lo+per);
  constexpr int cpr=NI/32;
#ifdef MMA_SWIZZLE
  mma_stage<NI>();
#endif
  // CTA-uniform iteration count permits deterministic split-K reduction.
  for(int base=lo;base<hi;base+=groups) {
    const int tile=base+wg,row=tile*16+lane/4;
    float d0=0,d1=0,d2=0,d3=0;
    if(tile<hi) for(int g=kg;g<cpr/4;g+=MMA_SPLIT) {
      uint4 q0={},q1={};__half2 mt0={},mt1={};
      if(row<w.n_out){q0=__ldg(w.codes+(int64_t)row*cpr+g*4+j);mt0=__ldg(w.meta+(int64_t)row*(cpr/4)+g);}
      if(row+8<w.n_out){q1=__ldg(w.codes+(int64_t)(row+8)*cpr+g*4+j);mt1=__ldg(w.meta+(int64_t)(row+8)*(cpr/4)+g);}
      const uint32_t words0[4]={q0.x,q0.y,q0.z,q0.w},words1[4]={q1.x,q1.y,q1.z,q1.w};
#pragma unroll
      for(int c=0;c<4;++c)
#pragma unroll
        for(int i=0;i<4;++i) {
          const uint32_t a0=mma_pair(__shfl_sync(0xffffffffu,words0[i],(lane&~3)+c),mt0,j);
          const uint32_t a1=mma_pair(__shfl_sync(0xffffffffu,words1[i],(lane&~3)+c),mt1,j);
          uint32_t b=0;const int col=lane/4;
          if(col<MT) {
#ifdef MMA_SWIZZLE
            b=reinterpret_cast<const uint32_t*>(u.sx4[col])[((g*4+c)*16+4*i+j)^(col*4)];
#else
            b=reinterpret_cast<const uint32_t*>(u.sx4[col])[(4*i+j)*cpr+g*4+c];
#endif
          }
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(a0),"r"(a1),"r"(b));
        }
    }
    if constexpr(MMA_SPLIT>1) {
      mma_partial[warp][lane][0]=d0;mma_partial[warp][lane][1]=d1;
      mma_partial[warp][lane][2]=d2;mma_partial[warp][lane][3]=d3;
      __syncthreads();
      if(kg==0) {
#pragma unroll
        for(int k=1;k<MMA_SPLIT;++k){d0+=mma_partial[warp+k][lane][0];d1+=mma_partial[warp+k][lane][1];d2+=mma_partial[warp+k][lane][2];d3+=mma_partial[warp+k][lane][3];}
      }
      __syncthreads();
    }
    if(kg==0 && tile<hi) {
      const int col=j*2;const float ds[4]={d0,d1,d2,d3};
#pragma unroll
      for(int i=0;i<4;++i) {
        const int rr=row+(i>=2?8:0),cc=col+(i&1);
        if(rr<w.n_out && cc<MT){if(w.yf)w.yf[cc*w.n_out+rr]=ds[i];else w.y[cc*w.n_out+rr]=__float2half(ds[i]);}
      }
    }
  }
}
__device__ __forceinline__ void gemv_run(const W2& w,int blk) {
  if(w.n_in==1024)gemm_mma<1024>(w,blk);
  else if(w.n_in==2048)gemm_mma<2048>(w,blk);
  else if(w.n_in==3072)gemm_mma<3072>(w,blk);
  else gemm_mma<128>(w,blk); // Fragment unit probe only.
}
