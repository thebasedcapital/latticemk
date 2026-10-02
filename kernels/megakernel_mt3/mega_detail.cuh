#ifdef PROFILE_DETAIL
__device__ __forceinline__ void detail_norm(const P2& p,bool embed,bool from_o,int sel){
  xresidual(p,embed,from_o);dt_trace(6);
  stage_norm(p.norms+(int64_t)sel*HID);dt_trace(7);
}
__device__ __forceinline__ void detail_bar(int* bar,int& expect){gbar(bar,expect);dt_trace(12);}
__global__ void __launch_bounds__(THREADS,1) mega_detail(const P2* pp){
  const P2& p=*pp;int cta=blockIdx.x,expect=NCTA;
  if(cta==0 && threadIdx.x==0){for(int i=0;i<15;++i)dt_cycles[i]=0;dt_epoch=clock64();}
  for(int l=0;l<NLAY;++l){
    detail_norm(p,l==0,false,l);
    gemv_run(p.w[l*4],cta);dt_trace(0);detail_bar(p.bar,expect);
    attn2(p,l,cta);
    detail_bar(p.bar,expect);
    pro_attnc(p);dt_trace(8);
    gemv_run(p.w[l*4+1],cta);dt_trace(1);detail_bar(p.bar,expect);
    detail_norm(p,false,true,NLAY+l);
    gemv_run(p.w[l*4+2],cta);dt_trace(2);detail_bar(p.bar,expect);
    pro_silu(p);dt_trace(9);
    gemv_run(p.w[l*4+3],cta);dt_trace(3);detail_bar(p.bar,expect);
  }
  detail_norm(p,false,false,4*NLAY);
  gemv_run(p.w[W_LM],cta);dt_trace(4);detail_bar(p.bar,expect);
  argp2(p,cta);__syncthreads();dt_trace(11);
  detail_bar(p.bar,expect);
  if(cta==0)argc2(p);__syncthreads();dt_trace(11);
  detail_bar(p.bar,expect);
  if(cta==0 && threadIdx.x==0)for(int i=0;i<15;++i)p.profile[i]=dt_cycles[i];
}
#endif
