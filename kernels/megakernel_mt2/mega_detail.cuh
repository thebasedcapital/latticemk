#ifdef PROFILE_DETAIL
__device__ __forceinline__ void detail_norm(const View& v,const __half* src,bool embed,int sel,int col){
  float r=xresidual(v,src,embed,col);dt_trace(6);
  stage_norm(v.norms+(int64_t)sel*HID,r,col);dt_trace(7);
}
__device__ __forceinline__ void detail_bar(int* bar,int& expect){gbar(bar,expect);dt_trace(12);}
__global__ void __launch_bounds__(THREADS,1) mega_detail(const P2* pp){
  const P2& p=*pp;int cta=blockIdx.x,expect=NCTA;
  if(cta==0 && threadIdx.x==0){for(int i=0;i<15;++i)dt_cycles[i]=0;dt_epoch=clock64();}
  for(int l=0;l<NLAY;++l){
    for(int m=0;m<MT;++m){View v=column(p,m);detail_norm(v,v.dout,l==0,l,m);}
    gemv_run(p.w[l*4],cta);dt_trace(0);detail_bar(p.bar,expect);
#if MT>1
    for(int m=0;m<MT;++m)append(column(p,m),l,position(p,m),cta);
    __syncthreads();dt_trace(10);detail_bar(p.bar,expect);
#endif
    for(int m=0;m<MT;++m){if(cta<NATTN)attn2(column(p,m),l,position(p,m),cta);__syncthreads();dt_trace(14);}
    detail_bar(p.bar,expect);
    for(int m=0;m<MT;++m){pro_attnc(column(p,m),m);dt_trace(8);}
    gemv_run(p.w[l*4+1],cta);dt_trace(1);detail_bar(p.bar,expect);
    for(int m=0;m<MT;++m){View v=column(p,m);detail_norm(v,v.oo,false,NLAY+l,m);}
    gemv_run(p.w[l*4+2],cta);dt_trace(2);detail_bar(p.bar,expect);
    for(int m=0;m<MT;++m){pro_silu(column(p,m),m);dt_trace(9);}
    gemv_run(p.w[l*4+3],cta);dt_trace(3);detail_bar(p.bar,expect);
  }
  for(int m=0;m<MT;++m){View v=column(p,m);detail_norm(v,v.dout,false,4*NLAY,m);}
  gemv_run(p.w[W_LM],cta);dt_trace(4);detail_bar(p.bar,expect);
  for(int m=0;m<MT;++m){argp2(column(p,m),cta);__syncthreads();dt_trace(11);}
  detail_bar(p.bar,expect);
  for(int m=0;m<MT;++m){if(cta==0)argc2(column(p,m),position(p,m));__syncthreads();dt_trace(11);}
  detail_bar(p.bar,expect);
  if(cta==0 && threadIdx.x==0)for(int i=0;i<15;++i)p.profile[i]=dt_cycles[i];
}
#endif
