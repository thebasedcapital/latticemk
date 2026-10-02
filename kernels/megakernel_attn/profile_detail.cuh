#ifdef PROFILE_DETAIL
__shared__ unsigned long long dt_cycles[15],dt_epoch;
__device__ __forceinline__ void dt_trace(int kind){
  if(blockIdx.x==0 && threadIdx.x==0){auto now=clock64();dt_cycles[kind]+=now-dt_epoch;dt_epoch=now;}
}
#else
__device__ __forceinline__ void dt_trace(int){}
#endif
