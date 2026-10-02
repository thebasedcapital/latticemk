// Coarse, dependency-enforced stage boundaries. Lower diagnostic occupancy
// provides registers for timers without production spills. Grouped x operands
// deliberately serialize load and compute to expose latency, not stall counters.
template<int NC>
__device__ __forceinline__ void stages(W2 w,unsigned long long* out) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32,cpr=w.n_in/32,row=blockIdx.x*WARPS+warp;
  if(row>=w.n_out)return;
  unsigned long long cycles[8]={};float sums[MT]={};__half2 resident[MT][16];
  auto b0=stamp(lane+1);auto b1=stamp((uint32_t)b0);cycles[7]=b1-b0;
  if constexpr(NC==1){
    auto a=stamp(lane+1);uint32_t dep=0;
#pragma unroll
    for(int m=0;m<MT;++m)
#pragma unroll
      for(int k=0;k<16;++k){resident[m][k]=as_h2(reinterpret_cast<const uint32_t*>(u.sx4[m])[k*32+lane]);dep^=bits(resident[m][k]);}
    auto b=stamp(dep);cycles[2]+=b-a;
  }
#pragma unroll
  for(int t=0;t<NC;++t){
    int c=lane+32*t;
    auto a=stamp(lane+1);uint4 q=__ldg(w.codes+(int64_t)row*cpr+c);__half2 meta=__ldg(w.meta+(int64_t)row*(cpr/4)+c/4);
    auto b=stamp(q.x^q.y^q.z^q.w^bits(meta));cycles[0]+=b-a;
    uint32_t qw[4]={q.x,q.y,q.z,q.w};__half2 decoded[16];uint32_t dep=0;
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
      for(int j=0;j<4;++j){
        __half2 code=__hsub2(as_h2(((qw[i]>>(4*j))&0x000f000fu)|0x64006400u),as_h2(0x64006400u));
        decoded[4*i+j]=__hfma2(code,__halves2half2(__low2half(meta),__low2half(meta)),__halves2half2(__high2half(meta),__high2half(meta)));dep^=bits(decoded[4*i+j]);
      }
    auto d=stamp(dep);cycles[1]+=d-b;
#pragma unroll
    for(int m=0;m<MT;++m){
      __half2 active[16];auto x0=stamp(dep);
      if constexpr(NC==1){
#pragma unroll
        for(int k=0;k<16;++k)active[k]=resident[m][k];
      }else{
        dep=0;
#pragma unroll
        for(int k=0;k<16;++k){active[k]=as_h2(reinterpret_cast<const uint32_t*>(u.sx4[m])[k*cpr+c]);dep^=bits(active[k]);}
      }
      auto x1=stamp(dep);if constexpr(NC>1)cycles[2]+=x1-x0;
      __half2 acc[4]={__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f)};
#pragma unroll
      for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<4;++j)acc[j]=__hfma2(decoded[4*i+j],active[4*i+j],acc[j]);
      auto f=stamp(bits(acc[0])^bits(acc[1])^bits(acc[2])^bits(acc[3]));cycles[3]+=f-x1;
      __half2 h=__hadd2(__hadd2(acc[0],acc[1]),__hadd2(acc[2],acc[3]));sums[m]+=__low2float(h)+__high2float(h);
      auto r=stamp(__float_as_uint(sums[m]));cycles[4]+=r-f;
    }
  }
  auto r0=stamp(__float_as_uint(sums[0]));
#pragma unroll
  for(int o=16;o>0;o>>=1)
#pragma unroll
    for(int m=0;m<MT;++m)sums[m]+=__shfl_xor_sync(0xffffffff,sums[m],o);
  uint32_t dep=0;
#pragma unroll
  for(int m=0;m<MT;++m)dep^=__float_as_uint(sums[m]);
  auto r1=stamp(dep);cycles[5]=r1-r0;
  if(lane==0)
#pragma unroll
    for(int m=0;m<MT;++m)w.yf[m*w.n_out+row]=sums[m];
  auto r2=stamp(dep);cycles[6]=r2-r1;
  if(lane==0)for(int k=0;k<8;++k)out[(blockIdx.x*32+warp)*8+k]=cycles[k];
}
__global__ void __launch_bounds__(128,1) profile_best(W2 w,unsigned long long* out){
  int cpr=w.n_in/32;
  for(int m=0;m<MT;++m)
    for(int i=threadIdx.x;i<cpr*16;i+=blockDim.x){int pair=i/cpr,c=i%cpr;reinterpret_cast<__half2*>(u.sx4[m])[i]=__halves2half2(w.x[m*w.n_in+c*32+(pair/4)*8+pair%4],w.x[m*w.n_in+c*32+(pair/4)*8+pair%4+4]);}
  __syncthreads();
  if(w.n_in==1024)stages<1>(w,out);else if(w.n_in==2048)stages<2>(w,out);else stages<3>(w,out);
}
extern "C" int shape_profile_best(int64_t codes,int64_t meta,int64_t x,int64_t y,int ni,int no,int64_t out){
  W2 w={(const uint4*)codes,(const __half2*)meta,(const __half*)x,nullptr,(float*)y,ni,no};
  profile_best<<<NCTA,128>>>(w,(unsigned long long*)out);return (int)cudaDeviceSynchronize();
}
