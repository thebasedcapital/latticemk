#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define HID 1024
#define NLAY 28
#define QROWS 2048
#define KVROWS 1024
#define QKV 4096
#define INTER 3072
#define GU 6144
#define VOCAB 151936
#define MAXPOS 8704
#define NCTA 36
#define NATTN 32            // CTAs that carry attention work
#define NARGP 36
#define W_LM (NLAY * 4)

#ifndef THREADS
#define THREADS 1024
#endif
#ifndef MT
#define MT 1
#endif
#define WARPS (THREADS / 32)
#define MAXIN 3072
#define NEG_INF (-1e30f)
#define RMS_EPS 1e-6f

#ifndef RPW2
#define RPW2 1
#endif
#ifndef RPW2_LM
#define RPW2_LM RPW2
#endif

struct W2 {
  const uint4* codes;   // [out][n_in/32]
  const __half2* meta;  // [out][n_in/128]
  const __half* x;      // matvec-only stage kernel input (unused otherwise)
  __half* y;
  float* yf;            // f32 out (lm_head) when non-null
  int n_in, n_out;
};

struct P2 {
  W2 w[W_LM + 1];
  const __half* emb;    // [VOCAB][HID]
  const __half* norms;  // [NORM_ROWS=113][HID]
  const float* rope;    // [MAXPOS][256]
  __half *qkv, *oo, *gu, *dout, *kc, *vc;
  float* part;          // [NATTN][130]
  float* logits;
  float* argp;          // [NARGP][2]
  int* tok;
  int* tok_hist;
  int* pos;             // device position base (incremented by steps at end)
  int* bar;             // global barrier counter
  int steps;
  int mode;
  int cap;
  #ifdef PROFILE
  unsigned long long* profile;
  #endif
};

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
  return *reinterpret_cast<__half2*>(&u);
}

__device__ __forceinline__ int ld_acq(const int* f) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(f)
               : "memory");
  return v;
}

__device__ __forceinline__ void red_rel(int* f, int a) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(f), "r"(a)
               : "memory");
}

// ---- SMEM pool (one CTA per SM, 48 KB default budget) ----------------------
__shared__ __align__(16) __half xloc[MT][HID];  // private residual stream
__shared__ __align__(16) union {
  uint4 sx4[MT][MAXIN / 8];    // transposed activation: 384 units = 6 KB
  float wpart[32][130]; // Keep v2's partition and accumulation order at 512 threads.
} u;
__shared__ __align__(16) __half qhead[128];
__shared__ __align__(16) __half kvloc[256];  // k_pos | v_pos of the pos row
#include "profile_detail.cuh"

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}


// grid barrier: `expect` is a per-CTA running target (+= NCTA per call).
__device__ __forceinline__ void gbar(int* ctr, int& expect) {
  __syncthreads();
  if (threadIdx.x == 0) {
    red_rel(ctr, 1);
    while (ld_acq(ctr) < expect) __nanosleep(64);
  }
  __syncthreads();
  expect += NCTA;
}

// ========================== GEMV (SMEM-x variant) ============================
// sx4 layout: 16B unit u (u<4) of chunk c (c<cpr) at sx4[u*cpr + c]. Lane
// owns chunks c = lane + 32*t (t<NC) so weight reads stay coalesced; its x
// quarters are sx4[u*cpr + c]: consecutive lanes -> consecutive units ->
// conflict-free LDS.128 (natural order would be a 4-way conflict).


#ifdef ORIGINAL
#include "gemm_original.cuh"
#elif defined(NONVOLATILE)
#include "gemm_nv.cuh"
#elif defined(INTERLEAVED)
#include "gemm.cuh"
#else
#include "gemm_preload.cuh"
#endif

struct View {
  const __half *emb,*norms;
  const float* rope;
  __half *qkv,*oo,*gu,*dout,*kc,*vc;
  float *part,*logits,*argp;
  int *tok,*tok_hist;
  int cap;
};
__device__ __forceinline__ View column(const P2& p,int m) {
  const int64_t cache=(int64_t)(p.mode?m:0)*NLAY*p.cap*KVROWS;
  return {p.emb,p.norms,p.rope,p.qkv+m*QKV,p.oo+m*HID,p.gu+m*GU,
          p.dout+m*HID,p.kc+cache,p.vc+cache,p.part+m*NATTN*130,
          p.logits+m*VOCAB,p.argp+m*NARGP*2,p.tok+m,p.tok_hist+m*p.cap,p.cap};
}
__device__ __forceinline__ int position(const P2& p,int m) {
  return p.mode?p.pos[m]:p.pos[0]+m;
}

// ---- prologues: produce u.sx4 for the following GEMV ------------------------

__device__ __forceinline__ void stage_store(int col,int idx,int cpr,uint4 x) {
  // A scalar half2 access puts each lane's chunk in a distinct SMEM bank.
  const int quarter=idx/cpr,c=idx%cpr;
  uint32_t* words=reinterpret_cast<uint32_t*>(u.sx4[col]);
  words[(4*quarter+0)*cpr+c]=x.x;
  words[(4*quarter+1)*cpr+c]=x.y;
  words[(4*quarter+2)*cpr+c]=x.z;
  words[(4*quarter+3)*cpr+c]=x.w;
}
#define SX4_WRITE(idx, halfs8)                          \
  do {                                                  \
    uint4 _pk;                                          \
    _pk.x = *reinterpret_cast<uint32_t*>(&(halfs8)[0]); \
    _pk.y = *reinterpret_cast<uint32_t*>(&(halfs8)[2]); \
    _pk.z = *reinterpret_cast<uint32_t*>(&(halfs8)[4]); \
    _pk.w = *reinterpret_cast<uint32_t*>(&(halfs8)[6]); \
    stage_store(col,idx,cpr,_pk);                        \
  } while (0)

// Preserve the four active warp sums and their padded XOR reduction order.
__shared__ float norm_part[MT][4], norm_r[MT];
__device__ void xresidual(const P2& p, bool embed, bool from_o) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int item = threadIdx.x; item < MT * (HID / 8); item += THREADS) {
    const int col = item / (HID / 8), chunk = item % (HID / 8), i = chunk * 8;
    float ss = 0.f;
    if (embed) {
      const __half* erow = p.emb + (int64_t)p.tok[col] * HID;
      uint4 v = *reinterpret_cast<const uint4*>(erow + i);
      *reinterpret_cast<uint4*>(xloc[col] + i) = v;
      __half2 a = as_h2(v.x), b = as_h2(v.y), c = as_h2(v.z), d = as_h2(v.w);
      float2 fa = __half22float2(a), fb = __half22float2(b);
      float2 fc = __half22float2(c), fd = __half22float2(d);
      ss += fa.x * fa.x + fa.y * fa.y + fb.x * fb.x + fb.y * fb.y +
            fc.x * fc.x + fc.y * fc.y + fd.x * fd.x + fd.y * fd.y;
    } else {
      const __half* src = (from_o ? p.oo : p.dout) + col * HID;
      uint4 xv = *reinterpret_cast<const uint4*>(xloc[col] + i);
      uint4 sv = *reinterpret_cast<const uint4*>(src + i);
      float t[8];
      const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
      const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 a = __half22float2(as_h2(xs[j]));
        float2 b = __half22float2(as_h2(sw[j]));
        t[2 * j] = a.x + b.x;
        t[2 * j + 1] = a.y + b.y;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) ss += t[j] * t[j];
      __half o[8];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        reinterpret_cast<__half2*>(o)[j] =
            __floats2half2_rn(t[2 * j], t[2 * j + 1]);
      *reinterpret_cast<uint4*>(xloc[col] + i) = *reinterpret_cast<uint4*>(o);
    }
    ss = warp_sum(ss);
    if (lane == 0) norm_part[col][chunk / 32] = ss;
  }
  __syncthreads();
  if (warp < MT) {
    float ss = lane < 4 ? norm_part[warp][lane] : 0.f;
    ss = warp_sum(ss);
    if (lane == 0) norm_r[warp] = rsqrtf(ss / HID + RMS_EPS);
  }
  __syncthreads();
}

// Stage all normalized columns with one publication barrier.
__device__ void stage_norm(const __half* wrow) {
  const int cpr = HID >> 5;
  for (int item = threadIdx.x; item < MT * cpr * 4; item += THREADS) {
    const int col = item / (cpr * 4), idx = item % (cpr * 4);
    const float r = norm_r[col];
    const int uu = idx / cpr, c = idx % cpr;
    const int i = (c * 4 + uu) * 8;
    uint4 xv = *reinterpret_cast<const uint4*>(xloc[col] + i);
    uint4 wv = *reinterpret_cast<const uint4*>(wrow + i);
    __half o[8];
    const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
    const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a = __half22float2(as_h2(xs[j]));
      float2 b = __half22float2(as_h2(ws[j]));
      reinterpret_cast<__half2*>(o)[j] =
          __floats2half2_rn(a.x * r * b.x, a.y * r * b.y);
    }
    SX4_WRITE(idx, o);
  }
  __syncthreads();
}

__device__ void pro_norm(const P2& p, bool embed, bool from_o, int wsel) {
  xresidual(p, embed, from_o);
  stage_norm(p.norms + (int64_t)wsel * HID);
}

// o-proj prologue: combine the 32 attention partials into sx (in = 2048).
// Head h has partials 2h (first half of positions) and 2h+1 (second half).
__device__ void pro_attnc(const P2& p) {
  const int cpr = QROWS >> 5;
  // A warp owns a head; aligned float2 loads keep the partial reads coalesced.
  for (int item = threadIdx.x; item < MT * (QROWS / 4); item += THREADS) {
    const int col = item / (QROWS / 4), chunk = item % (QROWS / 4);
    const int i = chunk * 4, h = i >> 7;
    const float* pa = p.part + col * NATTN * 130 + (h * 2 + 0) * 130;
    const float* pb = p.part + col * NATTN * 130 + (h * 2 + 1) * 130;
    float e0 = 0.f, e1 = 0.f, inv = 0.f;
    if ((threadIdx.x & 31) == 0) {
      const float m0 = pa[129], m1 = pb[129], M = fmaxf(m0, m1);
      e0 = m0 == NEG_INF ? 0.f : __expf(m0 - M);
      e1 = m1 == NEG_INF ? 0.f : __expf(m1 - M);
      const float l = e0 * pa[128] + e1 * pb[128];
      inv = l > 0.f ? 1.f / l : 0.f;
    }
    e0 = __shfl_sync(0xffffffffu, e0, 0);
    e1 = __shfl_sync(0xffffffffu, e1, 0);
    inv = __shfl_sync(0xffffffffu, inv, 0);
    uint32_t* words = reinterpret_cast<uint32_t*>(u.sx4[col]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int e = (i & 127) + j * 2;
      const float2 a = *reinterpret_cast<const float2*>(pa + e);
      const float2 b = *reinterpret_cast<const float2*>(pb + e);
      const __half2 v = __floats2half2_rn((e0 * a.x + e1 * b.x) * inv,
                                        (e0 * a.y + e1 * b.y) * inv);
      words[((i & 31) / 2 + j) * cpr + i / 32] =
          *reinterpret_cast<const uint32_t*>(&v);
    }
  }
  __syncthreads();
}

// down-proj prologue: sx = silu(gate) * up from gu[6144]. in = 3072.
__device__ void pro_silu(const P2& p) {
  const int cpr = INTER >> 5;
  for (int item = threadIdx.x; item < MT * cpr * 4; item += THREADS) {
    const int col = item / (cpr * 4), chunk = item % (cpr * 4);
    const int i = chunk * 8;
    const int idx = (chunk % 4) * cpr + chunk / 4;
    uint4 gv = *reinterpret_cast<const uint4*>(p.gu + col * GU + i);
    uint4 uv = *reinterpret_cast<const uint4*>(p.gu + col * GU + INTER + i);
    __half o[8];
    const uint32_t gs[4] = {gv.x, gv.y, gv.z, gv.w};
    const uint32_t us[4] = {uv.x, uv.y, uv.z, uv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 g = __half22float2(as_h2(gs[j]));
      float2 up = __half22float2(as_h2(us[j]));
      reinterpret_cast<__half2*>(o)[j] = __floats2half2_rn(
          (g.x / (1.f + __expf(-g.x))) * up.x,
          (g.y / (1.f + __expf(-g.y))) * up.y);
    }
    SX4_WRITE(idx, o);
  }
  __syncthreads();
}

// ============================ attention ======================================
// head-wise RMS norm over 128 elems + rope pairs (d, d+64), one warp.
// Qwen3 rope: for d < 64, (q[d], q[d+64]) -> (q[d]*cos - q[d+64]*sin,
// q[d+64]*cos + q[d]*sin) with cos/sin = rope[pos][d], rope[pos][128+d].
// Lane owns normed elems lane*4..+3; partners for lanes <16 live at lane+16.
__device__ __forceinline__ void norm_rope_row(const View& p, int layer, bool krow,
                                              int pos, const __half* src,
                                              __half* dst) {
  const int lane = threadIdx.x & 31;
  const __half* w =
      p.norms + (int64_t)((krow ? 3 * NLAY : 2 * NLAY) + layer) * HID;
  uint2 uq = *reinterpret_cast<const uint2*>(src + lane * 4);
  float2 a = __half22float2(as_h2(uq.x)), b = __half22float2(as_h2(uq.y));
  float v[4] = {a.x, a.y, b.x, b.y};
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  ss = warp_sum(ss);
  const float r = rsqrtf(ss / 128.f + RMS_EPS);
  float2 w0 = __half22float2(
      as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2]));
  float2 w1 = __half22float2(
      as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2 + 1]));
  v[0] *= r * w0.x; v[1] *= r * w0.y; v[2] *= r * w1.x; v[3] *= r * w1.y;
  const bool lo = lane < 16;
  const int pl = lo ? lane + 16 : lane - 16;
  __half o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = lane * 4 + j;
    const int d = lo ? e : e - 64;
    const float cs = p.rope[pos * 256 + d];
    const float sn = p.rope[pos * 256 + 128 + d];
    const float pv = __shfl_sync(0xffffffffu, v[j], pl);
    const float ro = lo ? v[j] * cs - pv * sn : v[j] * cs + pv * sn;
    o[j] = __float2half(ro);
  }
  *reinterpret_cast<uint2*>(dst + lane * 4) = *reinterpret_cast<uint2*>(o);
}

// per-CTA attention: head = cta>>1, position half = cta&1; warp partials ->
// SMEM -> one global partial row per CTA in p.part[cta].
__device__ void attn2(const View& p, int layer, int pos, int cta) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int h = cta >> 1, kvh = h >> 1, half = cta & 1;
  const int npos = pos + 1;
  const int mid = (npos + 1) >> 1;         // half0: [0,mid), half1: [mid,npos)
  const int a0 = half ? mid : 0, a1 = half ? npos : mid;

  // prologue: norm+rope this head's q; CTAs 0..7 append their kv head's
  // k (normed+roped) and v (raw) rows; half1 CTAs materialize the pos row
  // locally so nobody reads it from the caches this phase.
  if (warp == 0)
    norm_rope_row(p, layer, false, pos, p.qkv + h * 128, qhead);
  // appenders: CTA cta = 4*kvh covers kv head cta/4 (the first CTA of the
  // head pair for each of the 8 kv heads).
  const int appender = (cta % 4 == 0) ? cta / 4 : -1;
  if (MT == 1 && appender >= 0 && warp == 8)
    norm_rope_row(p, layer, true, pos, p.qkv + QROWS + appender * 128,
                  p.kc + ((int64_t)layer * p.cap + pos) * KVROWS +
                      appender * 128);
  if (MT == 1 && appender >= 0 && warp == 9) {
    const int lane8 = lane * 8;
    if (lane8 < 128)
      *reinterpret_cast<uint4*>(
          p.vc + ((int64_t)layer * p.cap + pos) * KVROWS + appender * 128 +
          lane8) =
          *reinterpret_cast<const uint4*>(p.qkv + QROWS + KVROWS +
                                          appender * 128 + lane8);
  }
  // the pair-half whose range contains pos builds that row locally
  const int cov = pos >= mid;
  if (half == cov && warp == 1)
    norm_rope_row(p, layer, true, pos, p.qkv + QROWS + kvh * 128, kvloc);
  if (half == cov && warp == 2) {
    const int lane8 = lane * 8;
    if (lane8 < 128)
      *reinterpret_cast<uint4*>(kvloc + 128 + lane8) =
          *reinterpret_cast<const uint4*>(p.qkv + QROWS + KVROWS + kvh * 128 +
                                          lane8);
  }
  __syncthreads();
  dt_trace(13);

  // Physical warps process the same 32 virtual partitions as batch-1 v2.
  const int len = a1 - a0;
  const int per = (len + 31) / 32;
  for (int vw = warp; vw < 32; vw += WARPS) {
  float* po = u.wpart[vw];
  const int w0 = a0 + vw * per, w1 = min(a1, w0 + per);
  const __half* kc = p.kc + (int64_t)layer * p.cap * KVROWS;
  const __half* vc = p.vc + (int64_t)layer * p.cap * KVROWS;
  const uint2 uq = *reinterpret_cast<const uint2*>(qhead + lane * 4);
  const __half2 qr0 = as_h2(uq.x), qr1 = as_h2(uq.y);
  float m = NEG_INF, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int t = w0; t < w1; ++t) {
    const __half* krow =
        t == pos ? kvloc : kc + (int64_t)t * KVROWS + kvh * 128;
    uint2 uk = *reinterpret_cast<const uint2*>(krow + lane * 4);
    __half2 prod = __hmul2(as_h2(uk.x), qr0);
    prod = __hfma2(as_h2(uk.y), qr1, prod);
    float s = __low2float(prod) + __high2float(prod);
    s = warp_sum(s) * 0.08838834764831845f;
    const float mn = fmaxf(m, s);
    const float corr = m == NEG_INF ? 0.f : __expf(m - mn);
    const float e = __expf(s - mn);
    const __half* vrow =
        t == pos ? kvloc + 128 : vc + (int64_t)t * KVROWS + kvh * 128;
    uint2 uv = *reinterpret_cast<const uint2*>(vrow + lane * 4);
    float2 v0 = __half22float2(as_h2(uv.x)), v1 = __half22float2(as_h2(uv.y));
    acc[0] = acc[0] * corr + e * v0.x;
    acc[1] = acc[1] * corr + e * v0.y;
    acc[2] = acc[2] * corr + e * v1.x;
    acc[3] = acc[3] * corr + e * v1.y;
    l = l * corr + e;
    m = mn;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) po[lane * 4 + j] = acc[j];
  if (lane == 0) { po[128] = l; po[129] = m; }
  }
  __syncthreads();
  dt_trace(5);

  // merge 32 warp partials -> one global partial per CTA (tid<130 covers
  // 128 acc elems + l + m).
  if (threadIdx.x < 130) {
    const int e = threadIdx.x;
    float M = NEG_INF;
    if (e < 130) {
#pragma unroll 4
      for (int w = 0; w < 32; ++w) M = fmaxf(M, u.wpart[w][129]);
    }
    if (M == NEG_INF) {
      if (e < 128) p.part[cta * 130 + e] = 0.f;
      else p.part[cta * 130 + e] = e == 128 ? 0.f : NEG_INF;
    } else if (e < 128) {
      float s = 0.f;
      for (int w = 0; w < 32; ++w) {
        const float mw = u.wpart[w][129];
        if (mw > NEG_INF) s += __expf(mw - M) * u.wpart[w][e];
      }
      p.part[cta * 130 + e] = s;
    } else if (e == 128) {
      float s = 0.f;
      for (int w = 0; w < 32; ++w) {
        const float mw = u.wpart[w][129];
        if (mw > NEG_INF) s += __expf(mw - M) * u.wpart[w][128];
      }
      p.part[cta * 130 + 128] = s;
      p.part[cta * 130 + 129] = M;
    }
  }
}

// ============================ argmax =========================================
__device__ void argp2(const P2& p, int cta) {
  constexpr int WP = WARPS / MT;
  __shared__ float sv[MT][WP];
  __shared__ int si[MT][WP];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int col = warp / WP, rowwarp = warp % WP;
  const int per = (VOCAB + NARGP - 1) / NARGP;
  const int a0 = cta * per, a1 = min(VOCAB, a0 + per);
  float bv = NEG_INF;
  int bi = a0;
  if (col < MT) {
    for (int i = a0 + rowwarp * 32 + lane; i < a1; i += WP * 32) {
      const float v = p.logits[col * VOCAB + i];
      if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) { sv[col][rowwarp] = bv; si[col][rowwarp] = bi; }
  }
  __syncthreads();
  if (warp < MT) {
    bv = lane < WP ? sv[warp][lane] : NEG_INF;
    bi = lane < WP ? si[warp][lane] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
      p.argp[warp * NARGP * 2 + 2 * cta] = bv;
      p.argp[warp * NARGP * 2 + 2 * cta + 1] = (float)bi;
    }
  }
}

__device__ void argc2(const P2& p) {
  const int lane = threadIdx.x & 31, col = threadIdx.x >> 5;
  if (col >= MT) return;
  float bv = NEG_INF;
  int bi = 0;
  for (int i = lane; i < NARGP; i += 32) {
    const float v = p.argp[col * NARGP * 2 + 2 * i];
    const int ix = (int)p.argp[col * NARGP * 2 + 2 * i + 1];
    if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) {
    p.tok[col] = bi;
    p.tok_hist[col * p.cap + position(p, col)] = bi;
  }
}


__device__ void append(const View& p,int l,int pos,int cta) {
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
  const int h=cta/4;
  if(cta>=NATTN || cta%4) return;
  if(warp==8) norm_rope_row(p,l,true,pos,p.qkv+QROWS+h*128,
                           p.kc+((int64_t)l*p.cap+pos)*KVROWS+h*128);
  if(warp==9 && lane<16)
    *reinterpret_cast<uint4*>(p.vc+((int64_t)l*p.cap+pos)*KVROWS+h*128+lane*8)=
      *reinterpret_cast<const uint4*>(p.qkv+QROWS+KVROWS+h*128+lane*8);
}
#include "mega_detail.cuh"
#ifdef PROFILE
__shared__ unsigned long long cycles[7],stamp;
#define TRACE(kind) do { if(cta==0 && threadIdx.x==0) { const auto now=clock64(); cycles[kind]+=now-stamp; stamp=now; } } while(0)
#else
#define TRACE(kind) do {} while(0)
#endif
__global__ void __launch_bounds__(THREADS,1) mega_mt(const P2* pp) {
  const P2& p=*pp;
  const int cta=blockIdx.x;
  int expect=NCTA;
#ifdef PROFILE
  if(cta==0 && threadIdx.x==0) {
    for(int i=0;i<7;++i) cycles[i]=0;
    stamp=clock64();
  }
#endif
  for(int l=0;l<NLAY;++l) {
    pro_norm(p,l==0,false,l);
    TRACE(6);
    gemv_run(p.w[l*4],cta);gbar(p.bar,expect);
    TRACE(0);
#if MT > 1
    for(int m=0;m<MT;++m) append(column(p,m),l,position(p,m),cta);
    gbar(p.bar,expect);
    TRACE(6);
#endif
    for(int m=0;m<MT;++m) {
      if(cta<NATTN) attn2(column(p,m),l,position(p,m),cta);
      __syncthreads();
    }
    gbar(p.bar,expect);
    TRACE(5);
    pro_attnc(p);
    TRACE(6);
    gemv_run(p.w[l*4+1],cta);gbar(p.bar,expect);
    TRACE(1);
    pro_norm(p,false,true,NLAY+l);
    TRACE(6);
    gemv_run(p.w[l*4+2],cta);gbar(p.bar,expect);
    TRACE(2);
    pro_silu(p);
    TRACE(6);
    gemv_run(p.w[l*4+3],cta);gbar(p.bar,expect);
    TRACE(3);
  }
  pro_norm(p,false,false,4*NLAY);
  TRACE(6);
  gemv_run(p.w[W_LM],cta);gbar(p.bar,expect);
  TRACE(4);
  argp2(p,cta);
  gbar(p.bar,expect);
  if(cta==0) argc2(p);
  gbar(p.bar,expect);
  TRACE(6);
#ifdef PROFILE
  if(cta==0 && threadIdx.x==0 && p.profile)
    for(int i=0;i<7;++i) p.profile[i]=cycles[i];
#endif
  if(cta==0 && threadIdx.x==0) {
    if(p.mode) for(int m=0;m<MT;++m) ++p.pos[m];
    else p.pos[0]+=MT;
  }
}

// Matvec-only persistent probe. Dedicated fixed activations avoid any
// cross-CTA data dependency. Includes staging, excludes attention/barriers.
__global__ void __launch_bounds__(THREADS,1) mt_matvec(const P2* pp) {
  const P2& p=*pp;
  for(int wi=0;wi<=W_LM;++wi) {
    const W2& w=p.w[wi];
    const int cpr=w.n_in/32;
    for(int m=0;m<MT;++m)
      for(int idx=threadIdx.x;idx<cpr*4;idx+=THREADS) {
        const int i=(idx%cpr*4+idx/cpr)*8;
        const uint4 x=*reinterpret_cast<const uint4*>(p.w[1].x+m*MAXIN+i);
        stage_store(m,idx,cpr,x);
      }
    __syncthreads();
    gemv_run(w,blockIdx.x);
    __syncthreads();
  }
}
// ============================ host side =====================================
static P2 h_p2;
static P2* d_p2 = nullptr;
static cudaStream_t g_s2 = nullptr;

static P2* dev2() {
  if (!d_p2) cudaMalloc(&d_p2, sizeof(P2));
  return d_p2;
}

// bufs: xn, qkv, oo, gu, dout, kc, vc, part, logits, argp, pos, bar
extern "C" int mt_init(const int64_t* bufs, const int64_t* codes,
                        const int64_t* meta, int64_t emb, int64_t norms,
                        int64_t rope, int64_t tok, int64_t tok_hist, int mode, int cap) {
  memset(&h_p2, 0, sizeof(h_p2));
  h_p2.mode=mode;
  h_p2.cap=cap;
  h_p2.w[0].x = (const __half*)bufs[0];  // xn: generic input for stage test
  h_p2.qkv = (__half*)bufs[1];
  h_p2.oo = (__half*)bufs[2];
  h_p2.gu = (__half*)bufs[3];
  h_p2.dout = (__half*)bufs[4];
  h_p2.kc = (__half*)bufs[5];
  h_p2.vc = (__half*)bufs[6];
  h_p2.part = (float*)bufs[7];
  h_p2.logits = (float*)bufs[8];
  h_p2.argp = (float*)bufs[9];
  h_p2.pos = (int*)bufs[10];
  h_p2.bar = (int*)bufs[11];
  h_p2.tok = (int*)tok;
  h_p2.tok_hist = (int*)tok_hist;
  h_p2.emb = (const __half*)emb;
  h_p2.norms = (const __half*)norms;
  h_p2.rope = (const float*)rope;
  const __half* xn = (const __half*)bufs[0];
  for (int l = 0; l < NLAY; ++l)
    for (int k = 0; k < 4; ++k) {
      W2& w = h_p2.w[l * 4 + k];
      w.codes = (const uint4*)codes[l * 4 + k];
      w.meta = (const __half2*)meta[l * 4 + k];
      w.yf = nullptr;
      switch (k) {
        case 0: w.x = xn; w.y = h_p2.qkv; w.n_in = HID; w.n_out = QKV; break;
        case 1: w.x = (const __half*)bufs[12]; w.y = h_p2.oo; w.n_in = QROWS; w.n_out = HID; break;
        case 2: w.x = xn; w.y = h_p2.gu; w.n_in = HID; w.n_out = GU; break;
        case 3: w.x = h_p2.gu; w.y = h_p2.dout; w.n_in = INTER; w.n_out = HID; break;
      }
    }
  W2& lm = h_p2.w[W_LM];
  lm.codes = (const uint4*)codes[W_LM];
  lm.meta = (const __half2*)meta[W_LM];
  lm.x = xn;
  lm.y = nullptr;
  lm.yf = h_p2.logits;
  lm.n_in = HID;
  lm.n_out = VOCAB;
  if (!g_s2) cudaStreamCreateWithFlags(&g_s2, cudaStreamNonBlocking);
  cudaMemcpy(dev2(), &h_p2, sizeof(P2), cudaMemcpyHostToDevice);
  return 0;
}


extern "C" int mt_run() {
  cudaMemsetAsync(h_p2.bar,0,sizeof(int),g_s2);
  mega_mt<<<NCTA,THREADS,0,g_s2>>>(dev2());
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) return (int)e;
  return (int)cudaStreamSynchronize(g_s2);
}
extern "C" int mt_time(int pos0,int iters,float* out) {
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  int positions[MT];for(int m=0;m<MT;++m) positions[m]=pos0;
  for(int i=0;i<iters;++i) {
    cudaMemcpyAsync(h_p2.pos,positions,(h_p2.mode?MT:1)*sizeof(int),cudaMemcpyHostToDevice,g_s2);
    cudaMemsetAsync(h_p2.bar,0,sizeof(int),g_s2);
    cudaEventRecord(a,g_s2);mega_mt<<<NCTA,THREADS,0,g_s2>>>(dev2());cudaEventRecord(b,g_s2);
    cudaEventSynchronize(b);cudaEventElapsedTime(out+i,a,b);
  }
  cudaEventDestroy(a);cudaEventDestroy(b);
  return (int)cudaGetLastError();
}
#ifdef PROFILE
extern "C" int mt_profile(int64_t output) {
  h_p2.profile=(unsigned long long*)output;
  cudaMemcpy(dev2(),&h_p2,sizeof(P2),cudaMemcpyHostToDevice);
  #ifdef PROFILE_DETAIL
  cudaMemsetAsync(h_p2.bar,0,sizeof(int),g_s2);
  mega_detail<<<NCTA,THREADS,0,g_s2>>>(dev2());
  return (int)cudaStreamSynchronize(g_s2);
  #else
  return mt_run();
  #endif
}
#endif
extern "C" int mt_time_gemv(int iters,float* out) {
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  for(int i=0;i<iters;++i) {
    cudaEventRecord(a,g_s2);mt_matvec<<<NCTA,THREADS,0,g_s2>>>(dev2());cudaEventRecord(b,g_s2);
    cudaEventSynchronize(b);cudaEventElapsedTime(out+i,a,b);
  }
  cudaEventDestroy(a);cudaEventDestroy(b);
  return (int)cudaGetLastError();
}
