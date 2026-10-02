// LM-09: compressed-KV decode megakernel for Qwen3-0.6B-Base, batch 1, sm_75.
// Fork of kernels/megakernel_v2/mega2.cu — identical GEMV/phase schedule;
// only the KV-cache storage and the attention read path differ.
//
// KV format (chosen by kvcodec/codec_lm09.py quality sweep):
//   K: asymmetric INT{KBITS} per-channel over KGRP-token groups
//      (codes [layer][tok][kvh][128 nibbles/bytes], meta half2 (scale,off) per
//      (group,kvh,channel)); the newest KRES tokens stay exact fp16 in a ring
//      kr[layer][slot][kvh][128] (slot = tok % RCAP). A K group is finalized by
//      its appender CTA at the step where it slides out of the residual.
//      KMODE_T=1 switches to per-token INT{KBITS} K groups of KGRP dims
//      (quantized on append, no residual ring).
//   V: asymmetric INT{VBITS} per-token, VGRP-dim groups along the head
//      (codes [layer][tok][kvh][VBITS*128/8 B], meta half2 per (tok,kvh,grp));
//      quantized on append. VRES>0 keeps a fp16 residual ring vr.
//   t == pos (current token) is always read from kvloc, as in v2.
//
// Attention split: 32 CTAs x (head = c>>1, half = c&1) — unchanged.
// Weight layout (wave-1 packed INT4-g128): codes[out][in/32] uint4,
// meta[out][in/128] half2 (scale, offset). Layer weights 4*l+{0,1,2,3} =
// qkv(4096x1024), o(1024x2048), gu(6144x1024), down(1024x3072);
// w[112] = lm_head (151936x1024, f32 out).
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
#define MAXPOS 16640
#define NCTA 36
#define NATTN 32            // CTAs that carry attention work
#define NARGP 36
#define W_LM (NLAY * 4)
#define NKVH 8

#define THREADS 1024
#define WARPS (THREADS / 32)
#define MAXIN 3072
#define NEG_INF (-1e30f)
#define RMS_EPS 1e-6f

// ---- KV codec parameters (must match the format picked by the ppl gate) ----
#ifndef KBITS
#define KBITS 8
#endif
#ifndef KMODE_T
#define KMODE_T 1      // 1: per-token K groups; 0: per-channel K groups
#endif
#ifndef KGRP
#define KGRP 128       // KMODE_T: dims/group. KMODE_C: tokens/group.
#endif
#ifndef KRES
#define KRES 0         // K fp16 residual window (KMODE_C only; mult of KGRP)
#endif
#ifndef VBITS
#define VBITS 4
#endif
#ifndef VGRP
#define VGRP 128
#endif
#ifndef VRES
#define VRES 0
#endif
#define KQROWB (KVROWS * KBITS / 8)       // packed K bytes/token/layer
#define VQROWB (KVROWS * VBITS / 8)       // packed V bytes/token/layer
#define KNGROUP (MAXPOS / KGRP)           // KMODE_C K groups/layer
#define VMETA_PT (KVROWS / VGRP)          // V meta half2 per token
#define RCAP (KRES + KGRP)                // K ring slots (KMODE_C)

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
  __half *qkv, *oo, *gu, *dout;
  uint8_t* kq;            // packed K codes [layer][tok][kvh][KBITS*128/8]
  __half2* kmeta;         // K scale/off (KMODE_C: [layer][grp][kvh][128] half2;
                          //              KMODE_T: [layer][tok][kvh][128/KGRP] half2)
  __half* kring;          // KMODE_C fp16 residual [layer][RCAP][KVROWS]
  uint8_t* vq;            // packed V codes [layer][tok][kvh][VBITS*128/8]
  __half2* vmeta;         // V scale/off [layer][tok][kvh][128/VGRP] half2
  __half* vring;          // VRES>0: fp16 residual [layer][RCAP][KVROWS]
  float* part;          // [NATTN][130]
  float* logits;
  float* argp;          // [NARGP][2]
  int* tok;
  int* tok_hist;
  int* pos;             // device position base (incremented by steps at end)
  int* bar;             // global barrier counter
  int steps;
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
__shared__ __align__(16) __half xloc[HID];  // private residual stream
__shared__ __align__(16) union {
  uint4 sx4[MAXIN / 8];    // transposed activation: 384 units = 6 KB
  float wpart[WARPS][130]; // attention warp partials = 16.6 KB
} u;
__shared__ __align__(16) __half qhead[128];
__shared__ __align__(16) __half kvloc[256];  // k_pos | v_pos of the pos row
__shared__ float red[WARPS];
__shared__ __half2 kfinmeta[128];  // K-group finalize: channel scale/off staging
__shared__ __half kvstg[128];      // append staging (warp 8, KMODE_T K row)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ float block_sum(float v) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = threadIdx.x < WARPS ? red[threadIdx.x] : 0.f;
  if (warp == 0) v = warp_sum(v);
  if (threadIdx.x == 0) red[0] = v;
  __syncthreads();
  v = red[0];
  __syncthreads();
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

__device__ __forceinline__ float chunk_xsum(int c, int cpr) {
  float s = 0.f;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    uint4 v = u.sx4[t * cpr + c];
    __half2 a = as_h2(v.x), b = as_h2(v.y), cc = as_h2(v.z), d = as_h2(v.w);
    s += __low2float(a) + __high2float(a) + __low2float(b) + __high2float(b) +
         __low2float(cc) + __high2float(cc) + __low2float(d) + __high2float(d);
  }
  return s;
}

__device__ __forceinline__ float dot4(uint4 q, __half2 meta, const uint4* xq,
                                      float xsum) {
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
  uint32_t xw[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    xw[4 * i + 0] = xq[i].x;
    xw[4 * i + 1] = xq[i].y;
    xw[4 * i + 2] = xq[i].z;
    xw[4 * i + 3] = xq[i].w;
  }
  const __half2 magic = as_h2(0x64006400u);
  __half2 acc[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) acc[a] = __float2half2_rn(0.f);
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __half2 v = __hsub2(
          as_h2(((w[i] >> (4 * j)) & 0x000F000Fu) | 0x64006400u), magic);
      acc[j] = __hfma2(v, as_h2(xw[4 * i + j]), acc[j]);
    }
  const __half2 s2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
  const float s = __low2float(s2) + __high2float(s2);
  return s * __low2float(meta) + __high2float(meta) * xsum;
}

// block-owned contiguous row range [r0, r1); warps stride WARPS, R rows deep.
template <int NC, int R>
__device__ __forceinline__ void gemv2(const W2& w, int r0, int r1) {
  const int cpr = w.n_in >> 5, ngrp = cpr >> 2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float xs[NC];
#pragma unroll
  for (int t = 0; t < NC; ++t) xs[t] = chunk_xsum(lane + 32 * t, cpr);
  for (int row0 = r0 + warp; row0 < r1; row0 += R * WARPS) {
    uint4 q[R][NC];
    __half2 mt[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r * WARPS < r1)
#pragma unroll
        for (int t = 0; t < NC; ++t) {
          const int c = lane + 32 * t;
          q[r][t] = __ldg(w.codes + (int64_t)(row0 + r * WARPS) * cpr + c);
          mt[r][t] =
              __ldg(w.meta + (int64_t)(row0 + r * WARPS) * ngrp + (c >> 2));
        }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
    for (int t = 0; t < NC; ++t) {
      const int c = lane + 32 * t;
      uint4 xq[4];
#pragma unroll
      for (int t2 = 0; t2 < 4; ++t2) xq[t2] = u.sx4[t2 * cpr + c];
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (row0 + r * WARPS < r1) acc[r] += dot4(q[r][t], mt[r][t], xq, xs[t]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int r = 0; r < R; ++r)
        acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    if (lane < R && row0 + lane * WARPS < r1) {
      float v = acc[0];
#pragma unroll
      for (int r = 1; r < R; ++r) v = lane == r ? acc[r] : v;
      const int row = row0 + lane * WARPS;
      if (w.yf) w.yf[row] = v;
      else w.y[row] = __float2half(v);
    }
  }
}

__device__ __forceinline__ void gemv_run(const W2& w, int blk) {
  const int chunk = (w.n_out + NCTA - 1) / NCTA;
  const int r0 = blk * chunk, r1 = min(w.n_out, r0 + chunk);
  if (r0 >= r1) return;
  const int nc = w.n_in >> 10;
  if (nc == 1) gemv2<1, RPW2>(w, r0, r1);
  else if (nc == 2) gemv2<2, RPW2>(w, r0, r1);
  else gemv2<3, RPW2>(w, r0, r1);
}

// ---- prologues: produce u.sx4 for the following GEMV ------------------------

#define SX4_WRITE(idx, halfs8)                          \
  do {                                                  \
    uint4 _pk;                                          \
    _pk.x = *reinterpret_cast<uint32_t*>(&(halfs8)[0]); \
    _pk.y = *reinterpret_cast<uint32_t*>(&(halfs8)[2]); \
    _pk.z = *reinterpret_cast<uint32_t*>(&(halfs8)[4]); \
    _pk.w = *reinterpret_cast<uint32_t*>(&(halfs8)[6]); \
    u.sx4[idx] = _pk;                                   \
  } while (0)

// residual-stream update + rms^-1 (xloc += src, or xloc = emb[tok]).
__device__ float xresidual(const P2& p, const __half* src, bool embed) {
  float ss = 0.f;
  const int tid = threadIdx.x;
  if (embed) {
    const __half* erow = p.emb + (int64_t)p.tok[0] * HID;
    for (int i = tid * 8; i < HID; i += THREADS * 8) {
      uint4 v = *reinterpret_cast<const uint4*>(erow + i);
      *reinterpret_cast<uint4*>(xloc + i) = v;
      __half2 a = as_h2(v.x), b = as_h2(v.y), c = as_h2(v.z), d = as_h2(v.w);
      float2 fa = __half22float2(a), fb = __half22float2(b);
      float2 fc = __half22float2(c), fd = __half22float2(d);
      ss += fa.x * fa.x + fa.y * fa.y + fb.x * fb.x + fb.y * fb.y +
            fc.x * fc.x + fc.y * fc.y + fd.x * fd.x + fd.y * fd.y;
    }
  } else {
    for (int i = tid * 8; i < HID; i += THREADS * 8) {
      uint4 xv = *reinterpret_cast<const uint4*>(xloc + i);
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
      *reinterpret_cast<uint4*>(xloc + i) = *reinterpret_cast<uint4*>(o);
    }
  }
  ss = block_sum(ss);
  return rsqrtf(ss / HID + RMS_EPS);
}

// sx = xloc * r * wrow, transposed; cpr = HID/32 = 32 -> 128 units.
__device__ void stage_norm(const __half* wrow, float r) {
  const int cpr = HID >> 5;
  for (int idx = threadIdx.x; idx < cpr * 4; idx += THREADS) {
    const int uu = idx / cpr, c = idx % cpr;
    const int i = (c * 4 + uu) * 8;
    uint4 xv = *reinterpret_cast<const uint4*>(xloc + i);
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

// qkv/gu/lm prologue: fold residual (+embed) + RMSNorm into sx.
__device__ void pro_norm(const P2& p, const __half* src, bool embed, int wsel) {
  const float r = xresidual(p, src, embed);
  stage_norm(p.norms + (int64_t)wsel * HID, r);
}

// o-proj prologue: combine the 32 attention partials into sx (in = 2048).
// Head h has partials 2h (first half of positions) and 2h+1 (second half).
__device__ void pro_attnc(const P2& p) {
  const int cpr = QROWS >> 5;  // 64
  for (int idx = threadIdx.x; idx < cpr * 4; idx += THREADS) {
    const int uu = idx / cpr, c = idx % cpr;
    const int i = (c * 4 + uu) * 8;
    const int h = i >> 7;  // 8-elem groups never straddle a head boundary
    const float* pa = p.part + (h * 2 + 0) * 130;
    const float* pb = p.part + (h * 2 + 1) * 130;
    const float m0 = pa[129], m1 = pb[129];
    const float M = fmaxf(m0, m1);
    const float e0 = m0 == NEG_INF ? 0.f : __expf(m0 - M);
    const float e1 = m1 == NEG_INF ? 0.f : __expf(m1 - M);
    const float l = e0 * pa[128] + e1 * pb[128];
    const float inv = l > 0.f ? 1.f / l : 0.f;
    __half o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = (i & 127) + j;
      o[j] = __float2half((e0 * pa[e] + e1 * pb[e]) * inv);
    }
    SX4_WRITE(idx, o);
  }
  __syncthreads();
}

// down-proj prologue: sx = silu(gate) * up from gu[6144]. in = 3072.
__device__ void pro_silu(const P2& p) {
  const int cpr = INTER >> 5;  // 96
  for (int idx = threadIdx.x; idx < cpr * 4; idx += THREADS) {
    const int uu = idx / cpr, c = idx % cpr;
    const int i = (c * 4 + uu) * 8;
    uint4 gv = *reinterpret_cast<const uint4*>(p.gu + i);
    uint4 uv = *reinterpret_cast<const uint4*>(p.gu + INTER + i);
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

// raw transposed copy for the matvec-only stage kernel.
__device__ void pro_raw(const __half* x, int n_in) {
  const int cpr = n_in >> 5;
  for (int idx = threadIdx.x; idx < cpr * 4; idx += THREADS) {
    const int uu = idx / cpr, c = idx % cpr;
    u.sx4[idx] = *reinterpret_cast<const uint4*>(x + (c * 4 + uu) * 8);
  }
  __syncthreads();
}

// ============================ attention ======================================
// head-wise RMS norm over 128 elems + rope pairs (d, d+64), one warp.
// Qwen3 rope: for d < 64, (q[d], q[d+64]) -> (q[d]*cos - q[d+64]*sin,
// q[d+64]*cos + q[d]*sin) with cos/sin = rope[pos][d], rope[pos][128+d].
// Lane owns normed elems lane*4..+3; partners for lanes <16 live at lane+16.
__device__ __forceinline__ void norm_rope_row(const P2& p, int layer, bool krow,
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

// ========================== KV codec helpers ==============================
// Lane-4 quantizer: quantize v[4] as part of a group of GRP consecutive dims
// (GRP/4 contiguous lanes hold the whole group). Min/max asymmetric RTN;
// meta = fp16 (scale, off), off = -zp*scale. Identical math to
// kvcodec/codec_lm09._rtn_rows.
template <int BITS, int GRP>
__device__ __forceinline__ void quant4(const float v[4], uint32_t& packed,
                                       __half2& meta) {
  float lo = fminf(fminf(v[0], v[1]), fminf(v[2], v[3]));
  float hi = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
  const int sw = GRP / 4;                 // lanes per group
#pragma unroll
  for (int o = 1; o < sw; o <<= 1) {
    lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
    hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
  }
  const int qmax = (1 << BITS) - 1;
  const float sc = (float)__float2half(fmaxf((hi - lo) / qmax, 1e-10f));
  const float zp = fminf(fmaxf(rintf(-lo / sc), 0.f), (float)qmax);
  const float off = (float)__float2half(-zp * sc);
  uint32_t q = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = (int)fminf(fmaxf(rintf(v[j] / sc) + zp, 0.f), (float)qmax);
    q |= (uint32_t)c << (BITS * j);
  }
  packed = q;
  meta = __floats2half2_rn(sc, off);
}

template <int BITS>
__device__ __forceinline__ void unpack4(uint32_t q, float c[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
    c[j] = (float)((q >> (BITS * j)) & ((1u << BITS) - 1u));
}

// Quantizer probe: same device routine as the append warps, fed identical
// fp16 inputs to compare packed codes and metadata against codec_lm09.py.
__global__ void kv_quant_probe(const __half* src, uint8_t* dst,
                               __half2* meta, int rows, int bits) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const uint2 raw = *reinterpret_cast<const uint2*>(src + row * 128 + lane * 4);
  const float2 a = __half22float2(as_h2(raw.x));
  const float2 b = __half22float2(as_h2(raw.y));
  const float vals[4] = {a.x, a.y, b.x, b.y};
  uint32_t codes; __half2 group;
  if (bits == 8) {
    quant4<8, 128>(vals, codes, group);
    *reinterpret_cast<uint32_t*>(dst + row * 128 + lane * 4) = codes;
  } else {
    quant4<4, 128>(vals, codes, group);
    *reinterpret_cast<uint16_t*>(dst + row * 64 + lane * 2) =
        (uint16_t)codes;
  }
  if (lane == 0) meta[row] = group;
}

#if !KMODE_T
// KMODE_C: quantize K group [g0, g0+KGRP) of kv head `kvh` from the fp16 ring.
// Warps 10-13: per-channel scale/off (one channel/lane) -> SMEM + global meta.
// Warps 14-31: pack codes, token u strided over warps, channel quad = lane.
__device__ void k_finalize(const P2& p, int layer, int kvh, int g0) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __half* ring = p.kring + (int64_t)layer * RCAP * KVROWS + kvh * 128;
  if (warp >= 10 && warp < 14) {
    const int ch = (warp - 10) * 32 + lane;
    float lo = 1e30f, hi = -1e30f;
    for (int u = 0; u < KGRP; ++u) {
      const float x =
          __half2float(ring[(int64_t)((g0 + u) % RCAP) * KVROWS + ch]);
      lo = fminf(lo, x);
      hi = fmaxf(hi, x);
    }
    const int qmax = (1 << KBITS) - 1;
    const float sc = (float)__float2half(fmaxf((hi - lo) / qmax, 1e-10f));
    const float zp = fminf(fmaxf(rintf(-lo / sc), 0.f), (float)qmax);
    const float off = (float)__float2half(-zp * sc);
    const __half2 m = __floats2half2_rn(sc, off);
    kfinmeta[ch] = m;
    p.kmeta[((int64_t)layer * KNGROUP + g0 / KGRP) * KVROWS + kvh * 128 + ch] =
        m;
  }
  __syncthreads();
  if (warp >= 14) {
    for (int u = warp - 14; u < KGRP; u += WARPS - 14) {
      const int cq = lane;                  // channel quad
      uint32_t q = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const __half2 m = kfinmeta[cq * 4 + j];
        const float sc = __low2float(m), off = __high2float(m);
        const float x = __half2float(
            ring[(int64_t)((g0 + u) % RCAP) * KVROWS + cq * 4 + j]);
        const float zpk = rintf(-off / sc);   // recover zp (exact for zp<=qmax)
        const int c = (int)fminf(fmaxf(rintf(x / sc) + zpk, 0.f),
                                 (float)((1 << KBITS) - 1));
        q |= (uint32_t)c << (KBITS * j);
      }
      if (KBITS == 4)
        *reinterpret_cast<uint16_t*>(
            p.kq + (int64_t)layer * MAXPOS * KQROWB + (g0 + u) * KQROWB +
            kvh * 64 + cq * 2) = (uint16_t)q;
      else
        *reinterpret_cast<uint32_t*>(
            p.kq + (int64_t)layer * MAXPOS * KQROWB + (g0 + u) * KQROWB +
            kvh * 128 + cq * 4) = q;
    }
  }
}
#endif  // !KMODE_T

// per-CTA attention: head = cta>>1, position half = cta&1; warp partials ->
// SMEM -> one global partial row per CTA in p.part[cta].
__device__ void attn2(const P2& p, int layer, int pos, int cta) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int h = cta >> 1, kvh = h >> 1, half = cta & 1;
  const int npos = pos + 1;
  const int mid = (npos + 1) >> 1;         // half0: [0,mid), half1: [mid,npos)
  const int a0 = half ? mid : 0, a1 = half ? npos : mid;
  // packed-region boundaries: K [0,nqk), V [0,nqv); tails are fp16 ring
  // (KMODE_C/VRES>0) or packed anyway (KMODE_T/VRES==0).
#if KMODE_T
  const int nqk = npos;                     // all K packed
#else
  const int nqk = max(0, (npos - KRES) / KGRP * KGRP);
#endif
#if VRES > 0
  const int nqv = max(0, npos - VRES);
#else
  const int nqv = npos;
#endif

  // prologue: norm+rope this head's q; CTAs 0..7 append their kv head's
  // quantized k/v rows; the half whose range covers pos builds it locally.
  if (warp == 0)
    norm_rope_row(p, layer, false, pos, p.qkv + h * 128, qhead);
  // appenders: CTA cta = 4*kvh covers kv head cta/4.
  const int appender = (cta % 4 == 0) ? cta / 4 : -1;
#if KMODE_T
  if (appender >= 0 && warp == 8) {
    // norm+rope into staging, then quantize per KGRP-dim group.
    norm_rope_row(p, layer, true, pos, p.qkv + QROWS + appender * 128, kvstg);
    __syncwarp();
    const uint2 uq = *reinterpret_cast<const uint2*>(kvstg + lane * 4);
    const float2 f0 = __half22float2(as_h2(uq.x));
    const float2 f1 = __half22float2(as_h2(uq.y));
    const float v[4] = {f0.x, f0.y, f1.x, f1.y};
    uint32_t qc; __half2 mt;
    quant4<KBITS, KGRP>(v, qc, mt);
    const int64_t row = (int64_t)layer * MAXPOS * KQROWB + pos * KQROWB +
                        appender * (128 * KBITS / 8);
    if (KBITS == 4)
      *reinterpret_cast<uint16_t*>(p.kq + row + lane * 2) = (uint16_t)qc;
    else
      *reinterpret_cast<uint32_t*>(p.kq + row + lane * 4) = qc;
    const int lpg = KGRP / 4;
    if (lane % lpg == 0)
      p.kmeta[((int64_t)layer * MAXPOS + pos) * (KVROWS / KGRP) +
              appender * (128 / KGRP) + lane / lpg] = mt;
  }
#else
  if (appender >= 0 && warp == 8)
    norm_rope_row(p, layer, true, pos, p.qkv + QROWS + appender * 128,
                  p.kring + ((int64_t)layer * RCAP + pos % RCAP) * KVROWS +
                      appender * 128);
  // group [b-KGRP, b) slides out of the residual window when
  // b = npos + 1 - KRES hits a group boundary: quantize it now so readers
  // find it packed from the next step on.
  if (appender >= 0) {
    const int b = npos + 1 - KRES;
    if (b > 0 && b % KGRP == 0) k_finalize(p, layer, appender, b - KGRP);
  }
#endif
  if (appender >= 0 && warp == 9) {
    // one 4-dim quad per lane: all 32 lanes cover the 128-dim V row
    const uint2 uvv = *reinterpret_cast<const uint2*>(
        p.qkv + QROWS + KVROWS + appender * 128 + lane * 4);
#if VRES > 0
    *reinterpret_cast<uint2*>(
        p.vring + ((int64_t)layer * RCAP + pos % RCAP) * KVROWS +
        appender * 128 + lane * 4) = uvv;
#endif
    const float2 fa = __half22float2(as_h2(uvv.x));
    const float2 fb = __half22float2(as_h2(uvv.y));
    const float v4[4] = {fa.x, fa.y, fb.x, fb.y};
    uint32_t qc; __half2 mt;
    quant4<VBITS, VGRP>(v4, qc, mt);
    if (VBITS == 4)
      *reinterpret_cast<uint16_t*>(
          p.vq + (int64_t)layer * MAXPOS * VQROWB + pos * VQROWB +
          appender * 64 + lane * 2) = (uint16_t)qc;
    else
      *reinterpret_cast<uint32_t*>(
          p.vq + (int64_t)layer * MAXPOS * VQROWB + pos * VQROWB +
          appender * 128 + lane * 4) = qc;
    const int lpg = VGRP / 4;
    if (lane % lpg == 0)
      p.vmeta[((int64_t)layer * MAXPOS + pos) * (KVROWS / VGRP) +
              appender * (128 / VGRP) + (lane * 4) / VGRP] = mt;
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

  // each warp takes a contiguous sub-range of [a0, a1)
  float* po = u.wpart[warp];
  const int len = a1 - a0;
  const int per = (len + WARPS - 1) / WARPS;
  const int w0 = a0 + warp * per, w1 = min(a1, w0 + per);
  const uint2 uq = *reinterpret_cast<const uint2*>(qhead + lane * 4);
  const __half2 qr0 = as_h2(uq.x), qr1 = as_h2(uq.y);
  const float q0 = __low2float(qr0), q1 = __high2float(qr0);
  const float q2 = __low2float(qr1), q3 = __high2float(qr1);
  const float qsum = q0 + q1 + q2 + q3;    // per-token K off term
  const int64_t kbase = (int64_t)layer * MAXPOS;
  float m = NEG_INF, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
#if !KMODE_T
  // per-K-group meta hoisted: qs[j] = q_j*scale_j, qz = sum(q*off)
  __half2 km[4];
  float qs[4] = {0, 0, 0, 0}, qz = 0.f;
  int gcur = -1;
#endif
  for (int t = w0; t < w1; ++t) {
    float s;
    if (t == pos || t >= nqk) {
      // fp16 K: pos row from kvloc, KMODE_C residual tail from the ring
      const __half* krow =
          t == pos ? kvloc
                   : p.kring + ((int64_t)layer * RCAP + t % RCAP) * KVROWS +
                         kvh * 128;
      uint2 uk = *reinterpret_cast<const uint2*>(krow + lane * 4);
      __half2 prod = __hmul2(as_h2(uk.x), qr0);
      prod = __hfma2(as_h2(uk.y), qr1, prod);
      s = __low2float(prod) + __high2float(prod);
    } else {
#if KMODE_T
      // per-token packed K: s = sc*sum(q*c) + off*qsum (one warp reduce)
      uint32_t uk;
      if (KBITS == 4)
        uk = *reinterpret_cast<const uint16_t*>(
            p.kq + kbase * KQROWB + t * KQROWB + kvh * 64 + lane * 2);
      else
        uk = *reinterpret_cast<const uint32_t*>(
            p.kq + kbase * KQROWB + t * KQROWB + kvh * 128 + lane * 4);
      const __half2 mt = p.kmeta[(kbase + t) * (KVROWS / KGRP) +
                                 kvh * (128 / KGRP) + (lane * 4) / KGRP];
      float c[4]; unpack4<KBITS>(uk, c);
      s = (q0 * c[0] + q1 * c[1] + q2 * c[2] + q3 * c[3]) * __low2float(mt) +
          __high2float(mt) * qsum;
#else
      // per-channel packed K: meta per (group, channel), hoisted per group
      const int g = t / KGRP;
      if (g != gcur) {
        gcur = g;
        const uint4 m4 = *reinterpret_cast<const uint4*>(
            p.kmeta + ((int64_t)layer * KNGROUP + g) * KVROWS + kvh * 128 +
            lane * 4);
        km[0] = as_h2(m4.x); km[1] = as_h2(m4.y);
        km[2] = as_h2(m4.z); km[3] = as_h2(m4.w);
        qs[0] = q0 * __low2float(km[0]); qs[1] = q1 * __low2float(km[1]);
        qs[2] = q2 * __low2float(km[2]); qs[3] = q3 * __low2float(km[3]);
        qz = q0 * __high2float(km[0]) + q1 * __high2float(km[1]) +
             q2 * __high2float(km[2]) + q3 * __high2float(km[3]);
      }
      uint32_t uk;
      if (KBITS == 4)
        uk = *reinterpret_cast<const uint16_t*>(
            p.kq + kbase * KQROWB + t * KQROWB + kvh * 64 + lane * 2);
      else
        uk = *reinterpret_cast<const uint32_t*>(
            p.kq + kbase * KQROWB + t * KQROWB + kvh * 128 + lane * 4);
      float c[4]; unpack4<KBITS>(uk, c);
      s = qs[0] * c[0] + qs[1] * c[1] + qs[2] * c[2] + qs[3] * c[3] + qz;
#endif
    }
    s = warp_sum(s) * 0.08838834764831845f;
    const float mn = fmaxf(m, s);
    const float corr = m == NEG_INF ? 0.f : __expf(m - mn);
    const float e = __expf(s - mn);
    float v0, v1, v2, v3;
    if (t == pos || t >= nqv) {
      const __half* vrow =
          t == pos ? kvloc + 128
                   : p.vring + ((int64_t)layer * RCAP + t % RCAP) * KVROWS +
                         kvh * 128;
      uint2 uv = *reinterpret_cast<const uint2*>(vrow + lane * 4);
      const float2 f0 = __half22float2(as_h2(uv.x));
      const float2 f1 = __half22float2(as_h2(uv.y));
      v0 = f0.x; v1 = f0.y; v2 = f1.x; v3 = f1.y;
    } else {
      uint32_t uk;
      if (VBITS == 4)
        uk = *reinterpret_cast<const uint16_t*>(
            p.vq + kbase * VQROWB + t * VQROWB + kvh * 64 + lane * 2);
      else
        uk = *reinterpret_cast<const uint32_t*>(
            p.vq + kbase * VQROWB + t * VQROWB + kvh * 128 + lane * 4);
      const __half2 mt = p.vmeta[(kbase + t) * (KVROWS / VGRP) +
                                 kvh * (128 / VGRP) + (lane * 4) / VGRP];
      float c[4]; unpack4<VBITS>(uk, c);
      const float sc = __low2float(mt), of = __high2float(mt);
      v0 = c[0] * sc + of; v1 = c[1] * sc + of;
      v2 = c[2] * sc + of; v3 = c[3] * sc + of;
    }
    acc[0] = acc[0] * corr + e * v0;
    acc[1] = acc[1] * corr + e * v1;
    acc[2] = acc[2] * corr + e * v2;
    acc[3] = acc[3] * corr + e * v3;
    l = l * corr + e;
    m = mn;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) po[lane * 4 + j] = acc[j];
  if (lane == 0) { po[128] = l; po[129] = m; }
  __syncthreads();

  // merge 32 warp partials -> one global partial per CTA (tid<130 covers
  // 128 acc elems + l + m).
  if (threadIdx.x < 130) {
    const int e = threadIdx.x;
    float M = NEG_INF;
    if (e < 130) {
#pragma unroll 4
      for (int w = 0; w < WARPS; ++w) M = fmaxf(M, u.wpart[w][129]);
    }
    if (M == NEG_INF) {
      if (e < 128) p.part[cta * 130 + e] = 0.f;
      else p.part[cta * 130 + e] = e == 128 ? 0.f : NEG_INF;
    } else if (e < 128) {
      float s = 0.f;
      for (int w = 0; w < WARPS; ++w) {
        const float mw = u.wpart[w][129];
        if (mw > NEG_INF) s += __expf(mw - M) * u.wpart[w][e];
      }
      p.part[cta * 130 + e] = s;
    } else if (e == 128) {
      float s = 0.f;
      for (int w = 0; w < WARPS; ++w) {
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
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int per = (VOCAB + NARGP - 1) / NARGP;
  const int a0 = cta * per, a1 = min(VOCAB, a0 + per);
  float bv = NEG_INF;
  int bi = a0;
  for (int i = a0 + threadIdx.x; i < a1; i += THREADS) {
    const float v = p.logits[i];
    if (v > bv) { bv = v; bi = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  __shared__ float sv[WARPS];
  __shared__ int si[WARPS];
  if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    bv = lane < WARPS ? sv[lane] : NEG_INF;
    bi = lane < WARPS ? si[lane] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { p.argp[2 * cta] = bv; p.argp[2 * cta + 1] = (float)bi; }
  }
}

__device__ void argc2(const P2& p, int pos) {
  // CTA 0 only: merge NARGP=36 partials (threads <36 each hold one slice).
  __shared__ float sv[WARPS];
  __shared__ int si[WARPS];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float bv = NEG_INF;
  int bi = 0;
  if (threadIdx.x < NARGP) {
    bv = p.argp[2 * threadIdx.x];
    bi = (int)p.argp[2 * threadIdx.x + 1];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    bv = lane < WARPS ? sv[lane] : NEG_INF;
    bi = lane < WARPS ? si[lane] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { p.tok[0] = bi; p.tok_hist[pos] = bi; }
  }
}


// ============================ main kernel ====================================
__global__ void __launch_bounds__(THREADS, 1) megakv_kernel(const P2* pp) {
  const P2& p = *pp;
  const int cta = blockIdx.x;
  const int base = p.pos[0];
  int expect = NCTA;
  int ph = 0;
  for (int step = 0; step < p.steps; ++step) {
    const int pos = base + step;
    for (int l = 0; l < NLAY; ++l) {
      // PH qkv: residual (embed for l==0, else += dout) + rmsnorm ln1 -> sx
      pro_norm(p, p.dout, l == 0, l);
      gemv_run(p.w[l * 4 + 0], cta);
      gbar(p.bar, expect);
      ++ph;
      // PH attn: q/k norm + rope + kv append + attention -> part[]
      if (cta < NATTN) attn2(p, l, pos, cta);
      gbar(p.bar, expect);
      ++ph;
      // PH o: combine partials -> sx; o GEMV -> oo
      pro_attnc(p);
      gemv_run(p.w[l * 4 + 1], cta);
      gbar(p.bar, expect);
      ++ph;
      // PH gu: residual += oo + rmsnorm ln2 -> sx; gu GEMV -> gu
      pro_norm(p, p.oo, false, NLAY + l);
      gemv_run(p.w[l * 4 + 2], cta);
      gbar(p.bar, expect);
      ++ph;
      // PH down: silu(gate)*up -> sx; down GEMV -> dout
      pro_silu(p);
      gemv_run(p.w[l * 4 + 3], cta);
      gbar(p.bar, expect);
      ++ph;
    }
    // lm_head: residual += dout + final norm -> sx; logits (f32)
    pro_norm(p, p.dout, false, 4 * NLAY);
    gemv_run(p.w[W_LM], cta);
    gbar(p.bar, expect);
    ++ph;
    argp2(p, cta);
    gbar(p.bar, expect);
    ++ph;
    if (cta == 0) argc2(p, pos);
    gbar(p.bar, expect);
    ++ph;
  }
  if (cta == 0 && threadIdx.x == 0) atomicAdd(p.pos, p.steps);
}



// ================== stage test: matvec-only persistent loops ================
// Kill-line probe: 113 GEMVs back to back at 36 CTAs x 1024 threads.

// empty-phase loop: pure grid-barrier cost, N barriers inside one launch.
__global__ void __launch_bounds__(THREADS, 1)
kv_barrieronly(const P2* pp, int n) {
  const P2& p = *pp;
  int expect = NCTA;
  for (int i = 0; i < n; ++i) gbar(p.bar, expect);
}

// _bar variant inserts one grid barrier per GEMV (the real dependency cost).
__global__ void __launch_bounds__(THREADS, 1)
kv_matvec(const P2* pp) {
  const P2& p = *pp;
  const int cta = blockIdx.x;
  for (int l = 0; l < NLAY; ++l)
    for (int k = 0; k < 4; ++k) {
      const W2& w = p.w[l * 4 + k];
      pro_raw(w.x, w.n_in);
      gemv_run(w, cta);
    }
  const W2& lm = p.w[W_LM];
  pro_raw(lm.x, lm.n_in);
  gemv_run(lm, cta);
}

__global__ void __launch_bounds__(THREADS, 1)
kv_matvec_bar(const P2* pp) {
  const P2& p = *pp;
  const int cta = blockIdx.x;
  int expect = NCTA;
  for (int l = 0; l < NLAY; ++l)
    for (int k = 0; k < 4; ++k) {
      const W2& w = p.w[l * 4 + k];
      pro_raw(w.x, w.n_in);
      gemv_run(w, cta);
      gbar(p.bar, expect);
    }
  const W2& lm = p.w[W_LM];
  pro_raw(lm.x, lm.n_in);
  gemv_run(lm, cta);
  gbar(p.bar, expect);
}

// ============================ host side =====================================
static P2 h_p2;
static P2* d_p2 = nullptr;
static int* d_pos2 = nullptr;
static int* d_bar2 = nullptr;
static cudaStream_t g_s2 = nullptr;

static P2* dev2() {
  if (!d_p2) cudaMalloc(&d_p2, sizeof(P2));
  return d_p2;
}

// bufs: xn, qkv, oo, gu, dout, kq, kmeta, kring, vq, vmeta, vring,
//       part, logits, argp, pos, bar, xpad
extern "C" int mkv_init(const int64_t* bufs, const int64_t* codes,
                        const int64_t* meta, int64_t emb, int64_t norms,
                        int64_t rope, int64_t tok, int64_t tok_hist) {
  memset(&h_p2, 0, sizeof(h_p2));
  h_p2.w[0].x = (const __half*)bufs[0];  // xn: generic input for stage test
  h_p2.qkv = (__half*)bufs[1];
  h_p2.oo = (__half*)bufs[2];
  h_p2.gu = (__half*)bufs[3];
  h_p2.dout = (__half*)bufs[4];
  h_p2.kq = (uint8_t*)bufs[5];
  h_p2.kmeta = (__half2*)bufs[6];
  h_p2.kring = (__half*)bufs[7];
  h_p2.vq = (uint8_t*)bufs[8];
  h_p2.vmeta = (__half2*)bufs[9];
  h_p2.vring = (__half*)bufs[10];
  h_p2.part = (float*)bufs[11];
  h_p2.logits = (float*)bufs[12];
  h_p2.argp = (float*)bufs[13];
  h_p2.pos = (int*)bufs[14];
  h_p2.bar = (int*)bufs[15];
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
        case 1: w.x = (const __half*)bufs[16]; w.y = h_p2.oo; w.n_in = QROWS; w.n_out = HID; break;
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

extern "C" int mkv_pos_set(int v) {
  return cudaMemcpy((void*)h_p2.pos, &v, sizeof(int), cudaMemcpyHostToDevice) ==
                 cudaSuccess
             ? 0
             : -1;
}

extern "C" int mkv_sync() {
  return cudaStreamSynchronize(g_s2) == cudaSuccess ? 0
                                                    : (int)cudaGetLastError();
}

extern "C" int mkv_mega(int steps) {
  h_p2.steps = steps;
  cudaMemcpy(dev2(), &h_p2, sizeof(P2), cudaMemcpyHostToDevice);
  cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
  megakv_kernel<<<NCTA, THREADS, 0, g_s2>>>(dev2());
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int mkv_time_mega(int steps, int iters, int pos0, float* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  h_p2.steps = steps;
  cudaMemcpy(dev2(), &h_p2, sizeof(P2), cudaMemcpyHostToDevice);
  for (int i = 0; i < iters; ++i) {
    mkv_pos_set(pos0);
    cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
    cudaEventRecord(e0, g_s2);
    megakv_kernel<<<NCTA, THREADS, 0, g_s2>>>(dev2());
    cudaEventRecord(e1, g_s2);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    out[i] = ms / steps;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int mkv_quant_probe(int64_t src, int64_t dst, int64_t meta,
                                 int rows, int bits) {
  if (bits != 4 && bits != 8) return -1;
  kv_quant_probe<<<(rows + 3) / 4, 128>>>(
      (const __half*)src, (uint8_t*)dst, (__half2*)meta, rows, bits);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---- stage-test timers (113 GEMVs, persistent loop) -------------------------
extern "C" float mkv_time_matvec(int iters, int use_bar) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  // warm
  if (use_bar) {
    cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
    kv_matvec_bar<<<NCTA, THREADS, 0, g_s2>>>(dev2());
  } else {
    kv_matvec<<<NCTA, THREADS, 0, g_s2>>>(dev2());
  }
  cudaStreamSynchronize(g_s2);
  cudaEventRecord(e0, g_s2);
  for (int i = 0; i < iters; ++i) {
    if (use_bar) {
      cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
      kv_matvec_bar<<<NCTA, THREADS, 0, g_s2>>>(dev2());
    } else {
      kv_matvec<<<NCTA, THREADS, 0, g_s2>>>(dev2());
    }
  }
  cudaEventRecord(e1, g_s2);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

extern "C" float mkv_time_barriers(int n, int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
  kv_barrieronly<<<NCTA, THREADS, 0, g_s2>>>(dev2(), n);
  cudaStreamSynchronize(g_s2);
  cudaEventRecord(e0, g_s2);
  for (int i = 0; i < iters; ++i) {
    cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
    kv_barrieronly<<<NCTA, THREADS, 0, g_s2>>>(dev2(), n);
  }
  cudaEventRecord(e1, g_s2);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}



extern "C" int mkv_matvec_run() {
  kv_matvec<<<NCTA, THREADS, 0, g_s2>>>(dev2());
  cudaStreamSynchronize(g_s2);
  return (int)cudaGetLastError();
}

extern "C" int mkv_matvec_run_bar() {
  cudaMemsetAsync(h_p2.bar, 0, sizeof(int), g_s2);
  kv_matvec_bar<<<NCTA, THREADS, 0, g_s2>>>(dev2());
  cudaStreamSynchronize(g_s2);
  return (int)cudaGetLastError();
}
