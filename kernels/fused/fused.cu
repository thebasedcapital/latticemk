// LM-08: fused separate-kernel INT4 decode engine for Qwen3-0.6B-Base,
// batch 1, sm_75. One CUDA graph per decode step; 169 nodes vs LM-03's ~285.
//
// Fusions vs the LM-03 separate-kernel engine (kernels/megakernel/mega.cu):
//   * RMSNorm -> GEMV prologue: each block re-normalizes the 1024-wide
//     activation while staging it into SMEM (PRO_NORM). Removes 2 norm
//     kernels/layer and the xn buffer.
//   * embed -> L0 qkv prologue (PRO_EMBNORM): x = emb[tok] is read directly,
//     removing the tick + embed kernels. pos is advanced by the lm_head
//     argmax epilogue instead.
//   * q_norm/k_norm + RoPE + KV append -> attention prologue: every attention
//     block re-normalizes/ropes the 16 q heads it needs (SMEM); the block
//     owning position `pos` additionally norms/ropes K and appends K,V into
//     the cache.
//   * attention split-K combine -> separate k_attnc kernel (measured faster
//     than a last-block in-kernel combine, which serializes behind stragglers)
//   * SiLU(gate)*up -> down-proj prologue (PRO_SILU), consuming gu directly.
//   * residual adds -> o_proj / down epilogues (EPI_RES): y = x += round(y);
//     layer 0's o epilogue reads emb[tok] as the residual base (res==null).
//   * final norm -> lm_head prologue; fused argmax: per-block best ->
//     atomicMax on a packed (value,index) key; last block writes tok,
//     tok_hist[pos], advances pos (EPI_AMAX).
//
// Per step: 28 * (qkv, attn, attnc, o, gu, down) + lm_head = 169 graph nodes.
// The INT4 GEMV row loop is verbatim from kernels/gemv.cu (I4 path).

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define HID 1024
#define NLAY 28
#define NQH 16
#define NKVH 8
#define HDIM 128
#define QROWS 2048
#define KVROWS 1024
#define QKV 4096
#define INTER 3072
#define GU 6144
#define VOCAB 151936
#define MAXPOS 8704
#define NSLICE_MAX 144
#define W_LM (NLAY * 4)

#define THREADS 256
#define WARPS (THREADS / 32)
#define NEG_INF (-1e30f)
#define RMS_EPS 1e-6f

// prologue / epilogue kinds (template params on the GEMV)
enum : int {
  PRO_PLAIN = 0,   // stage x as-is
  PRO_NORM = 1,    // stage rmsnorm(x; d.nw)
  PRO_EMBNORM = 2, // stage rmsnorm(emb[tok]; d.nw)
  PRO_SILU = 3,    // stage silu(x[i]) * x[in + i]
  EPI_PLAIN = 0,   // y[r] = h(acc)
  EPI_RES = 1,     // x[r] = res[r] + h(acc)   (res==null -> emb[tok])
  EPI_AMAX = 2,    // yf[r] = acc; block argmax -> amax; last block -> tok,pos
};

struct Weight {
  const uint32_t* codes;   // [out][n_in/8] packed nibbles
  const __half2* meta;     // [out][n_in/128] (scale, offset)
  const __half* x;         // activation source
  void* y;                 // out (half, or float when f32)
  int n_in, f32, nw;       // nw = norms-table row or -1
  const __half* res;       // EPI_RES residual base (null -> emb[tok])
};

struct P {
  Weight w[W_LM + 1];
  const __half* norms;     // [4*NLAY+1][HID]
  const __half* emb;       // [VOCAB][HID]
  const float* rope;       // [MAXPOS][256]: cos row then sin row
  __half *x, *qkv, *attn, *gu, *kc, *vc;
  float* part;             // [NSLICE_MAX][NQH][HDIM+2]
  float* logits;
  int* tok;
  int* tok_hist;           // argmax result per position
  int* pos;                // current decode position
  int* ctr;                // [NLAY] attention "blocks done" counters
  unsigned long long* amax;// argmax running key
  int* actr;               // argmax "blocks done" counter
  int nslice, ctx;
};

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
  return *reinterpret_cast<__half2*>(&u);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ============================ swappable matvec ==============================
// INT4-g128 GEMV: verbatim wave-1 core (kernels/gemv.cu I4 path), with the
// prologue/epilogue hooks described above. Desc stays by-value.
struct Desc {
  const void* codes;
  const __half2* meta;
  const __half* x;
  __half* y;
  float* yf;
  int out, in;
  // fusion payload (null-safe when unused by the template kind)
  const __half* nw;        // PRO_NORM/EMBNORM: norm weight row
  const __half* emb;       // PRO_EMBNORM / EPI_RES(res==null) source
  const int* tok;          // PRO_EMBNORM / EPI_RES token index
  const __half* res;       // EPI_RES residual base
  unsigned long long* amax;
  int* actr;
  int* otok;               // EPI_AMAX: tok, tok_hist, pos (3 pointers)
  int* ohist;
  int* opos;
};

struct Chunk {
  uint4 q;      // I4 uses q only
  __half2 meta; // (scale, offset)
};

__device__ __forceinline__ Chunk load_chunk(const Desc d, int row, int c,
                                            int cpr) {
  Chunk k;
  const int64_t rc = (int64_t)row * cpr + c;
  const int64_t g = (int64_t)row * (cpr >> 2) + (c >> 2);
  k.q = __ldg(reinterpret_cast<const uint4*>(d.codes) + rc);
  k.meta = __ldg(d.meta + g);
  return k;
}

__device__ __forceinline__ float dot_chunk(const Chunk& k, const __half2* xr,
                                           float xsum) {
  const uint32_t w[4] = {k.q.x, k.q.y, k.q.z, k.q.w};
  const __half2 magic = as_h2(0x64006400u);  // 1024.0
  __half2 acc[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) acc[a] = __float2half2_rn(0.f);
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __half2 v = __hsub2(as_h2(((w[i] >> (4 * j)) & 0x000F000Fu) | 0x64006400u),
                          magic);
      acc[j] = __hfma2(v, xr[4 * i + j], acc[j]);
    }
  const __half2 s2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
  const float s = __low2float(s2) + __high2float(s2);
  return s * __low2float(k.meta) + __high2float(k.meta) * xsum;
}

// argmax key: float -> order-preserving uint, high bits; (~idx) keeps the
// smallest index on ties, matching the torch argmax contract.
__device__ __forceinline__ unsigned long long amax_key(float v, int idx) {
  uint32_t b = __float_as_uint(v);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)b << 32) | (uint32_t)(~idx);
}

// x staging prologues: fill sx[0..n) (half) per template kind.
template <int PRO>
__device__ __forceinline__ void stage_x(const Desc& d, __half* sx,
                                        float* sred) {
  const int n = d.in, tid = threadIdx.x;
  if constexpr (PRO == PRO_PLAIN) {
    for (int i = tid * 8; i < n; i += THREADS * 8)
      *reinterpret_cast<uint4*>(sx + i) =
          *reinterpret_cast<const uint4*>(d.x + i);
    __syncthreads();
  } else if constexpr (PRO == PRO_NORM || PRO == PRO_EMBNORM) {
    const __half* src =
        PRO == PRO_EMBNORM ? d.emb + (int64_t)d.tok[0] * HID : d.x;
    // pass 1: stash x in SMEM + block sum of squares
    for (int i = tid * 4; i < n; i += THREADS * 4)
      *reinterpret_cast<uint2*>(sx + i) =
          *reinterpret_cast<const uint2*>(src + i);
    float ss = 0.f;
    for (int i = tid * 4; i < n; i += THREADS * 4) {
      float2 a =
          __half22float2(reinterpret_cast<const __half2*>(sx)[i >> 1]);
      float2 b = __half22float2(
          reinterpret_cast<const __half2*>(sx)[(i >> 1) + 1]);
      ss += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    }
    const int lane = tid & 31, warp = tid >> 5;
    ss = warp_sum(ss);
    if (lane == 0) sred[warp] = ss;
    __syncthreads();
    if (warp == 0) {
      ss = warp_sum(lane < WARPS ? sred[lane] : 0.f);
      if (lane == 0) sred[0] = ss;
    }
    __syncthreads();
    const float r = rsqrtf(sred[0] / (float)HID + RMS_EPS);
    // pass 2: sx[i] = x[i] * r * w[i]
    for (int i = tid * 4; i < n; i += THREADS * 4) {
      __half2 h0 = reinterpret_cast<const __half2*>(sx)[i >> 1];
      __half2 h1 = reinterpret_cast<const __half2*>(sx)[(i >> 1) + 1];
      __half2 w0 =
          as_h2(reinterpret_cast<const uint32_t*>(d.nw)[i >> 1]);
      __half2 w1 =
          as_h2(reinterpret_cast<const uint32_t*>(d.nw)[(i >> 1) + 1]);
      float2 a = __half22float2(h0), b = __half22float2(h1);
      float2 u = __half22float2(w0), v = __half22float2(w1);
      reinterpret_cast<__half2*>(sx)[i >> 1] =
          __floats2half2_rn(a.x * r * u.x, a.y * r * u.y);
      reinterpret_cast<__half2*>(sx)[(i >> 1) + 1] =
          __floats2half2_rn(b.x * r * v.x, b.y * r * v.y);
    }
    __syncthreads();
  } else {  // PRO_SILU: n = INTER, x = gu (gate | up)
    for (int i = tid * 4; i < n; i += THREADS * 4) {
      __half2 g0 = as_h2(reinterpret_cast<const uint32_t*>(d.x)[i >> 1]);
      __half2 g1 =
          as_h2(reinterpret_cast<const uint32_t*>(d.x)[(i >> 1) + 1]);
      __half2 u0 =
          as_h2(reinterpret_cast<const uint32_t*>(d.x + n)[i >> 1]);
      __half2 u1 =
          as_h2(reinterpret_cast<const uint32_t*>(d.x + n)[(i >> 1) + 1]);
      float2 g = __half22float2(g0), h = __half22float2(g1);
      float2 u = __half22float2(u0), v = __half22float2(u1);
      __half2 o0 = __floats2half2_rn(g.x / (1.f + __expf(-g.x)) * u.x,
                                     g.y / (1.f + __expf(-g.y)) * u.y);
      __half2 o1 = __floats2half2_rn(h.x / (1.f + __expf(-h.x)) * v.x,
                                     h.y / (1.f + __expf(-h.y)) * v.y);
      uint2 o;
      o.x = *reinterpret_cast<uint32_t*>(&o0);
      o.y = *reinterpret_cast<uint32_t*>(&o1);
      *reinterpret_cast<uint2*>(sx + i) = o;
    }
    __syncthreads();
  }
}

template <int NC, int R, int PRO, int EPI>
__device__ __forceinline__ void gemv_t(const Desc d, int row0_base, int stride,
                                       int n_out) {
  __shared__ __align__(16) __half sx[NC * 1024];
  __shared__ float sred[WARPS];
  __shared__ float sbv[WARPS];
  __shared__ int sbi[WARPS];
  const int n = d.in, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  stage_x<PRO>(d, sx, sred);

  __half2 xr[NC][16];
  float xsum[NC];
#pragma unroll
  for (int t = 0; t < NC; ++t) {
    const uint4* sp =
        reinterpret_cast<const uint4*>(sx + (lane + 32 * t) * 32);
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      uint4 v = sp[u];
      xr[t][4 * u + 0] = as_h2(v.x);
      xr[t][4 * u + 1] = as_h2(v.y);
      xr[t][4 * u + 2] = as_h2(v.z);
      xr[t][4 * u + 3] = as_h2(v.w);
    }
#pragma unroll
    for (int u = 0; u < 16; ++u)
      s += __low2float(xr[t][u]) + __high2float(xr[t][u]);
    xsum[t] = s;
  }

  float bv = NEG_INF;
  int bi = INT_MAX;
  const __half* resbase =
      (EPI == EPI_RES) ? (d.res ? d.res : d.emb + (int64_t)d.tok[0] * HID)
                       : nullptr;
  const int cpr = n / 32;
  for (int row0 = row0_base + warp; row0 < n_out; row0 += R * stride) {
    Chunk c[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r * stride < n_out)
#pragma unroll
        for (int t = 0; t < NC; ++t)
          c[r][t] = load_chunk(d, row0 + r * stride, lane + 32 * t, cpr);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = 0.f;
      if (row0 + r * stride < n_out)
#pragma unroll
        for (int t = 0; t < NC; ++t)
          acc[r] += dot_chunk(c[r][t], xr[t], xsum[t]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int r = 0; r < R; ++r)
        acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    if (lane < R && row0 + lane * stride < n_out) {
      const int ro = row0 + lane * stride;
      float v = acc[0];
#pragma unroll
      for (int r = 1; r < R; ++r) v = lane == r ? acc[r] : v;
      if (EPI == EPI_RES) {
        // match anorm_exec: residual += round-to-half of the GEMV result
        const float hv = __half2float(__float2half(v));
        d.y[ro] = __float2half(__half2float(resbase[ro]) + hv);
      } else if (EPI == EPI_AMAX) {
        d.yf[ro] = v;
        if (v > bv || (v == bv && ro < bi)) {
          bv = v;
          bi = ro;
        }
      } else {
        d.y[ro] = __float2half(v);
      }
    }
  }

  if constexpr (EPI == EPI_AMAX) {
    // warp then block argmax -> atomicMax(amax); last block emits the token.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      sbv[warp] = bv;
      sbi[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
      bv = lane < WARPS ? sbv[lane] : NEG_INF;
      bi = lane < WARPS ? sbi[lane] : INT_MAX;
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == 0) {
        atomicMax(d.amax, amax_key(bv, bi));
        __threadfence();
        if (atomicAdd(d.actr, 1) == gridDim.x - 1) {
          __threadfence();  // acquire: see every block's atomicMax
          const unsigned long long kk = __ldcg(d.amax);
          const int idx = (int)(~(uint32_t)kk);
          const int cur = __ldcg(d.opos);
          *d.otok = idx;
          d.ohist[cur] = idx;
          *d.opos = cur + 1;  // next step's position
          *d.actr = 0;
          *d.amax = 0;
        }
      }
    }
  }
}

#ifndef RPW
#define RPW 2
#endif

template <int NC, int PRO, int EPI>
__global__ void __launch_bounds__(THREADS, NC == 1 ? 4 : 2)
    k_gemv(const Desc d) {
  gemv_t<NC, RPW, PRO, EPI>(d, blockIdx.x * WARPS, gridDim.x * WARPS, d.out);
}

// ========================= end swappable matvec ==============================

// Per-head RMSNorm+RoPE into SMEM/global. src: head row; nw: norm weight row;
// rope_row: rope + pos*256; dst: 128-half output (SMEM for q, kc slot for k).
__device__ __forceinline__ void norm_rope_head(const __half* src,
                                               const __half* nw,
                                               const float* rope_row,
                                               __half* dst) {
  const int lane = threadIdx.x & 31;
  uint2 u = *reinterpret_cast<const uint2*>(src + lane * 4);
  float2 a = __half22float2(as_h2(u.x)), b = __half22float2(as_h2(u.y));
  float ss = a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
  ss = warp_sum(ss);
  const float r = rsqrtf(ss / 128.f + RMS_EPS);
  float2 w0 = __half22float2(
      as_h2(reinterpret_cast<const uint32_t*>(nw)[lane * 2]));
  float2 w1 = __half22float2(
      as_h2(reinterpret_cast<const uint32_t*>(nw)[lane * 2 + 1]));
  __half2 h0 = __floats2half2_rn(a.x * r * w0.x, a.y * r * w0.y);
  __half2 h1 = __floats2half2_rn(b.x * r * w1.x, b.y * r * w1.y);
  uint2 o;
  o.x = *reinterpret_cast<uint32_t*>(&h0);
  o.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(dst + lane * 4) = o;
  __syncwarp();
  // RoPE pairs (d, d+64), d in 0..63; cos row [0:64], sin row [128:192]
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = lane * 4 + j;
      const float c = rope_row[d], s = rope_row[128 + d];
      const float x1 = __half2float(dst[d]), x2 = __half2float(dst[d + 64]);
      dst[d] = __float2half(x1 * c - x2 * s);
      dst[d + 64] = __float2half(x2 * c + x1 * s);
    }
  }
  __syncwarp();
}

// One attention kernel per layer. gridDim.x = ceil(ctx / per) blocks,
// per = ceil(ctx / nslice). Block b covers positions [b*per, (b+1)*per).
// PREP=1: each block re-does q_norm+RoPE for all 16 q heads into SMEM;
//         the block owning `pos` additionally does k_norm+RoPE and appends
//         K,V at slot pos. PREP=0: q/k were already prepared by k_qkra
//         (ablation path); reads p.qkv directly.
// COMB=1: last-finishing block (atomic ctr) merges partials into attn.
// COMB=0: partials only; a separate k_attnc kernel merges.
// NT=256: warp handles 2 heads; NT=512: 1 head.
template <int PREP, int COMB, int NT>
__global__ void __launch_bounds__(NT, NT == 256 ? 4 : 2)
    k_attn(const P* pp, int layer) {
  const P& p = *pp;
  const int pos = p.pos[0];
  const int b = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = (p.ctx + p.nslice - 1) / p.nslice;
  const int a0 = b * per, a1 = min((b + 1) * per, p.ctx);
  const int hi = min(a1, pos + 1);

  __shared__ __align__(16) __half sq[NQH * HDIM];

  if constexpr (PREP) {
    const float* rrow = p.rope + (int64_t)pos * 256;
    for (int h = warp; h < NQH; h += NT / 32)
      norm_rope_head(p.qkv + h * HDIM, p.norms + (2 * NLAY + layer) * HID,
                     rrow, sq + h * HDIM);
    // owner block: normed+roped K row and the V copy -> cache slot pos
    const int owner = min(pos / per, (int)gridDim.x - 1);
    if (b == owner) {
      __half* kd = p.kc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
      __half* vd = p.vc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
      for (int h = warp; h < NKVH; h += NT / 32)
        norm_rope_head(p.qkv + QROWS + h * HDIM,
                       p.norms + (3 * NLAY + layer) * HID, rrow,
                       kd + h * HDIM);
      for (int i = tid * 8; i < KVROWS; i += NT * 8)
        *reinterpret_cast<uint4*>(vd + i) =
            *reinterpret_cast<const uint4*>(p.qkv + QROWS + KVROWS + i);
    }
    __syncthreads();
  }

  // split-K attention over this block's position slice
  float* base = p.part + (int64_t)b * NQH * (HDIM + 2);
  const __half* kc = p.kc + (int64_t)layer * MAXPOS * KVROWS;
  const __half* vc = p.vc + (int64_t)layer * MAXPOS * KVROWS;
  for (int h = warp; h < NQH; h += NT / 32) {
    const int kvh = h >> 1;
    const __half* q = PREP ? sq + h * HDIM : p.qkv + h * HDIM;
    uint2 uq = *reinterpret_cast<const uint2*>(q + lane * 4);
    const __half2 qr0 = as_h2(uq.x), qr1 = as_h2(uq.y);
    float m = NEG_INF, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = a0; t < hi; ++t) {
      const __half* krow = kc + (int64_t)t * KVROWS + kvh * 128;
      uint2 uk = *reinterpret_cast<const uint2*>(krow + lane * 4);
      __half2 prod = __hmul2(as_h2(uk.x), qr0);
      prod = __hfma2(as_h2(uk.y), qr1, prod);
      float s = __low2float(prod) + __high2float(prod);
      s = warp_sum(s) * 0.08838834764831845f;  // 1/sqrt(128)
      const float mn = fmaxf(m, s);
      const float corr = __expf(m - mn);
      const float e = __expf(s - mn);
      const __half* vrow = vc + (int64_t)t * KVROWS + kvh * 128;
      uint2 uv = *reinterpret_cast<const uint2*>(vrow + lane * 4);
      float2 v0 = __half22float2(as_h2(uv.x)), v1 = __half22float2(as_h2(uv.y));
      acc[0] = acc[0] * corr + e * v0.x;
      acc[1] = acc[1] * corr + e * v0.y;
      acc[2] = acc[2] * corr + e * v1.x;
      acc[3] = acc[3] * corr + e * v1.y;
      l = l * corr + e;
      m = mn;
    }
    float* po = base + h * (HDIM + 2);
#pragma unroll
    for (int j = 0; j < 4; ++j) po[lane * 4 + j] = acc[j];
    if (lane == 0) {
      po[128] = l;
      po[129] = m;
    }
  }

  if constexpr (!COMB) return;
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (tid == 0) last = atomicAdd(p.ctr + layer, 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (!last) return;

  // last block: combine slices -> attn[2048], reset the counter.
  // partials are first-read here for this kernel (L1 is invalidated at kernel
  // boundaries); the atomic counter orders them.
  __threadfence();
  if (tid == 0) p.ctr[layer] = 0;
  const int ns = gridDim.x;
  for (int h = warp; h < NQH; h += NT / 32) {
    float m = NEG_INF;
    for (int s = 0; s < ns; ++s)
      m = fmaxf(m, p.part[(int64_t)s * NQH * (HDIM + 2) + h * (HDIM + 2) + 129]);
    float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < ns; ++s) {
      const float* po = p.part + (int64_t)s * NQH * (HDIM + 2) + h * (HDIM + 2);
      const float e = __expf(po[129] - m);
      l += e * po[128];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += e * po[lane * 4 + j];
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      p.attn[h * 128 + lane * 4 + j] = __float2half(acc[j] * inv);
  }
}

// Standalone q_norm/k_norm + RoPE + KV append (LM-03's OP_QKRA, verbatim):
// retained for the "epilogue vs small kernel" ablation (PREP=0 path).
__global__ void __launch_bounds__(THREADS, 2) k_qkra(const P* pp, int layer) {
  const P& p = *pp;
  const int pos = p.pos[0];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int h = warp; h < 24; h += WARPS) {
    const __half* w =
        p.norms + (h < 16 ? 2 * NLAY + layer : 3 * NLAY + layer) * HID;
    __half* q = p.qkv + (h < 16 ? h * 128 : QROWS + (h - 16) * 128);
    uint2 u = *reinterpret_cast<const uint2*>(q + lane * 4);
    float2 a = __half22float2(as_h2(u.x)), b = __half22float2(as_h2(u.y));
    float ss = a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / 128.f + RMS_EPS);
    float2 w0 = __half22float2(
        as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2]));
    float2 w1 = __half22float2(
        as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2 + 1]));
    uint2 o;
    __half2 h0 = __floats2half2_rn(a.x * r * w0.x, a.y * r * w0.y);
    __half2 h1 = __floats2half2_rn(b.x * r * w1.x, b.y * r * w1.y);
    o.x = *reinterpret_cast<uint32_t*>(&h0);
    o.y = *reinterpret_cast<uint32_t*>(&h1);
    *reinterpret_cast<uint2*>(q + lane * 4) = o;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < 24 * 64; t += THREADS) {
    const int h = t / 64, d = t % 64;
    __half* q = p.qkv + (h < 16 ? h * 128 : QROWS + (h - 16) * 128);
    const float c = p.rope[pos * 256 + d], s = p.rope[pos * 256 + 128 + d];
    const float x1 = __half2float(q[d]), x2 = __half2float(q[d + 64]);
    q[d] = __float2half(x1 * c - x2 * s);
    q[d + 64] = __float2half(x2 * c + x1 * s);
  }
  __syncthreads();
  __half* kd = p.kc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
  __half* vd = p.vc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
  for (int i = threadIdx.x * 8; i < KVROWS; i += THREADS * 8) {
    *reinterpret_cast<uint4*>(kd + i) =
        *reinterpret_cast<const uint4*>(p.qkv + QROWS + i);
    *reinterpret_cast<uint4*>(vd + i) =
        *reinterpret_cast<const uint4*>(p.qkv + QROWS + KVROWS + i);
  }
}

// Standalone split-K combine: heads split across gridDim.x blocks (1/2/4/8),
// warps stride heads within a block's share.
__global__ void __launch_bounds__(THREADS, 4) k_attnc(const P* pp, int s0) {
  const P& p = *pp;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hn = NQH / (int)gridDim.x;
  const int h0 = blockIdx.x * hn;
  for (int h = h0 + warp; h < h0 + hn; h += WARPS) {
    float m = NEG_INF;
    for (int s = 0; s < s0; ++s)
      m = fmaxf(m, p.part[(int64_t)s * NQH * (HDIM + 2) + h * (HDIM + 2) + 129]);
    float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < s0; ++s) {
      const float* po = p.part + (int64_t)s * NQH * (HDIM + 2) + h * (HDIM + 2);
      const float e = __expf(po[129] - m);
      l += e * po[128];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += e * po[lane * 4 + j];
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      p.attn[h * 128 + lane * 4 + j] = __float2half(acc[j] * inv);
  }
}

// ======================== host side / engine ================================
static P h_p;
static P* d_p = nullptr;
static int* d_state = nullptr;    // [0]=pos
static int* d_ctr = nullptr;      // [NLAY]
static unsigned long long* d_amax = nullptr;
static int* d_actr = nullptr;
static cudaStream_t g_stream = nullptr;
static cudaGraphExec_t g_graph = nullptr;
static int g_attn_prep = 1;       // 1 = fused in k_attn; 0 = k_qkra + k_attn<0>
static int g_split_attnc = 1;     // 1 = separate k_attnc node (default)
static int g_attn_nt = 512;       // k_attn threads: 512 = 1 head/warp
static int g_attnc_grid = 8;      // k_attnc blocks: heads split 16/grid

static P* dev() {
  if (!d_p) cudaMalloc(&d_p, sizeof(P));
  return d_p;
}

// Attention slice split: 36 for short ctx, else 72 (swept on sm_75)
static int pick_nslice(int ctx, int want) {
  if (want > 0) return want;
  return ctx <= 512 ? 36 : 72;
}

static Desc mkdesc(int widx) {
  const Weight& w = h_p.w[widx];
  Desc d;
  d.codes = w.codes;
  d.meta = w.meta;
  d.x = w.x;
  d.y = reinterpret_cast<__half*>(w.y);
  d.yf = w.f32 ? reinterpret_cast<float*>(w.y) : nullptr;
  d.in = w.n_in;
  d.out = widx == W_LM ? VOCAB
                       : (widx % 4 == 0 ? QKV : (widx % 4 == 2 ? GU : HID));
  d.nw = w.nw >= 0 ? h_p.norms + w.nw * HID : nullptr;
  d.emb = h_p.emb;
  d.tok = h_p.tok;
  d.res = w.res;
  d.amax = d_amax;
  d.actr = d_actr;
  d.otok = h_p.tok;
  d.ohist = h_p.tok_hist;
  d.opos = h_p.pos;
  return d;
}

static int nattn_blocks() {
  const int per = (h_p.ctx + h_p.nslice - 1) / h_p.nslice;
  return per > 0 ? (h_p.ctx + per - 1) / per : 0;
}

static int g_plain_gemv = 0;   // ablation: plain PRO/EPI everywhere
static cudaGraphExec_t g_graph_gemv = nullptr;  // unused placeholder

static void launch_gemv(int widx, cudaStream_t s) {
  Desc d = mkdesc(widx);
  if (g_plain_gemv) {
    const int nc0 = d.in / 1024;
    const int want0 = (d.out + RPW * WARPS - 1) / (RPW * WARPS);
    const int cap0 = (nc0 == 1 ? 4 : 2) * 36;
    const int g0 = want0 < cap0 ? want0 : cap0;
    if (nc0 == 1) k_gemv<1, PRO_PLAIN, EPI_PLAIN><<<g0, THREADS, 0, s>>>(d);
    else if (nc0 == 2) k_gemv<2, PRO_PLAIN, EPI_PLAIN><<<g0, THREADS, 0, s>>>(d);
    else k_gemv<3, PRO_PLAIN, EPI_PLAIN><<<g0, THREADS, 0, s>>>(d);
    return;
  }
  const int nc = d.in / 1024;
  const int want = (d.out + RPW * WARPS - 1) / (RPW * WARPS);
  const int cap = (nc == 1 ? 4 : 2) * 36;
  const int grid = want < cap ? want : cap;
  const int kind = widx % 4;
  if (widx == W_LM) {
    k_gemv<1, PRO_NORM, EPI_AMAX><<<grid, THREADS, 0, s>>>(d);
  } else if (kind == 0) {  // qkv
    if (widx == 0)
      k_gemv<1, PRO_EMBNORM, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
    else
      k_gemv<1, PRO_NORM, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
  } else if (kind == 1) {  // o
    k_gemv<2, PRO_PLAIN, EPI_RES><<<grid, THREADS, 0, s>>>(d);
  } else if (kind == 2) {  // gate_up
    k_gemv<1, PRO_NORM, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
  } else {  // down
    k_gemv<3, PRO_SILU, EPI_RES><<<grid, THREADS, 0, s>>>(d);
  }
}

static void launch_attn(int layer, cudaStream_t s) {
  const int nb = nattn_blocks();
  const int nt = g_attn_nt;
  if (g_attn_prep) {
    if (nt == 512) {
      if (g_split_attnc) k_attn<1, 0, 512><<<nb, nt, 0, s>>>(dev(), layer);
      else k_attn<1, 1, 512><<<nb, nt, 0, s>>>(dev(), layer);
    } else {
      if (g_split_attnc) k_attn<1, 0, 256><<<nb, nt, 0, s>>>(dev(), layer);
      else k_attn<1, 1, 256><<<nb, nt, 0, s>>>(dev(), layer);
    }
  } else {
    k_qkra<<<1, THREADS, 0, s>>>(dev(), layer);
    if (nt == 512) {
      if (g_split_attnc) k_attn<0, 0, 512><<<nb, nt, 0, s>>>(dev(), layer);
      else k_attn<0, 1, 512><<<nb, nt, 0, s>>>(dev(), layer);
    } else {
      if (g_split_attnc) k_attn<0, 0, 256><<<nb, nt, 0, s>>>(dev(), layer);
      else k_attn<0, 1, 256><<<nb, nt, 0, s>>>(dev(), layer);
    }
  }
  if (g_split_attnc) k_attnc<<<g_attnc_grid, THREADS, 0, s>>>(dev(), nb);
}

static void step_kernels(cudaStream_t s) {
  for (int l = 0; l < NLAY; ++l) {
    launch_gemv(l * 4 + 0, s);
    launch_attn(l, s);
    launch_gemv(l * 4 + 1, s);
    launch_gemv(l * 4 + 2, s);
    launch_gemv(l * 4 + 3, s);
  }
  launch_gemv(W_LM, s);
}

extern "C" int fx_init(const int64_t* bufs, const int64_t* codes,
                       const int64_t* meta, int64_t emb, int64_t norms,
                       int64_t rope, int64_t tok, int64_t tok_hist,
                       int64_t nslice, int64_t ctx) {
  // bufs: x, qkv, attn, gu, kc, vc, part, logits
  memset(&h_p, 0, sizeof(h_p));
  h_p.x = (__half*)bufs[0];
  h_p.qkv = (__half*)bufs[1];
  h_p.attn = (__half*)bufs[2];
  h_p.gu = (__half*)bufs[3];
  h_p.kc = (__half*)bufs[4];
  h_p.vc = (__half*)bufs[5];
  h_p.part = (float*)bufs[6];
  h_p.logits = (float*)bufs[7];
  h_p.tok = (int*)tok;
  h_p.tok_hist = (int*)tok_hist;
  h_p.emb = (const __half*)emb;
  h_p.norms = (const __half*)norms;
  h_p.rope = (const float*)rope;
  h_p.ctx = (int)ctx;
  h_p.nslice = pick_nslice((int)ctx, (int)nslice);
  for (int l = 0; l < NLAY; ++l)
    for (int k = 0; k < 4; ++k) {
      Weight& w = h_p.w[l * 4 + k];
      w.codes = (const uint32_t*)codes[l * 4 + k];
      w.meta = (const __half2*)meta[l * 4 + k];
      w.f32 = 0;
      switch (k) {
        case 0:
          w.x = h_p.x;   // PRO_NORM reads d.x; PRO_EMBNORM reads emb[tok]
          w.y = h_p.qkv;
          w.n_in = HID;
          w.nw = l;      // ln1
          w.res = nullptr;
          break;
        case 1:
          w.x = h_p.attn;
          w.y = h_p.x;   // residual in-place
          w.n_in = QROWS;
          w.nw = -1;
          w.res = l == 0 ? nullptr : h_p.x;  // L0: base = emb[tok]
          break;
        case 2:
          w.x = h_p.x;
          w.y = h_p.gu;
          w.n_in = HID;
          w.nw = NLAY + l;  // ln2
          w.res = nullptr;
          break;
        default:
          w.x = h_p.gu;
          w.y = h_p.x;
          w.n_in = INTER;
          w.nw = -1;
          w.res = h_p.x;
          break;
      }
    }
  Weight& lm = h_p.w[W_LM];
  lm.codes = (const uint32_t*)codes[W_LM];
  lm.meta = (const __half2*)meta[W_LM];
  lm.x = h_p.x;
  lm.y = h_p.logits;
  lm.n_in = HID;
  lm.f32 = 1;
  lm.nw = 4 * NLAY;  // final norm
  lm.res = nullptr;
  if (!g_stream)
    cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
  if (!d_state) {
    cudaMalloc(&d_state, sizeof(int));
    cudaMemset(d_state, 0, sizeof(int));
  }
  if (!d_ctr) {
    cudaMalloc(&d_ctr, NLAY * sizeof(int));
    cudaMemset(d_ctr, 0, NLAY * sizeof(int));
  }
  if (!d_amax) {
    cudaMalloc(&d_amax, sizeof(unsigned long long));
    cudaMemset(d_amax, 0, sizeof(unsigned long long));
    cudaMalloc(&d_actr, sizeof(int));
    cudaMemset(d_actr, 0, sizeof(int));
  }
  h_p.pos = d_state;
  h_p.ctr = d_ctr;
  h_p.amax = d_amax;
  h_p.actr = d_actr;
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  return 0;
}

extern "C" int fx_pos_set(int v) {
  return cudaMemcpy(d_state, &v, sizeof(int), cudaMemcpyHostToDevice) ==
                 cudaSuccess
             ? 0
             : -1;
}

extern "C" void fx_set_plain_gemv(int v) { g_plain_gemv = v; }

extern "C" void fx_set_attn(int prep, int split_attnc, int nt,
                            int attnc_grid) {
  g_attn_prep = prep;
  g_split_attnc = split_attnc;
  g_attn_nt = nt;
  g_attnc_grid = attnc_grid;
}

extern "C" int fx_step() {
  step_kernels(g_stream);
  return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0 : -1;
}

extern "C" int fx_graph_build() {
  if (g_graph) cudaGraphExecDestroy(g_graph);
  step_kernels(g_stream);  // warm caches
  cudaStreamSynchronize(g_stream);
  cudaGraph_t g;
  cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
  step_kernels(g_stream);
  if (cudaStreamEndCapture(g_stream, &g) != cudaSuccess) return -2;
  return cudaGraphInstantiate(&g_graph, g, 0) == cudaSuccess ? 0 : -3;
}

extern "C" int fx_graph_launch() {
  cudaGraphLaunch(g_graph, g_stream);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int fx_sync() {
  return cudaStreamSynchronize(g_stream) == cudaSuccess
             ? 0
             : (int)cudaGetLastError();
}

extern "C" int fx_time_graph(int iters, int pos0, float* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int i = 0; i < iters; ++i) {
    fx_pos_set(pos0);
    cudaEventRecord(e0, g_stream);
    cudaGraphLaunch(g_graph, g_stream);
    cudaEventRecord(e1, g_stream);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    out[i] = ms;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return 0;
}

// Per-class timing of one step at position `pos` (no graph): sums per launch
// into out[0]=gemv ms, out[1]=attn ms, out[2]=combine/other ms.
extern "C" int fx_time_breakdown(int pos, float* out) {
  static cudaEvent_t e[169 * 2];
  static int init = 0;
  if (!init) {
    for (int i = 0; i < 169 * 2; ++i) cudaEventCreate(&e[i]);
    init = 1;
  }
  int ne = 0;
  fx_pos_set(pos);
  cudaStreamSynchronize(g_stream);
  float g_ms = 0.f, a_ms = 0.f, o_ms = 0.f;
  for (int l = 0; l < NLAY; ++l) {
    cudaEventRecord(e[ne++], g_stream);
    launch_gemv(l * 4 + 0, g_stream);
    cudaEventRecord(e[ne++], g_stream);
    launch_attn(l, g_stream);
    cudaEventRecord(e[ne++], g_stream);
    for (int k = 1; k < 4; ++k) launch_gemv(l * 4 + k, g_stream);
    cudaEventRecord(e[ne++], g_stream);
  }
  cudaEventRecord(e[ne++], g_stream);
  launch_gemv(W_LM, g_stream);
  cudaEventRecord(e[ne++], g_stream);
  cudaStreamSynchronize(g_stream);
  float ms;
  for (int l = 0; l < NLAY; ++l) {
    cudaEventElapsedTime(&ms, e[l * 3 + 0], e[l * 3 + 1]);
    g_ms += ms;  // qkv
    cudaEventElapsedTime(&ms, e[l * 3 + 1], e[l * 3 + 2]);
    a_ms += ms;  // attn + attnc (+qkra when unfused)
    cudaEventElapsedTime(&ms, e[l * 3 + 2], e[l * 3 + 3]);
    g_ms += ms;  // o + gu + down
  }
  cudaEventElapsedTime(&ms, e[NLAY * 3], e[NLAY * 3 + 1]);
  g_ms += ms;  // lm_head (+ argmax)
  out[0] = g_ms;
  out[1] = a_ms;
  out[2] = o_ms;
  return 0;  // static event pool: created once, reused
}

// Time one GEMV kind in isolation (prologue/epilogue cost ablation).
// kind: 0=qkv(L>0) 1=o 2=gu 3=down 4=lm_head; plain=1 forces PRO/EPI_PLAIN.
extern "C" float fx_time_gemv(int kind, int plain, int iters) {
  const int widx = kind == 4 ? W_LM : (3 * 4 + kind);  // layer 3 as sample
  Desc d = mkdesc(widx);
  const int nc = d.in / 1024;
  const int want = (d.out + RPW * WARPS - 1) / (RPW * WARPS);
  const int cap = (nc == 1 ? 4 : 2) * 36;
  const int grid = want < cap ? want : cap;
  auto launch = [&](cudaStream_t s) {
    if (plain) {
      if (nc == 1) k_gemv<1, PRO_PLAIN, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
      else if (nc == 2)
        k_gemv<2, PRO_PLAIN, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
      else k_gemv<3, PRO_PLAIN, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
      return;
    }
    switch (kind) {
      case 0:
        k_gemv<1, PRO_NORM, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
        break;
      case 1:
        k_gemv<2, PRO_PLAIN, EPI_RES><<<grid, THREADS, 0, s>>>(d);
        break;
      case 2:
        k_gemv<1, PRO_NORM, EPI_PLAIN><<<grid, THREADS, 0, s>>>(d);
        break;
      case 3:
        k_gemv<3, PRO_SILU, EPI_RES><<<grid, THREADS, 0, s>>>(d);
        break;
      default:
        k_gemv<1, PRO_NORM, EPI_AMAX><<<grid, THREADS, 0, s>>>(d);
        break;
    }
  };
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  launch(g_stream);
  cudaStreamSynchronize(g_stream);
  cudaEventRecord(e0, g_stream);
  for (int i = 0; i < iters; ++i) launch(g_stream);
  cudaEventRecord(e1, g_stream);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

// Attention-phase timing (prep variant via g_attn_prep): launches
// launch_attn(layer) each iteration at the current pos.
extern "C" float fx_time_attn(int layer, int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  launch_attn(layer, g_stream);
  cudaStreamSynchronize(g_stream);
  cudaEventRecord(e0, g_stream);
  for (int i = 0; i < iters; ++i) launch_attn(layer, g_stream);
  cudaEventRecord(e1, g_stream);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

// Raw attention-phase timing: launches only k_attn<prep,comb,nt>
// (no qkra/attnc) or, attnc>0, only k_attnc with attnc blocks.
extern "C" float fx_time_attn_raw(int prep, int comb, int attnc, int nt,
                                  int iters) {
  const int nb = nattn_blocks();
  auto launch = [&](cudaStream_t s) {
    if (attnc) {
      k_attnc<<<attnc, THREADS, 0, s>>>(dev(), nb);
      return;
    }
    if (nt == 512) {
      if (prep && comb) k_attn<1, 1, 512><<<nb, 512, 0, s>>>(dev(), 3);
      else if (prep) k_attn<1, 0, 512><<<nb, 512, 0, s>>>(dev(), 3);
      else if (comb) k_attn<0, 1, 512><<<nb, 512, 0, s>>>(dev(), 3);
      else k_attn<0, 0, 512><<<nb, 512, 0, s>>>(dev(), 3);
    } else {
      if (prep && comb) k_attn<1, 1, 256><<<nb, 256, 0, s>>>(dev(), 3);
      else if (prep) k_attn<1, 0, 256><<<nb, 256, 0, s>>>(dev(), 3);
      else if (comb) k_attn<0, 1, 256><<<nb, 256, 0, s>>>(dev(), 3);
      else k_attn<0, 0, 256><<<nb, 256, 0, s>>>(dev(), 3);
    }
  };
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  launch(g_stream);
  cudaStreamSynchronize(g_stream);
  cudaEventRecord(e0, g_stream);
  for (int i = 0; i < iters; ++i) launch(g_stream);
  cudaEventRecord(e1, g_stream);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

// GEMV-only step (113 launches, non-graph, one event pair per iter).
// Includes host-side launch latency; upper bound on GEMV phase.
extern "C" float fx_time_gemvseq(int pos0, int iters) {
  fx_pos_set(pos0);
  cudaStreamSynchronize(g_stream);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float total = 0.f;
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(e0, g_stream);
    for (int l = 0; l < NLAY; ++l)
      for (int k = 0; k < 4; ++k) launch_gemv(l * 4 + k, g_stream);
    launch_gemv(W_LM, g_stream);
    cudaEventRecord(e1, g_stream);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return total / iters;
}

// Clean teardown: destroy graphs and free device state so process exit
// doesn't hit the two-static-cudart teardown race with libmega loaded.
extern "C" int fx_shutdown() {
  if (g_graph) { cudaGraphExecDestroy(g_graph); g_graph = nullptr; }
  if (g_graph_gemv) { cudaGraphExecDestroy(g_graph_gemv); g_graph_gemv = nullptr; }
  cudaStreamSynchronize(g_stream);
  return 0;
}
