// LM-03: INT4 decode megakernel for Qwen3-0.6B-Base, batch 1, sm_75.
//
// Two engines share every device-side op body:
//   * "separate": ordinary kernels (k_task) launched op-by-op into one CUDA
//     graph per decode step — the tuned baseline and the kill-line denominator.
//   * "mega": one persistent launch (grid = NB co-resident CTAs) that walks the
//     packed task schedule emitted by kernels/megakernel/sched_gen.py and
//     synchronizes through global-memory flag counters (acquire loads /
//     release adds). Can run many decode steps inside one launch.
//
// INT4 matvec row-chunk logic is isolated in load_chunk_i4/dot_chunk_i4/
// gemv_exec so a future codec kernel drops in without restructuring.
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---- constants (mirror sched_gen.py) ---------------------------------------
#define HID 1024
#define NLAY 28
#define QROWS 2048
#define KVROWS 1024
#define QKV 4096
#define INTER 3072
#define GU 6144
#define VOCAB 151936
#define KVB 2048
#define MAXPOS 8704
#define NSLICE 72
#define NARGP 72
#define NORM_ROWS 113           // 28 ln1 + 28 ln2 + 28 qn + 28 kn + final
#define W_LM (NLAY * 4)

#define OP_TICK 0
#define OP_EMBED 1
#define OP_NORM 2
#define OP_ANORM 3
#define OP_QKRA 4
#define OP_ATTN 5
#define OP_ATTNC 6
#define OP_GEMV 7
#define OP_SILU 8
#define OP_ARGP 9
#define OP_ARGC 10

#define THREADS 256
#define WARPS (THREADS / 32)
#define NEG_INF (-1e30f)
#define RMS_EPS 1e-6f

struct Weight {
  const uint32_t* codes;   // [out][n_in/8] packed nibbles (pack.pack_int4 layout)
  const __half2* meta;     // [out][n_in/128] (scale, offset)
  const __half* x;         // activation source
  void* y;                 // out (half, or float when f32)
  int n_in, f32;
};

struct P {
  Weight w[W_LM + 1];
  const __half* norms;     // [NORM_ROWS][HID]
  const __half* emb;       // [VOCAB][HID]
  const float* rope;       // [MAXPOS][256]: cos row then sin row
  __half *x, *xn, *qkv, *attn, *oo, *gu, *dout, *kc, *vc;
  float* part;             // [NSLICE][16][130]
  float* logits;
  float* argp;             // [NARGP][2]
  int* tok;
  int* tok_hist;           // every argmax result, indexed by pos
  int* pos;                // current decode position (written by tick)
  int* flags;              // steps * fper counters
  int* pos_ctr;            // next position to issue
  const int* sched;        // packed blob
  int nslice, fper, f_argc, steps, ctx;
};

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
  return *reinterpret_cast<__half2*>(&u);
}

__device__ __forceinline__ int flag_get(const int* f) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
  return v;
}

__device__ __forceinline__ void flag_add(int* f, int a) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(f), "r"(a)
               : "memory");
}

// ============================ swappable matvec ==============================
// INT4-g128 GEMV: verbatim port of kernels/gemv.cu (I4 path). Wave-1 kernel
// passed Desc BY VALUE (param space = cmem) and had 0 local ops in SASS; the
// earlier rewrite read Weight fields through a global P* and spilled.
struct Desc {
  const void* codes;   // uint4 per 32-col chunk
  const __half2* meta; // (scale, offset) per 128-col group
  const __half* x;
  __half* y;
  float* yf;           // f32 output (lm_head) when non-null
  int out, in;
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

// R rows in flight per warp; wave-1 sweep found 2. The kernel version is
// grid-strided (base = blockIdx*WARPS, stride = gridDim*WARPS) exactly like
// gemv.cu; the megakernel version strides a block-owned range with WARPS.
template <int NC, int R>
__device__ __forceinline__ void gemv_t(const Desc d, int row0_base, int stride, int n_out) {
  __shared__ __align__(16) __half sx[NC * 1024];
  const int n = d.in, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid * 8; i < n; i += THREADS * 8)
    *reinterpret_cast<uint4*>(sx + i) = *reinterpret_cast<const uint4*>(d.x + i);
  __syncthreads();
  __half2 xr[NC][16];
  float xsum[NC];
#pragma unroll
  for (int t = 0; t < NC; ++t) {
    const uint4* sp = reinterpret_cast<const uint4*>(sx + (lane + 32 * t) * 32);
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
      float v = acc[0];
#pragma unroll
      for (int r = 1; r < R; ++r) v = lane == r ? acc[r] : v;
      if (d.yf) d.yf[row0 + lane * stride] = v;
      else d.y[row0 + lane * stride] = __float2half(v);
    }
  }
}

#ifndef RPW
#define RPW 2
#endif
// rows in flight per warp inside the persistent kernel; with 1 CTA/SM the
// warp count halves, so a deeper pipeline is affordable (registers are free
// up to 255 with unbounded launch_bounds).
#ifndef MKRPW
#define MKRPW 2
#endif
#ifndef MEGAOCC
#define MEGAOCC 1
#endif

__device__ __forceinline__ Desc wdesc(const P& p, int widx) {
  const Weight& w = p.w[widx];
  Desc d;
  d.codes = w.codes;
  d.meta = w.meta;
  d.x = w.x;
  d.y = reinterpret_cast<__half*>(w.y);
  d.yf = w.f32 ? reinterpret_cast<float*>(w.y) : nullptr;
  d.in = w.n_in;
  d.out = widx == W_LM ? VOCAB
                       : (widx % 4 == 0 ? QKV : (widx % 4 == 2 ? GU : HID));
  return d;
}

// megakernel path: block-owned contiguous range [r0,r1), warps stride WARPS.
__device__ __forceinline__ void gemv_exec(const P& p, int widx, int r0, int r1) {
  const Desc d = wdesc(p, widx);
  const int nc = d.in / 1024;
  if (nc == 1) gemv_t<1, MKRPW>(d, r0, WARPS, r1);
  else if (nc == 2) gemv_t<2, MKRPW>(d, r0, WARPS, r1);
  else gemv_t<3, MKRPW>(d, r0, WARPS, r1);
}

// separate-kernel path: Desc by value like gemv.cu — the kernel body is the
// wave-1 row loop (grid-stride).
template <int NC>
__global__ void __launch_bounds__(THREADS, NC == 1 ? 4 : 2)
    k_gemv(const Desc d) {
  gemv_t<NC, RPW>(d, blockIdx.x * WARPS, gridDim.x * WARPS, d.out);
}

// ========================= end swappable matvec ==============================

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ float block_sum(float v) {
  __shared__ float sh[WARPS];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = threadIdx.x < WARPS ? sh[threadIdx.x] : 0.f;
  if (warp == 0) v = warp_sum(v);
  if (threadIdx.x == 0) sh[0] = v;
  __syncthreads();
  v = sh[0];
  __syncthreads();
  return v;
}

__device__ void tick_exec(const P& p) {
  if (threadIdx.x == 0) {
    p.pos[0] = atomicAdd(p.pos_ctr, 1);  // returns the old value = this step's pos
  }
}

__device__ void embed_exec(const P& p) {
  const int t = p.tok[0];
  const __half* row = p.emb + (int64_t)t * HID;
  for (int i = threadIdx.x * 8; i < HID; i += THREADS * 8)
    *reinterpret_cast<uint4*>(p.x + i) = *reinterpret_cast<const uint4*>(row + i);
}

__device__ void norm_exec(const P& p, int wsel) {
  // xn[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i]; fp32 accumulate
  float ss = 0.f;
  for (int i = threadIdx.x * 4; i < HID; i += THREADS * 4) {
    __half2 h0 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2]);
    __half2 h1 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2 + 1]);
    float2 a = __half22float2(h0), b = __half22float2(h1);
    ss += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
  }
  ss = block_sum(ss);
  const float r = rsqrtf(ss / HID + RMS_EPS);
  const __half* w = p.norms + wsel * HID;
  for (int i = threadIdx.x * 4; i < HID; i += THREADS * 4) {
    __half2 h0 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2]);
    __half2 h1 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2 + 1]);
    __half2 w0 = as_h2(reinterpret_cast<const uint32_t*>(w)[i / 2]);
    __half2 w1 = as_h2(reinterpret_cast<const uint32_t*>(w)[i / 2 + 1]);
    float2 a = __half22float2(h0), b = __half22float2(h1);
    float2 u = __half22float2(w0), v = __half22float2(w1);
    reinterpret_cast<__half2*>(p.xn)[i / 2] = __floats2half2_rn(a.x * r * u.x, a.y * r * u.y);
    reinterpret_cast<__half2*>(p.xn)[i / 2 + 1] = __floats2half2_rn(b.x * r * v.x, b.y * r * v.y);
  }
}

// s0 selects the residual source: 0 -> oo (post-attn), 1 -> dout (post-mlp)
__device__ void anorm_exec(const P& p, int wsel, int s0) {
  const __half* src = s0 ? p.dout : p.oo;
  float t[4];
  float ss = 0.f;
  for (int i = threadIdx.x * 4; i < HID; i += THREADS * 4) {
    __half2 h0 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2]);
    __half2 h1 = as_h2(reinterpret_cast<const uint32_t*>(p.x)[i / 2 + 1]);
    __half2 s0h = as_h2(reinterpret_cast<const uint32_t*>(src)[i / 2]);
    __half2 s1h = as_h2(reinterpret_cast<const uint32_t*>(src)[i / 2 + 1]);
    float2 a = __half22float2(h0), b = __half22float2(h1);
    float2 c = __half22float2(s0h), d = __half22float2(s1h);
    t[0] = a.x + c.x; t[1] = a.y + c.y; t[2] = b.x + d.x; t[3] = b.y + d.y;
    ss += t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + t[3] * t[3];
    reinterpret_cast<__half2*>(p.x)[i / 2] = __floats2half2_rn(t[0], t[1]);
    reinterpret_cast<__half2*>(p.x)[i / 2 + 1] = __floats2half2_rn(t[2], t[3]);
  }
  ss = block_sum(ss);
  const float r = rsqrtf(ss / HID + RMS_EPS);
  const __half* w = p.norms + wsel * HID;
  for (int i = threadIdx.x * 4; i < HID; i += THREADS * 4) {
    float2 u = __half22float2(as_h2(reinterpret_cast<const uint32_t*>(w)[i / 2]));
    float2 v = __half22float2(as_h2(reinterpret_cast<const uint32_t*>(w)[i / 2 + 1]));
    float x0 = __half2float(p.x[i]), x1 = __half2float(p.x[i + 1]);
    float x2 = __half2float(p.x[i + 2]), x3 = __half2float(p.x[i + 3]);
    reinterpret_cast<__half2*>(p.xn)[i / 2] = __floats2half2_rn(x0 * r * u.x, x1 * r * u.y);
    reinterpret_cast<__half2*>(p.xn)[i / 2 + 1] = __floats2half2_rn(x2 * r * v.x, x3 * r * v.y);
  }
}


// q_norm/k_norm per head, RoPE (Qwen3 interleaved-half convention), KV append.
__device__ void qkra_exec(const P& p, int layer) {
  const int pos = p.pos[0];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // phase 1: head norms — heads 0..15 = q (row 2*NLAY+l), 16..23 = k (3*NLAY+l)
  for (int h = warp; h < 24; h += WARPS) {
    const __half* w = p.norms + (h < 16 ? 2 * NLAY + layer : 3 * NLAY + layer) * HID;
    __half* q = p.qkv + (h < 16 ? h * 128 : QROWS + (h - 16) * 128);
    uint2 u = *reinterpret_cast<const uint2*>(q + lane * 4);
    float2 a = __half22float2(as_h2(u.x)), b = __half22float2(as_h2(u.y));
    float ss = a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / 128.f + RMS_EPS);
    float2 w0 = __half22float2(as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2]));
    float2 w1 = __half22float2(as_h2(reinterpret_cast<const uint32_t*>(w)[lane * 2 + 1]));
    uint2 o;
    __half2 h0 = __floats2half2_rn(a.x * r * w0.x, a.y * r * w0.y);
    __half2 h1 = __floats2half2_rn(b.x * r * w1.x, b.y * r * w1.y);
    o.x = *reinterpret_cast<uint32_t*>(&h0);
    o.y = *reinterpret_cast<uint32_t*>(&h1);
    *reinterpret_cast<uint2*>(q + lane * 4) = o;
  }
  __syncthreads();
  // phase 2: rope on q heads (0..15) and k heads (16..23); pair (d, d+64)
  for (int t = threadIdx.x; t < 24 * 64; t += THREADS) {
    const int h = t / 64, d = t % 64;
    __half* q = p.qkv + (h < 16 ? h * 128 : QROWS + (h - 16) * 128);
    const float c = p.rope[pos * 256 + d], s = p.rope[pos * 256 + 128 + d];
    const float x1 = __half2float(q[d]), x2 = __half2float(q[d + 64]);
    q[d] = __float2half(x1 * c - x2 * s);
    q[d + 64] = __float2half(x2 * c + x1 * s);
  }
  __syncthreads();
  // phase 3: append K,V rows at pos
  __half* kd = p.kc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
  __half* vd = p.vc + ((int64_t)layer * MAXPOS + pos) * KVROWS;
  for (int i = threadIdx.x * 8; i < KVROWS; i += THREADS * 8) {
    *reinterpret_cast<uint4*>(kd + i) = *reinterpret_cast<const uint4*>(p.qkv + QROWS + i);
    *reinterpret_cast<uint4*>(vd + i) = *reinterpret_cast<const uint4*>(p.qkv + QROWS + KVROWS + i);
  }
}

// attention over kv positions [a0, a1) clamped to pos+1; warp handles 2 heads.
__device__ void attn_exec(const P& p, int layer, int s0, int a0, int a1) {
  const int pos = p.pos[0];
  const int hi = min(a1, pos + 1);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* base = p.part + (int64_t)s0 * 16 * 130;
  const __half* kc = p.kc + (int64_t)layer * MAXPOS * KVROWS;
  const __half* vc = p.vc + (int64_t)layer * MAXPOS * KVROWS;
  for (int h = warp; h < 16; h += WARPS) {
    const int kvh = h >> 1;
    const __half* q = p.qkv + h * 128;
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
      const float corr = __expf(m - mn);       // exp(-inf)=0 on first iter
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
    float* po = base + h * 130;
#pragma unroll
    for (int j = 0; j < 4; ++j) po[lane * 4 + j] = acc[j];
    if (lane == 0) { po[128] = l; po[129] = m; }
  }
}

// combine s0 = number of slices; writes attn[2048]
__device__ void attnc_exec(const P& p, int s0) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int h = warp; h < 16; h += WARPS) {
    float m = NEG_INF;
    for (int s = 0; s < s0; ++s)
      m = fmaxf(m, p.part[(int64_t)s * 16 * 130 + h * 130 + 129]);
    float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < s0; ++s) {
      const float* po = p.part + (int64_t)s * 16 * 130 + h * 130;
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

__device__ void silu_exec(const P& p, int a0, int a1) {
  for (int i = a0 + threadIdx.x; i < a1; i += THREADS) {
    const float g = __half2float(p.gu[i]);
    const float u = __half2float(p.gu[INTER + i]);
    p.gu[i] = __float2half(g / (1.f + __expf(-g)) * u);
  }
}

// per-slice argmax over logits[a0,a1); result -> argp[s0] = (val, idx)
__device__ void argp_exec(const P& p, int s0, int a0, int a1) {
  __shared__ float sv[WARPS];
  __shared__ int si[WARPS];
  float bv = NEG_INF;
  int bi = a0;
  for (int i = a0 + threadIdx.x; i < a1; i += THREADS) {
    const float v = p.logits[i];
    if (v > bv) { bv = v; bi = i; }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
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
    for (int o = 4; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { p.argp[2 * s0] = bv; p.argp[2 * s0 + 1] = (float)bi; }
  }
}

__device__ void argc_exec(const P& p, int s0) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float bv = NEG_INF;
  int bi = 0;
  for (int i = threadIdx.x; i < s0; i += THREADS) {
    const float v = p.argp[2 * i];
    const int idx = (int)p.argp[2 * i + 1];
    if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
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
    for (int o = 4; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { p.tok[0] = bi; p.tok_hist[p.pos[0]] = bi; }
  }
}

__device__ void task_exec(const P& p, int op, int warg, int s0, int a0, int a1) {
  switch (op) {
    case OP_TICK: tick_exec(p); break;
    case OP_EMBED: embed_exec(p); break;
    case OP_NORM: norm_exec(p, warg); break;
    case OP_ANORM: anorm_exec(p, warg, s0); break;
    case OP_QKRA: qkra_exec(p, warg); break;
    case OP_ATTN: attn_exec(p, warg, s0, a0, a1); break;
    case OP_ATTNC: attnc_exec(p, s0); break;
    case OP_GEMV: gemv_exec(p, warg, a0, a1); break;
    case OP_SILU: silu_exec(p, a0, a1); break;
    case OP_ARGP: argp_exec(p, s0, a0, a1); break;
    case OP_ARGC: argc_exec(p, s0); break;
  }
}

// ============================ mega kernel ====================================
// sched blob (i32): magic, nt, nb, fper, f_argc, rsvd | bcnt[nb] | boff[nb] |
// records: op,warg,s0,a0,a1,nw,ns,order then nw*(fid,val), ns*(fid,add).
// fid == -1 waits on (step-1)*fper + f_argc (previous step's argc_done).
__global__ void __launch_bounds__(THREADS, MEGAOCC) mega_kernel(const P* pp) {
  const P& p = *pp;
  const int* sched = p.sched;
  const int nb = sched[2];
  const int fper = sched[3];
  const int* btab = sched + 6;
  const int b = blockIdx.x;
  const int bcnt = btab[b];
  const int* rp = sched + 6 + 2 * nb + btab[nb + b];
  const int tid = threadIdx.x;
  for (int step = 0; step < p.steps; ++step) {
    int* fbase = p.flags + step * fper;
    const int* cur = rp;
    for (int t = 0; t < bcnt; ++t) {
      const int op = cur[0], warg = cur[1], s0 = cur[2], a0 = cur[3], a1 = cur[4];
      const int nw = cur[5], ns = cur[6];
      const int* wp = cur + 8;
      const int* sp = wp + 2 * nw;
      if (tid == 0 && nw) {
        for (int i = 0; i < nw; ++i) {
          const int fid = wp[2 * i], val = wp[2 * i + 1];
          if (fid < 0 && step == 0) continue;
          const int* f = fid < 0 ? p.flags + (step - 1) * fper + p.f_argc
                                 : fbase + fid;
          while (flag_get(f) < val) __nanosleep(64);
        }
      }
      __syncthreads();
      task_exec(p, op, warg, s0, a0, a1);
      __syncthreads();
      if (tid == 0)
        for (int i = 0; i < ns; ++i)
          flag_add(fbase + sp[2 * i], sp[2 * i + 1]);
      cur = sp + 2 * ns;
    }
  }
}

// ======================== separate-kernel engine ============================
static P h_p;
// GEMV kernels dispatch on input width like kernels/gemv.cu.
static int gemv_grid(int widx) {
  const int n_out =
      widx == W_LM ? VOCAB : (widx % 4 == 0 ? QKV : (widx % 4 == 2 ? GU : HID));
  const int nc = widx == W_LM || widx % 4 == 0 || widx % 4 == 2 ? 1 : 2;
  const int want = (n_out + RPW * WARPS - 1) / (RPW * WARPS);
  const int cap = (nc == 1 ? 4 : 2) * 36;
  return want < cap ? want : cap;
}

static void launch_gemv(int widx, cudaStream_t s) {
  const int grid = gemv_grid(widx);
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
  if (widx == W_LM || widx % 4 == 0 || widx % 4 == 2)
    k_gemv<1><<<grid, THREADS, 0, s>>>(d);
  else if (widx % 4 == 1)
    k_gemv<2><<<grid, THREADS, 0, s>>>(d);
  else
    k_gemv<3><<<grid, THREADS, 0, s>>>(d);
}

// One kernel per task kind; grid-parallel non-GEMV ops split by blockIdx.
__global__ void __launch_bounds__(THREADS, 2) k_task(const P* pp, int op, int warg,
                                                  int a0, int a1) {
  const P& p = *pp;
  const int b = blockIdx.x;
  if (op == OP_ATTN) {
    const int ns = p.nslice;
    const int per = (p.ctx + ns - 1) / ns;
    attn_exec(p, warg, b, b * per, min((b + 1) * per, p.ctx));
  } else if (op == OP_SILU) {
    const int per = INTER / 8;
    silu_exec(p, b * per, (b + 1) * per);
  } else if (op == OP_ARGP) {
    const int per = (VOCAB + NARGP - 1) / NARGP;
    argp_exec(p, b, b * per, min((b + 1) * per, VOCAB));
  } else {
    task_exec(p, op, warg, a0, a0, a1);
  }
}

// ------------------------------ host side -----------------------------------
static P* d_p = nullptr;
static int* d_sched = nullptr;
static int* d_flags = nullptr;
static int* d_state = nullptr;  // [0]=pos_ctr, [1]=pos
static int g_nb = 0, g_fper = 0, g_nt = 0;
static cudaStream_t g_stream = nullptr;
static cudaGraphExec_t g_graph = nullptr;
static int g_steps = 0;

static P* dev() {
  if (!d_p) cudaMalloc(&d_p, sizeof(P));
  return d_p;
}

static int n_attn_slices(int ctx, int ns) {
  const int per = (ctx + ns - 1) / ns;
  return per > 0 ? (ctx + per - 1) / per : 0;
}

// Enqueue one decode step as separate kernels on a stream. Mirrors
// sched_gen.build: same op order and row/slice splits, stream-ordered.
static void step_kernels(const P* pp, cudaStream_t s, int ctx, int ns) {
  k_task<<<1, THREADS, 0, s>>>(pp, OP_TICK, 0, 0, 0);
  k_task<<<1, THREADS, 0, s>>>(pp, OP_EMBED, 0, 0, 0);
  const int nattn = n_attn_slices(ctx, ns);
  for (int l = 0; l < NLAY; ++l) {
    if (l == 0) k_task<<<1, THREADS, 0, s>>>(pp, OP_NORM, l, 0, 0);
    launch_gemv(l * 4 + 0, s);
    k_task<<<1, THREADS, 0, s>>>(pp, OP_QKRA, l, 0, 0);
    k_task<<<nattn, THREADS, 0, s>>>(pp, OP_ATTN, l, 0, 0);
    k_task<<<1, THREADS, 0, s>>>(pp, OP_ATTNC, 0, nattn, 0);
    launch_gemv(l * 4 + 1, s);
    k_task<<<1, THREADS, 0, s>>>(pp, OP_ANORM, NLAY + l, 0, 0);
    launch_gemv(l * 4 + 2, s);
    k_task<<<8, THREADS, 0, s>>>(pp, OP_SILU, 0, 0, 0);
    launch_gemv(l * 4 + 3, s);
    k_task<<<1, THREADS, 0, s>>>(pp, OP_ANORM,
                                 l + 1 < NLAY ? l + 1 : NORM_ROWS - 1, 1, 0);
  }
  launch_gemv(W_LM, s);
  k_task<<<NARGP, THREADS, 0, s>>>(pp, OP_ARGP, 0, 0, 0);
  k_task<<<1, THREADS, 0, s>>>(pp, OP_ARGC, 0, NARGP, 0);
}

extern "C" int mx_init(const int64_t* bufs, const int64_t* codes,
                       const int64_t* meta, int64_t emb, int64_t norms,
                       int64_t rope, int64_t tok, int64_t tok_hist,
                       int64_t nslice, int64_t ctx) {
  // bufs: x, xn, qkv, attn, oo, gu, dout, kc, vc, part, logits, argp, -, pos
  memset(&h_p, 0, sizeof(h_p));
  h_p.x = (__half*)bufs[0];
  h_p.xn = (__half*)bufs[1];
  h_p.qkv = (__half*)bufs[2];
  h_p.attn = (__half*)bufs[3];
  h_p.oo = (__half*)bufs[4];
  h_p.gu = (__half*)bufs[5];
  h_p.dout = (__half*)bufs[6];
  h_p.kc = (__half*)bufs[7];
  h_p.vc = (__half*)bufs[8];
  h_p.part = (float*)bufs[9];
  h_p.logits = (float*)bufs[10];
  h_p.argp = (float*)bufs[11];
  h_p.pos = (int*)bufs[13];
  h_p.tok = (int*)tok;
  h_p.tok_hist = (int*)tok_hist;
  h_p.emb = (const __half*)emb;
  h_p.norms = (const __half*)norms;
  h_p.rope = (const float*)rope;
  h_p.nslice = (int)nslice;
  h_p.ctx = (int)ctx;
  for (int l = 0; l < NLAY; ++l)
    for (int k = 0; k < 4; ++k) {
      Weight& w = h_p.w[l * 4 + k];
      w.codes = (const uint32_t*)codes[l * 4 + k];
      w.meta = (const __half2*)meta[l * 4 + k];
      w.f32 = 0;
      switch (k) {
        case 0: w.x = h_p.xn; w.y = h_p.qkv; w.n_in = HID; break;
        case 1: w.x = h_p.attn; w.y = h_p.oo; w.n_in = QROWS; break;
        case 2: w.x = h_p.xn; w.y = h_p.gu; w.n_in = HID; break;
        case 3: w.x = h_p.gu; w.y = h_p.dout; w.n_in = INTER; break;
      }
    }
  Weight& lm = h_p.w[W_LM];
  lm.codes = (const uint32_t*)codes[W_LM];
  lm.meta = (const __half2*)meta[W_LM];
  lm.x = h_p.xn;
  lm.y = h_p.logits;
  lm.n_in = HID;
  lm.f32 = 1;
  if (!g_stream) cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
  if (!d_state) {
    cudaMalloc(&d_state, 2 * sizeof(int));
    cudaMemset(d_state, 0, 2 * sizeof(int));
  }
  h_p.pos_ctr = d_state;
  h_p.pos = d_state + 1;   // internal pos buffer; bufs[13] is IR-only
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  return 0;
}

extern "C" int mx_load_sched(const void* blob, int nbytes) {
  const int* h = (const int*)blob;
  if (h[0] != 0x4C4D3033) return -1;
  g_nt = h[1];
  g_nb = h[2];
  g_fper = h[3];
  h_p.f_argc = h[4];
  if (d_sched) cudaFree(d_sched);
  cudaMalloc(&d_sched, nbytes);
  cudaMemcpy(d_sched, blob, nbytes, cudaMemcpyHostToDevice);
  h_p.sched = d_sched;
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  return 0;
}

extern "C" int mx_flags_alloc(int nsteps) {
  if (d_flags) cudaFree(d_flags);
  cudaMalloc(&d_flags, (size_t)nsteps * g_fper * sizeof(int));
  cudaMemset(d_flags, 0, (size_t)nsteps * g_fper * sizeof(int));
  h_p.flags = d_flags;
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  g_steps = nsteps;
  return 0;
}

extern "C" int mx_pos_set(int v) {
  return cudaMemcpy(d_state, &v, sizeof(int), cudaMemcpyHostToDevice) ==
                 cudaSuccess
             ? 0
             : -1;
}

extern "C" int mx_reset_flags() {
  return cudaMemset(d_flags, 0, (size_t)g_steps * g_fper * sizeof(int)) ==
                 cudaSuccess
             ? 0
             : -1;
}

extern "C" int mx_step() {  // baseline, no graph (correctness path)
  step_kernels(dev(), g_stream, h_p.ctx, h_p.nslice);
  return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0 : -1;
}

extern "C" int mx_graph_build() {  // capture one step as a CUDA graph
  if (g_graph) cudaGraphExecDestroy(g_graph);
  step_kernels(dev(), g_stream, h_p.ctx, h_p.nslice);  // warm caches
  cudaStreamSynchronize(g_stream);
  cudaGraph_t g;
  cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
  step_kernels(dev(), g_stream, h_p.ctx, h_p.nslice);
  if (cudaStreamEndCapture(g_stream, &g) != cudaSuccess) return -2;
  return cudaGraphInstantiate(&g_graph, g, 0) == cudaSuccess ? 0 : -3;
}

extern "C" int mx_graph_launch() {
  cudaGraphLaunch(g_graph, g_stream);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int mx_mega(int steps) {  // one launch covering `steps` decode steps
  h_p.steps = steps;
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  mega_kernel<<<g_nb, THREADS, 0, g_stream>>>(dev());
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int mx_sync() {
  return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0
                                                      : (int)cudaGetLastError();
}

// Timed runs: each iteration resets pos, launches one graph replay (or one
// multi-step mega launch) between events, and writes ms-per-step to out[i].
extern "C" int mx_time_graph(int iters, int pos0, float* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int i = 0; i < iters; ++i) {
    mx_pos_set(pos0);
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

extern "C" int mx_time_mega(int steps, int iters, int pos0, float* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  h_p.steps = steps;
  cudaMemcpy(dev(), &h_p, sizeof(P), cudaMemcpyHostToDevice);
  for (int i = 0; i < iters; ++i) {
    mx_pos_set(pos0);
    mx_reset_flags();
    cudaEventRecord(e0, g_stream);
    mega_kernel<<<g_nb, THREADS, 0, g_stream>>>(dev());
    cudaEventRecord(e1, g_stream);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    out[i] = ms / steps;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return 0;
}

extern "C" void mx_dump() {
  printf("w[112]: codes=%p meta=%p n_in=%d f32=%d x=%p y=%p\n",
         (void*)h_p.w[112].codes, (void*)h_p.w[112].meta, h_p.w[112].n_in,
         h_p.w[112].f32, (void*)h_p.w[112].x, h_p.w[112].y);
  printf("logits=%p xn=%p argp=%p tok=%p\n", (void*)h_p.logits,
         (void*)h_p.xn, (void*)h_p.argp, (void*)h_p.tok);
}

extern "C" int mx_task_grid(int grid, int op, int warg, int a0, int a1) {
  if (op == OP_GEMV)
    launch_gemv(warg, g_stream);
  else
    k_task<<<grid, THREADS, 0, g_stream>>>(dev(), op, warg, a0, a1);
  return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0
                                                      : (int)cudaGetLastError();
}

extern "C" float mx_time_gemvs(int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, g_stream);
  for (int i = 0; i < iters; ++i)
    for (int l = 0; l < NLAY; ++l)
      for (int k = 0; k < 4; ++k)
        launch_gemv(l * 4 + k, g_stream);
  launch_gemv(W_LM, g_stream);
  cudaEventRecord(e1, g_stream);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return ms / iters;
}

extern "C" float mx_time_gemv1(int widx, int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  launch_gemv(widx, g_stream);  // warm occupancy
  cudaStreamSynchronize(g_stream);
  cudaEventRecord(e0, g_stream);
  for (int i = 0; i < iters; ++i) launch_gemv(widx, g_stream);
  cudaEventRecord(e1, g_stream);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return ms / iters;
}

// Per-class timing of one LM-03 separate-kernel step at position `pos`
// (no graph): sums per launch into out[0]=gemv ms, out[1]=attn ms,
// out[2]=all-other ms. Also returns the kernel count in out[3].
extern "C" int mx_time_breakdown(int pos, float* out) {
  static cudaEvent_t* e = nullptr;
  const int MAXE = 600;
  if (!e) {
    e = (cudaEvent_t*)malloc(MAXE * sizeof(cudaEvent_t));
    for (int i = 0; i < MAXE; ++i) cudaEventCreate(&e[i]);
  }
  int ne = 0;
  mx_pos_set(pos);
  cudaStreamSynchronize(g_stream);
  const int nattn = n_attn_slices(h_p.ctx, h_p.nslice);
  float g_ms = 0.f, a_ms = 0.f, o_ms = 0.f;
  auto timed = [&](int cat, auto fn) {
    cudaEventRecord(e[ne++], g_stream);
    fn();
    cudaEventRecord(e[ne++], g_stream);
  };
  timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_TICK, 0, 0, 0); });
  timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_EMBED, 0, 0, 0); });
  for (int l = 0; l < NLAY; ++l) {
    if (l == 0)
      timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_NORM, l, 0, 0); });
    timed(0, [&] { launch_gemv(l * 4 + 0, g_stream); });
    timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_QKRA, l, 0, 0); });
    timed(1, [&] { k_task<<<nattn, THREADS, 0, g_stream>>>(dev(), OP_ATTN, l, 0, 0); });
    timed(1, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_ATTNC, 0, nattn, 0); });
    timed(0, [&] { launch_gemv(l * 4 + 1, g_stream); });
    timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_ANORM, NLAY + l, 0, 0); });
    timed(0, [&] { launch_gemv(l * 4 + 2, g_stream); });
    timed(2, [&] { k_task<<<8, THREADS, 0, g_stream>>>(dev(), OP_SILU, 0, 0, 0); });
    timed(0, [&] { launch_gemv(l * 4 + 3, g_stream); });
    timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_ANORM,
                  l + 1 < NLAY ? l + 1 : NORM_ROWS - 1, 1, 0); });
  }
  timed(0, [&] { launch_gemv(W_LM, g_stream); });
  timed(2, [&] { k_task<<<NARGP, THREADS, 0, g_stream>>>(dev(), OP_ARGP, 0, 0, 0); });
  timed(2, [&] { k_task<<<1, THREADS, 0, g_stream>>>(dev(), OP_ARGC, 0, NARGP, 0); });
  cudaStreamSynchronize(g_stream);
  // bucket by op category: 0 gemv, 1 attn(+attnc), 2 everything else
  // categories recorded implicitly by order; recompute from event pairs:
  // simpler: categories encoded as we go — store per-pair cat list on host
  float ms;
  // recount in launch order (same sequence as above)
  int i = 0, cat;
  auto take = [&](int c) {
    cudaEventElapsedTime(&ms, e[i * 2], e[i * 2 + 1]);
    ++i;
    if (c == 0) g_ms += ms; else if (c == 1) a_ms += ms; else o_ms += ms;
  };
  take(2); take(2);
  for (int l = 0; l < NLAY; ++l) {
    if (l == 0) take(2);
    take(0); take(2); take(1); take(1); take(0); take(2); take(0); take(2);
    take(0); take(2);
  }
  take(0); take(2); take(2);
  out[0] = g_ms;
  out[1] = a_ms;
  out[2] = o_ms;
  out[3] = (float)ne / 2;
  return 0;
}

extern "C" int mx_shutdown() {
  if (g_graph) { cudaGraphExecDestroy(g_graph); g_graph = nullptr; }
  cudaStreamSynchronize(g_stream);
  return 0;
}
