# Wave 3 report — Lattice Megakernel (LM-08 fused graph engine, LM-03b megakernel retry)

2026-09-30 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · nvcc 13.3 · clocks unlocked (ratios from interleaved runs only)
Package reports: `reports/wave-3/LM-08.md`, `reports/wave-3/LM-03b.md`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-03b stage kill (matvec-only persistent loop >= 90% of separate GEMVs) | **pass** | 1.076 vs 1.273 ms (1.18x) without barriers; 0.891 with per-GEMV grid barriers (recorded) |
| LM-03b final kill (>= 1.1x LM-03 graph at ctx 128) | **pass** | 1.42x, reproduced by the integrator (1.640 / 1.644 vs 2.324 / 2.327 ms) |
| LM-08 (no kill line) | **done** | 1.36–1.43x over LM-03, reproduced |
| G3 megakernel (spec: >= 1.15x over the INT4 megakernel baseline) | **pass vs the LM-03 separate-graph engine** | megakernel-v2 1.42x / 1.50x / 1.44x at ctx 128 / 2k / 8k |

Engine of record: **megakernel-v2** (`kernels/megakernel_v2/mega2.cu`). It is at least as fast as the fused graph
engine at every context and 6% faster at 2k.

## 2. Status per work package (integrator re-checks)

| WP | status | checks re-run by integrator |
|---|---|---|
| LM-03b | done, passes both kill lines | correctness PASS (64/64 tokens x 3 prompts, max abs logit diff 0.138); ptxas `mega2_kernel` 64 regs, 0 spills, 20,096 B SMEM; `schedcheck` ACCEPT mk2_ctx{128,2048,8192}; 1 step per launch keeps the ratio (1.44x), so the gain is not launch amortization |
| LM-08 | done | correctness PASS (max abs logit diff 0.1125); two interleaved timing rounds |

## 3. Headline numbers (integrator runs, `bench/lm08/bench.py` and `bench/lm03b/bench.py`, 25 interleaved runs each) `[measured]`

| ctx | LM-03 separate graph | LM-08 fused graph (169 nodes) | megakernel-v2 (36 CTAs x 1024 thr) | v2 % roofline |
|---|---|---|---|---|
| 128 | 2.324 ms (430 tok/s) | 1.36–1.40x | **1.640 ms (610 tok/s), 1.42x** | 49.7 |
| 2048 | 3.456 ms (289 tok/s) | 2.447 ms, 1.41x | **2.298 ms (435 tok/s), 1.50x** | 59.0 |
| 8192 | 6.27 ms (159 tok/s) | 4.237 ms, 1.43x (vs a 6.04 ms baseline in its run) | **4.344 ms (230 tok/s), 1.44x** | 71.1 |

At ctx 2048 both runs saw the same baseline (3.449 vs 3.456 ms): v2 2.298 ms vs fused 2.447 ms, 6% `[measured]`.
At ctx 8192 the two are tied within the clock spread between processes.

Versus the wave-2 start of the day: 434 -> 610 tok/s at ctx 128, 166 -> 230 tok/s at ctx 8192 `[measured]`.

## 4. Honest negatives

- **Wave-1's 1.306 ms GEMV reference was a boost-clock artifact** of that harness; interleaved inside the engines the
  113 GEMVs take 1.71–1.74 ms (LM-08).
- **Fusion has a tax on the GEMV:** +164 us over the 113-GEMV sequence in the fused engine (norm prologue ~1.4 us,
  SiLU ~4.8 us, lm_head epilogue ~18 us). Still a net win against ~1 ms of separate kernels.
- **In-kernel attention combine lost** to a separate combine kernel (30–60 us/layer slower) in the graph engine.
- **Grid barriers are not free:** 143 per step at 1.04 us each is about 150 us of the 1.64 ms at ctx 128.
- ctx 128 is still at 50% of roofline (1.64 ms vs a 0.82 ms floor).

## 5. Open threads (next wave candidates)

- **v2 barrier reduction:** replace full-grid barriers with flag-grained dependencies or split arrive/wait
  (~150 us available at ctx 128); fix the attention position-half imbalance.
- **KV compression on v2 (LM-07 + D-02):** at ctx 8192 v2 is at 71% of roofline with KV bytes 940 MB/token vs
  317 MB of weights, so KV bytes are now the main cost at long context. Needs D-02 code, or start with KIVI-int4
  (ppl 15.46, 3.8x fewer KV bytes) as a known-good baseline in the kernel.
- The fused graph engine (LM-08) is a working fallback and a useful ablation reference; no further work planned on it.
