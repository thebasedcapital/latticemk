# Wave 2 report — Lattice Megakernel (LM-03, LM-05, LM-07)

2026-09-30 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · nvcc 13.3 · torch 2.11.0+cu128 · rustc 1.98.1
Package reports: `reports/wave-2/LM-03.md`, `reports/wave-2/LM-05.md`, `reports/wave-2/LM-07.md`.
Shared interfaces used: Schedule IR v1 (frozen in the wave-2 task contract), GPU lock `scripts/gpu.sh`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-03 kill line (mega >= 1.1x graph at ctx 128) | **killed, for the design built** | 0.96x (2.399 vs 2.311 ms), reproduced by the integrator; 0.80x at 2k, 0.66x at 8k |
| G5 correctness (LM-05) | **pass** | 0 false accepts / 0 false rejects over 10,000 fuzzed schedules; LM-03's 3 real schedules ACCEPT; 3 integrator mutations of a real schedule all REJECT |
| LM-07 (KV codec prep) | **done, no kill** | 2.4 b/coord at INT4 distortion is possible on K (Shannon floor 1.98 b) but not on V (3.35 b) |
| G3 (end-to-end megakernel) | **not reachable as built** | follows from the LM-03 kill |

## 2. Status per work package

| WP | status | tests (re-run by integrator) |
|---|---|---|
| LM-03 | killed at kill line after a valid baseline | `bench/lm03/check_correctness.py both`: PASS, 64/64 tokens x 3 prompts, max abs logit diff 0.129 `[measured]` |
| LM-05 | done | `cargo test --release`: 15/15; `schedcheck` on `kernels/megakernel/schedules/*.json`: 3/3 ACCEPT `[measured]` |
| LM-07 | done | `kvcodec/results.jsonl` 10 rows; fp16 row reproduces the wave-1 baseline ppl 12.66 `[measured]` |

Integrator interventions: LM-05's first run died on a provider connection error (resumed). LM-03's first KILL was
rejected because both of its engines used a spilling GEMV (a stale `#define RPW 12`; lm_head 1.30 ms vs 0.22 ms);
after porting the wave-1 GEMV verbatim (0 spills) the comparison was re-run.

## 3. Headline numbers

Decode step, batch 1, INT4-g128 weights, fp16 KV (`bench/lm03/bench.py`, 25 interleaved runs, SM clock 1830–1845 MHz)
`[measured]`:

| ctx | separate kernels + CUDA graph | megakernel (36 CTAs x 256 thr, 214 regs) | ratio | graph % roofline |
|---|---|---|---|---|
| 128 | 2.303 ms (434 tok/s) | 2.395 ms (418 tok/s) | 0.96x | 35.4 |
| 2048 | 3.433 ms (291 tok/s) | 4.292 ms (233 tok/s) | 0.80x | 39.5 |
| 8192 | 6.030 ms (166 tok/s) | 9.128 ms (110 tok/s) | 0.66x | 51.2 |

- The 113 INT4 GEMVs alone take 1.306 ms (wave 1); the other ~1.0 ms at ctx 128 is ~170 small non-matvec kernels
  (norms, q/k-norm, RoPE, KV append, attention, SiLU, residuals) `[derived]` from the two measurements.
- Roofline floor at ctx 128: ~332 MB/token / 406.7 GB/s = 0.82 ms, so the graph baseline is at 35% `[derived]`.

Validator (`validator/`, `schedcheck`): 49,932-task schedule checked in 186.8 ms median (p10 183.6, p90 192.2,
n=21) `[measured]`.

KV (`kvcodec/`), wikitext-2 ppl, fp16 12.66 `[measured]`:

| codec | b/coord | ppl |
|---|---|---|
| int8 per-token | 8.25 | 12.74 |
| KIVI-int4 (K per-channel, V per-token) | 4.25 | **15.46** |
| int4 per-token | 4.25 | 155.7 |
| KIVI-int2 | 2.25 | 166.3 |
| A2-k5 + RHT (stand-in, not D-02) | 2.625 | 433.7 |

Shannon floor at INT4 SQNR, incl. metadata `[derived, assumes i.i.d. given scale]`: K per-token 1.98 b (K is
heavy-tailed: kurtosis +39); V per-token 3.35 b (near-Gaussian, like the weights).

## 4. Honest negatives

- **The persistent megakernel loses to CUDA graphs on this card at every context.** Its register allocation is the
  maximum over all task types, which forces 1 CTA x 8 warps per SM; separate memory-bound kernels run 16–32 warps
  per SM. About 170 serialized flag handoffs cost about as much as the graph dispatch they replace. At 8k context,
  attention spread over 36 CTAs starves bandwidth (0.66x).
- **Wave 1's "+60% from a megakernel" was matvec-only.** The real decode step has ~1 ms of non-matvec work that the
  megakernel as built does not remove.
- **"Matched distortion to INT4" is a straw target for KV:** plain per-token INT4 KV has good SQNR but ppl 155.7.
  The honest bar is KIVI-int4 at 15.46. SQNR does not rank KV codecs (KIVI-int4 has worse K SQNR, 10x better ppl).
- **The A_n stand-in is far from the D-02 claim on KV** (ppl >= 434 at 2.375–2.625 b).

## 5. Open threads (need a gate decision)

- **A. Fuse inside the graph engine instead of a persistent kernel.** Fold RMSNorm into GEMV prologues;
  q/k-norm + RoPE + KV append into the qkv epilogue; SiLU·mul into the down-proj prologue; residual adds into
  epilogues. That cuts ~285 nodes to ~115 while keeping per-kernel occupancy. Targets the ~1 ms non-matvec time.
- **B. One bounded megakernel retry (occupancy variant):** keep x in SMEM so the matvec fits 64 regs, run
  1 CTA x 1024 threads per SM, and give attention its own warp group. Directly tests LM-03's stated root cause;
  kill if still < 1.1x at ctx 128.
- **C. D-02 on the K cache.** Blocked on the D-02 code (`machine-proof-builds/`). Plug-in steps are in
  `reports/wave-2/LM-07.md` section 5; the bar is KIVI-int4 ppl 15.46; if D-02 is K-only, pair it with int4-V.
- LM-04 residency remains low value on the 0.6B model (on-chip SMEM is < 1% of bytes/token).
