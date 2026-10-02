# Wave 1 report — Lattice Megakernel (LM-00, LM-01, LM-02)

2026-09-29/30 · Quadro RTX 4000 (sm_75) · driver 610.57.04 (CUDA 13.3) · nvcc 13.3 · torch 2.11.0+cu128
Spec: `Lattice Megakernel — Agent Build Spec` (Sep 29, 2026). Repo: `~/Projects/latticemk` (spec path
`/home/admin/lattice-megakernel/` does not exist on this machine).

## 1. Gate decision

| gate | decision | one-line reason |
|---|---|---|
| LM-00 hardware | **pass** | all fields read from device; read roofline 406.7 GB/s, p10–p90 spread 0.07–0.17% over 3 x 20 runs |
| G1 codec | **fail, unpassable** | matching INT4 error needs >= 3.49 b/w for any memoryless codec on these weights (18% max saving < 25%); A_n measured ~5% |
| Gate A (LM-01 kill) | **killed** | rate saving at matched error ~5% < 15% kill line |
| G2 matvec | **fail** | A4 codes 0.72–0.78x INT4; A2 codes 1.02–1.11x on the block shapes (74% of bytes), 1.24–1.39x only on lm_head |
| Gate B (LM-02 kill) | **killed for A4**, not triggered for A2 | A4 slower than INT4 on every shape; A2 is faster but fails G1/G4 |
| G4 quality | **fail** (measured early) | best lattice point PPL 17.23 at 2.875 b vs int4 13.43; at 2.375 b PPL 25.41 |
| G3, G5 | not started | LM-03 / LM-05 not started (see open threads) |

Per spec: the simplex path stops here; the INT4 megakernel (LM-03) remains a valid result on its own.

## 2. Status per work package

| WP | status | tests |
|---|---|---|
| LM-00 | done | `hw/hwprobe.cu` -> `hw/hw.json` |
| LM-01 | killed at kill line | `scripts/sweep_mse.py`, `scripts/rate_bound.py`, `scripts/eval_ppl.py`; H=I LDLQ == RTN exactly (int4, A4-k9) |
| LM-02 | killed (A4); A2 moot after LM-01 | `scripts/check_kernels.py`: ok, worst rel err 1.29e-3 over 7 formats x NC 1/2/3 x RHT |
| LM-03, LM-04, LM-05, LM-06 | not started | — |

## 3. Headline numbers

LM-00 (`hw/hwprobe.cu`, 20 runs each, 512 MiB buffers):
- 36 SMs, 64 KB SMEM/SM (48 KB default, 64 KB opt-in per block), 64 K regs/SM, 1024 threads/SM, 4 MB L2 `[measured]`
- read roofline 406.7 GB/s (p10 406.2, p90 406.9); copy 358.4–360.6 GB/s; spec sheet 416.1 GB/s `[measured]`
- launch overhead 1.67–1.70 us per back-to-back launch; CUDA-graph node 0.61 us `[measured]`
- clocks: cannot be locked (`nvidia-smi -lgc` needs root, sudo needs a password; application clocks deprecated on
  this driver). No power or thermal capping recorded during runs `[measured]`

LM-01 codec (22 matrices: layers 0/13/27 x 7 projections + 16384 lm_head rows):
- group-RMS-normalized weights have h(X) = 2.0466 bits vs 2.0471 for a Gaussian, excess kurtosis 0.085
  (-0.047 after RHT) `[measured]` (`scripts/rate_bound.py`)
- Shannon lower bound to match INT4-RTN-g128 error (20.27 dB): **3.49 b/w incl. scales**, i.e. at most 18% below
  INT4's 4.25 `[derived]`, assuming weights i.i.d. given the group scale `[assumed]`
- the same bound at INT3-level error (14.45 dB) is 2.40 b/w + 0.125: D-02's 2.41 b matches INT3-level error on
  weights, not INT4 `[derived]`
- A4 SQNR: 11.26 / 12.56 / 13.92 dB at 2.375 / 2.625 / 2.875 b; extrapolating (~5.4 dB/bit) to 20.27 dB gives
  ~4.04 b, a ~5% saving `[derived, extrapolated beyond measured range]`
- decode: one SMEM table lookup per d weights plus 2–4 integer ops; no per-weight search `[measured]` (SASS loop)
- perplexity, wikitext-2 test, fp32 12.66 (`scripts/eval_ppl.py`) `[measured]`:

  | config | b/w | +RHT+LDLQ | LDLQ only |
  |---|---|---|---|
  | int4 | 4.25 | 13.43 | 14.08 |
  | int3 | 3.25 | 16.40 | 21.99 |
  | A4-k11 | 2.875 | 17.23 | 29.67 |
  | A4-k10 | 2.625 | 19.97 | 35.06 |
  | A2-k5 | 2.625 | 21.38 | 52.32 |
  | A4-k9 | 2.375 | 25.41 | 59.57 |
  | int2 | 2.25 | 138.7 | — |

LM-02 matvec (`scripts/bench_gemv.py`, real quantized weights, 113-kernel CUDA graph, median of 7 interleaved
rounds, roofline 405.9 GB/s from the in-run read kernel) `[measured]`:

| config | b/w | ms/token | lm_head % roof | block shapes % roof | vs INT4: lm_head / blocks / token `[derived]` |
|---|---|---|---|---|---|
| int4 | 4.25 | 1.318 | 94.9 | 50.8 | 1 / 1 / 1 |
| A2-k4 | 2.125 | 1.145 | 66.0 | 28.3 | 1.39 / 1.11 / 1.15 |
| A2-k5 | 2.625 | 1.221 | 72.4 | 32.1 | 1.24 / 1.02 / 1.08 |
| A4-k9 | 2.375 | 1.777 | 38.3 | 22.1 | 0.72 / 0.78 / 0.74 |
| A4-k10 | 2.625 | 1.777 | 41.9 | 23.5 | 0.72 / 0.75 / 0.74 |

Block shapes carry 74% of INT4 bytes per token, lm_head 26% `[derived]`.

## 4. Honest negatives

- **D-02 does not transfer to weights.** The 2.41 b vs 4.00 b result sits below the Shannon floor for these
  (near-Gaussian) weights at INT4's error. No codec can pass G1 as specified.
- **Lattice structure buys nothing in a LUT decoder.** Lloyd-trained 4-D tables beat A4 by 0.43–0.63 dB SQNR
  (lloyd4-k10 PPL 348.9 vs A4-k10 1458 under RTN).
- **LUT decode is SMEM-bank-bound.** A4 tables of 512–2048 uint2 entries conflict (38–42% of roofline). Tried and
  rejected: rows in flight 4 and 8 (no gain, 8 spills), 4 independent accumulators (accuracy only), a 256-entry pair
  LUT for A2-k4 (half the LDS count, 1.7x slower).
- **RTN is unusable below 3 b** on this 0.6B model: A4-k9 RTN PPL 3756; even int3 RTN 41.4.
- **Fused RHT prologue** costs 0.3–0.6 ms/token when every block recomputes it (e.g. int4 1.367 -> 1.690 ms).

## 5. Open threads and spec deviations

Deviations (all disclosed; none changes a gate outcome):
1. D-02 codec code (`machine-proof-builds/`) is not on this machine and no online peer has it. LM-01 used my own
   A_n truncated-ball LUT code (`lmk/codebooks.py`). The Shannon-bound result covers any memoryless codec,
   D-02 included; a D-02 exploiting cross-weight dependence is the one case it does not cover.
2. Codec in Python + CUDA, not Rust; no frozen `codec/FORMAT.md` and no bit-exact roundtrip test (killed first).
3. llama.cpp 4-bit baseline not built: the local CUDA 13.3 install has no cuBLAS. Nsight Compute is not installed;
   DRAM throughput is derived from bytes/time, not profiled.
4. Benchmarks use 7 interleaved rounds (spread < 1% for INT4; VQ drifts with boost clock), not 20.
5. `reports/` rows are Markdown tables, not the JSONL benchmark-row interface.

Proposals for the next wave (need a gate decision):
- **A. INT4 megakernel (LM-03 + LM-05) as the main line.** Block shapes stream at 50.8% of roofline vs 94.9% for
  lm_head; bringing all 316.6 MB/token to 95% gives ~0.82 ms, ~1220 matvec tok/s (+60%) `[derived]`. Measured
  per-node costs (1.7 us launch, 0.61 us graph node) x 113 nodes = 69–192 us of the 1.318 ms `[derived]`.
- **B. INT3 + RHT + LDLQ as the compression lever:** 3.25 b, PPL 16.40, 24% fewer bytes than INT4, LUT-free
  decode. Needs a kernel; quality cost vs INT4 is +22% PPL, so it fails G4 as written.
- **C. LM-04 residency** is unaffected by the codec kill and can run on INT4.
