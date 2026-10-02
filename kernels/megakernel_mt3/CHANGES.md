# LM-23 combined multi-token kernel

```text
mt2                    mt3
serial attention  ->   LM-20 batch sharing / causal warp-private KV sharing
serial prologues  ->   LM-21 all-column norm / staging / combine / SiLU / argmax
separate append   ->   fused append and attention, one fewer grid barrier/layer
GEMM / ABI / layouts    unchanged
```

## Inventory

- `mega_mt.cu` merges LM-20 and LM-21 onto mt2. The only textual conflict removes obsolete norm-reduction scratch while preserving LM-20's per-column Q/KV and partial-merge scratch. `MERGE.md` records the choices and global partial-row compatibility.
- M1 `pro_norm` and M5 batch attention have explicit outlined boundaries to keep production spill-free. Causal attention remains outlined at M2-5 and inline at M1. Thread counts are unchanged, 1024 at M1 and 512 at M2-5. The final production register counts are 64/117/127/128/128 [measured, `build.sh selected`, final build logs].
- `sched_mt.py` follows LM-20's fused schedule and writes the 20 schedules and validator manifest under `bench/lm23/`. Both modes, M1-5, ctx128/2048 ACCEPT [measured, `bench/lm23/schedules.json`].
- `mega_detail.cuh` combines the all-column detail signatures with the fused attention phase. It is not the production timing source.
- `build.sh` retains the exact selected compiler flags, including `--fmad=false`. `gate_contract.json` declares those flags, production thread counts and the package's own M1 sequential reference.
- `gemm.cuh`, `gemm_nv.cuh`, `gemm_original.cuh`, `gemm_preload.cuh` and `profile_detail.cuh` are unchanged mt2 files. LM-22 contributes no GEMM change.
- `bench/lm23/` has strict correctness, raw-bit dump association, same-process mt2/mt3/v2 timing, the measured lower-thread trade, raw JSON/JSONL, schedules and derived break-even analysis. All M1-5 production timings use CUDA events, 27 interleaved samples and recorded SM clocks.

## Gate companion changes

`gate/debug.py` adds detection and instrumentation for the all-column P2 prologue. Existing scalar paths are unchanged. `gate/contract.py` and `gate/layers.py` select the trusted tier-3 calibration library by compiler/thread contract, not by a candidate-specified calibration path. Default LM-14 keeps its original baseline; the accepted noncontracting contract uses mt2. Unknown contracts fail closed. The same 213 positions, full/local probes, max-absolute/RMS metrics and k=1.25 procedure apply. `gate/README.md` documents both changes.

The initial mt3 and unchanged mt2 tier-3 full-chain failures were identical, 19 failures each. Contract-keyed calibration resolves them without changing kernel arithmetic or the multiplier. Final mt3 causal M4, mt2 causal M4, default LM-14 causal M4 and v2 all PASS tiers 1-3. M1 dumping preserves logit bits at 48 positions with complete finite hidden/attention arrays [measured, `bench/lm23/{regression-tiers,contract-tiers}.jsonl`, `dump-m1.json`].

Correctness passes, but every measured M4/M1 curve misses the <=1.5 wave target. The report gives the full timing matrix, including regressions at long-context independent batch M3/M5, and the conditional speculative-decoding decision. See `reports/wave-8/LM-23.md`.
