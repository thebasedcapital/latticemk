# LM-22 fork changes

```text
production default -> unchanged mt2 CUDA-core GEMM
-DMMA              -> experimental INT4 -> fp16 A/B -> FP32 m16n8k8
merge recommendation: no GEMM change
```

- `mega_mt.cu`, GEMM include selection: add opt-in `MMA` without changing the default include, GEMM call sites, attention, prologues, ABI or shared-memory pool. The unified diff against mt2 is `mega_mt.diff`.
- `gemm_mma.cuh`, `mma_pair`: decode existing adjacent-column nibbles through `0x6400` and apply each group's FP16 scale/offset directly to A registers.
- `gemm_mma.cuh`, `mma_stage`: register-backed in-place transpose and column XOR bank swizzle remove conflicting B-fragment addresses without changing external staging.
- `gemm_mma.cuh`, `gemm_mma<NI>`: coalesced 128-weight group loads, direct PTX fragments, padded zero B columns, FP32 accumulation, output tails and optional deterministic split-K reduction.
- `gemm_mma.cuh`, `gemv_run`: select the compile-time supported width so staging address arithmetic does not use runtime integer division. The 128-width branch is used by the tiny CPU-reference probe.
- `gemm_probe.cu`, `shape`: stage exactly the production adjacent half2 activation pairs and exercise either copied mt2 GEMM or MMA in isolation.
- `gemm_probe.cu`, `shape_time`: CUDA-event timing and observable output for CPU checks, using the five unchanged packed matrix shapes.
- `build_probe.sh`: build CUDA, direct MMA and swizzled MMA with `--fmad=false`, selectable split factor and M.
- `build.sh`, variant selection: add all-MMA diagnostic builds at 512 threads; keep selected production flags/thread counts unchanged; use `--fmad=false` for copied profile/detail variants.
- `gate_contract.json`: record the unchanged selected-production compiler/thread/sequential-reference contract for the generic gate.
- `sched_mt.py`, final evidence path only: direct hypothetical fork schedule results to `bench/lm22`, not the baseline's directory. Emission/interface semantics are unchanged; this script was not run.
- `bench/lm22/engine.py`, library path only: point the copied identical Python interface at this fork.
- `bench/lm22/check_gate.py` and `detail_profile.py`: copied baseline tools retained for scoped reproduction; not run after the integrator stopped full-pass work.
- `bench/lm22/fragment_check.py`: CPU-exact dyadic oracle for fragment mapping, M1-8 padding, row tails and split-K.
- `bench/lm22/shape_probe.py`: actual GPTQ-row CPU checks before 27-round same-process shape interleaving; save every sample and clock.
- `bench/lm22/sass.py`: reproduce static fragment-body instruction counts and parse retained build logs into compiler resource evidence.
- `bench/lm22/*.json`: measured fragment, shape and compiler evidence; `shapes-dynamic-width.json` preserves the earlier bank-swizzle session.

`gemm_preload.cuh`, `gemm_nv.cuh`, `gemm.cuh`, `gemm_original.cuh`, `mega_detail.cuh` and `profile_detail.cuh` are unchanged copies. No weight files were repacked. The experimental all-MMA M4/M5 persistent builds spill and were never timed. See `reports/wave-8/LM-22.md` for the performance kill, integrator stop decision and unmeasured limits.
