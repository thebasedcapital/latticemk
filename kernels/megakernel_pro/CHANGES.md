# LM-21 changes against megakernel_mt2

```text
per-column CTA prologues -> column/element flattened prologues
attention partial format -> unchanged [column][32][130]
GEMM + attention + append -> unchanged
schedule/global barriers -> unchanged
```

- Shared `red` and `block_sum`: remove the now-unused single-row reduction scratch and helper.
- `xresidual` and `norm_part`/`norm_r`: update all residual columns in one pass and reduce their four active warp partials concurrently, retaining the original floating-point order.
- `stage_norm` and `pro_norm`: stage every normalized column with one publication barrier.
- `pro_attnc`: give each head a full warp, broadcast unchanged softmax scales, use aligned float2 partial loads, and stage all outputs with one publication barrier.
- `pro_silu`: flatten column/element work across all CTA threads, read contiguous chunks, and publish with one barrier.
- `argp2`: allocate disjoint warp groups to columns and reduce all per-CTA argmax partials after one shared-memory barrier.
- `argc2`: merge each column's 36 global partials in its own warp and write token/history without a CTA scratch reduction.
- `mega_mt`: replace serial owned-phase invocations with all-column invocations and remove the redundant per-column argmax CTA barriers. Preserve all grid barriers and attention/KV invocations.
- `mega_detail.cuh`: profile the changed all-column invocations without altering attention trace boundaries.
- `build.sh`: use `--fmad=false` for diagnostic profiles as well as selected builds.
- `bench/lm21/engine.py`: use fork libraries with the unchanged constructor and ABI.
- `bench/lm21/check_gate.py`: copy the LM-18 strict gate to the fork.
- `bench/lm21/bench.py`: interleave the unmodified selected mt2 and pro full passes in one process.
- `bench/lm21/detail_profile.py` and `build_profile.sh`: interleave exact-contract baseline and fork diagnostic builds and retain phase clocks.
- `gate_contract.json`: declare the fork's actual compiler flags, thread counts, and M1 sequential library for the integrator's generic gate fix.
- `bench/lm21/summarize.py`: derive pass deltas and owned-phase M4-minus-M1 increments from the paired sample files.

The requested unified kernel diff is `mega_mt.diff`. No GEMM header, `attn2`, `append`, weight/KV layout, external ABI, or schedule-generator change is part of this fork.
