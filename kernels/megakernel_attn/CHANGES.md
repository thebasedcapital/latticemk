# LM-20 changes

```text
Changed: attention, append and their barrier schedule
Kept:    GEMMs, pro_attnc, all other prologues, ABI and buffer layouts
```

- Shared attention storage `u.wpart`, `qhead`, `kvloc`: two independent batch groups, all causal local rows, inherited 32-partition layout. Causal partial merge also runs two query columns concurrently through these groups.
- `attn_setup`: head-wise Q/K norm and RoPE plus local new-row materialization for each active query.
- `attn_append_local`: copy locally computed K/V to the unchanged cache exactly once per new row.
- `attn_batch`: distribute sequence/head/half work items across all 36 CTAs and two warp groups per CTA at M>1.
- `attn_causal`: warp-private register K/V row tiles compute all M queries over the union of their original partition ranges, skipping empty gaps. Q operands and softmax states remain in registers; each loaded row serves all matching queries. No CTA tile barriers serialize virtual partitions. Per-query FP order and causal masks remain unchanged.
- `attn_merge_weights`, `attn_merge`: compute each softmax correction once per warp partial, share it across elements, and stride over all 130 entries even in narrower thread groups; preserve serial summation order.
- `attn2`: dispatch concurrent batch or shared-load causal attention.
- Removed separate `append` and the M>1 append grid barrier in `mega_mt`: attention consumes local in-block K/V instead of waiting for other CTAs' cache writes.
- `mega_detail.cuh`: instrument the same fused attention path and remove the obsolete append phase.
- `sched_mt.py`: emit the fused schedule, concurrent batch ownership and local causal reads; validator interface unchanged.
- `build.sh`: detailed/coarse profiles use the same `--fmad=false` contract as production. Final thread counts match mt2 at every M, 1024 for M1 and 512 for M2-5. The discarded contiguous design's M5 256-thread exception is removed.
- `gate_contract.json`: bind sequential and jitter builds to the candidate's numerical/compiler contract.
- `build_reference.sh`: compile unmodified mt2 detailed profiles with the same compiler flags into this fork, leaving mt2 untouched.

Minimal out-of-region edits are the attention-specific shared arrays and profile call sites. `pro_attnc`, GEMM headers and all host ABI functions are unchanged. Batch attention inlines; causal attention outlines at M>1 to keep register lifetimes separate, and inlines at M1.

The engine bridge in `bench/lm20/engine.py` differs from LM-18 only in its library directory. `check_gate.py` retains the strict sequential tolerance of 0.0. `bench.py` and `detail_profile.py` run same-process interleaved comparisons at context 128 with 27 samples per row.

`context_bench.py` checks raw M1/M4 logit bits against mt2 and mt2 sequential steps on nonzero seeded long-context KV, then measures paired production and phase curves. `analyze.py` derives slice targets, logical historical-KV address bytes over all 28 layers, and the first tested winning context. Logical bytes are not a DRAM counter.

Final performance selection: both paths beat mt2 M4 at context 128; causal also wins at 2048 and 8192. No measured losing context warrants an added dispatch threshold. The sweep only bounds the first sampled winning context (128), not an exact crossover below it. Neither attention scaling slice target passes; see `reports/wave-8/LM-20.md`.
