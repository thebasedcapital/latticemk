# LM-23 merge

```text
mt2 source/layouts
  + LM-20 attention, fused KV append, schedule
  + LM-21 all-column prologues and epilogues
  + phase boundaries at M1 norm and M5 batch attention
  = mt3, unchanged host ABI / GEMM / global partial rows
```

## Source conflict

A three-way `diff3 -m` merge used `megakernel_attn/mega_mt.cu`, `megakernel_mt2/mega_mt.cu`, and `megakernel_pro/mega_mt.cu`. It produced one textual conflict, in shared declarations next to `u`.

- Keep LM-20's `qhead[ATTROWS][128]`, `kvloc[ATTROWS][256]`, `merge_weight[ATTGROUP][32]` and `merge_max[ATTGROUP]`.
- Remove `red[WARPS]`, as LM-21 removes its only consumer, `block_sum`.
- Keep LM-21's separate `norm_part[MT][4]` and `norm_r[MT]`.

No other source conflict required a manual choice. The merged main loop calls LM-21's all-column helpers and LM-20's single all-column `attn2`, without a separate append phase.

## Attention storage contract

The shared `u.wpart` changes from `[32][130]` to `[ATTGROUP][32][130]`, and local Q/KV rows become per-column storage. These are CTA-private scratch, not the interface read by `pro_attnc`.

The global partial row is unchanged. Element 0 through 127 stores the unnormalized accumulator, 128 stores the softmax denominator, and 129 stores the maximum. The address is `p.part + col*NATTN*130 + (2*head+half)*130`. LM-20's `attn_merge` writes it; LM-21's float2 `pro_attnc` reads it. No conversion, padding, cache-layout change or additional synchronization is needed at this boundary. The existing ending grid barrier publishes every partial row.

## Schedule and detail wrapper

Use LM-20's fused `sched_mt.py`, not LM-21's unchanged mt2 schedule. Its output directory and validator manifest now belong to `bench/lm23/`. The runtime and IR both remove the separate append barrier. All 20 schedules, both modes, M1-5 and contexts 128/2048, ACCEPT [measured, `sched_mt.py` to `bench/lm23/schedules.json`].

The detail wrapper uses LM-21's all-column signatures with LM-20's fused attention call. This fixes a non-conflicting companion-file mismatch that a source-only diff would miss. Detail timings are not used for the LM-23 pass-latency decision.

## Structural register fixes

The unmodified merged selected build spilled at M1, 20 bytes stores/20 bytes loads, and M5, 8/8 bytes. M2-4 had zero spills [measured, initial `build.sh selected` compiler output]. Outlining batch attention removed M5 spills. Outlining attention at M1 did not work: batch-only outlining retained 20/20 bytes; outlining both modes worsened it to 28/48 bytes [measured, exploratory `build.sh selected 1 5` and `selected 1` compiler outputs].

The selected M1 instead outlines `pro_norm`; both attention paths stay inline at M1. M5 outlines batch attention and retains LM-20's outlined causal attention. M2-4 retain the input forks' attention qualifiers. Production threads remain 1024 at M1 and 512 at M2-5. Every production entry and outlined helper has zero stack/spill bytes [measured, final `build-selected-m*.log`].

`bench/lm23/build_trade.py` builds spill-free inline alternatives at 512 threads for M1 and 256 for M5. `trade.py` verifies their bits and measures them against selected boundaries. Selected wins all ten paired mode/context/M comparisons. These comparisons change threads and call boundaries together, so they measure the practical thread trade, not isolated call overhead. No spilled build was timed. The report retains exact costs and samples.
