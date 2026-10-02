# Measured results

Batch-1 greedy decode on a Quadro RTX 4000, Turing sm_75, with unlocked clocks. The read roofline in `hw/hw.json` is 406.791 GB/s; benchmark drivers use the 406.7 GB/s reference reported in `reports/wave-1.md`. The percentages below are byte-model estimates based on packed weights and full fp16 KV reads divided by elapsed time, not DRAM counters. Headline comparisons use same-session alternating samples at each context; a later repetition is not silently mixed into these pairs.

## Paired decode against llama.cpp Q4_0

All throughput is tokens/s, calculated from the median per-token latency. llama.cpp uses FA on, fp16 KV, commit `6011c34ce6099646ccdf0d39a61c6e681477c178`. Both engines have batch size one. The GPTQ kernel has better WikiText-2 perplexity than the Q4_0 pair in each model. Each table entry below comes from the two adjacent rows for that context in the named JSONL file. Speed ratio divides the corresponding row's tokens/s, with no cross-session comparisons.

| Model | Context | Megakernel GPTQ | llama.cpp Q4_0 | Paired speed ratio | Byte-model roofline, GPTQ / Q4_0 | Source |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 128 | 578.6 | 352.7 | 1.64x | 47.1% / 33.8% | `bench/lm11/results.jsonl`, GPTQ and Q4_0 FA-on adjacent rows, ctx 128, runs 27/30 |
| Qwen3-0.6B | 2048 | 416.6 | 284.3 | 1.47x | 56.5% / 42.7% | `bench/lm11/results.jsonl`, GPTQ and Q4_0 FA-on adjacent rows, ctx 2048, runs 27/30 |
| Qwen3-0.6B | 8192 | 221.6 | 187.6 | 1.18x | 68.4% / 60.7% | `bench/lm11/results.jsonl`, GPTQ and Q4_0 FA-on adjacent rows, ctx 8192, runs 27/30 |
| Qwen3-1.7B | 128 | 250.0 | 197.5 | 1.27x | 57.1% / 51.6% | `bench/lm12/results.jsonl`, GPTQ `paired_with=Q4_0` and adjacent Q4_0 FA-on rows, ctx 128 |
| Qwen3-1.7B | 2048 | 211.4 | 173.7 | 1.22x | 59.7% / 54.8% | `bench/lm12/results.jsonl`, GPTQ `paired_with=Q4_0` and adjacent Q4_0 FA-on rows, ctx 2048 |
| Qwen3-1.7B | 8192 | 145.7 | 133.2 | 1.09x | 66.4% / 65.1% | `bench/lm12/results.jsonl`, GPTQ `paired_with=Q4_0` and adjacent Q4_0 FA-on rows, ctx 8192 |

At 1.7B/context 8192 the engines were unloaded between alternating timing slices to fit the card. Loading lies outside both timing windows (`bench/lm12/results.jsonl`, `memory_mode`). The ctx-128 1.7B integrator repetition gave 251.2 versus 199.2 tokens/s in `reports/wave-5.md`; it is a separate pair, not a replacement for the recorded three-context set.

## Quality on WikiText-2 test

The llama-perplexity `-c 2048` protocol scores the second half of 146 non-overlapping windows, with fp16 KV. The GPTQ scores come from fake-quant fp16 GGUF files carrying the kernel's packed-weight reconstruction; these are *weight quality under llama.cpp arithmetic*, not a perplexity result from the decode kernel itself. The EXL2 quality uses its own corrected ExLlamaV2 forward on the same scoring positions. Lower is better.

| Model | Weights / quality engine | PPL | Source |
|---|---|---:|---|
| Qwen3-0.6B | GPTQ INT4-g128 fake-quant GGUF | 12.4435 | `bench/lm11/ppl.json`, `int4gptq-fake/f16` |
| Qwen3-0.6B | llama.cpp Q4_0 | 12.9617 | `bench/lm11/ppl.json`, `q4_0/f16` |
| Qwen3-0.6B | llama.cpp Q4_K_M | 12.1664 | `bench/lm11/ppl.json`, `q4_k_m/f16` |
| Qwen3-0.6B | ExLlamaV2 EXL2 4.65 decoder bpw, 6-bit head | 12.876691 | `bench/lm13/ppl_parts.jsonl`, 4.65bpw-causal, 146-window row |
| Qwen3-1.7B | GPTQ INT4-g128 fake-quant GGUF | 9.2371 | `bench/lm12/ppl.json`, `int4gptq-fake/f16` |
| Qwen3-1.7B | llama.cpp Q4_0 | 9.7140 | `bench/lm12/ppl.json`, `q4_0/f16` |
| Qwen3-1.7B | llama.cpp Q4_K_M | 8.8768 | `bench/lm12/ppl.json`, `q4_k_m/f16` |

Q4_K_M beats GPTQ quality at both sizes. It is shown for context, *not* as a quality-matched speed claim. The EXL2 4.65 bpw 0.6B run reached 192.0 / 106.6 / 34.6 tokens/s at contexts 128 / 2048 / 8192 (`bench/lm13/results.jsonl`, `quant_model_dir=baselines/exl2/qwen3-0.6b-4.65bpw-causal`). That pair used alternating same-process timing against v2, but a different timing mechanism and weight rate; see `reports/wave-5/LM-13.md` for the accounting.

## Correctness and static schedules

| Check | Result | Source |
|---|---|---|
| 0.6B teacher-forced v2.1, HF fp32 fake-GPTQ | PASS; max absolute logit diff 0.2114; zero hard flips; one near-tie flip; bitwise repeated logits | `bench/lm11/gate_v21.json`, `gptq`, `gptq_repeat`, `pass` |
| 1.7B teacher-forced v2.1, CPU HF fp32 fake-GPTQ | PASS; max absolute logit diff 0.05154; zero hard flips and nonfinite steps; bitwise repeated logits | `bench/lm12/gate_v21.json`, `first`, `second`, `pass` |
| Schedule validator | 0 false accepts and 0 false rejects over 10,000 fuzzed schedules; three original schedules accepted | `reports/wave-2.md`, LM-05 gate and tests; `validator/` |

Both gates use the same forced token sequence for reference and kernel, 64 steps per prompt across three prompts. The reported pass does not assert that free-running greedy generations are identical after a near-tie divergence. The kernel's timed `mega2_kernel` uses 36 CTAs by 1024 threads with 64 registers and no ptxas spills (`reports/wave-3/LM-03b.md`, `reports/wave-5/LM-12.md`).

## Failed ideas and limits

| Experiment | Measured observation | Source |
|---|---|---|
| Lattice A_n codes on weights | INT4-matched error needs at least 3.49 bits/weight including scales under the memoryless, independent-given-group-scale bound; best A4 point 17.23 PPL at 2.875 bits versus 13.43 for INT4 at 4.25 bits. Different older fp32 full-window PPL protocol from the tables above. | `reports/wave-1.md`, sections 1 and 3; `scripts/rate_bound.py`, `scripts/eval_ppl.py` |
| First persistent kernel | 0.96x of same-session CUDA graph at ctx 128: 415.1 versus 432.4 tokens/s; register-limited occupancy. | `bench/lm03b/results.jsonl`, first three ctx-128 rows; `reports/wave-2.md` |
| Occupancy fix | 1.42x versus separate graph at ctx 128: 612.0 versus 432.4 tokens/s. Kept. | `bench/lm03b/results.jsonl`, first three ctx-128 rows |
| Compressed KV kernel | 0.72x at ctx 8192: 6.288 versus 4.549 ms; correctness also fails with max absolute logit diff 3.22. Diagnostic, not an alternative engine. | `bench/lm09/results.jsonl`, ctx-8192 rows; `reports/wave-4/LM-09.md` |
| Finer sync | 0.93x at ctx 128: 1.865 versus 1.731 ms. | `bench/lm10/results.jsonl`, ctx-128 rows |
| llama.cpp q8_0 KV on Q4_K_M | 164.0 versus 184.5 tokens/s at ctx 8192 with fp16 KV; q4_0 KV instead raises PPL to 49.3823. | `bench/lm11/results.jsonl`, Q4_K_M FA-on fp16/q8_0 KV ctx-8192 rows; `bench/lm11/ppl.json`, `q4_k_m/q4_0+q4_0` |
| 1.7B fp16 dot accumulation | Original half partials produced max finite logit diff 24.6176 and 21 nonfinite steps; timed fp32 accumulation passed without spills. | `bench/lm12/gate_v21.json`, `fp16_dot_diagnostic`, `first`; `reports/wave-5/LM-12.md` |
| Stock ExLlamaV2 0.3.2 Qwen3 | Q/K head norm treated as LayerNorm and SDPA multi-token prefill lacked causal masking; first-window PPL 59.498, then 9.63618 with both config overrides. | `reports/wave-5/LM-13.md`, `bench/lm13/ppl.py` |

## Waves 6-7: mutation power and gate tiers

The base logit gate missed real faults. Extra probes helped, but a tolerance check against HF cannot distinguish every small single-layer bug from legitimate arithmetic changes. The cumulative gate now adds race/conditional probes, calibrated per-layer comparisons, and optional bitwise regression against the last accepted library. Regression is appropriate only for changes declared numerics-preserving.

| Check | Result | Source and selector |
|---|---|---|
| Mutation campaign, Qwen3-0.6B | Base gate kills 72.1%; with extra tests 89.1%; 22 non-equivalent mutants survive | `mutation/results.jsonl`, last row per `mutant_id`, `model="0.6B"`, exclude `family="control"`, equivalent and compile-failure rows; `mutation/summarize.py`, `status(..., "gpu_stage")` versus `status(..., "stage")` |
| Mutation campaign, Qwen3-1.7B | Base gate kills 71.9%; with extra tests 83.3%; 19 non-equivalent mutants survive | Same source and deduplication, `model="1.7B"` |
| Originals through cumulative tiers | 12/12 pass, including self-baseline regression | `gate/validation-originals.jsonl`, `gate.protocol="lm17-three-tier-fullchain-v1"`, latest `(engine,m,repeat)` for engines `v2`, `scale`, `mt` and the recorded M values; `gate.pass` |
| Known survivors, layer-only check | 34/41 rejected, 82.9%; 7 missed | `gate/campaign-results.jsonl`, latest `index` with `regression_enabled=true`, `cohort="survivor"`; `gate.tiers.3.layers.pass` |
| Known survivors, bitwise regression | 41/41 rejected | Same cohort; `gate.tiers.3.regression.pass` |
| Previously killed controls | 30/30 still rejected | Same deduplication, `cohort="previous-kill"`; `gate.pass` |

The layer-only misses include rounding changes inside calibrated envelopes, lm_head-only faults outside the dumps, and a position-dependent fault outside the layer probe. Exact regression catches this cohort, but cannot validate an intentional arithmetic change. Package reports: [LM-16](../reports/wave-6/LM-16.md) and [LM-17](../reports/wave-7/LM-17.md). The PPT feasibility package produced no accuracy result; its faithful run was compute-blocked, not an accuracy failure. See [LM-19](../reports/wave-7/LM-19.md) and `ppt/results.jsonl`, which is empty.

## Waves 6-8: multi-token trajectory

The pass-time target was M=4 at no more than 1.5x M=1, defined in [wave 6, gate decision](../reports/wave-6.md#1-gate-decision) and retained in [wave 8, gate decision](../reports/wave-8.md#1-gate-decision). No measured context met it. The table uses committed result files, not the separate integrator repetitions in the wave reports.

| Kernel / mode | Context | M4/M1 time | Decision | Source and selector |
|---|---:|---:|---|---|
| Wave 6 mt, batch | 128 | 2.56x | Miss | `bench/lm14/decision-128.json`, `context`, `t_over_t1.batch.4`, `kill` |
| Wave 7 mt2, batch | 128 | 2.19x | Miss | `bench/lm18/decision-128.json`, `context`, `t_over_t1.batch.4`, `kill` |
| Wave 8 mt3, batch | 128 | 1.83x | Miss | `bench/lm23/analysis.json`, `ratios[mode="batch",context=128].mt3_m4_over_m1`, `target_met` |
| Wave 8 mt3, causal | 128 | 1.89x | Miss | Same `ratios`, `mode="causal",context=128` |
| Wave 8 mt3, causal | 2048 | 2.46x | Miss | Same `ratios`, `mode="causal",context=2048` |
| Wave 8 mt3, causal | 8192 | 2.81x | Miss | Same `ratios`, `mode="causal",context=8192` |
| Wave 8 mt3, batch | 2048 | 3.28x | Miss | Same `ratios`, `mode="batch",context=2048` |

The merged kernel keeps sequential exactness in both modes for M=1 through M=5. All entries in `bench/lm23/gate.json` have `pass_gate=true`, `max_diff_sequential=0.0`, and `bitwise_repeat=true`; the largest `max_diff_hf` is 0.1875. Cumulative gate records are in `bench/lm23/tiers.jsonl` and `contract-tiers.jsonl`. The merge parallelized attention and per-column work, not the GEMM. The tensor-core experiment was rejected because it broke sequential exactness. Package reports: [LM-14](../reports/wave-6/LM-14.md), [LM-18](../reports/wave-7/LM-18.md), [LM-20](../reports/wave-8/LM-20.md), [LM-21](../reports/wave-8/LM-21.md), [LM-22](../reports/wave-8/LM-22.md), and [LM-23](../reports/wave-8/LM-23.md).

For a causal pass with one anchor and three drafts, the derived break-even accepted-draft means are strictly above 0.89 / 1.46 / 1.81 at contexts 128 / 2048 / 8192. Source: `bench/lm23/analysis.json`, `breakeven[k=4,context=128|2048|8192].mandatory_anchor_plus_bonus_min_accepted_drafts_strictly_greater_than`. These assume negligible CPU drafting and omit host/rollback overhead. They are not measured end-to-end speedups or thresholds for the adaptive policy below.

## Wave 8: turn-aware KV compaction

Compaction evicts complete turns and preserves logical positions while shortening the physical live cache. It changes the retained context, so it is not lossless full-history compression or an answer-quality result. See the [LM-24 package report](../reports/wave-8/LM-24.md).

| Measurement | Before | After | Change | Source and selector |
|---|---:|---:|---:|---|
| Decode with logical context 8192 | 8192 live rows, 216.4 tok/s | 2048 live rows, 405.8 tok/s | +87.6% throughput | `bench/lm24/timing.json`, `decode[name="v2-long"]` and `decode[name="kvc-compacted"]`, `logical`, `physical`, `tokens_per_s`; derived ratio |
| Session of 16384 tokens, including 8 compactions | 79.3 s | 44.8 s | 1.77x | `bench/lm24/session.json`, `tokens`, `compactions`, `totals_wall_s.v2-extended`, `totals_wall_s.kvc`, `speedup` |
| Copy cost for retaining 1024 / 2048 rows | n/a | 1.28 / 2.56 ms GPU time | Host wall time is separate | `bench/lm24/timing.json`, `copies[retained=1024|2048].gpu_ms` |
| Decode versus short-cache v2 | v2 at 2048 rows | Compacted cache at 2048 rows | 0.999x throughput | `bench/lm24/timing.json`, `compact_ratio` |
| No-compaction control | Original v2 logits | Bitwise unchanged | 0.993-0.995x throughput | `bench/lm24/timing.json`, `bitwise_no_compaction`, `no_event_ratios` |

The masked-HF correctness run covers 384 steps and two evictions, not the whole session. It reaches max logit error 0.1904 within bound 0.2158, repeats bitwise, and reports one near-tie flip. Source: `bench/lm24/correctness.json`, `no_events_steps`, `cases[repeat=0].events`, `.max_diff`, `.bound`, `.bitwise_repeat`, `.flips`. The long-session result proves timing and finite execution, not full-session agreement with HF: `bench/lm24/session.json`, `finite`, `extended_baseline_bitwise`. Final physical cache length is 4608 at logical position 16384, from `final_physical` and `final_logical`; the live cache is not fixed at its post-compaction length throughout the session.

## Wave 9: lossless prompt-lookup speculation

CPU longest-suffix lookup drafts tokens from the prompt and accepted output. The unchanged mt3 causal kernel verifies them; logical-position rollback hides rejected cache rows. Decode wall time includes drafting, index maintenance, host/device copies, synchronization and rollback. It excludes loading and prompt prefill, so this is not full-request latency. See the [LM-25 package report](../reports/wave-9/LM-25.md).

| Category | v2 tok/s | Speculative tok/s | Speedup, 95% prompt-bootstrap CI | Accepted drafts/pass | Verdict |
|---|---:|---:|---|---:|---|
| Code edits | 457.3 | 645.7 | 1.412x [1.252, 1.565] | 1.595 | Ship for evaluated workload |
| RAG-style answers | 453.4 | 650.9 | 1.436x [1.322, 1.563] | 1.753 | Ship for evaluated workload |
| Summarization | 453.9 | 417.0 | 0.919x [0.878, 0.975] | 0.445 | Disable |
| Open continuation control | 456.7 | 539.3 | 1.181x [1.087, 1.301] | 1.051 | Repetitive-text caveat |

Every table number comes from `spec/analysis.json`, `categories.code|rag|summarization|chat`, selecting `throughput_tps.v2`, `throughput_tps.spec`, `speedup_v2`, `speedup_v2_ci95`, and `mean_accepted`. Raw timing and acceptance traces are in `spec/results.jsonl`, selected by `category`, `id`, and nested `runs[].repeat` / `runs[].variant`. Aggregation uses total output tokens divided by summed per-prompt mean wall time. The confidence intervals resample paired prompts, not independent repeats.

Each category has 20 prompts and three repeats: `spec/analysis.json`, `categories.*.prompts` and `.repeats`. The CI procedure uses 10000 prompt resamples with seed 25, from `bootstrap_prompt_resamples` and `seed`. The shipping rule requires at least 1.15x versus v2 with CI lower bound strictly above 1.0, defined in [LM-25, gate decision](../reports/wave-9/LM-25.md#gate-decision-and-status). Code and RAG clear it; summarization slows down.

Lossless means identical to the same-build mt3 M=1 greedy target, **not v2**. All 80 prompts and 240 repeat pairs match that target: `spec/analysis.json`, `identity_reference`, `identity_prompts_passed`, `identity_repeat_pairs_passed`; raw `spec/results.jsonl`, `runs[variant="spec"].identity_mt1` and `runs[].tokens`. V2 and mt3 diverge on 11 prompts, from `v2_diverged_prompts`; `spec/divergence.jsonl` records each first divergence. The original v2 identity requirement failed and was corrected rather than hidden. V2 remains the speed baseline.

The `chat` key labels open continuation, not instruct-style chat. Its mean repeated generated four-gram fraction is 41.78%, from `spec/analysis.json`, `categories.chat.output_repeated_fourgram_fraction_mean`. Repetition inflates lookup acceptance; the control's speedup is not evidence of a general chat gain. Greedy decoding only was evaluated. No long-session compaction/speculation combination or long-context speculative speedup is claimed.

## Reproduction wall time

The original waves 1-5 `./reproduce.sh all` run completed using cached checkpoints and packed GPTQ weights. The table below describes that historical command, not a run of the newly extended `all`. Stage times are wall seconds, including GPU-lock waits and per-stage setup where applicable. Build/download setup sits in the overall time rather than in a stage. These are not fresh-checkout calibration estimates. The waves 6-9 stage runs and remaining scope are recorded in [the wrap-up run log](WRAPUP.md).

| Stage | Wall seconds | Source |
|---|---:|---|
| Correctness gates | 81 | `publish/reproduction_times.jsonl`, `gate` |
| Static validation, all schedule families | 0 (whole-second timer) | `publish/reproduction_times.jsonl`, `validate` |
| Paired benchmarks, both sizes and all contexts | 83 | `publish/reproduction_times.jsonl`, `bench` |
| WikiText-2 perplexity | 536 | `publish/reproduction_times.jsonl`, `ppl` |
| Original four charts | 2 | `publish/reproduction_times.jsonl`, first `charts` row |
| Original waves 1-5 end-to-end command | 716 | `publish/reproduction_times.jsonl`, first `all` row |

The new stages each completed on cached artifacts. These are stage-body wall seconds, including builds inside the stage and GPU-lock waits but excluding the common setup. Sources select the latest row for each stage in `publish/reproduction_times.jsonl`. These separate invocations do not establish a new end-to-end `all` wall time.

| New stage | Wall seconds | Exit code | Source |
|---|---:|---:|---|
| Schedule regeneration and manifest checks | 13 | 0 | `publish/reproduction_times.jsonl`, latest `schedules` |
| Cumulative gates, v2 and mt3 causal M=4 | 102 | 0 | `publish/reproduction_times.jsonl`, latest `gate-tiers` |
| Multi-token correctness and short timing | 52 | 0 | `publish/reproduction_times.jsonl`, latest `multitoken` |
| Compaction correctness, timing and session | 157 | 0 | `publish/reproduction_times.jsonl`, latest `compaction` |
| Fixed speculative subset and full analysis | 44 | 0 | `publish/reproduction_times.jsonl`, latest `spec` |
| Seven charts from recorded data | 3.339 | 0 | `publish/reproduction_times.jsonl`, latest `charts` |

Charts: [speed](charts/speed_vs_llamacpp.png), [quality/speed](charts/ppl_vs_speed.png), [experiments](charts/waves.png), [byte-model roofline](charts/roofline.png), [multi-token trajectory](charts/multitoken.png), [KV compaction](charts/compaction.png), and [speculative decoding](charts/speculative.png). Regenerate from recorded rows with `.venv/bin/python publish/make_charts.py`.
