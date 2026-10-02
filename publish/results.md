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

## Reproduction wall time

`./reproduce.sh all` completed on the machine above using cached checkpoints and packed GPTQ weights. Stage times are wall seconds, including GPU-lock waits and per-stage setup where applicable. The build/download setup sits in the overall time rather than in a stage. These are not estimates of a fresh calibration run.

| Stage | Wall seconds | Source |
|---|---:|---|
| Correctness gates | 81 | `publish/reproduction_times.jsonl`, `gate` |
| Static validation, all schedule families | 0 (whole-second timer) | `publish/reproduction_times.jsonl`, `validate` |
| Paired benchmarks, both sizes and all contexts | 83 | `publish/reproduction_times.jsonl`, `bench` |
| WikiText-2 perplexity | 536 | `publish/reproduction_times.jsonl`, `ppl` |
| Four charts | 2 | `publish/reproduction_times.jsonl`, `charts` |
| End-to-end command | 716 | `publish/reproduction_times.jsonl`, `all` |

Charts: [speed](charts/speed_vs_llamacpp.png), [quality/speed](charts/ppl_vs_speed.png), [experiments](charts/waves.png), [byte-model roofline](charts/roofline.png). Regenerate from recorded rows with `.venv/bin/python publish/make_charts.py`.
