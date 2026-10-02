# Numeric claims check

`T1` to `T11` refer to the draft thread's post bodies; `N` refers to the research note. Repeated values share a row. Reported precision is shown beside the exact stored value; ratios and percentages marked derived are computed in the cited reports from paired rows. Model names, format names, file paths, post indices and version identifiers are identifiers, not measured values. The headings' character counts are checked separately below. No unresolved numeric mismatches.

| Draft claim and location | Repo source and exact value | Check |
|---|---|---|
| Quadro RTX 4000, Turing sm_75, 36 SMs, 8 GB nominal; `N`, `T1`, `T2`, `T11` | `hw/hw.json`: `name="Quadro RTX 4000"`, `compute_capability="7.5"`, `sm_count=36`, `global_mem_bytes=8149073920`; `reports/wave-5.md`: Quadro RTX 4000, Turing, 8 GB | Match. GB is marketed nominal capacity, not byte conversion. |
| 2018 card; `T1`, `N` | Quadro RTX 4000 identity: `hw/hw.json`; launch year 2018 is supplied explicitly in the verified project brief, not measured or stated in a repo report | External historical descriptor, not a benchmark claim; provenance disclosed here. |
| Read ceiling 406.7 GB/s and specification 416.1 GB/s; `N` | `reports/wave-1.md` §3: 406.7 GB/s and 416.1 GB/s. `hw/hw.json`: raw median `406.791`, spec `416.1` | Match report convention. Read median in JSON rounds to 406.8 at one decimal; the project's published 406.7 follows its report. |
| Batch size 1, ctx 128 / 2048 / 8192; `T1`, `T3`, `T4`, `T6`, `T7`, `T10`, `N` | `bench/lm11/results.jsonl` GPTQ/Q4_0 paired rows: `batch=1`, `context=128,2048,8192`; same in `bench/lm12/results.jsonl` | Match. Thread's `8k` means context 8192. |
| Engine 36 CTAs × 1024 threads, 64 registers, 0 spills; `T2`, `T6`, `N` | `reports/wave-3/LM-03b.md` §4: `36 CTAs × 1024 threads`, `64 regs, 0 spills`; `reports/wave-5/LM-12.md` §2 confirms scaled kernel same layout | Match. |
| GPTQ INT4 group size 128, fp16 KV; `T2`, `N` | `reports/wave-4.md` §1: GPTQ INT4-g128, fp16 KV; `reports/wave-5/LM-12.md` §§2-3 scaled GPTQ and F16 KV | Match. |
| RTN/GPTQ interleaved ratio 0.9969 at ctx 128; `N` | `reports/wave-4/LM-11.md` §3: RTN 1.7178 ms, GPTQ 1.7232 ms, RTN/GPTQ `0.9969` | Match. |
| Validator 0 false accepts and 0 false rejects on 10,000 fuzzed schedules; `N` | `reports/wave-2.md` §1: `0 false accepts / 0 false rejects over 10,000 fuzzed schedules` | Match. |
| Timing 27 v2 and 30 llama samples per context; `N` | `reports/wave-4/LM-11.md` §3 paired GPTQ rows: 27 and 30; `reports/wave-5/LM-12.md` §3: 27 and 30. `runs` keys in paired JSONL agree | Match. |
| V2 8 steps per timed launch, llama-bench 32 generated tokens per sample; `N` | `bench/lm11/interleave.py`: `mk2_time_mega(8, 1, ...)`, `--llama-n` default `32`; `bench/lm12/interleave.py`: same call and default; `bench/lm11/results.jsonl` matched llama rows `n_gen=32` | Match. |
| 0.6B GPTQ / Q4_0 tok/s at 128: 578.6 / 352.7, 1.640x or 1.64x; `T1`, `T3`, `N` | `bench/lm11/results.jsonl` paired rows 49-50: `578.6179868174606 / 352.71799965694976`; `reports/wave-4/LM-11.md` §3: `578.6 / 352.7`, ratio `1.640` | Match, rounded report display. |
| 0.6B GPTQ / Q4_0 tok/s at 2048: 416.6 / 284.3, 1.465x; `T3`, `N` | `bench/lm11/results.jsonl` rows 51-52: `416.60279187073576 / 284.3129984488926`; `reports/wave-4/LM-11.md`: `416.6 / 284.3`, ratio `1.465` | Match. |
| 0.6B GPTQ / Q4_0 tok/s at 8192: 221.6 / 187.6, 1.181x or 1.18x; `T1`, `T3`, `N` | `bench/lm11/results.jsonl` rows 53-54: `221.55578358664258 / 187.5864992949919`; `reports/wave-4/LM-11.md`: `221.6 / 187.6`, ratio `1.181` | Match. |
| 1.7B GPTQ / Q4_0 tok/s at 128: 250.0 / 197.5, 1.266x or 1.27x; `T4`, `N` | `bench/lm12/results.jsonl` rows 1-2: `249.96826455572224 / 197.50299927089716`; `reports/wave-5/LM-12.md` §3: `250.0 / 197.5 / 1.266` | Match. Do not substitute separate integrator rerun `251.2 / 199.2`. |
| 1.7B GPTQ / Q4_0 tok/s at 2048: 211.4 / 173.7, 1.217x or 1.22x; `T4`, `N` | `bench/lm12/results.jsonl` rows 5-6: `211.35429482724425 / 173.6789907876024`; `reports/wave-5/LM-12.md` §3: `211.4 / 173.7 / 1.217` | Match. |
| 1.7B GPTQ / Q4_0 tok/s at 8192: 145.7 / 133.2, 1.094x or 1.09x; `T4`, `N` | `bench/lm12/results.jsonl` rows 9-10: `145.65088935751848 / 133.15049968268988`; `reports/wave-5/LM-12.md` §3: `145.7 / 133.2 / 1.094` | Match. Rows explicitly say `memory_mode=unload-megakernel-between-rounds`. |
| 0.6B GPTQ / Q4_0 / Q4_K_M PPL 12.4435 / 12.9617 / 12.1664; `T3` rounded 12.44 / 12.96, `N` full | `bench/lm11/ppl.json`: `int4gptq-fake/f16=12.4435`, `q4_0/f16=12.9617`, `q4_k_m/f16=12.1664` | Match. Q4_K_M better PPL, not matched. |
| 1.7B GPTQ / Q4_0 / Q4_K_M PPL 9.2371 / 9.7140 / 8.8768; `T4` rounded 9.24 / 9.71, `N` full | `bench/lm12/ppl.json`: `9.2371`, `9.714`, `8.8768`; `reports/wave-5/LM-12.md` §3 writes Q4_0 as `9.7140` | Match report formatting; Q4_K_M better PPL. |
| PPL protocol 146 windows of 2048 tokens, second half scored; `N`, `T3` | `bench/lm11/ppl.json` `protocol` field; `bench/lm12/ppl.json` `protocol` field; `reports/wave-5/LM-13.md` §3 explains same positions for EXL2 | Match. |
| 0.6B gate max 0.2114, one near-tie and no hard flips; `N` | `reports/wave-4/LM-11.md` §§1,3: `0.2114`, one permitted near-tie, zero hard flips, bitwise repeated logits; `bench/lm11/gate_v21.json` | Match. |
| 1.7B fp16 finite max diff 24.6176; fp32 max diff 0.05154; `T9`, `N` | `reports/wave-5/LM-12.md` §§3-4: `24.6176` then NaNs, corrected `0.05154`, no hard flips, bitwise repeat; `bench/lm12/gate_v21.json` | Match. |
| 0.6B roofline GPTQ 47.1% / 68.4%; Q4_0 33.8% / 60.7% at 128 / 8192; `N` | `bench/lm11/results.jsonl` paired rows 49-50 and 53-54: `47.13407153658504 / 68.43011045958988` and `33.84900545690026 / 60.659512800161465` | Match one-decimal rounding. Derived bytes/time, not counters. |
| 1.7B roofline GPTQ 57.1% / 66.4%; Q4_0 51.6% / 65.1% at 128 / 8192; `N` | `bench/lm12/results.jsonl` paired rows 1-2 and 9-10: `57.07842445404724 / 66.37962206038101` and `51.62912044251319 / 65.08547182154238` | Match one-decimal rounding. |
| EXL2 decoder 4.65 bpw with 6-bit head, 0.6B PPL 12.876691; `T8`, `T10`, `N` | `reports/wave-5/LM-13.md` §§1,3: decoder 4.65 bpw, 6-bit head, test PPL `12.876691`, effective head rate `6.53 bpw`; `bench/lm13/results.jsonl` rows 6,8,10 | Match. Decoder target is not whole-model bit rate. |
| Corrected EXL2 tok/s 192.0 / 106.6 / 34.6 at 128 / 2048 / 8192; `T8`, `T10`, `N` | `bench/lm13/results.jsonl` EXL2 rows 6,8,10: `192.04414295912989 / 106.61871731066584 / 34.646001678615065`; `reports/wave-5/LM-13.md` §3 rounds to `192.0 / 106.6 / 34.6` | Match. Paired only with its adjacent v2 rows, not with LM-11 table. |
| Shannon floor >=3.49 bits/weight, INT4 4.25 bits/weight, max 18% saving, extrapolated A4 ~5%; `T5`, `N` | `reports/wave-1.md` §§1,3: `3.49`, `4.25`, `18%`; A4 ~5% is an extrapolation from measured SQNR, explicitly `[derived, extrapolated beyond measured range]` | Match conditional bound; an earlier thread draft called the A4 saving measured and was corrected. |
| A4-k11 2.875 bits, PPL 17.23; INT4 4.25 bits, PPL 13.43; A4 0.74x INT4 GEMV throughput; `N` | `reports/wave-1.md` §3, quality and matvec tables: `2.875 / 17.23`, `4.25 / 13.43`, A4-k9/k10 token ratios `0.74` | Match; A4-k11 PPL and A4-k9/k10 speed are distinct evaluated configurations, not one measured joint point. |
| First megakernel v1 0.96x / 0.80x / 0.66x at 128 / 2048 / 8192; 214 regs, 256 threads, eight warps/SM; `T6`, `N` | `reports/wave-2.md` §§1,3-4: `0.96 / 0.80 / 0.66`, `36 CTAs × 256 threads, 214 regs`, `8 warps/SM` | Match. |
| v2 1.42x / 1.50x / 1.44x at those contexts, 32 warps/SM; `T6`, `N` | `reports/wave-3.md` §§1,3; `reports/wave-3/LM-03b.md` §§3-4: `1.42 / 1.50 / 1.44`, `32 warps/SM` | Match, separate graph ablation rather than llama.cpp comparison. |
| 143 barriers at 1.04 microseconds each; `N` | `reports/wave-3/LM-03b.md` §§3-4: `143`, empty grid barrier `1.04 µs` | Match empty-barrier probe, not profiler attribution. |
| KV cache 939.524 -> 367.002 MB/token, saving 572.522 MB/token at ctx 8192; `T7`, `N` | `reports/wave-4/LM-09.md` §3 table and discussion: `939.524 -> 367.002`, `572.522` | Match modeled packed/read bytes. |
| Compressed KV 6.288 vs 4.549 ms/token and 0.724x v2 speed; `T7`, `N` | `reports/wave-4/LM-09.md` §§1,3: `6.288 / 4.549`, `0.724x`; `bench/lm09/results.jsonl` | Match, explicitly diagnostic because gate failed. |
| KV codec PPL 12.806, integrated logit difference 3.2162 versus limit 0.4900; `N` | `reports/wave-4/LM-09.md` §§1,3: `12.806`, `3.2162`, `0.4900`; quality screen and integrated gate are different protocols | Match, with protocol distinction. |
| llama.cpp q8 KV 184.5 -> 164.0 tok/s, PPL near 12.17; q4 KV PPL 49.3823; `N` | `reports/wave-4/LM-11.md` §3: Q4_K_M/f16 `184.5 / 12.1664`, Q4_K_M/q8_0 `164.0 / 12.1658`, q4_0 `49.3823`; `bench/lm11/ppl.json` | Match; q8 comparison not paired across clock sessions, so only an observed direction, not a claimed paired speed ratio. |
| Sync fork 1.8649 vs 1.7309 ms/token, 0.9282x at 128; `T7`, `N` | `reports/wave-4/LM-10.md` §§1,3: fork `1.8649`, v2 `1.7309`, `0.9282x`; `bench/lm10/results.jsonl` | Match paired ablation. |
| ExLlamaV2 version 0.3.2, first-window PPL unmodified 59.498, headnorm-only 38.260, both fixes 9.63618, HF 9.63661; `T8`, `N` | `reports/wave-5/LM-13.md` §3: exact sequence `59.498 / 38.260 / 9.63618 / 9.63661`, installed version in `reports/wave-5.md` | Match. |

## Thread length audit

The character count includes each post body's spaces, punctuation, newline separating a chart marker, and the marker itself. `## Post` headings and this explanatory text are excluded. The counts are `247, 278, 231, 262, 262, 269, 239, 279, 267, 267, 246` for posts `1` through `11`; all are at most `280`. The research note is within the requested word range. No draft was posted.

## Thread v2

Integrator rewrite of `publish/x_thread.md` (12 posts). Lengths come from `publish/check_thread.py` (all at most 280). The "Thread length audit" above refers to the superseded v1 draft.

| post | claim in draft | source | value at source |
|---|---|---|---|
| 1, 3 | 579 vs 353 tok/s, 1.64x; 1.47x, 1.18x | `bench/lm11/results.jsonl`, `reports/wave-4/LM-11.md` | 578.6 / 352.7, 1.640x; 1.465x; 1.181x |
| 1 | 2018 card | assignment brief (Quadro RTX 4000 release) | not a repo measurement |
| 2 | 36 blocks of 1024 threads | `reports/wave-3/LM-03b.md`, ptxas | 36 CTAs x 1024, 64 regs |
| 2 | checker for races and deadlocks | `reports/wave-2/LM-05.md` | 0 false accepts / rejects, 10,000 cases |
| 3 | 1.7B 1.27x, 1.22x, 1.09x | `bench/lm12/results.jsonl`, `reports/wave-5/LM-12.md` | 1.266x, 1.217x, 1.094x |
| 4 | PPL 12.44 vs 12.96; 9.24 vs 9.71; Q4_K_M better | `bench/lm11/ppl.json`, `bench/lm12/ppl.json` | 12.4435 / 12.9617; 9.2371 / 9.714; Q4_K_M 12.1664, 8.8768 |
| 5 | near 2.4 bits | spec (D-02 claim) | 2.41 b/coord |
| 5 | Gaussian within 0.0005 bits; >= 3.49 bits | `reports/wave-1.md`, `scripts/rate_bound.py` | h(X) 2.0466 vs 2.0471; 3.49 b/w |
| 6 | 0.96x; 214 registers; 8 warps/SM; 64 regs at 1024 threads; 1.42x | `reports/wave-2.md`, `reports/wave-3.md` | 0.96x; 214 regs at 256 threads/CTA, 1 CTA/SM; 1.42x |
| 7 | 6.25-bit cache, 573 MB less, 0.72x, failed gate | `reports/wave-4/LM-09.md`, `bench/lm09/results.jsonl` | 6.25 b/coord; 939.524 - 367.002 = 572.522 MB; 0.724x; max logit diff 3.2162 |
| 7 | llama.cpp q8_0 KV slower than f16 | `reports/wave-4/LM-11.md` | Q4_K_M FA-on 319.1 vs 340.7 (ctx 128), 164.0 vs 184.5 (8k) |
| 8 | 0.93x; race passed schedcheck, kernel did not implement the IR; caught by run-to-run disagreement | `reports/wave-4/LM-10.md` sec. 4 | 0.9282x; single cumulative counter, "run-to-run unstable greedy counts ... even while schedcheck accepted the IR" |
| 9 | fp16 accumulation: 24.6 error then NaN; fp32: 0.05, 64 regs, no spills | `reports/wave-5/LM-12.md`, `bench/lm12/gate_v21.json` | 24.6176 max finite, NaN late; 0.05154; 64 regs, 0 spills |
| 10 | ExLlamaV2 0.3.2 bugs; 59.5 vs 9.64; 3-6x slower | `reports/wave-5/LM-13.md` | 59.498 vs 9.63661; 3.03x / 3.90x / 6.35x |
| 11 | interleaved timing; teacher-forced gate; bitwise repeat | `reports/wave-4.md` (gate v2.1) | as stated |
| 12 | about 12 minutes | `publish/reproduction_times.jsonl` | 716 s, models and GPTQ weights cached |
