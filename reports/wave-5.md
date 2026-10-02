# Wave 5 report — scale to Qwen3-1.7B (LM-12), ExLlamaV2 EXL2 baseline (LM-13)

2026-09-30 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · llama.cpp `6011c34c` · ExLlamaV2 0.3.2 · agents on `openai-codex/gpt-6-sol`
Package reports: `reports/wave-5/LM-12.md`, `reports/wave-5/LM-13.md`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-12 1.7B correctness (v2.1) | **pass** | max abs logit diff 0.0515, 0 flips, bitwise repeat; re-run by integrator |
| LM-12 speed vs llama.cpp Q4_0 at better PPL | **pass, narrower** | 1.27x / 1.22x / 1.09x at ctx 128 / 2k / 8k; integrator re-run at ctx 128: 1.26x |
| LM-12 4B stretch | **not reached** | bf16 checkpoint needs 7.7 GB vs 4.9 GB free; needs 32-head mapping and masked GEMV chunks |
| LM-13 ExLlamaV2 EXL2 | **done** | v2 3.0x / 3.9x / 6.4x faster at 3.5% better PPL |

## 2. Integrator checks

- `bench/lm12/check_gate_v21.py` re-run: PASS (0.0515). `schedcheck` ACCEPT for `kernels/megakernel_scale/schedules/scale_ctx{128,2048,8192}.json`.
- `bench/lm12/interleave.py 128 --llama bench/lm12/qwen3-1.7b-Q4_0.gguf`: **v2-1.7B-GPTQ 3.981 ms (251.2 tok/s, 57.4% roofline) vs llama.cpp Q4_0 FA-on 5.019 ms (199.2 tok/s, 52.1%), 1830 MHz** `[measured]`.

## 3. Headline numbers

| model | engine | PPL (llama-perplexity -c 2048) | ctx 128 | ctx 2048 | ctx 8192 |
|---|---|---|---|---|---|
| Qwen3-0.6B | v2 + GPTQ INT4 | 12.44 | 578.6 | 416.6 | 221.6 |
| Qwen3-0.6B | llama.cpp Q4_0 FA-on | 12.96 | 352.7 | 284.3 | 187.6 |
| Qwen3-0.6B | ExLlamaV2 EXL2 4.65 bpw | 12.88 (own eval) | 192.0 | 106.6 | 34.6 |
| Qwen3-1.7B | v2-scale + GPTQ INT4 | 9.24 | 1.27x | 1.22x | 1.09x (vs Q4_0) |
| Qwen3-1.7B | llama.cpp Q4_0 FA-on | 9.71 | 199.2 (integrator) | — | — |
| Qwen3-1.7B | llama.cpp Q4_K_M | 8.88 | not quality-matched (better PPL) | | |

Absolute 1.7B tok/s per context: `bench/lm12/results.jsonl` `[measured]`.

## 4. Honest negatives

- **The advantage shrinks with model size and context:** 1.65x (0.6B, ctx 128) -> 1.26x (1.7B, ctx 128) -> 1.09x
  (1.7B, 8k). llama.cpp reaches 52% of roofline on 1.7B vs 34% on 0.6B: a larger share of its time is streaming, so
  fixed per-kernel overheads matter less.
- **fp16 dot products broke on 1.7B:** v2's half2 FMA partials gave max logit diff 24.6 and NaN logits; the 1.7B
  kernel accumulates in fp32 (same bytes, 0 spills). The 0.6B engine still uses fp16 partials and passes its gate (0.39).
- **ExLlamaV2 0.3.2 mis-runs Qwen3 out of the box:** q/k head norm treated as LayerNorm (Qwen3 uses RMSNorm) and
  causal masking skipped in its SDPA prefill path (first-window PPL 59.5 vs HF 9.64). Fixed via config overrides
  only (`config.arch.lm.headnorm='rmsnorm'`, `config.no_sdpa=True`); upstream not edited. Its low speed here is
  largely host dispatch on a tiny model.
- At ctx 8192 on 1.7B, engines are unloaded between alternating timing slices (VRAM), disclosed in LM-12.
- Q4_K_M has better PPL than our GPTQ at both sizes; it is never claimed as a matched comparison.

## 5. Publish verdict

Worth publishing as an honest engineering note: a reproducible, correctness-gated result against the standard
engine on an old consumer-class card, with the negative results (lattice weights under the Shannon floor, compressed
KV that does not pay, finer sync that loses, fp16 accumulation collapse, ExLlamaV2 Qwen3 bugs) as research content.
Not a headline "Nx faster" claim: the gain is 1.1x–1.65x depending on model size and context.
