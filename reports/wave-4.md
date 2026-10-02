# Wave 4 report — Lattice Megakernel (LM-09 KV compression, LM-10 barriers, LM-11 external baselines)

2026-09-30 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · llama.cpp `6011c34c` (CUDA, sm_75) · clocks unlocked
Package reports: `reports/wave-4/LM-09.md`, `reports/wave-4/LM-10.md`, `reports/wave-4/LM-11.md`.
Agents: first run on `bifrost/devin/swe-2` (two provider connection drops), resumed on `openai-codex/gpt-6-sol`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-09 KV compression (>= 1.3x v2 at ctx 8192) | **killed** | 0.72x (6.288 vs 4.549 ms) despite 572 MB/token fewer KV bytes; kernel also fails correctness (max abs logit diff 3.22) |
| LM-10 barrier reduction (>= 1.05x v2 at ctx 128) | **killed** | 0.93x (1.865 vs 1.731 ms); passes correctness after fixing a race of its own |
| LM-11 external baselines | **done** | v2 + GPTQ beats llama.cpp Q4_0 at lower perplexity at every context |
| Correctness gate | **revised to v2.1** | teacher-forced, max abs logit diff <= 0.5 and <= 1.25x v2, near-ties listed, bitwise run-to-run determinism |

Engine of record unchanged: **megakernel-v2**, now with **GPTQ INT4-g128 weights** (`bench/lm11/weights_int4_gptq.pt`,
same bytes and speed as RTN: 0.997x interleaved).

## 2. Status and integrator checks

| WP | status | integrator check |
|---|---|---|
| LM-09 | killed | read report; quantizer bit-exact vs Python, bug is downstream (unresolved, moot after kill) |
| LM-10 | killed | read report; fork has 20 B spill stores, not a v2 replacement |
| LM-11 | done | re-ran the headline pair at ctx 128: **v2-GPTQ 1.697 ms (589.4 tok/s) vs llama.cpp Q4_0 FA-on 2.799 ms (357.3 tok/s), 1.65x, 1830 MHz** `[measured, bench/lm11/interleave.py]` |

## 3. Headline numbers

Perplexity, one protocol (`llama-perplexity -c 2048`, wikitext-2 test, 146 windows) `[measured, bench/lm11/ppl.json]`:

| weights | PPL |
|---|---|
| HF fp16 | 11.20 |
| llama.cpp Q8_0 | 11.23 |
| llama.cpp Q4_K_M | 12.17 |
| **v2 INT4-g128 GPTQ** | **12.44** |
| llama.cpp Q4_0 | 12.96 |
| v2 INT4-g128 RTN | 14.66 |

Decode tok/s, batch 1, f16 KV, same-session paired runs `[measured, bench/lm11/results.jsonl]`:

| ctx | v2 + GPTQ | llama.cpp Q4_0 FA-on | ratio | llama.cpp Q4_K_M FA-on (unpaired) |
|---|---|---|---|---|
| 128 | 578.6 | 352.7 | 1.64x | 340.7 |
| 2048 | 416.6 | 284.3 | 1.47x | 277.3 |
| 8192 | 221.6 | 187.6 | 1.18x | 184.5 |

Other engines (not clock-paired, see LM-11): HF fp16 eager 31 tok/s, `torch.compile` 95.5, ExLlamaV2 fp16 (no quant) 73.5 at ctx 128.

## 4. Honest negatives

- **Compressed KV does not speed up decode on this card, for us or for llama.cpp.** LM-09: 0.72x with 61% fewer KV
  bytes. llama.cpp Q4_K_M with q8_0 KV: 164.0 vs 184.5 tok/s at 8k (f16 KV). With q4_0 KV the PPL goes 12.17 -> 49.38.
- **Finer-grained sync lost** to v2's plain grid barriers (0.93x); per-phase counters added work and spills.
- The v2 advantage shrinks with context (1.64x -> 1.18x): attention over an fp16 cache dominates at 8k.
- Q4_K_M has better PPL than v2-GPTQ (12.17 vs 12.44), so v2 is not quality-matched to it; the claimed pair is Q4_0.
- ExLlamaV2 was run unquantized (fp16), not in its native EXL2 4-bit format; that comparison is not yet fair.

## 5. Next (wave 5)

- **Scale beyond 0.6B:** a result on one tiny model invites "launch overhead only". Port v2 to Qwen3-1.7B (and 4B if it
  fits in 8 GB) and repeat the paired llama.cpp comparison.
- **ExLlamaV2 EXL2 4-bit baseline**, quality measured.
- **Publication package:** one-command reproduction, charts, research note, X thread draft.
