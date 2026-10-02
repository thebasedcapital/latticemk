# Wave 9 report — lossless speculative decoding on the multi-token kernel (LM-25)

2026-10-02 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · agent on `openai-codex/gpt-6.1-sol`.
Package report: `reports/wave-9/LM-25.md`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| Kill line: some category >= 1.15x over v2 with CI lower bound > 1.0 | **pass** | code 1.41x [1.25, 1.56], RAG 1.44x [1.32, 1.56] |
| Lossless output (vs its own target, mt3 M=1 greedy) | **pass** | 80/80 prompts, 240/240 repeat pairs identical |
| Ship per category | code, RAG: **ship**; summarization: **disable** (0.92x) | continuation control 1.18x, but its text is repetitive |

## 2. Integrator checks

- Re-ran `spec/run.py` on prompts 8-10 of code, RAG and summarization (3 repeats each): output identical to mt3
  greedy on all 9; per-prompt spec/v2 speed ratios within about 1% of the agent's (e.g. code-10 1.792 vs 1.792,
  rag-08 2.299 vs 2.271, summarization-10 0.920 vs 0.907) `[measured]`.
- Spec corrected by the integrator mid-run: the identity reference is mt3's own greedy decoding, not v2. v2 and mt3
  round differently and flip near-ties: 11 of 80 prompts diverge (code-04 at generated index 1, v2 top-2 margin
  0.0147, mt3 0.0028). Both engines pass the HF gate; v2 remains the speed baseline.

## 3. Headline numbers

Qwen3-0.6B, greedy, prompts 128-1901 tokens, 64-256 generated tokens, 20 public-source prompts per category,
decode wall time including CPU drafting, host-device copies and KV rollback `[measured, spec/results.jsonl;
derived, spec/analyze.py]`:

| category | v2 tok/s | speculative tok/s | speedup [95% CI] | accepted drafts per pass | verdict |
|---|---|---|---|---|---|
| code edits | 457.3 | 645.7 | 1.41x [1.25, 1.56] | 1.60 | ship |
| RAG-style answers | 453.4 | 650.9 | 1.44x [1.32, 1.56] | 1.75 | ship |
| summarization | 453.9 | 417.0 | 0.92x [0.88, 0.98] | 0.44 | disable |
| open continuation (control) | 456.7 | 539.3 | 1.18x [1.09, 1.30] | 1.05 | see negatives |

Drafting: CPU longest-suffix lookup over prompt + output, adaptive k = 1..5; verification by mt3 causal mode;
rollback by resetting the logical position (rejected rows are overwritten before reuse). No kernel change.

## 4. Honest negatives

- **Summarization gets slower:** its acceptance (0.44) is below the 0.60 threshold once host overhead is counted.
- **The continuation control is not open chat:** 42% of its generated 4-grams repeat, which inflates acceptance.
  Do not read 1.18x as a chat speedup; instruct-style chat was not evaluated.
- Greedy decoding only; no sampling. No 8k-context speculative workload.
- The identity requirement against v2 failed on 11 of 80 prompts; that requirement was the integrator's spec error
  (different engine, legitimate rounding), not a speculation bug.

## 5. Where the project ends

This was the last research wave. Next is a wrap-up wave with no new kernels: refresh `reproduce.sh` for waves 6-9,
regenerate charts, and draft a second public thread (KV compaction + lossless speculative decoding on code/RAG).
