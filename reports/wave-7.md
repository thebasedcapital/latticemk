# Wave 7 report — gate tiers (LM-17), multi-token stall diagnosis (LM-18), PPT feasibility (LM-19)

2026-10-02 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · all three agents on `openai-codex/gpt-6.1-sol`.
Package reports: `reports/wave-7/LM-17.md`, `reports/wave-7/LM-18.md`, `reports/wave-7/LM-19.md`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-17 originals pass all tiers | **pass** | v2, 1.7B scale, multi-token M=1 and M=4: 12/12 runs PASS on tiers 1-3 |
| LM-17 tier 3 power >= 90% on wave-6 survivors | **fail without regression mode, pass with it** | per-layer check 34/41 (82.9%); bitwise regression 41/41 |
| LM-18 kill line (M=4 pass <= 1.5x M=1) | **killed** | 2.19x after fixes (wave 6: 2.59x) |
| LM-19 PPT at 1.7B beats matched baselines by >= 2 points | **undecided, compute-blocked** | one faithful problem not finished after ~2 GPU-h; ~395 GPU-h projected for 100 problems |

## 2. Integrator checks

- LM-17: `gate/run.py --engine v2 --tier 3` on the original kernel: PASS, exit 0 (tier 1 max diff 0.2114, one near-tie
  listed) `[measured]`. Wave-6 survivor `0.6B-bounds-rope_pos_next-L0-P1` re-run through the gate without
  `--baseline`: tiers 1 and 2 PASS, tier 3 FAIL, exit 1 (caught by the per-layer check, 34.6 s) `[measured]`.
- LM-19: `ppt/test_sampler.py` re-run on CPU: 7/7 pass; swap acceptance 0.3684 vs target 0.3679 `[measured]`.

## 3. Headline numbers

LM-17 three-tier gate (`gate/run.py`, `gate/README.md`) `[measured, gate/campaign-results.jsonl]`:

| | result |
|---|---|
| tier 1 | gate v2.1, unchanged tolerances |
| tier 2 | wave-6 extra tests (sampler, contexts 129/257/1100 + a 1.7B context reaching every attention slice, adversarial prompts, lm_head permutation, multistep, repeats, timing jitter) |
| tier 3a | per-layer hidden state and attention output vs HF, full-run and local (injected) comparisons, tolerances calibrated on the original kernels |
| tier 3b | `--baseline`: bitwise logits vs the last accepted kernel, for changes declared numerics-preserving |
| wave-6 survivors (41) | tiers 1/2 catch 0; tier 3a 34 (82.9%); tier 3b 41 (100%) |
| previously killed mutants (30) | 30/30 still caught (first kill: tier 1 27, tier 2 3) |
| warm gate time per candidate | v2 45.8 s, 1.7B scale 169.2 s, mt M=1 27.8 s, mt M=4 42.3 s |

LM-18, Qwen3-0.6B batch mode, ctx 128, 27 paired runs `[measured, bench/lm18]`: M=1 1.606 ms, M=2 2.316 ms,
M=4 3.510 ms (2.19x). Extra time at M=4 over M=1: GEMMs +0.85 ms, batch attention +0.57 ms, other +0.75 ms, of
which the attention-output combine `pro_attnc` alone is +0.36 ms. `--fmad=false` restores bitwise sequential
equality at no measurable speed cost.

LM-19, Qwen3-1.7B-Base, paper's GSM8K schedule (K=6, alpha 2-4, B=192, 10 local rounds, swaps every round)
`[measured, ppt/run.py; derived, ppt/feasibility.py]`: 83/160 rounds of one problem in ~1 GPU-h of continuation,
197k generated + 262k scored tokens so far; projected ~673k generated tokens per problem; 100 problems need
~395 GPU-h in HF here, ~43 GPU-h on an ideal batch kernel (PPT alone), ~216 GPU-h including matched baselines.

## 4. Honest negatives

- **The per-layer check misses 7 of 41 known bugs:** four rounding changes inside the calibrated envelopes, two
  lm_head-only faults outside the layer dumps, one position >= 128 fault the layer probe does not reach. Only bitwise
  regression catches all 41, and it cannot judge an intentional arithmetic change.
- **Tier 3 is expensive on 1.7B** (169 s per candidate): practical as an acceptance check, not on every edit.
- **Multi-token cost is mostly serial non-GEMM code.** Waves 6-7 optimized the GEMM, which is only ~40% of the extra
  M=4 time; attention runs sequences one after another and the attention-output combine loops per token.
- **PPT is out of reach on this card at the paper's settings,** even with a perfect batch kernel; six replicas also
  exceed 8 GB in HF above a 768-token horizon. No accuracy number exists.
- No MMA (tensor-core) variant of the multi-token GEMM was tried in wave 7.

## 5. Open threads

1. **Gate policy:** tiers 1-2 on every kernel edit (~30 s on 0.6B), tier 3 before accepting a kernel, `--baseline`
   for refactors that should not change numerics. Extend the layer dumps to lm_head and add a position >= 128 probe
   to close 3 of the 7 misses.
2. **Multi-token, if pursued:** parallelize batch attention across sequences and the `pro_attnc` combine across
   columns before touching the GEMM again; the combine fix alone is worth ~0.25 ms (LM-18 estimate), not enough
   for the 1.5x line by itself.
3. **PPT:** test the method on rented hardware at 4B, or drop it for this project.
