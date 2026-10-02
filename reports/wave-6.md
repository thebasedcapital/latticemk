# Wave 6 report — multi-token pass (LM-14), gate mutation analysis (LM-16)

2026-10-02 · Quadro RTX 4000 (sm_75) · driver 610.57.04 · agents: LM-14 on `openai-codex/gpt-6.1-sol`;
LM-16 on `openai-codex/gpt-6.1-sol`, then `anthropic/claude-sonnet-5-5`, finished by the integrator.
Package reports: `reports/wave-6/LM-14.md`, `reports/wave-6/LM-16.md`.

## 1. Gate decision

| gate | decision | one line |
|---|---|---|
| LM-14 kill line (M=4 pass <= 1.5x M=1 at ctx 128) | **killed** | 2.56x (agent), 2.59x (integrator re-run), after one accepted bounded retry |
| LM-14 correctness | **pass** | both modes, M=1-5: max diff vs HF 0.130, multi-token = sequential bitwise, bitwise repeat, 12 schedules ACCEPT |
| LM-16 gate adequacy | **gate v2.1 insufficient alone** | kills 72% of non-equivalent single faults; 89% (0.6B) / 83% (1.7B) with the new extra tests |

## 2. Integrator checks

- `bench/lm14/check_gate.py` re-run: PASS for causal and batch modes, M=1-5; max diff vs HF 0.1302 (v2: 0.2114),
  sequential difference 0.0, bitwise repeat `[measured]`.
- `bench/lm14/bench.py --ctx 128 --runs 27` re-run (1740 MHz): M1 1.584 ms, M2 2.350, M3 3.237, M4 4.096, M5 5.002;
  v2 M1 1.665 ms; t4/t1 = 2.59 `[measured]`. Result files restored to the agent's recorded run afterwards.
- LM-16: final reruns completed by the integrator; `mutation/summarize.py` written and run (numbers below).

## 3. Headline numbers

LM-14, Qwen3-0.6B, ctx 128, batch mode (independent sequences) `[measured, integrator re-run]`:

| M | pass time | aggregate tok/s | vs v2 single stream |
|---|---|---|---|
| 1 | 1.584 ms | 631 | 1.05x |
| 2 | 2.350 ms | 851 | 1.42x |
| 4 | 4.096 ms | 977 | 1.63x |
| 5 | 5.002 ms | 1000 | 1.66x |

Each extra token costs about 0.85 ms, close to linear `[derived]`. The new M=1 kernel is 5% faster than v2 and
closer to HF (0.130 vs 0.211). The isolated GEMM phase runs at 232 GB/s for M=1 and 77 GB/s for M=4 (LM-14).

LM-16 kill matrix, `mutation/summarize.py` `[measured]`:

| | 0.6B gate | 0.6B +extra | 1.7B gate | 1.7B +extra |
|---|---|---|---|---|
| synchronization | 63.0% | 100% | 76.0% | 100% |
| attention | 62.5% | 92.5% | 68.2% | 81.8% |
| bounds | 84.4% | 93.3% | 79.4% | 91.2% |
| dequant | 100% | 100% | 100% | 100% |
| precision | 43.3% | 46.7% | 33.3% | 33.3% |
| all | 72.1% | 89.1% | 71.9% | 83.3% |

## 4. Honest negatives

- **Multi-token pass does not pay for speculative decoding yet.** At 2.59x for 4 tokens, drafting would need more
  than about 2.6 accepted tokens per pass to break even. An integrator throughput bound puts the GEMM phase at about
  1.1-1.2x for M=4; about 2.5 ms per pass is unexplained, most likely latency in dependent shared-memory-load to FMA
  chains. The kill reflects this implementation, not a Turing limit.
- **Batch mode still gains throughput** (1.63x aggregate at M=4), which helps parallel sampling (PPT) even under the
  kill, but not single-stream latency.
- **Gate v2.1 missed every race and every conditional fault** on its base cases; bitwise repeat over two runs is
  not race detection. Timing jitter, longer contexts and the sampler check close those gaps.
- **41 real single-layer bugs survive every test** (halved attention scale or RoPE off by one at one layer,
  single-site precision changes). They move logits by 0.008-0.31 against the original kernel but stay inside the
  engine's own noise against HF (0.21 on 0.6B, 0.05 on 1.7B).
- LM-16 needed two model switches (a Codex access error, then an agent stuck on a correct but long wait) before the
  integrator finished it.

## 5. Open threads

1. **Gate tier 2:** adopt `mutation/extra_tests.py` for every kernel change (about 16-31 s per candidate).
2. **Gate tier 3:** per-layer hidden-state comparison against HF, plus bitwise regression against the last accepted
   kernel for numerics-preserving changes. Either would close the 41-survivor blind spot.
3. **LM-14 next:** stall analysis of one GEMM shape at M=1 vs M=4 before any rewrite; if the 0.85 ms per extra
   token drops toward the ~0.1-0.2 ms throughput bound, speculative decoding becomes viable.
4. **Use batch mode now:** Parallel Power Tempering (arXiv 2609.38104) at K=4 on Qwen3-1.7B can run on the batch
   kernel at its measured 1.63x aggregate throughput (0.6B; 1.7B not yet ported).
