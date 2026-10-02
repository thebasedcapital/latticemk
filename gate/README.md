# Reusable kernel correctness gate

```
candidate library
  | tier 1: unchanged v2.1 logits, near ties, bitwise repeat
  | tier 2: sampler, long contexts, adversarial prompts, head rows,
  |         launch equality, extra repeats, timing jitter, KL
  | tier 3a: every layer's hidden state and attention output vs HF
  | tier 3b: optional accepted-library bitwise regression
  v
PASS only when every requested cumulative tier passes
```

## Run a candidate

Run from the repository root. Every command using CUDA must hold the shared GPU lock.

```sh
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine v2 --tier 3
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine scale --tier 3
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine mt --m 1 --mode batch --tier 3
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine mt --m 4 --mode batch --tier 3
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine v2 --lib path/candidate.so --tier 3
scripts/gpu.sh timeout 285 .venv/bin/python gate/run.py --engine v2 --lib path/candidate.so \
    --tier 3 --baseline kernels/megakernel_v2/libmega2.so --output gate/results.jsonl
```

`--tier` defaults to 3. Tiers are cumulative and fail fast between tiers. Exit 0 means PASS, exit 1 means numerical failure or a missing prerequisite, with the reason in the JSON result. `--output` appends the same complete result as one JSONL line. `--baseline` declares the change numerics-preserving, requires tier 3, and adds exact logit equality against that accepted library. Do not use regression mode to judge intentional numerical rewrites.

The wave-7 mutation campaign meets its detection-power target only with bitwise regression enabled. Layer-only checks fall below that target; a PASS without `--baseline` is not evidence of equivalent detection power for an intentional arithmetic rewrite. See [the measured decision and seven layer-only misses](../reports/wave-7/LM-17.md).

Keep the timeout inside `gpu.sh`, so lock waiting does not consume the GPU job deadline. A timeout has exit 124 and is not a PASS. The validation driver distinguishes a missing result caused by timeout from an observed numerical failure.

Candidates need the engine's existing ABI. Tier 2 jitter and tier 3 dumps require matching CUDA source. A library resolves to its same-basename `.cu`, or use `--lib candidate.so --source matching.cu` for an explicit hash-bound association. The default libraries resolve to their known source files. Never substitute the original source for an unrelated candidate. The gate fails if it cannot instrument the candidate. Copies, shared libraries and compiler logs live in ignored `gate/build/`; original packages are read-only.

## References and tolerances

Tier 1 uses the cached wave-6 HF fp32 fake-GPTQ references in `mutation/reference-0.6B.pt` and `mutation/reference-1.7B.pt`. Generate missing caches using the existing `mutation/gate.py prepare` workflow. The gate never recalibrates its v2.1 bounds on the candidate. V2 retains `min(0.5, 1.25 * RTN-control-error)` and scale retains the absolute 0.5 bound. Near-tie handling and two fresh base passes are unchanged. Mt retains the existing LM-14 HF, v2-relative and sequential checks.

For a fresh checkout, first produce the packed GPTQ weights using the repository quickstart. Then prepare missing references. Scale's HF stage is CPU-only:

```sh
scripts/gpu.sh timeout 285 .venv/bin/python mutation/gate.py prepare --model 0.6B
.venv/bin/python mutation/gate.py prepare --model 1.7B --phase hf
scripts/gpu.sh timeout 285 .venv/bin/python mutation/gate.py prepare --model 1.7B --phase gpu
```

Tier 2 copies the wave-6 extra scenarios. Sampler tokens must equal the argmax of the kernel's own logits, including head row permutations. Contexts are 129, 257 and 1100, plus a varied 2049-token scale case. All 32 warp ranges in both attention halves receive KV rows in that scale case, at least two per range under the kernel's ceiling-divided partition. Generate its CPU-only reference once:

```sh
.venv/bin/python gate/prepare_context.py
```

Extra context passes repeat three times beyond the first pass. Two passes through a separately compiled jitter candidate compare bitwise with matching unjittered output. The race probe adds pseudo-random nanosleep around CTA barriers and warp-entry skew. It increases exposure; no finite repeat count proves race freedom. KL retains the wave-6 1.25-times-control limit.

Tier 3a dumps all 28 actual hidden states and attention outputs during token teacher-forcing. A second binding local probe injects the HF incoming state rounded to FP16 at each layer, retaining candidate QKV, RoPE and KV history to isolate local defects. Full-chain and local probes calibrate independently. Both must pass. Each layer has separate max-absolute and RMS bounds, each `k=1.25` times the accepted kernel's worst error across every prompt token and the first 63 forced outputs in each of the three base cases. That is 213 positions. Candidates never widen thresholds.

`gate/layers.py` and `gate/debug.py` generate hash-bound HF and accepted-kernel calibration caches, including their own implementation hashes. `gate/calibration.json` records all measured per-layer values after validation. Before layer comparison, an uninjected debug-copy probe must match selected-library logit bits. Missing dumps, missing source, and source/build mismatches fail closed.

Mt tier-3 calibration follows the candidate's `gate_contract.json` compiler flags and thread table. The default LM-14 contract still calibrates against `kernels/megakernel_mt/`. The accepted `--fmad=false`, 1024-thread M1 and 512-thread M2-5 contract calibrates against `kernels/megakernel_mt2/`. The same 213 positions, separate full/local envelopes, max-absolute/RMS metrics and `k=1.25` rule apply to both. Unknown contracts fail closed; candidates cannot nominate their own calibration library. Cache keys bind the accepted library, source, headers and compiler contract through the debug manifest. The per-contract measured envelopes live in `gate/cache/layers-calibration-*.json`.

The mt debug builder supports the scalar-column prologue and the all-column `P2` prologue used by mt3. For the latter, it dumps each actual rounded half2 from attention staging and injects HF inputs through the existing all-column norm storage. It does not replace production attention or GEMM. Both hooks must reproduce selected-library logit bits before layer scoring. `bench/lm23/check_dump.py` also exercises the M1 all-column hook with dumping enabled and checks complete finite hidden/attention arrays.

## Validate the gate

```sh
.venv/bin/python mutation/summarize.py --survivors
.venv/bin/python gate/campaign.py prepare
.venv/bin/python gate/campaign.py prebuild
.venv/bin/python gate/validate.py
.venv/bin/python gate/campaign.py evaluate --index 0
.venv/bin/python gate/campaign.py evaluate-all
.venv/bin/python gate/summarize.py
```

The campaign rebuilds all 41 survivors and a deterministic seed-17 sample of 30 previous kills, six per fault family. `prebuild` compiles debug and jitter copies on CPU before GPU evaluation. `validate.py` and the campaign evaluation actions acquire the GPU lock internally; do not wrap them in another `gpu.sh`. Each `evaluate` command runs one bounded GPU job. `evaluate-all` groups two v2 candidates or one scale candidate into jobs capped at 285 seconds, releasing the lock between batches. Index order is survivors sorted by id, then sampled previous kills sorted by id. Outputs append to `gate/campaign-results.jsonl`. The campaign uses bitwise regression by default; `--without-baseline` measures layer-only tier 3 power. Do not conflate those results. A missing source, failed compile or harness error is not a numerical mutant kill.

Per-tier `wall_s` values are host wall time. First-use builds and reference calibration can dominate a cold run. Report warm evaluation separately, and do not treat these correctness costs as production decode latency. See `reports/wave-7/LM-17.md` for the measured gate decision and remaining survivors.
