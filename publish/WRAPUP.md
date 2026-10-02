# Wrap-up run log

```text
reproduce.sh all
  existing gate / validate / bench / ppl  unchanged entry points
  + gate-tiers / multitoken / compaction / spec / schedules
  + charts  recorded data -> publication PNGs

published evidence  unchanged
fresh replications  -> publish/reproduction/
```

## What ran

Each new stage completed as a separate invocation of `./reproduce.sh <stage>` on the Quadro RTX 4000 using cached checkpoints, packed weights and reference caches. GPU commands ran singly through `scripts/gpu.sh`, with inner timeouts and `--timing` for measurements. The existing package drivers were reused, not rewritten. The selected mt3 build uses its existing compiler/thread contract; its mt2 timing and calibration prerequisite is built if missing. Missing reference caches use the packages' existing preparation scripts.

The table is generated from the latest stage rows in `publish/reproduction_times.jsonl`. Stage-body wall time includes local builds and GPU-lock waits, but excludes the common build/download setup. Exit codes describe the completed stage bodies. Chart rendering was timed separately with the same Python executable.

| Stage | Wall seconds | Exit code | Recorded UTC, source selector |
|---|---:|---:|---|
| `schedules` | 13 | 0 | `2026-10-02T16:25:24Z` |
| `gate-tiers` | 102 | 0 | `2026-10-02T16:35:23Z` |
| `multitoken` | 52 | 0 | `2026-10-02T16:29:02Z` |
| `compaction` | 157 | 0 | `2026-10-02T16:32:01Z` |
| `spec` | 44 | 0 | `2026-10-02T16:33:04Z` |
| `charts` | 3.339 | 0 | `2026-10-02T16:33:42.724562Z` |

The common setup ran for every stage invocation. These cached runs do not measure fresh model downloads, GPTQ calibration or cold gate calibration. The historical `all` timing in `reproduction_times.jsonl` predates the added stages and remains labeled as waves 1-5 in `results.md`.

## Runtime evidence

- `gate-tiers` runs cumulative tier 3 on v2 and mt3 causal M=4, so each result includes tiers 1, 2 and 3. Both latest results pass every tier. Source: `publish/reproduction/gate-tiers.jsonl`, final v2 and mt rows, `pass` and `tiers.*.pass`. No bitwise `--baseline` was requested; this is not a new mutation-power campaign.
- `multitoken` built selected mt3 and ran `bench/lm23/check_gate.py` for both modes and every supported M. All 10 cases passed with zero sequential difference. The repeated deterministic gate output equaled the published `bench/lm23/gate.json`. The first capture wrapper omitted unchanged output; it now archives identical successful JSON too, verified by a scoped identical-output smoke. Fresh ctx-128 timing is in `publish/reproduction/bench/lm23/causal-128.json` and `results.jsonl`. All 5 seeded-KV proof rows pass. The measured M=1 and M=4 medians were 1.500 and 2.838 ms, giving 1.893x. This still misses the research target. Published charts retain the original research rows rather than mixing this replication into them.
- `compaction` built kvc and the extended session baseline, passed `check_correctness.py` and the short `bench.py`, then completed the actual teacher-forced session. `publish/reproduction/bench/lm24/timing.json`, `passed`, is true. `session.json` records 16384 tokens, 8 events, finite final logits and a bitwise-matching extended baseline. Fresh totals were 79.591 s for v2 and 45.059 s for kvc, including events, a 1.766x gain. The deterministic correctness output matched the published `bench/lm24/correctness.json`. This is one scripted session, not an answer-quality evaluation.
- `spec` selected the first prompt in each category with `--start 0 --count 1 --repeats 3`, without `--smoke`. Fixed IDs: `code-00`, `rag-00`, `summarization-00`, `chat-00`. All 12 own-target identity repeat pairs passed in `publish/reproduction/spec.jsonl`. This small subset is a runtime/identity check, not a new category estimate. Its summarization prompt can win while the full category loses. `spec/analyze.py` successfully regenerated `spec/analysis.json` from the unchanged full `spec/results.jsonl`, asserting full-corpus identities, repeat completeness, gate success and spill-free resource logs.
- `schedules` regenerated attention, mt3 and kvc schedules with their original generators. All 45 schedules match their committed file list, byte count, sha256 and schedcheck ACCEPT verdict: `kernels/megakernel_attn/schedules/MANIFEST.json`, `bench/lm23/schedules/MANIFEST.json`, `kernels/megakernel_kvc/schedules/MANIFEST.json`. The check mode never rewrites manifests. A scoped corrupted-hash smoke verified that a valid schedule with an incorrect committed hash is rejected while leaving the manifest unchanged.

`scripts/reproduce_capture.py` archives fresh package JSONs and only newly appended JSONL rows under `publish/reproduction/<original path>`, then restores the original evidence on success or failure. A scoped failed-driver smoke verified restoration, new-row retention and forwarding of the nonzero exit. This keeps research charts reproducible after short validation runs.

## Charts and documentation

The chart generator produced the existing speed, quality/speed and roofline charts, the expanded waves chart, and new compaction, speculative and multi-token charts. The new/updated PNGs were image-read after final generation. All have the requested white canvas, takeaway title, subtitle and source footnote; no clipping or overlapping labels was observed.

- `compaction.png`: decode rows from `bench/lm24/timing.json`; chunk curve and total wall time from `bench/lm24/session.json`. Variable chunk sizes are disclosed, and logical RoPE positions stay unchanged.
- `speculative.png`: category speed ratios, confidence bounds and accepted drafts from `spec/analysis.json`. Summarization is disabled; continuation is labeled a repetitive-text control, not general chat. Identity is mt3 M=1, not v2.
- `multitoken.png`: causal mt3 pass medians for each plotted M/context from `bench/lm23/results.jsonl`; target lines are derived from the corresponding M=1 median. The missed target is explicit.
- `waves.png`: waves 1-9, each measured ratio against its own baseline from package result files. Correctness audits and the compute-blocked PPT campaign have no decode speed ratio. Tensor-core probes have no valid full-pass ratio. Wave 1 has report-only lattice timings and no public raw timing result file located, so its old report-parsed numeric bar is replaced with an explicit source-gap label. No report numbers were copied into a fake result file.

`publish/results.md` now includes sourced waves 6-9 sections and the new stage timings. README Results adds compaction and workload-conditional speculation; Quickstart lists all stages, fixed subset selection and the separation between fresh evidence and published full results.

## Failures and scope limits

No new stage remains blocked. The first `gate-tiers` stage body passed, but its outer shell then reported `pl: command not found`. The likely cause was an edit to the script while Bash was still reading it [INFERENCE]. A clean invocation after edits completed successfully; the timing table uses that later row. The earlier stage-body row remains in the append-only log rather than being deleted.

The newly expanded `./reproduce.sh all` was not run end to end in this wrap-up. All its new stage bodies were exercised individually; the old full paired-benchmark/perplexity campaign was not repeated. No new end-to-end wall time is claimed. Project-wide validation remains the integrator's separate check after all delegated changes land.

The wave-1 raw timing source gap is the only chart evidence limitation. Cold reference preparation and calibration paths were wired to existing scripts but not exercised because caches were present. No new kernels, research experiments, accuracy claims or private `research/` content were added.
