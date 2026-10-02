"""Render LM-17's report from saved gate evidence, never from expected outcomes."""
import json
import hashlib
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'gate'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def numerical(row):
    gate = row['gate']
    return 'error' not in gate and not gate.get('infrastructure_error') and bool(gate.get('tiers'))

def calibrations():
    implementation = hashlib.sha256((HERE / 'layers.py').read_bytes()).hexdigest()
    selected = {}
    for path in sorted((HERE / 'cache').glob('layers-calibration-*.json')):
        value = json.loads(path.read_text())
        if value.get('implementation_sha256') != implementation:
            continue
        signature = value['signature']
        key = (signature['engine'], signature['m'], value['probe'])
        selected[key] = value
    values = list(selected.values())
    if values:
        (HERE / 'calibration.json').write_text(json.dumps(values, indent=2) + '\n')
    return values



def main():
    original = {}
    for row in rows(HERE/'validation-originals.jsonl'):
        if row['gate'].get('protocol') == 'lm17-three-tier-fullchain-v1':
            original[(row['engine'],row['m'],row['repeat'])] = row
    campaign = {}
    for row in rows(HERE/'campaign-results.jsonl'):
        if row.get('regression_enabled'):
            campaign[row['index']] = row
    campaign = list(campaign.values())
    survivors = [r for r in campaign if r['cohort']=='survivor']
    previous = [r for r in campaign if r['cohort']=='previous-kill']
    killed = [r for r in survivors if numerical(r) and not r['gate']['pass']]
    infra = [r for r in campaign if not numerical(r)]
    configs = [('v2',1),('scale',1),('mt',1),('mt',4)]
    accepted = len(survivors)==41 and len(previous)==30 and not infra and len(killed)>=37 and all(
        all(original.get((e,m,i),{}).get('gate',{}).get('pass',False) for i in range(1,4)) for e,m in configs)
    text = ['# LM-17 correctness gate', '', '```',
            '                          wave 6                LM-17',
            'base logit checks         v2.1                  unchanged tier 1',
            'race / conditional probes extra suite           tier 2 + repeats + scale 2049',
            'layer visibility          none                  all 28 hidden + attention dumps',
            'numerics-preserving       no regression         exact accepted-library bits', '```', '',
            '## Gate decision', '',
            ('PASS (regression mode)' if accepted else 'NOT ACCEPTED (regression mode)') + f'. {len(killed)}/41 wave-6 survivors rejected by the cumulative three-tier gate with bitwise regression enabled. '+
            f'{len(survivors)}/41 survivors and {len(previous)}/30 sampled previous kills were evaluated. '+
            f'{len(infra)} infrastructure failures. `[measured, gate/campaign.py -> gate/campaign-results.jsonl; derived, gate/summarize.py]`', '',
            'Bitwise regression is explicitly requested with `--baseline`. Layer-only power is reported separately below; it is not interchangeable with regression-mode power.', '',
            '## False alarms', '', '| engine | all tiers incl. self-baseline |', '|---|---|']
    for e,m in configs:
        passed = sum(original.get((e,m,i),{}).get('gate',{}).get('pass',False) for i in range(1,4))
        text.append(f'| {e} M={m} batch | {passed}/3 |')
    text += ['', 'Counts are `[measured, gate/validate.py -> gate/validation-originals.jsonl]`. Each invocation starts a fresh process, compares every requested cumulative tier and includes a true bitwise self-baseline check.', '',
             '## Mutation power', '', '| cohort | tier 1 first kill | tier 2 first kill | tier 3 first kill | survives | harness error |', '|---|---:|---:|---:|---:|---:|']
    for label,group in [('41 survivors',survivors),('30 previous kills',previous)]:
        count={1:0,2:0,3:0,'survives':0,'error':0}
        for row in group:
            if not numerical(row): count['error']+=1;continue
            failed=next((i for i in (1,2,3) if str(i) in row['gate']['tiers'] and not row['gate']['tiers'][str(i)]['pass']),None)
            count[failed if failed else 'survives']+=1
        text.append(f"| {label} | {count[1]} | {count[2]} | {count[3]} | {count['survives']} | {count['error']} |")
    reached = [r for r in survivors if numerical(r) and '3' in r['gate']['tiers']]
    layer_kills = [r for r in reached if not r['gate']['tiers']['3'].get('layers',{}).get('pass',True)]
    regression_kills = [r for r in reached if not r['gate']['tiers']['3'].get('regression',{}).get('pass',True)]
    layer_power = 'PASS' if len(layer_kills)>=37 else ('FAIL' if len(reached)==41 else 'NOT ESTABLISHED')
    text += ['', f'Tier 3a rejects {len(layer_kills)}/{len(reached)} survivors reaching tier 3. Tier 3b rejects {len(regression_kills)}/{len(reached)}. Both tests execute even when one fails, so these counts overlap. `[measured, gate/run.py -> gate/campaign-results.jsonl]`', '',
             f'Layer-only power: **{layer_power}**. {len(layer_kills)}/41 ({100*len(layer_kills)/41:.1f}%) is below the binding 90% target (at least 37/41).' if layer_power=='FAIL' else f'Layer-only power: **{layer_power}** against the binding 90% target (at least 37/41).', '',
             'The rebuilt cohort is all 41 non-equivalent wave-6 survivors. The 30 previous kills use seed 17 and six samples from each fault family. Every CUDA build is a copy under `gate/build/`; original packages and wave-6 results remain untouched. `[derived selection, gate/campaign.py]`', '',
             '## Runtime', '', '| configuration | first total s | warm tier 1 s | warm tier 2 s | warm tier 3 s | warm total s |', '|---|---:|---:|---:|---:|---:|']
    for e,m in configs:
        measured=[r['gate'] for (ee,mm,repeat),r in original.items() if (ee,mm)==(e,m) and repeat in (2,3) and r['gate'].get('pass')]
        first_gate=original.get((e,m,1),{}).get('gate',{})
        first=f"{first_gate['wall_s']:.3f}" if first_gate.get('pass') else 'not measured'
        values=[]
        for key in ('1','2','3'):
            v=[g['tiers'][key]['wall_s'] for g in measured if key in g['tiers']]
            values.append(f'{statistics.median(v):.3f}' if v else 'not measured')
        total=[g['wall_s'] for g in measured]
        warm=f'{statistics.median(total):.3f}' if total else 'not measured'
        text.append(f"| {e} M={m} batch | {first} | {' | '.join(values)} | {warm} |")
    text += ['', '### Per-candidate runtime', '',
             'Seconds are `[measured, gate/run.py -> gate/campaign-results.jsonl]`. Higher tiers skipped after a rejection are marked `not run`, not zero.', '',
             '| cohort | candidate | tier 1 s | tier 2 s | tier 3 s | gate total s |',
             '|---|---|---:|---:|---:|---:|']
    for row in sorted(campaign,key=lambda r:r['index']):
        if not numerical(row):
            continue
        g = row['gate']
        elapsed = [f"{g['tiers'][str(t)]['wall_s']:.3f}" if str(t) in g['tiers'] else 'not run' for t in (1,2,3)]
        text.append(f"| {row['cohort']} | `{row['mutant_id']}` | {' | '.join(elapsed)} | {g['wall_s']:.3f} |")
    text += ['', 'Host wall seconds are `[measured, gate/validate.py -> gate/validation-originals.jsonl; derived medians, gate/summarize.py]`. Warm columns are medians of repeats 2 and 3, after first-use reference/calibration caches exist. First-run totals retain their observed setup costs. These are correctness-gate costs, not decode speed. GPU lock waiting is excluded from the gate result clock. Every candidate also retains its own per-tier time in the campaign JSONL.', '',
             '## Layer calibration', '',
             'The mandatory full-chain probe teacher-forces tokens but never replaces hidden states. The debug copy dumps every one of the 28 actual post-MLP residuals and 28 FP16 pre-o-projection attention vectors against HF. A second binding local probe replaces each layer incoming state with the HF state rounded to FP16, while retaining candidate QKV, Q/K norms, RoPE and KV history. It isolates local arithmetic and temporal attention defects from cross-layer drift. Full-chain and local probes calibrate independently.', '',
             'The calibration set is every prompt token plus the first 63 HF forced outputs for each of the three wave-6 base prompts. Every probe and layer has independent maximum-absolute and RMS bounds. Each bound is `k=1.25` times the known-correct kernel\'s worst corresponding error over the set, with no candidate-specific widening. `[assumed policy, gate/layers.py]`', '',
             'Reference cache keys hash teacher tokens and fake-quant weights. Calibration keys also hash the accepted library, source, headers and debug build flags. Before injecting HF states, the debug copy must reproduce selected-library logits bitwise on a short uninjected probe. Explicit `--source` associations are hash-bound to the selected library. Unsupported source patterns fail closed. `[derived implementation, gate/debug.py and gate/layers.py]`', '',
             'Tier 1 keeps the original limits. V2 uses `min(0.5, 1.25 * RTN-control-error)`, scale uses 0.5, and mt preserves LM-14\'s HF/v2-relative and sequential checks. Tier 2 copies wave-6 extras and adds three extra repeats plus the varied 2049-token scale reference. Every warp range in both attention halves receives at least two KV rows in that case. `[derived slice coverage, gate/prepare_context.py slice_rows; assumed scenarios, gate/suites.py and gate/mt_gate.py]`', '',
             'Tier 3b compares raw float32 logit bits for base and extra contexts against the accepted library. It rejects signed-zero differences as well as any changed numeric value. `[measured CPU edge smoke; implementation gate/suites.py regression]`', '']
    observed = calibrations()
    if observed:
        text += ['| calibration | worst hidden max abs | worst attention max abs |', '|---|---:|---:|']
        for value in observed:
            sig = value['signature']
            h = max(layer[0] for layer in value['observed'][0])
            a = max(layer[0] for layer in value['observed'][1])
            text.append(f"| {sig['engine']} M={sig['m']} {value['probe']} | {h:.8f} | {a:.8f} |")
        text += ['', "Observed differences are `[measured, gate/layers.py calibration -> gate/calibration.json]`. That tracked JSON retains every layer's observed max-absolute/RMS value and fixed bound, plus hashes and calibration tokens.", '']
    text += ['## Remaining survivors', '']
    remaining=[r for r in survivors if numerical(r) and r['gate']['pass']]
    if not remaining and len(survivors)==41:
        text.append('None in regression mode.')
    elif not remaining:
        text.append('No evaluated survivor remains, but the cohort is incomplete; missing candidates are not kills.')
    else:
        for row in remaining:
            text.append(f"- `{row['mutant_id']}`: {row['operator']} at layer {row['layer']}; no binding check rejected it on the exercised cases.")
    layer_remaining=[r for r in reached if r not in layer_kills]
    text += ['', 'Layer-only remaining cases, including cases caught by regression:', '']
    faults = {'half_score':'attention scores rounded to FP16', 'rope_pos_next':'Q/K RoPE position incremented by one',
              'fp16_partial':'row partial sums rounded to FP16', 'half_partial':'first row partial rounded to FP16',
              'fp16_dot':'FP16 rather than FP32 dot accumulation', 'round_zero':'output conversion uses round toward zero',
              'truncate_reduction':'warp reduction additions rounded to FP16',
              'long_skip_last':'last attention position excluded when position >=128'}
    for row in layer_remaining:
        if row['operator']=='long_skip_last':
            cause = 'the pos>=128 branch is never exercised by the base layer probes; extended contexts pass the end-logit tolerance but fail bitwise regression'
        elif row['layer']==-1:
            cause = 'lm_head only; layer dumps stop before the head'
        else:
            cause = 'numeric change remains inside both fixed HF-calibrated full-chain and local error envelopes'
        drift = row['gate']['tiers']['1']['first'].get('orig_max_diff')
        text.append(f"- `{row['mutant_id']}`: {faults.get(row['operator'],row['operator'])}; {cause}. Base max logit drift vs accepted kernel is {drift:.8f}. `[measured, gate/core.py pass_cases]`")
    text += ['', '## Honest negatives', '',
             'Finite repeats and timing jitter increase race exposure but do not prove race freedom. Local-layer tolerance checks cannot reliably separate every small rounding change from the accepted kernel\'s own HF error. Head-only faults are outside layer dump coverage. Bitwise regression is binding only when a change is declared numerics-preserving; using it for intentional arithmetic rewrites would reject valid changes.', '',
             'Mt has no fused multi-step device-launch ABI. Its equality check exercises causal chunking or mixed-prompt batch feedback against isolated single-sequence behavior for twelve passes; it does not pretend to test a nonexistent fused launch.', '',
             '## Open threads', '']
    text.append('Regression-mode acceptance is established. Layer-only power misses the binding target; do not use this campaign to claim >=90% power for intentional arithmetic changes without a baseline.' if accepted and layer_power=='FAIL' else ('No acceptance blocker remains in the requested regression-mode campaign.' if accepted else 'Acceptance is not established by the current evidence. See missing configuration/cohort counts and any infrastructure failures above.'))
    for row in infra:
        text.append(f"- `{row['mutant_id']}`: {row['gate'].get('error',row['gate'].get('stderr','unknown harness failure'))}")
    path = ROOT/'reports/wave-7/LM-17.md'
    path.write_text('\n'.join(text)+'\n')
    print(json.dumps({'accepted':accepted,'survivors_evaluated':len(survivors),'survivors_killed':len(killed),
                      'layer_kills':len(layer_kills),'regression_kills':len(regression_kills),'previous_evaluated':len(previous),'infrastructure_errors':len(infra)}))


if __name__ == '__main__':
    main()
