"""Category throughput, paired prompt-bootstrap CIs, and break-even accounting."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VARIANTS = ('v2', 'mt1', 'spec')


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def summarize(rows):
    tokens = np.array([r['generate_tokens'] for r in rows])
    wall = np.array([[np.mean([run['wall_s'] for run in r['runs'] if run['variant'] == variant])
                      for variant in VARIANTS] for r in rows])
    rng = np.random.default_rng(25)
    indices = rng.integers(0, len(rows), (10000, len(rows)))
    sampled_wall = wall[indices].sum(axis=1)
    sampled_tokens = tokens[indices].sum(axis=1)
    tps = tokens.sum()/wall.sum(axis=0)
    headline = wall[:, 0].sum()/wall[:, 2].sum()
    ci = interval(sampled_wall[:, 0]/sampled_wall[:, 2])
    traces = [t for r in rows for run in r['runs'] if run['variant'] == 'spec' for t in run['trace']]
    accepted = Counter(t['accepted'] for t in traces)
    ks = Counter(t['k'] for t in traces)
    per_prompt = []
    for row in rows:
        trace = [t for run in row['runs'] if run['variant'] == 'spec' for t in run['trace']]
        per_prompt.append([sum(t['accepted'] for t in trace), len(trace)])
    a = np.array(per_prompt)
    sampled_a = a[indices].sum(axis=1)
    mean_a = sum(t['accepted'] for t in traces)/len(traces)
    by_k = {}
    for k in range(1, 6):
        subset = [t for t in traces if t['k'] == k]
        if subset:
            by_k[k] = {'passes': len(subset), 'mean_accepted': np.mean([t['accepted'] for t in subset]),
                       'mean_verify_ms': np.mean([t['verify_ns'] for t in subset])/1e6}
    context_traces = []
    for row in rows:
        for run in row['runs']:
            if run['variant'] != 'spec':
                continue
            context = row['prompt_tokens']
            for step in run['trace']:
                context_traces.append((context, step))
                context += step['accepted']+1
    m4_context = {}
    for name, low, high in [('short', 1, 256), ('middle', 257, 1200), ('long', 1201, 2304)]:
        subset = [(ctx, t) for ctx, t in context_traces if low <= ctx <= high and t['k'] == 4]
        if subset:
            m4_context[name] = {'passes': len(subset),
                                'actual_context_range': [min(c for c, _ in subset), max(c for c, _ in subset)],
                                'mean_accepted': float(np.mean([t['accepted'] for _, t in subset])),
                                'mean_verify_ms': float(np.mean([t['verify_ns'] for _, t in subset])/1e6)}
    # Host-inclusive observed break-even versus this run's plain M1. This is a
    # cost diagnostic, not a counterfactual GPU-event measurement.
    m1_ms = wall[:, 1].sum()/tokens.sum()*1000
    spec_ms = wall[:, 2].sum()*1000/(len(traces)/3)
    setup_ns = sum(run['lookup_setup_ns'] for r in rows for run in r['runs'] if run['variant'] == 'spec')
    cpu_ns = setup_ns+sum(t['draft_ns']+t['index_update_ns']+t['rollback_ns'] for t in traces)
    repeated = []
    for row in rows:
        output = next(run['tokens'] for run in row['runs'] if run['variant'] == 'spec')
        ngrams = [tuple(output[i:i+4]) for i in range(len(output)-3)]
        repeated.append(1-len(set(ngrams))/len(ngrams))
    return {'tag': 'derived', 'prompts': len(rows), 'repeats': 3,
            'throughput_tps': {v: float(tps[i]) for i, v in enumerate(VARIANTS)},
            'throughput_ci95': {v: interval(sampled_tokens/sampled_wall[:, i]) for i, v in enumerate(VARIANTS)},
            'speedup_v2': float(headline), 'speedup_v2_ci95': ci,
            'speedup_mt1': float(wall[:, 1].sum()/wall[:, 2].sum()),
            'speedup_mt1_ci95': interval(sampled_wall[:, 1]/sampled_wall[:, 2]),
            'ship': bool(headline >= 1.15 and ci[0] > 1.0),
            'mean_accepted': float(mean_a), 'mean_accepted_ci95': interval(sampled_a[:, 0]/sampled_a[:, 1]),
            'acceptance_histogram': dict(sorted(accepted.items())), 'k_distribution': dict(sorted(ks.items())),
            'by_k': by_k, 'mean_k': float(np.mean([t['k'] for t in traces])),
            'm4_by_context': m4_context,
            'draft_us_per_pass': float(np.mean([t['draft_ns'] for t in traces])/1000),
            'index_update_us_per_pass': float(np.mean([t['index_update_ns'] for t in traces])/1000),
            'lookup_setup_ms_per_decode': setup_ns/(len(rows)*3)/1e6,
            'rollback_us_per_pass': float(np.mean([t['rollback_ns'] for t in traces])/1000),
            'cpu_wall_fraction': cpu_ns/1e9/(wall[:, 2].sum()*3),
            'mean_target_verify_ms': float(np.mean([t['verify_ns'] for t in traces])/1e6),
            'mean_spec_wall_ms_per_pass': float(spec_ms),
            'measured_policy_break_even_accepted': float(spec_ms/m1_ms-1),
            'output_repeated_fourgram_fraction_mean': float(np.mean(repeated)),
            'prefill_s_mean': {v: float(np.mean([r['prefill_s'][v] for r in rows])) for v in ('v2', 'mt3')},
            'prefill_inclusive_tps': {v: float(tokens.sum()/sum(r['prefill_s']['v2' if v == 'v2' else 'mt3']+wall[j, i] for j, r in enumerate(rows))) for i, v in enumerate(VARIANTS)},
            'prompt_tokens_range': [min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows)],
            'generate_tokens_range': [int(tokens.min()), int(tokens.max())]}


def main():
    rows = [json.loads(line) for line in (HERE/'results.jsonl').read_text().splitlines()]
    assert len(rows) == 80 and len({r['id'] for r in rows}) == 80
    for row in rows:
        assert not row['smoke']
        assert len(row['runs']) == 9
        assert row['library_sha256'] == rows[0]['library_sha256']
        reference = {r['variant']: r for r in row['runs'] if r['repeat'] == 0}
        for repeat in range(3):
            runs = {r['variant']: r for r in row['runs'] if r['repeat'] == repeat}
            assert set(runs) == set(VARIANTS)
            assert runs['spec']['tokens'] == runs['mt1']['tokens'], row['id']
            assert len(runs['spec']['tokens']) == row['generate_tokens']
            for variant in VARIANTS:
                assert runs[variant]['tokens'] == reference[variant]['tokens'], row['id']
            assert [(t['k'], t['accepted']) for t in runs['spec']['trace']] == [(t['k'], t['accepted']) for t in reference['spec']['trace']]
            assert all(1 <= t['k'] <= 5 and 0 <= t['accepted'] < t['k'] for t in runs['spec']['trace'])
            assert sum(t['accepted']+1 for t in runs['spec']['trace']) == row['generate_tokens']
    divergence = [json.loads(line) for line in (HERE/'divergence.jsonl').read_text().splitlines()]
    assert len(divergence) == 80 and {r['id'] for r in divergence} == {r['id'] for r in rows}
    spills = []
    for m in range(1, 6):
        text = (ROOT/f'kernels/megakernel_mt3/build-selected-m{m}.log').read_text()
        resources = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', text)
        assert resources and all(all(int(x) == 0 for x in r) for r in resources)
        spills.append({'m': m, 'stack_spill_bytes': [0, 0, 0]})
    gate = [json.loads(line) for line in (HERE/'gate.jsonl').read_text().splitlines()]
    assert len(gate) == 1 and gate[0]['pass']
    result = {'tag': 'derived', 'seed': 25, 'bootstrap_prompt_resamples': 10000,
              'identity_reference': 'plain mt3 M1, identical library/build contract',
              'identity_prompts_passed': len(rows), 'identity_repeat_pairs_passed': len(rows)*3,
              'v2_diverged_prompts': sum(r['diverged'] for r in divergence),
              'v2_divergence_fraction': sum(r['diverged'] for r in divergence)/len(rows),
              'zero_spill_build_logs': spills, 'gate_pass': True, 'categories': {}}
    for category in ('code', 'rag', 'summarization', 'chat'):
        subset = [r for r in rows if r['category'] == category]
        assert len(subset) == 20
        result['categories'][category] = summarize(subset)
        result['categories'][category]['v2_diverged_prompts'] = sum(r['diverged'] for r in divergence if r['category'] == category)
    result['worth_shipping_on_card'] = any(r['ship'] for r in result['categories'].values())
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
