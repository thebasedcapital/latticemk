"""Bounded prompt batches, three interleaved repeats and strict token identity."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from runtime import Target, ROOT

HERE = Path(__file__).resolve().parent


def identity(actual, expected):
    try:
        assert actual == expected, 'greedy token identity failure'
    except AssertionError:
        first = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), min(len(actual), len(expected)))
        return {'pass': False, 'first_mismatch': first,
                'actual': actual[first] if first < len(actual) else None,
                'expected': expected[first] if first < len(expected) else None}
    return {'pass': True}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=['code', 'rag', 'summarization', 'chat'], required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=HERE/'results.jsonl')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.repeats < 3 and not args.smoke:
        parser.error('at least three repeats required')
    rows = [json.loads(line) for line in (HERE/'prompts.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['category'] == args.category][args.start:args.start+args.count]
    if not rows:
        parser.error('empty prompt batch')
    torch.manual_seed(25)
    target = Target()
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [ROOT/'kernels/megakernel_v2/libmega2.so'] +
                [ROOT/f'kernels/megakernel_mt3/libmt{m}.so' for m in range(1, 6)]}
    failed = False
    for prompt in rows:
        ids = prompt['prompt_ids']
        count = min(12, prompt['generate_tokens']) if args.smoke else prompt['generate_tokens']
        v2_state, v2_prefill = target.prefill(ids, 'v2')
        mt_state, mt_prefill = target.prefill(ids, 'mt1')
        # Warm all actual paths, then restore the prompt KV before every timed run.
        for variant in ('v2', 'mt1', 'spec'):
            target.restore(v2_state if variant == 'v2' else mt_state, variant)
            target.decode(ids, 8, variant)
        runs = []
        for repeat in range(args.repeats):
            order = ['v2', 'mt1', 'spec']
            order = order[repeat % 3:]+order[:repeat % 3]
            if repeat % 2:
                order.reverse()
            outputs = {}
            for variant in order:
                target.restore(v2_state if variant == 'v2' else mt_state, variant)
                output, metrics = target.decode(ids, count, variant)
                outputs[variant] = output
                runs.append({'variant': variant, 'repeat': repeat, 'order': order,
                             'tokens': output, 'tokens_per_s': count/metrics['wall_s'], **metrics})
            checks = {variant: identity(outputs[variant], outputs['v2']) for variant in ('mt1', 'spec')}
            speculative_check = identity(outputs['spec'], outputs['mt1'])
            for run in runs[-3:]:
                run['identity_v2'] = checks.get(run['variant'], {'pass': True})
                if run['variant'] == 'spec':
                    run['identity_mt1'] = speculative_check
            failed |= not speculative_check['pass']
        row = {'tag': 'measured', 'work_package': 'LM-25', 'id': prompt['id'],
               'category': prompt['category'], 'prompt_tokens': len(ids),
               'generate_tokens': count, 'seed': 25, 'smoke': args.smoke,
               'prefill_s': {'v2': v2_prefill, 'mt3': mt_prefill},
               'library_sha256': manifest, 'runs': runs,
               'identity_pass': all(r.get('identity_mt1', {'pass': True})['pass'] for r in runs),
               'v2_divergence': checks['mt1']}
        with args.output.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps({'id': row['id'], 'identity_pass': row['identity_pass'],
                          'prefill_s': row['prefill_s'],
                          'mean_tps': {v: sum(r['tokens_per_s'] for r in runs if r['variant'] == v)/args.repeats for v in ('v2', 'mt1', 'spec')}}), flush=True)
        del v2_state, mt_state
    if failed:
        raise SystemExit('BUG: one or more strict token-identity assertions failed; see retained results')


if __name__ == '__main__':
    main()
