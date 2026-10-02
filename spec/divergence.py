"""Replay first v2/mt3 divergence, outside all timed throughput runs."""
import argparse
import json
from pathlib import Path

import torch
from runtime import Target
from run import identity

HERE = Path(__file__).resolve().parent


def top_two(logits):
    values, ids = torch.topk(logits, 2)
    return {'tokens': ids.cpu().tolist(), 'logits': values.cpu().tolist(),
            'margin': float((values[0]-values[1]).cpu())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=['code', 'rag', 'summarization', 'chat'], required=True)
    args = parser.parse_args()
    prompts = {r['id']: r for r in map(json.loads, (HERE/'prompts.jsonl').read_text().splitlines())}
    rows = [r for r in map(json.loads, (HERE/'results.jsonl').read_text().splitlines()) if r['category'] == args.category and not r['smoke']]
    target = None
    for row in rows:
        runs = {r['variant']: r for r in row['runs'] if r['repeat'] == 0}
        assert runs['spec']['tokens'] == runs['mt1']['tokens'], 'spec/target identity bug'
        check = identity(runs['mt1']['tokens'], runs['v2']['tokens'])
        result = {'tag': 'measured', 'id': row['id'], 'category': row['category'],
                  'spec_identity_mt1': True, 'diverged': not check['pass'],
                  'first_position': check.get('first_mismatch'), 'v2_top_two': None,
                  'mt3_top_two': None}
        if not check['pass']:
            if target is None:
                target = Target()
            ids = prompts[row['id']]['prompt_ids']
            pos = len(ids)-1
            index = check['first_mismatch']
            # Replay each engine's own prompt history and the common generated
            # prefix. At the first flip neither has seen the other engine's branch.
            target.prefill(ids, 'v2')
            target.v2.pos_set(pos)
            target.v2.set_tok(ids[-1])
            torch.cuda.synchronize()
            target.v2.mega(index+1)
            result['v2_top_two'] = top_two(target.v2.bufs['logits'])
            target.prefill(ids, 'mt1')
            history = [ids[-1]]+runs['mt1']['tokens'][:index]
            for i, token in enumerate(history):
                target.pass_mt([token], pos+i)
            result['mt3_top_two'] = top_two(target.mt[1].bufs['logits'])
            assert result['v2_top_two']['tokens'][0] == check['expected']
            assert result['mt3_top_two']['tokens'][0] == check['actual']
        with (HERE/'divergence.jsonl').open('a') as handle:
            handle.write(json.dumps(result)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
