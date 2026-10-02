"""Reusable correctness gate. Run all checks through the selected cumulative tier."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import torch
import core
import suites


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def run(engine, library=None, tier=3, baseline=None, m=1, mode='batch'):
    torch.set_num_threads(12)
    start = time.perf_counter()
    if engine == 'mt':
        import mt_gate
        result = mt_gate.run(library, tier, baseline, m, mode)
        result['protocol'] = 'lm17-three-tier-fullchain-v1'
        return result
    library = Path(library or core.LIBRARIES[engine]).resolve()
    cache = core.load_cache(engine)
    module = core.make_module(engine, library)
    gpu, shared = core.resources(engine, module)
    result = {'engine': engine, 'library': str(library), 'tiers': {},
              'protocol': 'lm17-three-tier-fullchain-v1'}
    first, logits = suites.tier1(module, gpu, shared, cache)
    result['tiers']['1'] = first
    if tier >= 2 and first['pass']:
        result['tiers']['2'] = suites.tier2(engine, library, module, gpu, shared, cache, logits)
    if tier >= 3 and all(t['pass'] for t in result['tiers'].values()):
        import layers
        third_start = time.perf_counter()
        layer = layers.check(engine, library, module, gpu, shared, cache)
        third = {'pass': layer['pass'], 'layers': layer}
        if baseline:
            third['regression'] = suites.regression(engine, baseline, library, gpu, shared, cache, logits)
            third['pass'] &= third['regression']['pass']
        third['wall_s'] = time.perf_counter() - third_start
        result['tiers']['3'] = third
    result['pass'] = len(result['tiers']) == tier and all(t['pass'] for t in result['tiers'].values())
    result['wall_s'] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('v2', 'scale', 'mt'), required=True)
    parser.add_argument('--lib', type=Path)
    parser.add_argument('--source', type=Path, help='Explicit matching CUDA source, hash-bound to --lib')
    parser.add_argument('--tier', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--baseline', type=Path, help='Declare numerics preserving; tier 3 additionally requires bitwise logits')
    parser.add_argument('--m', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--mode', choices=('batch', 'causal'), default='batch')
    parser.add_argument('--output', type=Path, help='Append one complete result to JSONL')
    args = parser.parse_args()
    if args.baseline and args.tier != 3:
        parser.error('--baseline requires --tier 3')
    if args.source and not args.lib:
        parser.error('--source requires --lib')
    try:
        if args.source:
            import debug
            debug.associate(args.lib, args.source)
        result = run(args.engine, args.lib, args.tier, args.baseline, args.m, args.mode)
    except Exception as exc:
        result = {'engine': args.engine, 'library': str(args.lib), 'pass': False,
                  'error': f'{type(exc).__name__}: {exc}'}
    text = json.dumps(json_safe(result), allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('a') as out:
            out.write(text + '\n')
    print(text, flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
