"""Binding tier checks. Tolerances come unchanged from the v2.1 reference cache."""
import hashlib
import importlib.util
import math
from pathlib import Path
import time

import torch
import core
from core import ROOT, HERE, MODEL, LIBRARIES, pass_cases
import extra_tests


def tier1(module, gpu, shared, cache):
    stamp = time.perf_counter()
    first, logs = pass_cases(module, gpu, shared, cache['cases'], bound=cache['bound'], fail_fast=True)
    if first['kill']:
        return {'pass': False, 'wall_s': time.perf_counter() - stamp, 'first': first}, logs
    repeat, _ = pass_cases(module, gpu, shared, cache['cases'], prior=logs)
    passed = not first['nonfinite'] and not first['over_bound'] and not any(
        e['kind'] == 'hard' for e in first['events']) and repeat['bitwise_repeat']
    return {'pass': passed, 'wall_s': time.perf_counter() - stamp,
            'bound': cache['bound'], 'first': first, 'repeat': repeat}, logs


def candidate_source(engine, library):
    import debug
    return debug.resolve_source(engine, library)[0]


def jitter_module(engine, library):
    spec = importlib.util.spec_from_file_location('gate_mutator', ROOT / 'mutation/mutate.py')
    mutator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutator)
    text = candidate_source(engine, library).read_text()
    key = hashlib.sha256((engine + text).encode()).hexdigest()[:16]
    folder = HERE / 'build' / ('jitter-' + key)
    folder.mkdir(parents=True, exist_ok=True)
    cu, lib = folder / 'candidate.cu', folder / 'candidate.so'
    if not lib.exists():
        cu.write_text(mutator.jitter_text(text))
        if mutator.compile_cu(MODEL[engine], cu, lib):
            raise RuntimeError(f'Jitter compile failed: {lib.with_suffix(".build.log")}')
    return core.make_module(engine, lib, folder / 'bridge')


@torch.no_grad()
def tier2(engine, library, module, gpu, shared, cache, stored):
    stamp = time.perf_counter()
    stages, sample_errors = {}, []
    first, _ = pass_cases(module, gpu, shared, cache['cases'], check_sampling=True)
    sample_errors.extend(first['sample_errors'])
    longctx = [c for c in cache['extra'] if c['name'].startswith('context')]
    if engine == 'scale':
        path = HERE / 'cache' / 'scale-context-2049.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run gate/prepare_context.py first')
        longctx.append(torch.load(path, weights_only=False, map_location='cpu'))
    adverse = [c for c in cache['extra'] if c['name'].startswith('adversarial')]

    def check(cases, weights):
        res, logs = pass_cases(module, weights, shared, cases, bound=cache['bound'],
                               fail_fast=True, check_sampling=True)
        sample_errors.extend(res['sample_errors'])
        return res, logs

    context, ctx_logs = check(longctx, gpu)
    stages['context'] = {'pass': not context['kill'], **context}
    res, _ = check(adverse, gpu)
    stages['adversarial'] = {'pass': not res['kill'], **res}
    permutation = {'pass': True, 'wall_s': 0., 'variants': 0}
    for perm in extra_tests.head_permutations(torch.cat([c['forced'] for c in cache['cases']])):
        weights = extra_tests.permuted_head(gpu, perm)
        res, _ = check(extra_tests.permuted_cases(cache['cases'], perm), weights)
        permutation['pass'] &= not bool(res['kill'])
        permutation['wall_s'] += res['wall_s']
        permutation['variants'] = permutation['variants'] + 1
        del weights
    stages['head_perm'] = permutation
    start = time.perf_counter()
    stages['multistep'] = {**extra_tests.multistep(module, gpu, shared, cache['cases'][2]),
                           'wall_s': time.perf_counter() - start}
    repeat_ok, repeat_s = not context['kill'], 0.
    if repeat_ok:
        for _ in range(3):
            res, _ = pass_cases(module, gpu, shared, longctx, prior=ctx_logs)
            repeat_ok &= res['bitwise_repeat']
            repeat_s += res['wall_s']
    stages['repeat'] = {'pass': repeat_ok, 'wall_s': repeat_s, 'extra_passes': 3}
    start = time.perf_counter()
    jitter = jitter_module(engine, library)
    jit_ok = True
    for _ in range(2):
        res, _ = pass_cases(jitter, gpu, shared, cache['cases'], prior=stored)
        jit_ok &= res['bitwise_repeat']
    stages['jitter'] = {'pass': jit_ok, 'wall_s': time.perf_counter() - start, 'passes': 2}
    stages['sampling'] = {'pass': not sample_errors, 'errors': sample_errors[:6], 'count': len(sample_errors)}
    stages['distribution'] = {'pass': math.isfinite(first['mean_kl']) and first['mean_kl'] <= cache['kl_bound'],
                              'mean_kl': first['mean_kl'], 'bound': cache['kl_bound']}
    return {'pass': all(s['pass'] for s in stages.values()), 'wall_s': time.perf_counter() - stamp,
            'stages': stages}


def regression(engine, baseline, library, gpu, shared, cache, candidate_logs):
    stamp = time.perf_counter()
    module = core.make_module(engine, baseline, HERE / 'build' / 'baseline-bridge')
    _, baseline_logs = pass_cases(module, gpu, shared, cache['cases'])
    equal = all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                for a, b in zip(candidate_logs, baseline_logs))
    extras = list(cache['extra'])
    if engine == 'scale':
        extras.append(torch.load(HERE / 'cache/scale-context-2049.pt', weights_only=False, map_location='cpu'))
    candidate = core.make_module(engine, library, HERE / 'build' / 'regression-candidate')
    _, extra_logs = pass_cases(candidate, gpu, shared, extras)
    _, baseline_extra = pass_cases(module, gpu, shared, extras)
    equal &= all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                 for a, b in zip(extra_logs, baseline_extra))
    return {'pass': equal, 'wall_s': time.perf_counter() - stamp,
            'baseline': str(Path(baseline).resolve()), 'cases': 3 + len(extras),
            'positions': 192 + sum(c['steps'] for c in extras), 'comparison': 'bitwise logits'}
