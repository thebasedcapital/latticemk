"""Tier 3a: every hidden state and pre-o_proj attention against HF fp32 fake quant.

The mandatory full probe teacher-forces tokens only. It dumps the candidate's
actual FP16 post-MLP residuals and pre-o_proj attention with no hidden injection.
The additional binding local probe overwrites every layer's incoming residual
with the HF incoming state rounded to FP16. Both probes keep candidate-built
QKV, Q/K norm, RoPE and KV histories, never copied from the original. Local
comparison isolates layer faults that whole-chain numerical drift can hide.

Calibration uses all prompt tokens plus the first 63 HF teacher-forced outputs
of each of the three wave-6 base prompts, i.e. every position in the 64-step
logit gate. k=1.25 applies independently to each layer's worst max-absolute and
RMS error over that set. No candidate-specific widening is permitted.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import debug

K = 1.25
VERSION = 'hf-local-layer-v3'


def _core():
    import core
    return core


def _cases(cache):
    return [dict(c, tokens=torch.cat((c['prompt'].long(), c['forced'][:c['steps'] - 1].long())))
            for c in cache['cases']]


def _key(engine, cache):
    model = 'scale' if engine == 'scale' else 'v2'
    bench = ROOT / ('bench/lm12' if model == 'scale' else 'bench/lm11')
    meta = {'version': VERSION, 'model': model, 'k': K,
            'implementation_sha256': debug.digest(Path(__file__)),
            'weights_sha256': debug.digest(bench / 'weights_int4_gptq_deq.pt'),
            'cases': [{'name': c.get('name', str(i)), 'tokens': c['tokens'].tolist()}
                      for i, c in enumerate(_cases(cache))]}
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:20]
    return key, meta


@torch.no_grad()
def reference(engine, cache):
    """Reproducible CPU HF cache. Evaluates causal full sequences, no GPU model allocation."""
    key, metadata = _key(engine, cache)
    folder = HERE / 'cache'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'layers-hf-{key}.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False), key
    from transformers import AutoModel
    if engine == 'scale':
        sys.path.insert(0, str(ROOT / 'bench/lm12'))
        import scale
        snapshot, inter = scale.SNAPSHOT, 6144
    else:
        from lmk.model import SNAPSHOT
        snapshot, inter = SNAPSHOT, 3072
    torch.set_num_threads(8)
    net = AutoModel.from_pretrained(snapshot, dtype=torch.float32, attn_implementation='eager').eval()
    bench = ROOT / ('bench/lm12' if engine == 'scale' else 'bench/lm11')
    deq = torch.load(bench / 'weights_int4_gptq_deq.pt', map_location='cpu')
    for i, layer in enumerate(net.layers):
        qkv = deq[f'L{i}.qkv']
        for name, lo, hi in [('q', 0, 2048), ('k', 2048, 3072), ('v', 3072, 4096)]:
            getattr(layer.self_attn, name + '_proj').weight.data = qkv[lo:hi]
        layer.self_attn.o_proj.weight.data = deq[f'L{i}.o']
        layer.mlp.gate_proj.weight.data = deq[f'L{i}.gu'][:inter]
        layer.mlp.up_proj.weight.data = deq[f'L{i}.gu'][inter:]
        layer.mlp.down_proj.weight.data = deq[f'L{i}.down']
    del deq
    captured = {}
    hooks = []
    for i, layer in enumerate(net.layers):
        def incoming(module, args, kwargs, index=i):
            value = args[0] if args else kwargs['hidden_states']
            captured['input'][index] = value[0].detach().clone()
        def outgoing(module, args, value, index=i):
            if isinstance(value, tuple):
                value = value[0]
            captured['hidden'][index] = value[0].detach().clone()
        def attn_input(module, args, index=i):
            captured['attention'][index] = args[0][0].detach().clone()
        hooks.extend([layer.register_forward_pre_hook(incoming, with_kwargs=True),
                      layer.register_forward_hook(outgoing),
                      layer.self_attn.o_proj.register_forward_pre_hook(attn_input)])
    rows = []
    started = time.perf_counter()
    for c in _cases(cache):
        captured = {kind: {} for kind in ['input', 'hidden', 'attention']}
        net(input_ids=c['tokens'][None], use_cache=False)
        rows.append({'name': c.get('name', ''), 'tokens': c['tokens'],
                     **{kind: torch.stack([values[i] for i in range(28)], dim=1)
                        for kind, values in captured.items()}})
    for hook in hooks:
        hook.remove()
    value = {'metadata': metadata, 'cases': rows, 'hf_wall_s': time.perf_counter() - started,
             'script': 'gate/layers.py reference'}
    torch.save(value, path)
    del net
    gc.collect()
    return value, key


def _buffers(m, hid):
    return (torch.empty((m, 28, hid), dtype=torch.float16, device='cuda'),
            torch.empty((m, 28, hid), dtype=torch.float32, device='cuda'),
            torch.empty((m, 28, 2048), dtype=torch.float32, device='cuda'))


def _stats(actual, expected):
    diff = (actual.cpu() - expected).abs()
    return torch.stack((diff.amax(dim=-1), diff.square().mean(dim=-1).sqrt()), dim=-1)


@torch.no_grad()
def verify_association(engine, library, built, data, refs, m, mode):
    """Without injection, the debug build must reproduce this candidate's logits."""
    debug.bind(built, None, None, None)
    outputs = []
    for selected in [library, built]:
        rows = []
        if engine == 'mt':
            from mt_bridge import Engine
            for c in refs['cases']:
                obj = Engine(m, mode, *data, cap=32, library=selected)
                for pos in range(0, 2 * m if mode == 'causal' else 2,
                                 m if mode == 'causal' else 1):
                    tokens = [int(c['tokens'][min(pos + col if mode == 'causal' else pos,
                                                 len(c['tokens']) - 1)]) for col in range(m)]
                    rows.append(obj.run(tokens, pos if mode == 'causal' else [pos] * m).cpu())
                del obj
        else:
            core = _core()
            module = core.make_module(engine, selected, folder=built.parent / ('association-' + Path(selected).stem))
            gpu, shared = data
            for c in refs['cases']:
                obj = module.Engine2(32, gpu, *shared)
                for pos, token in enumerate(c['tokens'][:8]):
                    obj.pos_set(pos)
                    obj.set_tok(int(token))
                    obj.mega(1)
                    rows.append(obj.logits().cpu())
                del obj
        outputs.append(torch.stack(rows))
    if not torch.equal(outputs[0], outputs[1]):
        delta = float((outputs[0] - outputs[1]).abs().max())
        raise RuntimeError(f'candidate source/debug build does not reproduce selected library logits: max_abs={delta}')
    return {'bitwise_equal': True, 'rows': len(outputs[0]),
            'script': 'gate/layers.py verify_association'}


@torch.no_grad()
def collect(engine, library, data, refs, m=1, mode='causal', local=False):
    """Return each token/layer's max-absolute and RMS errors, all columns exercised."""
    built, manifest = debug.build(engine, library, m=m)
    association = verify_association(engine, library, built, data, refs, m, mode)
    manifest = {**manifest, 'association_check': association}
    hid = 2048 if engine == 'scale' else 1024
    inputs, hidden, attention = _buffers(m, hid)
    results = []
    if engine == 'mt':
        from mt_bridge import Engine
        # Causal passes pack consecutive forced tokens; batch columns rotate cases.
        if mode == 'causal':
            groups = [[c] for c in refs['cases']]
        else:
            groups = [[refs['cases'][(shift + col) % len(refs['cases'])] for col in range(m)]
                      for shift in range(len(refs['cases']))]
        for group in groups:
            length = max(len(c['tokens']) for c in group)
            cap = length + m + 8
            obj = Engine(m, mode, *data, cap=cap, library=built)
            rows = []
            stride = m if mode == 'causal' else 1
            for pos in range(0, length, stride):
                expected_hidden, expected_attention, tokens = [], [], []
                valid = []
                for col in range(m):
                    c = group[0] if mode == 'causal' else group[col]
                    index = pos + col if mode == 'causal' else pos
                    valid.append(index < len(c['tokens']))
                    index = min(index, len(c['tokens']) - 1)
                    inputs[col].copy_(c['input'][index])
                    tokens.append(int(c['tokens'][index]))
                    expected_hidden.append(c['hidden'][index])
                    expected_attention.append(c['attention'][index])
                hidden.fill_(float('nan')); attention.fill_(float('nan'))
                debug.bind(built, inputs if local else None, hidden, attention)
                obj.run(tokens, pos if mode == 'causal' else [pos] * m)
                h = _stats(hidden, torch.stack(expected_hidden))
                a = _stats(attention, torch.stack(expected_attention))
                for col, take in enumerate(valid):
                    if take:
                        rows.append(torch.stack((h[col], a[col])))
            results.append({'name': '/'.join(c['name'] for c in group), 'errors': torch.stack(rows)})
            del obj
    else:
        core = _core()
        module = core.make_module(engine, built, folder=built.parent / 'bridge')
        gpu, shared = data
        for c in refs['cases']:
            obj = module.Engine2(len(c['tokens']) + 8, gpu, *shared)
            rows = []
            for pos, token in enumerate(c['tokens']):
                inputs[0].copy_(c['input'][pos])
                hidden.fill_(float('nan')); attention.fill_(float('nan'))
                debug.bind(built, inputs if local else None, hidden, attention)
                obj.pos_set(pos)
                obj.set_tok(int(token))
                obj.mega(1)
                rows.append(torch.stack((_stats(hidden[0], c['hidden'][pos]),
                                         _stats(attention[0], c['attention'][pos]))))
            results.append({'name': c['name'], 'errors': torch.stack(rows)})
            del obj
    debug.bind(built, None, None, None)
    return results, manifest


def _original(engine, m):
    if engine == 'mt':
        return ROOT / f'kernels/megakernel_mt/libmt{m}.so'
    return debug.ORIGINAL[engine][0]


def calibration(engine, data, cache, m=1, mode='causal', local=False):
    torch.set_num_threads(8)
    refs, key = reference(engine, cache)
    original = _original(engine, m)
    _, signature = debug.build(engine, original, m=m)
    stamp = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    probe = 'local' if local else 'full'
    path = HERE / 'cache' / f'layers-calibration-{engine}-m{m}-{mode}-{probe}-{key}-{stamp}.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False), refs
    started = time.perf_counter()
    results, signature = collect(engine, original, data, refs, m, mode, local)
    all_errors = torch.cat([r['errors'] for r in results])
    if not torch.isfinite(all_errors).all():
        raise RuntimeError('known-correct debug kernel produced missing/nonfinite layer dumps')
    observed = all_errors.amax(dim=0)
    bounds = observed * K
    value = {'version': VERSION, 'probe': probe, 'k': K, 'observed': observed, 'bounds': bounds,
             'implementation_sha256': debug.digest(Path(__file__)),
             'results': results, 'signature': signature, 'reference_metadata': refs['metadata'],
             'wall_s': time.perf_counter() - started, 'script': 'gate/layers.py calibration',
             'metrics': ['max_abs', 'rms'], 'signals': ['hidden', 'attention']}
    torch.save(value, path)
    summary = {k: v for k, v in value.items() if k not in ['results', 'observed', 'bounds']}
    summary.update(observed=observed.tolist(), bounds=bounds.tolist(), cache=str(path.relative_to(ROOT)))
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return value, refs


def _probe_check(engine, library, data, cache, m=1, mode='causal', local=False):
    started = time.perf_counter()
    # Resolve first: unsupported candidates must not accidentally run originals.
    debug.resolve_source(engine, library)
    cal, refs = calibration(engine, data, cache, m, mode, local)
    results, signature = collect(engine, library, data, refs, m, mode, local)
    failures = []
    maxima = torch.zeros_like(cal['bounds'])
    for row in results:
        errors = row['errors']
        maxima = torch.maximum(maxima, errors.amax(dim=0))
        bad = ~torch.isfinite(errors) | (errors > cal['bounds'][None])
        for signal, name in enumerate(['hidden', 'attention']):
            for layer in range(28):
                positions = torch.nonzero(bad[:, signal, layer].any(dim=-1)).flatten()
                if len(positions):
                    p = int(positions[0])
                    failures.append({'case': row['name'], 'signal': name, 'layer': layer,
                                     'first_position_index': p, 'failing_positions': len(positions),
                                     'error': errors[p, signal, layer].tolist(),
                                     'bound': cal['bounds'][signal, layer].tolist()})
    result = {'pass': not failures, 'wall_s': time.perf_counter() - started, 'failures': failures,
              'k': K, 'comparison': ('local-HF-input-fp16; candidate-KV-history' if local else
                                     'full-uninjected-hidden; teacher-forced-tokens; candidate-KV-history'),
              'metrics': ['max_abs', 'rms'], 'maxima': maxima.tolist(), 'signature': signature,
              'calibration_signature': cal['signature'], 'script': 'gate/layers.py check'}
    probe = 'local' if local else 'full'
    evidence = HERE / 'build' / ('layers-evidence-' + signature['library_sha256'][:20] + f'-m{m}-{mode}-{probe}.json')
    def finite(value):
        if isinstance(value, float) and not __import__('math').isfinite(value):
            return str(value)
        if isinstance(value, list):
            return [finite(v) for v in value]
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        return value
    result = finite(result)
    evidence.write_text(json.dumps(result, indent=2, allow_nan=False))
    result['evidence'] = str(evidence.relative_to(ROOT))
    return result

def _check(engine, library, data, cache, m=1, mode='causal'):
    started = time.perf_counter()
    full = _probe_check(engine, library, data, cache, m, mode, local=False)
    local = _probe_check(engine, library, data, cache, m, mode, local=True)
    result = {'pass': full['pass'] and local['pass'], 'full': full, 'local': local,
              'wall_s': time.perf_counter() - started, 'script': 'gate/layers.py check',
              'implementation_sha256': debug.digest(Path(__file__))}
    signature = full['signature']
    evidence = HERE / 'build' / ('layers-evidence-' + signature['library_sha256'][:20] + f'-m{m}-{mode}.json')
    evidence.write_text(json.dumps(result, indent=2, allow_nan=False))
    result['evidence'] = str(evidence.relative_to(ROOT))
    return result



def check(engine, library, module, gpu, shared, cache):
    return _check(engine, library, (gpu, shared), cache)


def check_mt(library, m, mode, data, cache):
    return _check('mt', library, data, cache, m, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['v2', 'scale'], required=True)
    parser.add_argument('--lib', type=Path)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    core = _core()
    cache = core.load_cache(args.engine)
    if args.reference_only:
        refs, key = reference(args.engine, cache)
        print(json.dumps({'reference_key': key, 'hf_wall_s': refs['hf_wall_s'], 'script': 'gate/layers.py reference'}))
    else:
        library = args.lib or _original(args.engine, 1)
        module = core.make_module(args.engine, library)
        gpu, shared = core.resources(args.engine, module)
        print(json.dumps(check(args.engine, library, module, gpu, shared, cache)))
