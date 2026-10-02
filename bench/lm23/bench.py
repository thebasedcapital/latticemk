"""Bounded same-process mt2/mt3 full-pass timings and nonzero-KV bit proof."""
import argparse
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from engine import Engine, ROOT, shared, lm03b
HERE = Path(__file__).resolve().parent


def bitwise(a, b):
    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctx', type=int, required=True)
    parser.add_argument('--mode', choices=['batch', 'causal'], required=True)
    parser.add_argument('--samples', type=int, default=27)
    args = parser.parse_args()
    if args.samples < 25:
        parser.error('at least 25 samples required')
    ctx, mode, cap = args.ctx, args.mode, args.ctx + 8
    data = shared()
    caches = 5 if mode == 'batch' else 1
    rng = torch.Generator(device='cuda').manual_seed(23)
    cache = {k: torch.empty(caches*28*cap*1024, dtype=torch.half, device='cuda').normal_(0, .125, generator=rng) for k in ('kc', 'vc')}
    keys = [(v, m) for m in range(1, 6) for v in ('mt2', 'mt3')]
    engines = {k: Engine(k[1], mode, *data, cap=cap, cache_buffers=cache,
                        library=ROOT/f'kernels/megakernel_{k[0]}/libmt{k[1]}.so') for k in keys}
    proof = []
    tokens = [100, 101, 102, 103, 104]
    for m in range(1, 6):
        pos = [ctx-1]*m if mode == 'batch' else ctx-1
        old = engines['mt2', m].run(tokens[:m], pos).cpu()
        new = engines['mt3', m].run(tokens[:m], pos).cpu()
        repeated = engines['mt3', m].run(tokens[:m], pos).cpu()
        sequential = []
        for col in range(m):
            e = engines['mt3', 1]
            if mode == 'batch':
                # Point M1 at exactly this sequence's historical cache.
                offset = col*28*cap*1024
                e = Engine(1, mode, *data, cap=cap,
                           cache_buffers={k:t[offset:offset+28*cap*1024] for k,t in cache.items()})
            sequential.append(e.run([tokens[col]], [ctx-1] if mode == 'batch' else ctx-1+col)[0].cpu())
        sequential = torch.stack(sequential)
        row = dict(m=m, bitwise_vs_mt2=bitwise(old,new), bitwise_repeat=bitwise(new,repeated),
                   bitwise_sequential=bitwise(new,sequential), max_diff_sequential=float((new-sequential).abs().max()),
                   nonfinite=int((~torch.isfinite(new)).sum()))
        proof.append(row)
        if not all(row[k] for k in ('bitwise_vs_mt2','bitwise_repeat','bitwise_sequential')) or row['nonfinite']:
            raise RuntimeError(f'nonzero-KV proof failed: {row}')
    base = None
    if ctx in (128, 2048):
        base = lm03b.Engine2(8704, *data)
        for name in ('kc', 'vc'):
            base.bufs[name].view(28,8704,1024)[:,:cap].copy_(cache[name][:28*cap*1024].view(28,cap,1024))
        keys.append(('v2',1))
    def measure(key):
        if key[0] == 'v2':
            base.set_tok(100)
            out = (ctypes.c_float*1)()
            rc = lm03b._lib2.mk2_time_mega(1,1,ctx-1,out)
            if rc: raise RuntimeError(rc)
            return out[0]
        e = engines[key]
        e.bufs['tok'].fill_(100)
        return e.time(ctx)[0]
    for _ in range(3):
        for key in keys: measure(key)
    samples = {k:[] for k in keys}
    clocks = []
    for i in range(args.samples):
        order = keys[i%len(keys):]+keys[:i%len(keys)]
        if i%2: order.reverse()
        for key in order: samples[key].append(measure(key))
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    rows = []
    for variant,m in keys:
        values = samples[variant,m]
        row = dict(tag='measured', experiment='full_pass', work_package='LM-23', variant=variant,
                   mode=mode, context=ctx, m=m, median_ms=float(np.median(values)),
                   p10_ms=float(np.percentile(values,10)), p90_ms=float(np.percentile(values,90)),
                   samples_ms=values, samples=len(values), sm_clock_samples=clocks,
                   sm_clock_mhz=int(np.median(clocks)), threads=1024 if m==1 else 512,
                   method='Same-process rotating/reversing interleaved CUDA events, 3 warmups, nonzero seeded FP16 KV; token 100 reset before each full pass. v2 uses its own compiler contract.')
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','sm_clock_samples','method')}),flush=True)
    result = dict(context=ctx, mode=mode, proof=proof, timings=rows)
    (HERE/f'{mode}-{ctx}.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'results.jsonl').open('a') as f:
        for row in rows: f.write(json.dumps(row)+'\n')


if __name__ == '__main__':
    main()
