"""Paired causal production/attention curves, plus bitwise long-context smoke."""
import argparse
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from engine import Engine,shared,ROOT
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lm20_context_reference',ROOT/'bench/lm18/engine.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
CATEGORIES=['qkv','o','gate_up','down','lm_head','attention_scan','residual_rmsnorm','norm_staging','attention_combine','silu_staging','kv_append','argmax','grid_barriers','attention_prologue_norm_rope_kv','attention_merge']

def clock():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip())

def seed(engines):
    source=engines['mt2',1]
    rng=torch.Generator(device='cuda');rng.manual_seed(20)
    for name in ('kc','vc'):
        source.bufs[name].normal_(0,.125,generator=rng)
        for key,e in engines.items():
            if key!=('mt2',1):e.bufs[name].copy_(source.bufs[name])
    for e in engines.values():e.bufs['tok'].fill_(100)
    torch.cuda.synchronize()

def bitwise(a,b):
    return bool(torch.equal(a.view(torch.int32),b.view(torch.int32)))

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ctx',type=int,required=True);parser.add_argument('--samples',type=int,default=27);args=parser.parse_args()
    ctx=args.ctx;n=args.samples;data=shared();keys=[(v,m) for m in (1,4) for v in ('mt2','attention')]
    engines={k:(reference.Engine if k[0]=='mt2' else Engine)(k[1],'causal',*data,cap=ctx+8) for k in keys}
    seed(engines)
    tokens=[100,101,102,103]
    multi={v:engines[v,4].run(tokens,ctx-1).cpu() for v in ('mt2','attention')}
    sequential=[]
    first_base=engines['mt2',1].run([100],ctx-1)[0].cpu()
    first_tile=engines['attention',1].run([100],ctx-1)[0].cpu()
    sequential.append(first_base)
    for col in range(1,4):sequential.append(engines['mt2',1].run([tokens[col]],ctx-1+col)[0].cpu())
    sequential=torch.stack(sequential)
    smoke=dict(m1_bitwise_vs_mt2=bitwise(first_base,first_tile),m4_bitwise_vs_mt2=bitwise(multi['mt2'],multi['attention']),m4_bitwise_vs_mt2_sequential=bitwise(multi['attention'],sequential),max_diff_sequential=float((multi['attention']-sequential).abs().max()),nonfinite=int((~torch.isfinite(multi['attention'])).sum()),method='Nonzero seeded FP16 KV at all 28 layers; tokens 100,101,102,103; raw FP32 bit comparisons.')
    print(json.dumps(dict(context=ctx,smoke=smoke)),flush=True)
    if not all(smoke[k] for k in ('m1_bitwise_vs_mt2','m4_bitwise_vs_mt2','m4_bitwise_vs_mt2_sequential')) or smoke['nonfinite']:
        (HERE/f'context-{ctx}.json').write_text(json.dumps(dict(context=ctx,smoke=smoke),indent=2)+'\n');raise SystemExit('long-context bitwise smoke FAIL')
    for _ in range(3):
        for k in keys:engines[k].time(ctx)
    samples={k:[] for k in keys};clocks=[]
    for i in range(n):
        for k in (keys if i%2 else keys[::-1]):samples[k].append(engines[k].time(ctx)[0])
        clocks.append(clock())
    full=[]
    for v,m in keys:
        a=samples[v,m]
        full.append(dict(variant=v,m=m,median_ms=float(np.median(a)),p10_ms=float(np.percentile(a,10)),p90_ms=float(np.percentile(a,90)),samples_ms=a,sm_clock_samples=clocks,runs=n,method='Same-process interleaved CUDA-event full passes; --fmad=false.'))
    del engines;gc.collect();torch.cuda.empty_cache()
    engines={k:Engine(k[1],'causal',*data,variant='referencedetail' if k[0]=='mt2' else 'detail',cap=ctx+8) for k in keys}
    seed(engines)
    out=torch.zeros(15,dtype=torch.int64,device='cuda')
    def profile(k):
        e=engines[k];e.bufs['tok'].fill_(100);e.bufs['pos'].fill_(ctx-1);e.activate();e.lib.mt_profile.argtypes=[ctypes.c_int64]
        rc=e.lib.mt_profile(out.data_ptr())
        if rc:raise RuntimeError(rc)
        return out.cpu().tolist()
    for _ in range(3):
        for k in keys:profile(k)
    samples={k:[] for k in keys};clocks=[]
    for i in range(n):
        for k in (keys if i%2 else keys[::-1]):samples[k].append(profile(k))
        clocks.append(clock())
    phases=[]
    for v,m in keys:
        raw=np.asarray(samples[v,m]);ms=raw/(np.asarray(clocks)[:,None]*1000);med=np.median(ms,axis=0)
        phases.append(dict(variant=v,m=m,categories=CATEGORIES,phase_median_ms=med.tolist(),phase_p10_ms=np.percentile(ms,10,axis=0).tolist(),phase_p90_ms=np.percentile(ms,90,axis=0).tolist(),attention_ms=float(sum(med[i] for i in (5,10,13,14))),samples_cycles=samples[v,m],sm_clock_samples=clocks,runs=n,method='Same-process interleaved CTA0 clock64; attention includes scan/prologue/append/partial merge, excludes ending grid barrier. Diagnostic, not CUDA-event attribution.'))
    result=dict(context=ctx,mode='causal',smoke=smoke,full_pass=full,phases=phases)
    (HERE/f'context-{ctx}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(context=ctx,full_pass=[{k:v for k,v in r.items() if k not in ('samples_ms','sm_clock_samples')} for r in full],attention=[dict(variant=r['variant'],m=r['m'],attention_ms=r['attention_ms']) for r in phases])),flush=True)
if __name__=='__main__':main()
