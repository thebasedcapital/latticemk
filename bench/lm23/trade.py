"""Compare selected structural boundaries against spill-free lower-thread inlining."""
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from engine import Engine, ROOT, shared
HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    data = shared()
    results = []
    for mode,ctx in [('batch',128),('causal',128),('batch',2048),('causal',2048),('causal',8192)]:
        cap = ctx+8
        count = 5 if mode == 'batch' else 1
        cache = {k:torch.empty(count*28*cap*1024,dtype=torch.half,device='cuda').normal_(0,.125) for k in ('kc','vc')}
        keys = [(v,m) for m in (1,5) for v in ('selected','inline_less_threads')]
        engines = {k:Engine(k[1],mode,*data,cap=cap,cache_buffers=cache,
                           library=ROOT/(f'kernels/megakernel_mt3/libmt{k[1]}.so' if k[0]=='selected' else f'gate/build/lm23-thread-trade/libmt{k[1]}.so')) for k in keys}
        for m in (1,5):
            pos = ctx-1 if mode=='causal' else [ctx-1]*m
            a = engines['selected',m].run(list(range(100,100+m)),pos)
            b = engines['inline_less_threads',m].run(list(range(100,100+m)),pos)
            if not torch.equal(a.view(torch.int32),b.view(torch.int32)):
                raise RuntimeError(f'thread trade changes bits {mode} ctx{ctx} M{m}')
        def measure(k):
            engines[k].bufs['tok'].fill_(100)
            return engines[k].time(ctx)[0]
        for _ in range(3):
            for k in keys:measure(k)
        samples = {k:[] for k in keys};clocks=[]
        for i in range(27):
            order = keys[i%len(keys):]+keys[:i%len(keys)]
            if i%2:order.reverse()
            for k in order:samples[k].append(measure(k))
            clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
        for m in (1,5):
            old,new = samples['inline_less_threads',m],samples['selected',m]
            row = dict(tag='measured',mode=mode,context=ctx,m=m,selected_threads=1024 if m==1 else 512,
                       inline_threads=512 if m==1 else 256,bitwise_equal=True,selected_median_ms=float(np.median(new)),
                       inline_median_ms=float(np.median(old)),selected_samples_ms=new,inline_samples_ms=old,
                       selected_p10_p90=np.percentile(new,[10,90]).tolist(),inline_p10_p90=np.percentile(old,[10,90]).tolist(),
                       sm_clock_samples=clocks,samples=27,method='Same-process rotating/reversing CUDA events. Thread count and phase boundary change together; not isolated call overhead.')
            row['derived_selected_minus_inline_ms'] = row['selected_median_ms']-row['inline_median_ms']
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if 'samples' not in k and 'method' not in k}),flush=True)
        del engines,cache;gc.collect();torch.cuda.empty_cache()
    (HERE/'trade.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
