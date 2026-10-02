"""Same-process interleaved production event timing against unmodified mt2."""
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from engine import Engine, shared, ROOT
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lm20_reference',ROOT/'bench/lm18/engine.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

@torch.no_grad()
def main():
    data=shared()
    keys=[(variant,mode,m) for mode in ('causal','batch') for m in (1,2,4) for variant in ('mt2','attention')]
    engines={k:(reference.Engine if k[0]=='mt2' else Engine)(k[2],k[1],*data,cap=136) for k in keys}
    for e in engines.values():
        e.bufs['tok'].fill_(100)
    for _ in range(3):
        for k in keys:engines[k].time(128,2)
    samples={k:[] for k in keys};clocks=[]
    for i in range(27):
        for k in (keys if i%2 else keys[::-1]):samples[k].append(float(np.median(engines[k].time(128,2))))
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    rows=[]
    for variant,mode,m in keys:
        a=samples[variant,mode,m]
        row=dict(variant=variant,mode=mode,m=m,context=128,runs=27,median_ms=float(np.median(a)),p10_ms=float(np.percentile(a,10)),p90_ms=float(np.percentile(a,90)),samples_ms=a,sm_clock_samples=clocks,method='CUDA events; same-process interleaved; --fmad=false; two launches per sample')
        rows.append(row);print(json.dumps(row),flush=True)
    (HERE/'timing.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
