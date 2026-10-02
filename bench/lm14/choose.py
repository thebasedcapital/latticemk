"""Measure the CUDA-core and padded Turing MMA candidates in one process."""
import json
from pathlib import Path
import numpy as np
import torch
from engine import Engine,shared
import subprocess

def clock():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip())

@torch.no_grad()
def main():
    data=shared()
    engines=[Engine(4,'batch',*data),Engine(4,'batch',*data,variant='mma'),Engine(4,'batch',*data,variant='initial'),Engine(4,'batch',*data,variant='fp32')]
    for e in engines:e.bufs['tok'].fill_(100)
    logs=[e.run([100]*4,[0]*4).cpu() for e in engines]
    for e in engines:
        for _ in range(3):e.time(128)
    samples=[[],[],[],[]];clocks=[]
    for r in range(27):
        for i in ((0,1,2,3) if r%2==0 else (3,2,1,0)):samples[i]+=engines[i].time(128)
        clocks.append(clock())
    result={'context':128,'m':4,'mode':'batch','max_logit_diff_candidates':float((logs[0]-logs[1]).abs().max()),'mma_finite':bool(torch.isfinite(logs[1]).all()),'candidates':[]}
    for name,s in zip(('cuda-core-retry-1024','mma-retry-1024','cuda-core-initial-512','fp32-row-retry-1024'),samples):
        med,p10,p90=map(float,np.percentile(s,[50,10,90]))
        result['candidates'].append(dict(kernel=name,median_ms=med,p10_ms=p10,p90_ms=p90,samples_ms=s,runs=27,sm_clock_samples=clocks))
    result['selected']='cuda-core' if np.median(samples[0])<np.median(samples[1]) else 'mma'
    Path(__file__).with_name('choose.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
