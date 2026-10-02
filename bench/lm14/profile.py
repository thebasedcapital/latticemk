"""Interleaved CTA-0 clock64 attribution, excluded from production ratios."""
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from engine import Engine,shared

@torch.no_grad()
def main():
    data=shared();keys=[(mode,m) for mode in ('causal','batch') for m in (1,4)]
    engines={k:Engine(k[1],k[0],*data,variant='profile',cap=136) for k in keys}
    out=torch.zeros(3,dtype=torch.int64,device='cuda');samples={k:[] for k in keys};clocks=[]
    def run(k):
        e=engines[k];e.bufs['tok'].fill_(100);e.bufs['pos'].fill_(127);e.activate()
        e.lib.mt_profile.argtypes=[ctypes.c_int64]
        rc=e.lib.mt_profile(out.data_ptr())
        if rc:raise RuntimeError(f'profile CUDA {rc}')
        return out.cpu().tolist()
    for _ in range(3):
        for k in keys:run(k)
    for i in range(27):
        for k in (keys if i%2 else keys[::-1]):samples[k].append(run(k))
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    results=[]
    for mode,m in keys:
        raw=np.asarray(samples[mode,m]);cycles=np.median(raw,axis=0)
        ms=raw/(np.asarray(clocks)[:,None]*1000)
        result=dict(mode=mode,m=m,context=128,weight_cycles=float(cycles[0]),attention_cycles=float(cycles[1]),other_cycles=float(cycles[2]),fractions=(cycles/cycles.sum()).tolist(),phase_median_ms=np.median(ms,axis=0).tolist(),phase_p10_ms=np.percentile(ms,10,axis=0).tolist(),phase_p90_ms=np.percentile(ms,90,axis=0).tolist(),samples_cycles=samples[mode,m],runs=27,sm_clock_samples=clocks,method='CTA0 thread0 clock64; derived ms use round SM clock, not CUDA events. GEMM and attention include ending grid barrier; other includes prologues, append barriers and argmax. Diagnostic build excluded from production speed ratios.')
        results.append(result);print(json.dumps(result),flush=True)
    Path(__file__).with_name('profile.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
