"""Paired clock64 attribution copied from LM-18 with exact compiler contract."""
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from engine import Engine,shared
HERE=Path(__file__).resolve().parent
CATEGORIES=['qkv','o','gate_up','down','lm_head','attention_scan','residual_rmsnorm','norm_staging','attention_combine','silu_staging','kv_append','argmax','grid_barriers','attention_prologue_norm_rope_kv','attention_merge']
@torch.no_grad()
def main():
    data=shared();keys=[(variant,mode,m) for mode in ('causal','batch') for m in (1,4) for variant in ('baseline_detail','detail')]
    engines={k:Engine(k[2],k[1],*data,variant=k[0],cap=136) for k in keys}
    out=torch.zeros(15,dtype=torch.int64,device='cuda');samples={k:[] for k in keys};clocks=[]
    def run(k):
        e=engines[k];e.bufs['tok'].fill_(100);e.bufs['pos'].fill_(127);e.activate();e.lib.mt_profile.argtypes=[ctypes.c_int64]
        rc=e.lib.mt_profile(out.data_ptr())
        if rc:raise RuntimeError(rc)
        return out.cpu().tolist()
    for _ in range(3):
        for k in keys:run(k)
    for i in range(27):
        for k in (keys if i%2 else keys[::-1]):samples[k].append(run(k))
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    results=[]
    for variant,mode,m in keys:
        raw=np.asarray(samples[variant,mode,m]);ms=raw/(np.asarray(clocks)[:,None]*1000)
        row=dict(variant=variant,mode=mode,m=m,context=128,categories=CATEGORIES,phase_median_ms=np.median(ms,axis=0).tolist(),phase_p10_ms=np.percentile(ms,10,axis=0).tolist(),phase_p90_ms=np.percentile(ms,90,axis=0).tolist(),samples_cycles=samples[variant,mode,m],runs=27,sm_clock_samples=clocks,method='CTA0 thread0 clock64, --fmad=false, separate grid barriers, diagnostic boundaries, no additive event-time claim')
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('samples_cycles','sm_clock_samples')}),flush=True)
    (HERE/'detail-profile.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
