"""Pair the numerical compiler contract against the contracted preload build."""
import json
import subprocess
from pathlib import Path
import numpy as np
from engine import Engine,shared
HERE=Path(__file__).resolve().parent

def main():
    data=shared();engines={(v,m):Engine(m,'batch',*data,variant=v,cap=136) for v in ('preload','') for m in (1,4)}
    keys=list(engines);samples={k:[] for k in keys};clocks=[]
    for e in engines.values():e.bufs['tok'].fill_(100);e.time(128,3)
    for r in range(27):
        order=keys[r%4:]+keys[:r%4]
        if r%2:order=order[::-1]
        for k in order:samples[k]+=engines[k].time(128)
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    rows=[]
    for (v,m),s in samples.items():
        row=dict(experiment='contraction_cost',variant='fmad-true' if v else 'fmad-false',m=m,runs=27,median_ms=float(np.median(s)),samples_ms=s,sm_clock_samples=clocks)
        rows.append(row)
        with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','sm_clock_samples')}),flush=True)
    (HERE/'contraction-cost.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
