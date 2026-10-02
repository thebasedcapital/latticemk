"""Interleave selected real 113-matrix GEMM-only passes at M=1,2,4."""
import json
import subprocess
from pathlib import Path
import numpy as np
from engine import Engine,shared
HERE=Path(__file__).resolve().parent

def main():
    data=shared();ms=(1,2,4);engines={m:Engine(m,'batch',*data,cap=136) for m in ms}
    samples={m:[] for m in ms};clocks=[]
    for e in engines.values():e.time_gemv(3)
    for r in range(27):
        order=ms[r%3:]+ms[:r%3]
        if r%2:order=order[::-1]
        for m in order:samples[m]+=engines[m].time_gemv()
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    size=sum(t.numel()*t.element_size() for v in data[0].values() for t in v.values());rows=[]
    for m,s in samples.items():
        med,p10,p90=map(float,np.percentile(s,[50,10,90]))
        row=dict(experiment='gemm_probe',m=m,median_ms=med,p10_ms=p10,p90_ms=p90,weight_bytes=size,gbps=size/med/1e6,runs=27,samples_ms=s,sm_clock_samples=clocks,method='113 real packed matrices; includes fixed activation staging, excludes attention/grid barriers; spill-free selected preload')
        rows.append(row)
        with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','sm_clock_samples')}),flush=True)
    out={'rows':rows,'t_over_t1':{r['m']:r['median_ms']/rows[0]['median_ms'] for r in rows}}
    (HERE/'gemv.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['t_over_t1']),flush=True)
if __name__=='__main__':main()
