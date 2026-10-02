"""Interleave real 113-matrix weight-only passes at M=1 and M=4."""
import json
import subprocess
from pathlib import Path
import numpy as np
from engine import Engine,shared

def main():
    data=shared();engines=[Engine(m,'batch',*data,cap=136) for m in (1,4)]
    samples=[[],[]];clocks=[]
    for e in engines:e.time_gemv(3)
    for r in range(27):
        for i in ((0,1) if r%2==0 else (1,0)):samples[i]+=engines[i].time_gemv()
        clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
    size=sum(t.numel()*t.element_size() for v in data[0].values() for t in v.values())
    rows=[]
    for m,s in zip((1,4),samples):
        med,p10,p90=map(float,np.percentile(s,[50,10,90]))
        rows.append(dict(m=m,median_ms=med,p10_ms=p10,p90_ms=p90,weight_bytes=size,gbps=size/med/1e6,pct_roofline=size/med/1e6/406.7*100,runs=27,samples_ms=s,sm_clock_samples=clocks,method='113 real packed matrices; persistent GEMM only, including fixed activation staging, without attention or grid barriers'))
    out={'rows':rows,'t4_over_t1':rows[1]['median_ms']/rows[0]['median_ms']}
    Path(__file__).with_name('gemv.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
