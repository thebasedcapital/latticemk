"""Alternate every M and v2 in one process; stop at the binding kill line."""
import argparse
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
from engine import Engine, shared, lm03b
HERE=Path(__file__).resolve().parent

def clock():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ctx',type=int,default=128);ap.add_argument('--runs',type=int,default=27);a=ap.parse_args()
    data=shared();pk,emb,norms,rope=data
    base=lm03b.Engine2(8704,pk,emb,norms,rope);base.set_tok(100)
    engines={(mode,m):Engine(m,mode,*data,cap=a.ctx+8) for mode in ('causal','batch') for m in range(1,6)}
    for e in engines.values():e.bufs['tok'].fill_(100)
    def baseline():
        out=(ctypes.c_float*1)()
        rc=lm03b._lib2.mk2_time_mega(1,1,a.ctx-1,out)
        if rc:raise RuntimeError(f'baseline timer {rc}')
        return out[0]
    keys=list(engines)+[('v2',1)]
    samples={k:[] for k in keys};clocks=[]
    for _ in range(3):
        for k in keys:baseline() if k[0]=='v2' else engines[k].time(a.ctx)
    for r in range(a.runs):
        order=keys[r%len(keys):]+keys[:r%len(keys)]
        if r%2:order=order[::-1]
        for k in order:samples[k].append(baseline() if k[0]=='v2' else engines[k].time(a.ctx)[0])
        clocks.append(clock())
    weight_bytes=sum(t.numel()*t.element_size() for v in pk.values() for t in v.values())
    medians={}
    for k,s in samples.items():
        mode,m=k;median,p10,p90=map(float,np.percentile(s,[50,10,90]));medians[k]=median
        # Byte model counts logical query reads, not unique physical DRAM traffic.
        positions=m*a.ctx+(m*(m-1)//2 if mode=='causal' else 0)
        kv_bytes=28*1024*4*positions;gbps=(weight_bytes+kv_bytes)/median/1e6
        row=dict(commit='nogit',work_package='LM-14',model='Qwen3-0.6B',kernel='megakernel-v2' if mode=='v2' else f'megakernel-mt-{mode}',context=a.ctx,batch=m if mode=='batch' else 1,m_tokens=m,tokens_per_s=m*1000/median,gbps=gbps,pct_roofline=gbps/406.7*100,sm_clock_mhz=int(np.median(clocks)),driver='610.57.04',runs=len(s),median_ms=median,p10_ms=p10,p90_ms=p90,weight_bytes=weight_bytes,kv_bytes=kv_bytes,samples_ms=s,sm_clock_samples=clocks,threads=512 if mode!='v2' and m==5 else 1024,revision='bounded-retry')
        with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({x:row[x] for x in ('kernel','m_tokens','median_ms','p10_ms','p90_ms','tokens_per_s','sm_clock_mhz')}),flush=True)
    decision={'context':a.ctx,'t4_over_t1':medians['batch',4]/medians['batch',1],'m1_batch_over_v2':medians['batch',1]/medians['v2',1],'m1_causal_over_v2':medians['causal',1]/medians['v2',1],'t_over_t1':{mode:{m:medians[mode,m]/medians[mode,1] for m in range(1,6)} for mode in ('causal','batch')}}
    decision['kill']=decision['t4_over_t1']>1.5
    (HERE/f'decision-{a.ctx}.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
    if decision['kill']:raise SystemExit('Binding LM-14 kill line reached. No larger contexts or scale port.')
if __name__=='__main__':main()
