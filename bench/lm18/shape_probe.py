"""Interleaved single-shape timing and clock64 latency diagnostics."""
import argparse
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',default=['original']);ap.add_argument('--runs',type=int,default=27);ap.add_argument('--profile',action='store_true');a=ap.parse_args()
    packed=torch.load(ROOT/'bench/lm11/weights_int4_gptq.pt',map_location='cpu')
    shapes={}
    for kind,ni,no in [('gate_up',1024,6144),('down',3072,1024)]:
        key='L0.gu' if kind=='gate_up' else 'L0.down'
        shapes[kind]=(ni,no,{k:v.cuda() for k,v in packed[key].items()})
    libs={(v,m):ctypes.CDLL(str(ROOT/f'kernels/megakernel_mt2/libshape_{v}_m{m}.so')) for v in a.variants for m in range(1,5)}
    for lib in libs.values():
        lib.shape_time.argtypes=[ctypes.c_int64]*4+[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_float)]
        lib.shape_profile.argtypes=[ctypes.c_int64]*4+[ctypes.c_int]*2+[ctypes.c_int64]
    for name,(ni,no,pk) in shapes.items():
        g=torch.Generator(device='cuda').manual_seed(1818)
        x=torch.randn((4,ni),generator=g,device='cuda',dtype=torch.half)*0.1;y=torch.empty((4,no),device='cuda',dtype=torch.float32)
        keys=list(libs);samples={k:[] for k in keys};clocks=[]
        def run(k):
            out=ctypes.c_float();rc=libs[k].shape_time(pk['codes'].data_ptr(),pk['meta'].data_ptr(),x.data_ptr(),y.data_ptr(),ni,no,64,ctypes.byref(out))
            if rc:raise RuntimeError(rc)
            return out.value
        for k in keys:run(k)
        for r in range(a.runs):
            order=keys[r%len(keys):]+keys[:r%len(keys)]
            if r%2:order=order[::-1]
            for k in order:samples[k].append(run(k))
            clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
        for (v,m),s in samples.items():
            row=dict(experiment='one_shape',variant=v,m=m,shape=name,n_in=ni,n_out=no,median_ms=float(np.median(s)),p10_ms=float(np.percentile(s,10)),p90_ms=float(np.percentile(s,90)),samples_ms=s,runs=a.runs,sm_clock_samples=clocks,cache='64 repetitions of one real matrix; warm L2; includes activation staging and launch')
            if a.profile and v=='original':
                out=torch.zeros((36,32,6),dtype=torch.int64,device='cuda')
                rc=libs[v,m].shape_profile(pk['codes'].data_ptr(),pk['meta'].data_ptr(),x.data_ptr(),y.data_ptr(),ni,no,out.data_ptr())
                if rc:raise RuntimeError(rc)
                raw=out.cpu().numpy().reshape(-1,6);raw=raw[np.any(raw,axis=1)]
                row['clock64']={'categories':['weight_load_wait','dequant','x_load_wait','fma','local_reduce','warp_reduce'],'median_cycles':np.median(raw,axis=0).tolist(),'p10_cycles':np.percentile(raw,10,axis=0).tolist(),'p90_cycles':np.percentile(raw,90,axis=0).tolist(),'warp_count':len(raw),'raw_cycles':raw.tolist(),'note':'first row per warp; dependency-enforced boundaries around every operand and FMA; includes clock/branch instrumentation overhead, not production stall percentages'}
            with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({k:val for k,val in row.items() if k not in ('samples_ms','sm_clock_samples','clock64')}),flush=True)
            if 'clock64' in row:print(json.dumps({k:val for k,val in row['clock64'].items() if k!='raw_cycles'}),flush=True)
if __name__=='__main__':main()
