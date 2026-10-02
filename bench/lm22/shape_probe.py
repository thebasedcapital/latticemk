"""Five real layer-0 shapes; CPU check before interleaved CUDA-event timing."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from fragment_check import library,call
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=27);ap.add_argument('--variants',nargs='+',default=['cuda','mma1','mma4','mma8']);a=ap.parse_args()
    packed=torch.load(ROOT/'bench/lm11/weights_int4_gptq.pt',map_location='cpu');libs={(v,m):library(v,m) for v in a.variants for m in (1,2,4)};results=[]
    for kind,key,ni,no in [('qkv','L0.qkv',1024,4096),('o','L0.o',2048,1024),('gate_up','L0.gu',1024,6144),('down','L0.down',3072,1024),('lm_head','lm_head',1024,151936)]:
        pk={k:v.cuda() for k,v in packed[key].items()};g=torch.Generator().manual_seed(22)
        xc=(torch.randn((4,ni),generator=g,dtype=torch.float32)*0.1).half();x=xc.cuda();y=torch.empty((4,no),device='cuda',dtype=torch.float32)
        # Small real row subset validates non-binary scales/offsets against CPU.
        words=packed[key]['codes'][:35].numpy().astype(np.uint32).reshape(35,ni//8,1);sh=np.arange(4,dtype=np.uint32)*4
        q=np.stack([(words>>sh)&15,(words>>(sh+16))&15],axis=-1).reshape(35,ni)
        meta=packed[key]['meta'][:35].float().numpy();weights=(q*np.repeat(meta[...,0],128,1)+np.repeat(meta[...,1],128,1)).astype(np.float16).astype(np.float32)
        reference=xc.float().numpy()@weights.T;checks={}
        for k,lib in libs.items():
            call(lib,pk,x,y);out=y[:k[1],:35].cpu().numpy();err=float(np.max(np.abs(out-reference[:k[1]])))
            assert err<(0.015 if k[0]=='cuda' else 0.0002),(kind,k,err)
            checks[str(k)]=err
        samples={k:[] for k in libs};keys=list(libs);clocks=[]
        for r in range(a.runs):
            order=keys[r%len(keys):]+keys[:r%len(keys)]
            if r%2:order=order[::-1]
            for k in order:samples[k].append(call(libs[k],pk,x,y,32))
            clocks.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader,nounits'],text=True).strip()))
        for k,s in samples.items():
            row=dict(shape=kind,variant=k[0],m=k[1],n_in=ni,n_out=no,median_ms=float(np.median(s)),p10_ms=float(np.percentile(s,10)),p90_ms=float(np.percentile(s,90)),samples_ms=s,sm_clock_samples=clocks,cpu_max_abs=checks[str(k)],runs=a.runs,method='same process rotated interleave; 32 launches; includes staging; warm L2 except head')
            results.append(row);print(json.dumps({z:v for z,v in row.items() if z not in ('samples_ms','sm_clock_samples')}),flush=True)
        (HERE/'shapes.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
