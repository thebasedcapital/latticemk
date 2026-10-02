"""Serialized dependency latency per shape, separate from production timing."""
import ctypes
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def main():
    pk=torch.load(ROOT/'bench/lm11/weights_int4_gptq.pt',map_location='cpu');rows=[]
    for name,key,ni,no in [('qkv','L0.qkv',1024,4096),('o','L0.o',2048,1024),('gate_up','L0.gu',1024,6144),('down','L0.down',3072,1024),('lm_head','lm_head',1024,151936)]:
        w={k:v.cuda() for k,v in pk[key].items()};torch.manual_seed(1818)
        x=torch.randn((4,ni),device='cuda',dtype=torch.half)*.1;y=torch.empty((4,no),device='cuda',dtype=torch.float32)
        outputs={};libs={}
        for m in (1,4):
            lib=ctypes.CDLL(str(ROOT/f'kernels/megakernel_mt2/libshape_preload_m{m}.so'))
            lib.shape_profile_best.argtypes=[ctypes.c_int64]*4+[ctypes.c_int]*2+[ctypes.c_int64]
            libs[m]=lib;outputs[m]=torch.zeros((36,32,8),device='cuda',dtype=torch.int64)
        samples={m:[] for m in libs}
        for r in range(27):
            for m in ((1,4) if r%2 else (4,1)):
                out=outputs[m];out.zero_();rc=libs[m].shape_profile_best(w['codes'].data_ptr(),w['meta'].data_ptr(),x.data_ptr(),y.data_ptr(),ni,no,out.data_ptr())
                if rc:raise RuntimeError(rc)
                raw=out.cpu().numpy().reshape(-1,8);raw=raw[np.any(raw,axis=1)];samples[m].append(np.median(raw,axis=0))
        for m,s in samples.items():
            raw=np.asarray(s);cycles=np.median(raw,axis=0);nc=ni//1024
            # Each coarse region has one clock-boundary floor. NC=1 stages x once
            # per matrix/warp rather than once per output row. Amortize by rows.
            boundaries=np.array([nc,nc,1 if nc==1 else nc*m,nc*m,nc*m,1,1])
            corrected=np.maximum(0,cycles[:7]-cycles[7]*boundaries)
            repeats=(no+36*(32 if m==1 else 16)-1)//(36*(32 if m==1 else 16))
            if nc==1:corrected[2]/=repeats
            row=dict(experiment='stage_profile',shape=name,m=m,n_in=ni,n_out=no,runs=27,categories=['weight_load_wait','dequant','activation_load_wait','fma_issue','local_reduce','warp_reduce','writeback_issue','clock_floor'],median_cycles=cycles.tolist(),corrected_amortized_cycles=corrected.tolist(),clock_regions=boundaries.tolist(),activation_reuse_rows=repeats,samples_cycles=raw.tolist(),diagnostic_threads=128,method='144 sampled warps; coarse operand-dependent clocks; x/FMA regrouped to expose serialized latency. Clock floor subtracted, NC1 x amortized. Not production retired-stall accounting, not additive CUDA-event time.')
            rows.append(row)
            with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='samples_cycles'}),flush=True)
    (HERE/'stages.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
