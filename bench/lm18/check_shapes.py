"""Localize whether register preloading changes a GEMM's arithmetic."""
import ctypes
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]

def main():
    pk=torch.load(ROOT/'bench/lm11/weights_int4_gptq.pt',map_location='cpu')
    results=[]
    for name,ni,no in [('gu',1024,6144),('down',3072,1024)]:
        w={k:v.cuda() for k,v in pk[f'L0.{name}'].items()}
        torch.manual_seed(1818);x=torch.randn((4,ni),device='cuda',dtype=torch.half);outputs={}
        for m in range(1,5):
            lib=ctypes.CDLL(str(ROOT/f'kernels/megakernel_mt2/libshape_preload_m{m}.so'))
            lib.shape_time.argtypes=[ctypes.c_int64]*4+[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_float)]
            cols=[]
            for off in (range(4) if m==1 else (0,)):
                y=torch.empty((m,no),device='cuda',dtype=torch.float32);ms=ctypes.c_float()
                rc=lib.shape_time(w['codes'].data_ptr(),w['meta'].data_ptr(),x[off:].data_ptr(),y.data_ptr(),ni,no,1,ctypes.byref(ms))
                if rc:raise RuntimeError(rc)
                cols.append(y.cpu())
            outputs[m]=torch.cat(cols)
        for m in range(2,5):
            row=dict(shape=name,m=m,max_diff=float((outputs[m]-outputs[1][:m]).abs().max()),bitwise=torch.equal(outputs[m],outputs[1][:m]));results.append(row);print(json.dumps(row),flush=True)
    Path(__file__).with_name('shape-correctness.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
