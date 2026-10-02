"""Packed layout, PTX fragments, zero padding, tails, and split-K CPU oracle."""
import argparse, ctypes, json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def library(v,m):
    lib=ctypes.CDLL(str(ROOT/f'kernels/megakernel_mma/libshape_{v}_m{m}.so'))
    lib.shape_time.argtypes=[ctypes.c_int64]*4+[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_float)]
    return lib

def call(lib,pk,x,y,reps=1):
    ms=ctypes.c_float();rc=lib.shape_time(pk['codes'].data_ptr(),pk['meta'].data_ptr(),x.data_ptr(),y.data_ptr(),x.shape[1],y.shape[1],reps,ctypes.byref(ms))
    if rc:raise RuntimeError(f'CUDA error {rc}')
    return ms.value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',default=['mma1']);ap.add_argument('--ms',nargs='+',type=int,default=[1,2,4]);a=ap.parse_args()
    rng=np.random.default_rng(22);results=[]
    for ni,no in [(128,17),(1024,35),(2048,33),(3072,16)]:
        q=rng.integers(0,16,(no,ni),dtype=np.int32);pairs=q.reshape(no,ni//8,4,2)
        words=((pairs[...,0].astype(np.uint32) << (np.arange(4)*4).astype(np.uint32)).sum(-1,dtype=np.uint32) | (pairs[...,1].astype(np.uint32) << (16+np.arange(4)*4).astype(np.uint32)).sum(-1,dtype=np.uint32))
        meta=np.empty((no,ni//128,2),dtype=np.float16);meta[...,0]=rng.choice([0.03125,0.0625,0.125],meta.shape[:2]);meta[...,1]=rng.choice([-0.25,-0.5,-1.0],meta.shape[:2])
        w=(q.astype(np.float32)*np.repeat(meta[...,0].astype(np.float32),128,axis=1)+np.repeat(meta[...,1].astype(np.float32),128,axis=1)).astype(np.float16).astype(np.float32)
        pk={'codes':torch.from_numpy(words.view(np.int32)).cuda(),'meta':torch.from_numpy(meta).cuda()}
        for m in a.ms:
            xc=rng.choice([-0.125,-0.0625,0.0625,0.125],(m,ni)).astype(np.float16);x=torch.from_numpy(xc).cuda();ref=xc.astype(np.float32)@w.T
            for v in a.variants:
                y=torch.full((m,no),float('nan'),device='cuda');call(library(v,m),pk,x,y);out=y.cpu().numpy();error=float(np.max(np.abs(out-ref)))
                assert error==0.0,(v,m,ni,no,error)
                results.append(dict(variant=v,m=m,n_in=ni,n_out=no,max_abs_cpu=error))
    (HERE/'fragments.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'pass':True,'cases':len(results),'max_abs_cpu':0.0}))
if __name__=='__main__':main()
