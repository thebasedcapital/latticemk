"""Prove M1 dump instrumentation preserves raw logit bits with dumping enabled."""
import json
import sys
from pathlib import Path
import torch
from engine import Engine, ROOT, shared
sys.path.insert(0, str(ROOT/'gate'))
import debug
HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    library = ROOT/'kernels/megakernel_mt3/libmt1.so'
    built,manifest = debug.build('mt',library,m=1)
    data = shared()
    hidden = torch.empty((1,28,1024),dtype=torch.float32,device='cuda')
    attention = torch.empty((1,28,2048),dtype=torch.float32,device='cuda')
    debug.bind(built,None,hidden,attention)
    refs = [torch.load(HERE.parent/'lm14'/f'reference-{p}.pt',map_location='cpu') for p in range(3)]
    rows = 0
    for ref in refs:
        tokens = ref['prompt'].tolist()+ref['tokens'][:8].tolist()
        old = Engine(1,'causal',*data,cap=256,library=library)
        new = Engine(1,'causal',*data,cap=256,library=built)
        for pos,token in enumerate(tokens):
            a = old.run([token],pos)
            hidden.fill_(float('nan')); attention.fill_(float('nan'))
            b = new.run([token],pos)
            if not torch.equal(a.view(torch.int32),b.view(torch.int32)):
                raise RuntimeError(f'dump changed logit bits at position {pos}')
            if not bool(torch.isfinite(hidden).all() and torch.isfinite(attention).all()):
                raise RuntimeError('incomplete or nonfinite actual dumps')
            rows += 1
    result = dict(tag='measured',m=1,mode='causal',bitwise_equal=True,dumping_enabled=True,
                  finite_complete_dumps=True,positions=rows,manifest=manifest)
    (HERE/'dump-m1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}))


if __name__ == '__main__':
    main()
