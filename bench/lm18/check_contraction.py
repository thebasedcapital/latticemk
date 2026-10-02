"""One prompt tests whether disabling FP32 contraction removes M4 drift."""
import json
from pathlib import Path
import torch
from engine import Engine,shared,lm03b
HERE=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    ref=torch.load(HERE.parent/'lm14/reference-2.pt',map_location='cpu');data=shared();ids=ref['prompt'].tolist()
    base=lm03b.Engine2(256,*data);base.prefill(ids[:-1]);seeds={k:base.bufs[k].view(28,8704,1024)[:,:256].clone() for k in ('kc','vc')};del base
    inputs=[ids[-1]]+ref['tokens'][:63].tolist();rows=[]
    for variant in ('preload','nocontract'):
        outputs={}
        for m in (1,4):
            e=Engine(m,'causal',*data,variant=variant)
            for k in seeds:e.bufs[k].view(28,256,1024).copy_(seeds[k])
            out=[]
            for s in range(0,64,m):out.append(e.run(inputs[s:s+m],len(ids)-1+s).cpu())
            outputs[m]=torch.cat(out);del e
        row=dict(variant=variant or 'selected',prompt=2,max_diff_sequential=float((outputs[4]-outputs[1]).abs().max()),bitwise_sequential=torch.equal(outputs[1],outputs[4]),max_diff_hf=float((outputs[4]-ref['logits']).abs().max()))
        rows.append(row);print(json.dumps(row),flush=True)
    (HERE/'contraction.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
