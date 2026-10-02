"""Gate both modes at every M against HF and sequential M=1 outputs."""
import gc
import json
from pathlib import Path
import torch
from engine import Engine, shared, lm03b
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    refs=[torch.load(HERE.parent/'lm14'/f'reference-{p}.pt',map_location='cpu') for p in range(3)]
    data=shared();base=lm03b.Engine2(256,*data)
    seeds=[];baseline=[];baseline_error=0.
    for ref in refs:
        ids=ref['prompt'].tolist();base.prefill(ids[:-1])
        seeds.append({k:base.bufs[k].view(28,8704,1024)[:,:256].clone() for k in ('kc','vc')})
        output=[]
        for step in range(64):
            base.set_tok(ids[-1] if step==0 else int(ref['tokens'][step-1]));base.mega()
            output.append(base.logits().cpu())
        baseline.append(torch.stack(output))
        baseline_error=max(baseline_error,float((baseline[-1]-ref['logits']).abs().max()))
    del base;gc.collect();torch.cuda.empty_cache()
    all_results=[]
    own_sequential=[]
    for pi,ref in enumerate(refs):
        e=Engine(1,'causal',*data)
        for k in ('kc','vc'):e.bufs[k].view(28,256,1024).copy_(seeds[pi][k])
        ids=ref['prompt'].tolist();position=len(ids)-1
        output=[]
        for step in range(64):output.append(e.run([ids[-1] if step==0 else int(ref['tokens'][step-1])],position+step)[0].cpu())
        own_sequential.append(torch.stack(output));del e
    for mode in ('causal','batch'):
        for m in range(1,6):
            def execute():
                e=Engine(m,mode,*data);out=[]
                if mode=='causal':
                    for pi,ref in enumerate(refs):
                        for k in ('kc','vc'):e.bufs[k].view(28,256,1024).copy_(seeds[pi][k])
                        ids=ref['prompt'].tolist();inputs=[ids[-1]]+ref['tokens'][:63].tolist();chunks=[]
                        for s in range(0,64,m):
                            ts=inputs[s:s+m];ts+= [ts[-1]]*(m-len(ts))
                            chunks.append(e.run(ts,len(ids)-1+s).cpu()[:min(m,64-s)])
                        out.append(torch.cat(chunks))
                else:
                    # Rotate all three prompts across columns, so each column sees each prompt.
                    for rotate in range(3):
                        pis=[(c+rotate)%3 for c in range(m)]
                        for c,pi in enumerate(pis):
                            for k in ('kc','vc'):e.bufs[k].view(m,28,256,1024)[c].copy_(seeds[pi][k])
                        steps=[]
                        for s in range(64):
                            tokens=[int(refs[pi]['prompt'][-1]) if s==0 else int(refs[pi]['tokens'][s-1]) for pi in pis]
                            positions=[len(refs[pi]['prompt'])-1+s for pi in pis]
                            steps.append(e.run(tokens,positions).cpu())
                        stacked=torch.stack(steps)
                        out.extend(stacked[:,c] for c in range(m))
                del e;gc.collect();torch.cuda.empty_cache();return out
            first,second=execute(),execute()
            prompt_indices=list(range(3)) if mode=='causal' else [(c+r)%3 for r in range(3) for c in range(m)]
            metrics=[];near=[];hard=[];repeat=True;seqdiff=0.;nonfinite=0
            for si,(a,b,pi) in enumerate(zip(first,second,prompt_indices)):
                repeat &= torch.equal(a,b)
                seqdiff=max(seqdiff,float((a-own_sequential[pi]).abs().max()))
                error=float((a-refs[pi]['logits']).abs().max());metrics.append(error)
                nonfinite+=int((~torch.isfinite(a)).sum())
                for step in range(64):
                    target=refs[pi]['logits'][step];mine=int(a[step].argmax());top=target.topk(2)
                    if mine!=int(top.indices[0]):
                        diff=float((a[step]-target).abs().max());margin=float(top.values[0]-top.values[1])
                        event=dict(sequence=si,prompt=pi,column=si%m if mode=='batch' else None,step=step,diff=diff,margin=margin,mine=mine,reference=int(top.indices[0]))
                        (near if margin<2*diff else hard).append(event)
            passed=max(metrics)<=0.5 and max(metrics)<=1.25*baseline_error and not hard and repeat and not nonfinite and seqdiff==0.0
            result=dict(mode=mode,m=m,max_diff_hf=max(metrics),per_sequence_diff=metrics,baseline_v2_diff=baseline_error,max_diff_sequential=seqdiff,sequential_tolerance=0.0,bitwise_repeat=repeat,nonfinite=nonfinite,near_ties=near,hard_flips=hard,pass_gate=passed)
            all_results.append(result);print(json.dumps(result),flush=True)
            (HERE/'gate.json').write_text(json.dumps(all_results,indent=2)+'\n')
    if not all(r['pass_gate'] for r in all_results):raise SystemExit('LM-23 gate FAIL')
if __name__=='__main__':main()
