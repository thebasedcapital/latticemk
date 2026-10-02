"""LM-14's unchanged numeric gate plus mt-aware wave-6 extras.

Causal columns are consecutive teacher-forced positions in one cache, not
independent generation. Batch columns have independent caches and positions.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'gate'), str(ROOT/'mutation')]
from mt_bridge import Engine, shared, lm03b
import extra_tests


def release():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def seeds_for(cases, data, cap):
    """Use the unchanged v2 prefill, truncating only allocation padding."""
    seeds = []
    base = lm03b.Engine2(cap, *data)
    for case in cases:
        base.prefill(case['prompt'][:-1])
        seeds.append({k:base.bufs[k].view(28,8704,1024)[:,:cap].cpu().clone() for k in ('kc','vc')})
    del base
    release()
    return seeds


@torch.no_grad()
def execute(library, m, mode, data, cases, seeds, cap, sampling=False, stride=None):
    e = Engine(m, mode, *data, cap=cap, library=library)
    outputs, indices, sample_errors = [], [], []
    if mode == 'causal':
        for pi, case in enumerate(cases):
            for k in ('kc','vc'):
                e.bufs[k].view(28,cap,1024).copy_(seeds[pi][k])
            inputs = [int(case['prompt'][-1])] + case['forced'][:case['steps']-1].tolist()
            lines = []
            for s in range(0, case['steps'], stride or m):
                tokens = inputs[s:s+m]
                tokens += [tokens[-1]] * (m-len(tokens))
                out = e.run(tokens, len(case['prompt'])-1+s).cpu()
                used = min(stride or m, case['steps']-s)
                if sampling:
                    actual = e.bufs['tok'].cpu()
                    for c in range(used):
                        if int(actual[c]) != int(out[c].argmax()):
                            sample_errors.append({'case':pi,'step':s+c,'column':c})
                lines.append(out[:used])
            outputs.append(torch.cat(lines));indices.append(pi)
    else:
        for rotate in range(len(cases)):
            pis = [(c+rotate)%len(cases) for c in range(m)]
            for c, pi in enumerate(pis):
                for k in ('kc','vc'):
                    e.bufs[k].view(m,28,cap,1024)[c].copy_(seeds[pi][k])
            # Cases of differing length are padded by repeating their last forced
            # token, but only their requested steps are scored.
            lines = []
            for s in range(max(cases[pi]['steps'] for pi in pis)):
                tokens = [int(cases[pi]['prompt'][-1]) if s == 0 else int(cases[pi]['forced'][min(s-1,cases[pi]['steps']-1)]) for pi in pis]
                positions = [len(cases[pi]['prompt'])-1+s for pi in pis]
                out = e.run(tokens, positions).cpu()
                if sampling:
                    actual = e.bufs['tok'].cpu()
                    for c, pi in enumerate(pis):
                        if s < cases[pi]['steps'] and int(actual[c]) != int(out[c].argmax()):
                            sample_errors.append({'case':pi,'step':s,'column':c})
                lines.append(out)
            stacked = torch.stack(lines)
            outputs.extend(stacked[:cases[pi]['steps'],c] for c,pi in enumerate(pis));indices.extend(pis)
    del e
    release()
    return outputs, indices, sample_errors


def metrics(outputs, indices, cases, bound, prior=None, sequential=None, distribution=False):
    errors, near, hard = [], [], []
    nonfinite, repeat, seqdiff, kl_sum, count = 0, True, 0., 0., 0
    for si, (actual, pi) in enumerate(zip(outputs, indices)):
        target = cases[pi]['reference']
        nonfinite += int((~torch.isfinite(actual)).sum())
        error = float((actual-target).abs().max());errors.append(error)
        if prior is not None:
            repeat &= torch.equal(actual, prior[si])
        if sequential is not None:
            seqdiff = max(seqdiff,float((actual-sequential[pi]).abs().max()))
        top = target.topk(2,dim=1)
        for step in (actual.argmax(1) != top.indices[:,0]).nonzero().flatten().tolist():
            diff = float((actual[step]-target[step]).abs().max())
            margin = float(top.values[step,0]-top.values[step,1])
            event = {'sequence':si,'prompt':pi,'step':step,'diff':diff,'margin':margin,'mine':int(actual[step].argmax()),'reference':int(top.indices[step,0])}
            (near if margin < 2*diff else hard).append(event)
        if distribution:
            lp,lq = target.double().log_softmax(1),actual.double().log_softmax(1)
            kl_sum += float((lp.exp()*(lp-lq)).sum());count += len(actual)
    maximum = max(errors)
    return {'pass':maximum <= bound and not hard and repeat and not nonfinite and seqdiff <= .1,
            'max_diff_hf':maximum,'per_sequence_diff':errors,'near_ties':near,'hard_flips':hard,'nonfinite':nonfinite,
            'bitwise_repeat':repeat,'max_diff_sequential':seqdiff,'sequential_tolerance':.1,'bound':bound,
            'mean_kl':kl_sum/count if count else None}


@torch.no_grad()
def multistep(library,m,mode,data,cases,seeds,cap):
    case,seed=cases[0],seeds[0]
    if mode == 'causal':
        small = {**case,'steps':12}
        # A full causal pass must equal sliding single-position passes. The
        # future columns are masked, and column zero uses the identical kernel.
        # Supply twelve teacher-forced positions, not speculative generations.
        a,_,_ = execute(library,m,mode,data,[small],[seed],cap)
        b,_,_ = execute(library,m,mode,data,[small],[seed],cap,stride=1)
        return {'pass':torch.equal(a[0],b[0]),'logits_equal':torch.equal(a[0],b[0]),'semantics':'teacher-forced causal chunks versus single-position sliding passes'}
    def feedback(pis):
        e=Engine(m,mode,*data,cap=cap,library=library)
        for c,pi in enumerate(pis):
            for k in ('kc','vc'):e.bufs[k].view(m,28,cap,1024)[c].copy_(seeds[pi][k])
        tokens=[int(cases[pi]['prompt'][-1]) for pi in pis];lines=[];history=[]
        for s in range(12):
            out=e.run(tokens,[len(cases[pi]['prompt'])-1+s for pi in pis]).cpu()
            sampled=e.bufs['tok'].cpu().clone()
            lines.append(out);history.append(sampled);tokens=sampled.tolist()
        del e;release()
        return torch.stack(lines),torch.stack(history)
    pis=[c%len(cases) for c in range(m)]
    batched=feedback(pis)
    isolated={pi:feedback([pi]*m) for pi in set(pis)}
    logits_equal=all(torch.equal(batched[0][:,c],isolated[pi][0][:,c]) for c,pi in enumerate(pis))
    tokens_equal=all(torch.equal(batched[1][:,c],isolated[pi][1][:,c]) for c,pi in enumerate(pis))
    return {'pass':logits_equal and tokens_equal,'logits_equal':logits_equal,'tokens_equal':tokens_equal,'semantics':'mixed-prompt independent batch feedback versus isolated replicated-prompt feedback, twelve passes; mt has no fused multi-step ABI'}


@torch.no_grad()
def run(library=None,tier=2,baseline=None,m=1,mode='causal'):
    if m not in range(1,6) or mode not in ('causal','batch') or tier not in (1,2,3):
        raise ValueError('tier 1..3, M 1..5, mode causal/batch required')
    torch.set_num_threads(12)
    library=Path(library or ROOT/f'kernels/megakernel_mt/libmt{m}.so').resolve()
    started=time.perf_counter();data=shared();tiers={}
    refs=[torch.load(ROOT/f'bench/lm14/reference-{p}.pt',map_location='cpu',weights_only=False) for p in range(3)]
    cases=[{'name':f'prompt-{p}','prompt':r['prompt'],'forced':r['tokens'],'reference':r['logits'],'steps':64} for p,r in enumerate(refs)]
    seeds=seeds_for(cases,data,256)
    base=lm03b.Engine2(256,*data);base_error=0.
    for pi,case in enumerate(cases):
        base.prefill(case['prompt'][:-1]);lines=[]
        for s in range(64):
            base.set_tok(int(case['prompt'][-1]) if s==0 else int(case['forced'][s-1]));base.mega();lines.append(base.logits().cpu())
        base_error=max(base_error,float((torch.stack(lines)-case['reference']).abs().max()))
    del base;release()
    sequential,_,_=execute(ROOT/'kernels/megakernel_mt/libmt1.so',1,'causal',data,cases,seeds,256)
    first,indices,_=execute(library,m,mode,data,cases,seeds,256)
    second,_,_=execute(library,m,mode,data,cases,seeds,256)
    result=metrics(first,indices,cases,min(.5,1.25*base_error),prior=second,sequential=sequential)
    result.update(baseline_v2_diff=base_error,mode=mode,m=m,wall_s=time.perf_counter()-started,script='gate/mt_gate.py:run tier1; unchanged bench/lm14/check_gate.py criteria')
    regression={'requested':bool(baseline),'pass':None,'coverage':'base and context/adversarial cases','wall_s':0.}
    if baseline:
        regression_stamp=time.perf_counter()
        previous,_,_=execute(Path(baseline),m,mode,data,cases,seeds,256)
        regression['pass']=all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(first,previous))
        result['baseline_bitwise_informational']=regression['pass']
        regression['wall_s']+=time.perf_counter()-regression_stamp
    tiers['1']=result
    if tier >= 2 and result['pass']:
        from core import load_cache
        from mt_build import build
        cache=load_cache('v2');stamp=time.perf_counter();checks={}
        extra=cache['extra'];cap=max(len(c['prompt']) for c in extra)+max(c['steps'] for c in extra)+m
        extra_seeds=seeds_for(extra,data,cap)
        a,ids,sample=execute(library,m,mode,data,extra,extra_seeds,cap,True)
        repeats=[]
        for _ in range(3):
            b,_,repeat_samples=execute(library,m,mode,data,extra,extra_seeds,cap,True)
            sample.extend(repeat_samples)
            repeats.append(metrics(a,ids,extra,cache['bound'],prior=b))
        checks['context_adversarial_repeat']={'pass':all(r['pass'] for r in repeats),'extra_repeats':3,'runs':repeats}
        if baseline:
            regression_stamp=time.perf_counter()
            previous,_,_=execute(Path(baseline),m,mode,data,extra,extra_seeds,cap)
            regression['pass'] &= all(torch.equal(x.view(torch.int32),y.view(torch.int32)) for x,y in zip(a,previous))
            regression['wall_s']+=time.perf_counter()-regression_stamp
        checks['sampling']={'pass':not sample,'errors':sample}
        # The distribution rule covers the original 192 teacher-forced steps.
        wave=cache['cases'];wave_seeds=seeds_for(wave,data,256)
        wave_out,wave_ids,wave_sample=execute(library,m,mode,data,wave,wave_seeds,256,True)
        dist=metrics(wave_out,wave_ids,wave,cache['bound'],distribution=True)
        dist['kl_bound']=cache['kl_bound'];dist['pass'] &= dist['mean_kl'] <= cache['kl_bound'];checks['distribution']=dist
        checks['sampling']['errors']+=wave_sample;checks['sampling']['pass'] &= not wave_sample
        checks['multistep']=multistep(library,m,mode,data,wave,wave_seeds,256)
        perm_results=[]
        for perm in extra_tests.head_permutations(torch.cat([c['forced'] for c in wave])):
            perm_data=(extra_tests.permuted_head(data[0],perm),*data[1:])
            perm_cases=extra_tests.permuted_cases(wave,perm)
            out,pis,samples=execute(library,m,mode,perm_data,perm_cases,wave_seeds,256,True)
            checked=metrics(out,pis,perm_cases,cache['bound']);checked['sample_errors']=samples;checked['pass'] &= not samples
            perm_results.append(checked);del perm_data;release()
        checks['head_perm']={'pass':all(r['pass'] for r in perm_results),'variants':perm_results}
        jitter=build(library,m)
        jitter_results=[]
        for _ in range(2):
            out,_,samples=execute(jitter,m,mode,data,extra,extra_seeds,cap,True)
            same=all(torch.equal(x,y) for x,y in zip(a,out));jitter_results.append({'pass':same and not samples,'bitwise_equal':same,'sample_errors':samples})
        checks['jitter']={'pass':all(r['pass'] for r in jitter_results),'runs':jitter_results,'library':str(jitter)}
        tiers['2']={'pass':all(c['pass'] for c in checks.values()),'checks':checks,'wall_s':time.perf_counter()-stamp,'script':'gate/mt_gate.py:run tier2; mutation/extra_tests.py scenarios and rules'}
        if tier >= 3 and tiers['2']['pass']:
            import layers
            layer_result=layers.check_mt(library,m,mode,data,cache)
            tiers['3']={'pass':layer_result['pass'] and (regression['pass'] if baseline else True),'layers':layer_result,'regression':regression,'wall_s':layer_result['wall_s']+regression['wall_s'],'script':'gate/mt_gate.py tier3: layer calibration and optional bitwise baseline'}
    for number in range(1,tier+1):
        if str(number) not in tiers:
            tiers[str(number)]={'pass':False,'wall_s':0.,'not_run':'prior tier failed'}
    return {'engine':'mt','library':str(library),'m':m,'mode':mode,'tiers':tiers,'pass':all(t['pass'] for t in tiers.values()),'wall_s':time.perf_counter()-started,'script':'gate/mt_gate.py'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib',type=Path);parser.add_argument('--tier',type=int,default=2);parser.add_argument('--baseline',type=Path)
    parser.add_argument('--m',type=int,default=1);parser.add_argument('--mode',choices=('causal','batch'),default='causal')
    parser.add_argument('--repeat',type=int,default=1);parser.add_argument('--output',type=Path)
    args=parser.parse_args();passed=True
    for repeat in range(args.repeat):
        row=run(args.lib,args.tier,args.baseline,args.m,args.mode);row['repeat']=repeat+1
        text=json.dumps(row,allow_nan=False);print(text,flush=True)
        if args.output:
            with args.output.open('a') as file:file.write(text+'\n')
        passed &= row['pass']
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
