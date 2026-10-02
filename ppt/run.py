"""Run only through scripts/gpu.sh; JSONL rows resume by model/method/problem."""
import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.request

from sampler import GSM8K, HFEngine, Record, ppt

ROOT = Path(__file__).resolve().parent


def answer(text):
    matches = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if not matches:
        matches = re.findall(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}', text)
    if not matches:
        return None
    try:
        value=Decimal(matches[-1].replace(',','')).normalize()
        return str(value) if value else '0'
    except InvalidOperation:
        return None


def prepare():
    path = ROOT/'gsm8k.json'
    if path.exists():
        return json.loads(path.read_text())
    url = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
    raw = urllib.request.urlopen(url,timeout=60).read()
    rows = [json.loads(line) for line in raw.splitlines()]
    indices = random.Random(719).sample(range(len(rows)),200)
    data = dict(source=url,sha256=hashlib.sha256(raw).hexdigest(),seed=719,total=len(rows),
                problems=[dict(id=i,question=rows[i]['question'],gold=answer(rows[i]['answer'])) for i in indices])
    path.write_text(json.dumps(data,indent=2)+'\n')
    return data


def save(path,data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data)+'\n'); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',choices=['1.7B','0.6B'],default='1.7B')
    ap.add_argument('--count',type=int,default=200)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--seconds',type=int,default=650)
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--prepare',action='store_true')
    args = ap.parse_args()
    data = prepare()
    if args.prepare:
        print(json.dumps({k:v for k,v in data.items() if k!='problems'})); return
    outpath = ROOT/('pilot.jsonl' if args.pilot else 'results.jsonl')
    rows = [json.loads(l) for l in outpath.read_text().splitlines()] if outpath.exists() else []
    done = {(r['model'],r['problem'],r['method']) for r in rows}
    started = time.monotonic(); deadline = started+args.seconds
    engine = HFEngine('Qwen/Qwen3-'+args.model+'-Base',GSM8K['powers'],719)
    print('model loaded',args.model,'seconds',time.monotonic()-started,flush=True)
    for problem in data['problems'][args.start:args.start+args.count]:
        pid = problem['id']; prompt = engine.prompt(problem['question'])
        cp = ROOT/f'checkpoint-{args.model}-{pid}.json'
        key = (args.model,pid,'ppt')
        if key not in done:
            state = json.loads(cp.read_text()) if cp.exists() else None
            counts = dict(generated=0,prefill=0,decode_steps=0,wall_seconds=0)
            if state:
                counts = state['counts']
                engine.generator.set_state(engine.torch.tensor(state['torch_rng'],dtype=engine.torch.uint8))
            else:
                engine.generator.manual_seed(719+pid)
            base = {k:getattr(engine,k) for k in ('generated','prefill','decode_steps')}
            t0 = time.monotonic()
            milestones = list(state.get('milestones', [])) if state else []
            if state and not milestones:
                milestones.append(dict(stage=state['stage'],round=state['round'],**counts))
            def checkpoint(s):
                s['counts'] = {k:counts[k]+getattr(engine,k)-base[k] for k in base}
                s['counts']['wall_seconds'] = counts['wall_seconds']+time.monotonic()-t0
                s['torch_rng'] = engine.generator.get_state().tolist()
                if s['round']==-1:
                    milestones.append(dict(stage=s['stage'],round=s['round'],**s['counts'],
                        lengths=[len(r['tokens']) for r in s['records']],
                        terminals=[r['terminal'] for r in s['records']]))
                s['milestones']=milestones
                save(cp,s)
            try:
                record,stats = ppt(engine,prompt,719+pid,checkpoint=checkpoint,state=state,deadline=deadline)
            except TimeoutError:
                print('checkpointed PPT',pid,flush=True); return
            final = json.loads(cp.read_text())
            text = engine.text(record)
            row = dict(model=args.model,method='ppt',task='gsm8k',problem=pid,gold=problem['gold'],
                       answer=answer(text),correct=answer(text)==problem['gold'],text=text,
                       returned_tokens=len(record.tokens),terminal=record.terminal,config=GSM8K,
                       **final['counts'],stats=stats,seed=719+pid,milestones=final.get('milestones',[]))
            with outpath.open('a') as f:
                f.write(json.dumps(row)+'\n'); f.flush()
            rows.append(row); done.add(key); cp.unlink()
            print(json.dumps({k:v for k,v in row.items() if k not in ('text','stats')}),flush=True)
        if args.pilot:
            if time.monotonic()>=deadline:
                return
            # Natural-cost pilots quantify baseline throughput before token matching.
            for name,alpha in [('standard',1.0),('low_temperature',4.0),('greedy',0.0)]:
                if (args.model,pid,name) in done:
                    continue
                before = engine.generated; pf = engine.prefill; t0 = time.monotonic()
                record = engine.generate(prompt,[(Record(),alpha,3072)])[0]
                text = engine.text(record)
                row=dict(model=args.model,method=name,task='gsm8k',problem=pid,gold=problem['gold'],
                         answer=answer(text),correct=answer(text)==problem['gold'],text=text,
                         generated=engine.generated-before,prefill=engine.prefill-pf,
                         wall_seconds=time.monotonic()-t0,returned_tokens=len(record.tokens),terminal=record.terminal)
                with outpath.open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
                if time.monotonic()>=deadline: return
        else:
            from controls import independent, single_chain
            budget = next(r['generated'] for r in rows if r['model']==args.model and r['problem']==pid and r['method']=='ppt')
            for family,alpha in [('standard',1.0),('low_temperature',4.0),('greedy',0.0),('single_chain',4.0)]:
                names = ['standard','best_of_n','majority_vote'] if family=='standard' else [family]
                if all((args.model,pid,name) in done for name in names):
                    continue
                control_cp=ROOT/f'control-{args.model}-{pid}-{family}.json'
                saved=json.loads(control_cp.read_text()) if control_cp.exists() else None
                counts=saved['counts'] if saved else dict(generated=0,prefill=0,decode_steps=0,wall_seconds=0)
                if saved:
                    engine.generator.set_state(engine.torch.tensor(saved['torch_rng'],dtype=engine.torch.uint8))
                else:
                    engine.generator.manual_seed(719+pid+10000*(['standard','low_temperature','greedy','single_chain'].index(family)+1))
                base={k:getattr(engine,k) for k in ('generated','prefill','decode_steps')}
                t0=time.monotonic()
                def control_checkpoint(s):
                    accumulated={k:counts[k]+getattr(engine,k)-base[k] for k in base}
                    accumulated['wall_seconds']=counts['wall_seconds']+time.monotonic()-t0
                    save(control_cp,dict(state=s,counts=accumulated,torch_rng=engine.generator.get_state().tolist()))
                try:
                    if family=='single_chain':
                        result=single_chain(engine,prompt,budget,answer,control_checkpoint,
                            state=saved['state'] if saved else None,deadline=deadline,seed=719+pid)
                    else:
                        result=independent(engine,prompt,budget,alpha,answer,control_checkpoint,
                            state=saved['state'] if saved else None,deadline=deadline)
                except TimeoutError:
                    print('checkpointed control',pid,family,flush=True); return
                measured=json.loads(control_cp.read_text())['counts']
                assert measured['generated']==budget
                for name in names:
                    if (args.model,pid,name) in done: continue
                    selected=result.get('best' if name=='best_of_n' else 'last')
                    prediction=result['answer'] if name=='single_chain' else result['vote'] if name=='majority_vote' else selected['answer'] if selected else None
                    text=result['text'] if name=='single_chain' else selected['text'] if selected else ''
                    row=dict(model=args.model,method=name,task='gsm8k',problem=pid,gold=problem['gold'],
                        answer=prediction,correct=prediction==problem['gold'],text=text,**measured,
                        budget=budget,truncated_tokens=result['truncated'],temperature=0 if alpha==0 else 1/alpha,
                        independent_samples=len(result.get('candidates',[])),seed=719+pid,
                        compute_group=family)
                    with outpath.open('a') as f: f.write(json.dumps(row)+'\n')
                    rows.append(row); done.add((args.model,pid,name))
                    print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
                control_cp.unlink()
                if time.monotonic()>=deadline: return
        if time.monotonic()>=deadline: return


if __name__ == '__main__':
    main()
