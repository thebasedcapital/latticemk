"""Cost-only calibration of fixed-horizon single-chain MH, not an accuracy run."""
import argparse
import json
import time
from run import prepare, answer, ROOT, save
from sampler import HFEngine, GSM8K
from controls import single_chain, independent

ap=argparse.ArgumentParser(); ap.add_argument('--count',type=int,default=2)
ap.add_argument('--budget',type=int,default=8192); ap.add_argument('--seconds',type=int,default=510)
ap.add_argument('--control-smoke',action='store_true')
args=ap.parse_args(); path=ROOT/'pilot.jsonl'; data=prepare()
rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
done={r['problem'] for r in rows if r['model']=='1.7B' and r['method']=='single_chain'}
engine=HFEngine('Qwen/Qwen3-1.7B-Base',GSM8K['powers'],719)
if args.control_smoke:
    proof=[]
    for problem in data['problems'][:2]:
        prompt=engine.prompt(problem['question'])
        for name,alpha in [('standard',1.),('low_temperature',4.),('greedy',0.)]:
            before=engine.generated; prefill=engine.prefill; started=time.monotonic()
            result=independent(engine,prompt,256,alpha,answer)
            assert result['spent']==engine.generated-before==256
            proof.append(dict(model='1.7B',method=name,problem=problem['id'],budget=256,
                generated=engine.generated-before,prefill=engine.prefill-prefill,
                seconds=time.monotonic()-started,candidates=len(result['candidates']),
                charged_incomplete_tokens=result['truncated']))
    save(ROOT/'control-proof.json',proof)
    print(json.dumps(dict(control_cost_proof=proof)),flush=True)
deadline=time.monotonic()+args.seconds
for problem in data['problems'][:args.count]:
    if problem['id'] in done: continue
    t0=time.monotonic(); before=engine.generated; pf=engine.prefill; ds=engine.decode_steps
    cp=ROOT/f'single-cost-{problem["id"]}.json'
    saved=json.loads(cp.read_text()) if cp.exists() else None
    counts=saved['counts'] if saved else dict(generated=0,prefill=0,decode_steps=0,wall_seconds=0)
    if saved:
        engine.generator.set_state(engine.torch.tensor(saved['torch_rng'],dtype=engine.torch.uint8))
    def checkpoint(s):
        payload=dict(state=s,counts=dict(generated=counts['generated']+engine.generated-before,
            prefill=counts['prefill']+engine.prefill-pf,decode_steps=counts['decode_steps']+engine.decode_steps-ds,
            wall_seconds=counts['wall_seconds']+time.monotonic()-t0),torch_rng=engine.generator.get_state().tolist())
        save(cp,payload)
    try:
        result=single_chain(engine,engine.prompt(problem['question']),args.budget,answer,checkpoint,
            state=saved['state'] if saved else None,deadline=deadline,seed=719+problem['id'])
    except TimeoutError:
        print('single-chain cost checkpoint',problem['id'],flush=True); break
    measured=json.loads(cp.read_text())['counts']
    row=dict(model='1.7B',method='single_chain',task='gsm8k',problem=problem['id'],gold=problem['gold'],
        answer=result['answer'],correct=result['answer']==problem['gold'],text=result['text'],**measured,
        returned_tokens=len(result['record']['tokens']) if result['record'] else 0,
        terminal=result['record']['terminal'] if result['record'] else False,
        attempts=result['attempts'],accepts=result['accepts'],truncated_tokens=result['truncated'],
        cost_pilot_budget=args.budget,phase='throughput-calibration-only')
    with path.open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
    cp.unlink()
    if time.monotonic()>=deadline: break
