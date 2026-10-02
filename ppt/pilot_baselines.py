"""Measure natural response lengths and baseline decode costs before authorizing a run."""
import argparse
import json
from pathlib import Path
import time
from run import prepare, answer, ROOT
from sampler import HFEngine, GSM8K, Record

ap=argparse.ArgumentParser(); ap.add_argument('--model',default='1.7B'); ap.add_argument('--count',type=int,default=2)
args=ap.parse_args(); data=prepare(); path=ROOT/'pilot.jsonl'
rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
done={(r['model'],r['problem'],r['method']) for r in rows}
engine=HFEngine('Qwen/Qwen3-'+args.model+'-Base',GSM8K['powers'],719)
for problem in data['problems'][:args.count]:
    prompt=engine.prompt(problem['question'])
    for name,alpha in [('standard',1.),('low_temperature',4.),('greedy',0.)]:
        if (args.model,problem['id'],name) in done: continue
        t0=time.monotonic(); before=engine.generated; pf=engine.prefill
        record=engine.generate(prompt,[(Record(),alpha,3072)])[0]
        text=engine.text(record)
        row=dict(model=args.model,method=name,task='gsm8k',problem=problem['id'],gold=problem['gold'],
            answer=answer(text),correct=answer(text)==problem['gold'],text=text,generated=engine.generated-before,
            prefill=engine.prefill-pf,wall_seconds=time.monotonic()-t0,returned_tokens=len(record.tokens),terminal=record.terminal)
        with path.open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
