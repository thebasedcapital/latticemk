"""Decode a checkpoint on CPU without changing or grading unfinished records."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from run import ROOT

ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='checkpoint-1.7B-566.json')
args=ap.parse_args(); cp=json.loads((ROOT/args.checkpoint).read_text())
tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base',local_files_only=True)
records=[]
for k,r in enumerate(cp['records']):
    text=tokenizer.decode(r['tokens'],skip_special_tokens=False)
    records.append(dict(rung=k,power=[2,2.3,2.6,3,3.5,4][k],tokens=len(r['tokens']),
                        terminal=r['terminal'],beginning=text[:600],ending=text[-1200:]))
result=dict(stage=cp['stage'],round=cp['round'],counts=cp['counts'],records=records)
(ROOT/'checkpoint-inspection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
