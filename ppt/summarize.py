"""Wilson intervals and exact paired McNemar tests, no normal approximation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def wilson(correct,total):
    if not total:
        return None
    z=1.959963984540054; p=correct/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-radius,center+radius]


def paired(ppt,baseline):
    common=sorted(ppt.keys() & baseline.keys())
    wins=sum(ppt[k]['correct'] and not baseline[k]['correct'] for k in common)
    losses=sum(baseline[k]['correct'] and not ppt[k]['correct'] for k in common)
    discordant=wins+losses
    p=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant) if discordant else 1.0
    return dict(n=len(common),wins=wins,losses=losses,p=p,
                delta=(wins-losses)/len(common) if common else None)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='ppt/results.jsonl'); args=ap.parse_args()
    path=Path(args.input); rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    groups=defaultdict(dict)
    for r in rows:
        key=(r['model'],r['task'],r['method'])
        if r['problem'] in groups[key]: raise ValueError('duplicate result key')
        groups[key][r['problem']]=r
    summary=[]; gates=[]
    for (model,task,method),items in sorted(groups.items()):
        values=list(items.values()); n=len(values); correct=sum(r['correct'] for r in values)
        summary.append(dict(model=model,task=task,method=method,n=n,correct=correct,
                            accuracy=correct/n,ci95=wilson(correct,n),
                            tokens_per_problem=statistics.mean(r['generated'] for r in values),
                            prefill_tokens_per_problem=statistics.mean(r['prefill'] for r in values),
                            wall_seconds=sum(r['wall_seconds'] for r in values),
                            seconds_per_problem=statistics.mean(r['wall_seconds'] for r in values),
                            decode_tokens_per_second=sum(r['generated'] for r in values)/sum(r['wall_seconds'] for r in values)))
    for model,task,method in groups:
        if method!='ppt': continue
        alternatives=[r for r in summary if r['model']==model and r['task']==task and r['method']!='ppt']
        if not alternatives: continue
        best=max(alternatives,key=lambda r:r['accuracy'])
        comparison=paired(groups[(model,task,'ppt')],groups[(model,task,best['method'])])
        enough=comparison['n']>=100
        gates.append(dict(model=model,task=task,best_baseline=best['method'],**comparison,
                          decision=('build' if comparison['delta']>=.02 and comparison['p']<.05 else 'no-build') if enough else 'insufficient-sample'))
    result=dict(rows=len(rows),summary=summary,gates=gates)
    dest=path.with_suffix('.summary.json'); dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
