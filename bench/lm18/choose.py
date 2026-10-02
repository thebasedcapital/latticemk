"""Alternate original, nonvolatile and interleaved full-pass/probe candidates."""
import json
from pathlib import Path
import numpy as np
from engine import Engine,shared
HERE=Path(__file__).resolve().parent

def main():
    data=shared()
    engines={(v,m):Engine(m,'batch',*data,variant=v,cap=136) for v in ('original','nonvolatile','preload') for m in (1,4)}
    for e in engines.values():e.bufs['tok'].fill_(100)
    keys=list(engines);samples={(kind,k):[] for kind in ('full','gemm') for k in keys}
    for e in engines.values():e.time(128,3);e.time_gemv(3)
    for r in range(27):
        order=keys[r%len(keys):]+keys[:r%len(keys)]
        if r%2:order=order[::-1]
        for k in order:
            samples['full',k]+=engines[k].time(128)
            samples['gemm',k]+=engines[k].time_gemv()
    rows=[]
    for (kind,(v,m)),s in samples.items():
        row=dict(experiment='candidate_selection',kind=kind,variant=v,m=m,context=128,runs=27,median_ms=float(np.median(s)),p10_ms=float(np.percentile(s,10)),p90_ms=float(np.percentile(s,90)),samples_ms=s,diagnostic_spills_bytes=8 if v=='interleaved' and m==4 and kind=='full' else 0)
        rows.append(row)
        with (HERE/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
    (HERE/'choose.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
