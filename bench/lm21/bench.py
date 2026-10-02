"""Same-process interleaved selected mt2/pro comparison at context 128."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from engine import Engine, shared, ROOT
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('baseline_engine', ROOT/'bench/lm18/engine.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

@torch.no_grad()
def main():
    data = shared()
    keys = [(variant, mode, m) for mode in ('causal', 'batch') for m in (1, 2, 4) for variant in ('mt2', 'pro')]
    engines = {k: (baseline.Engine if k[0]=='mt2' else Engine)(k[2],k[1],*data,cap=136) for k in keys}
    for e in engines.values(): e.bufs['tok'].fill_(100)
    for _ in range(3):
        for k in keys: engines[k].time(128,1)
    samples = {k:[] for k in keys}
    for i in range(27):
        for k in (keys if i%2 else keys[::-1]): samples[k].extend(engines[k].time(128,1))
    rows=[]
    for k in keys:
        values=samples[k]
        row=dict(variant=k[0],mode=k[1],m=k[2],context=128,runs=27,median_ms=float(np.median(values)),p10_ms=float(np.percentile(values,10)),p90_ms=float(np.percentile(values,90)),samples_ms=values)
        rows.append(row);print(json.dumps(row),flush=True)
    (HERE/'timing.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
