"""Derive latency ratios and acceptance thresholds from retained CUDA-event rows."""
import hashlib
import json
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    builds = []
    for m in range(1,6):
        text = (ROOT/f'kernels/megakernel_mt3/build-selected-m{m}.log').read_text()
        stats = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',text)
        if not stats or any(int(x) for row in stats for x in row):
            raise RuntimeError(f'M{m} has nonzero stack/spills')
        resources = re.findall(r'Used (\d+) registers, used \d+ barriers, (\d+) bytes smem',text)
        mega = resources[1]
        builds.append(dict(tag='measured',m=m,threads=1024 if m==1 else 512,
                           registers=int(mega[0]),static_smem_bytes=int(mega[1]),
                           stack_bytes=0,spill_store_bytes=0,spill_load_bytes=0,
                           source=f'kernels/megakernel_mt3/build-selected-m{m}.log'))
    matrix = []
    ratios = []
    breakeven = []
    proofs = []
    for mode,ctx in [('batch',128),('batch',2048),('causal',128),('causal',2048),('causal',8192)]:
        obj = json.loads((HERE/f'{mode}-{ctx}.json').read_text())
        proofs.extend(dict(mode=mode,context=ctx,**r) for r in obj['proof'])
        rows = {(r['variant'],r['m']):r for r in obj['timings']}
        t1 = rows['mt3',1]['median_ms']
        for m in range(1,6):
            old,new = rows['mt2',m],rows['mt3',m]
            matrix.append(dict(tag='derived',mode=mode,context=ctx,m=m,mt2_ms=old['median_ms'],
                               mt3_ms=new['median_ms'],delta_ms=new['median_ms']-old['median_ms'],
                               new_over_old=new['median_ms']/old['median_ms'],
                               new_over_m1=new['median_ms']/t1,p10_ms=new['p10_ms'],p90_ms=new['p90_ms'],
                               sm_clock_mhz=new['sm_clock_mhz']))
        ratios.append(dict(tag='derived',mode=mode,context=ctx,
                           mt2_m4_over_m1=rows['mt2',4]['median_ms']/rows['mt2',1]['median_ms'],
                           mt3_m4_over_m1=rows['mt3',4]['median_ms']/t1,
                           target_met=rows['mt3',4]['median_ms']/t1<=1.5,
                           mt3_m1_over_v2=t1/rows['v2',1]['median_ms'] if ('v2',1) in rows else None))
        if mode == 'causal':
            for k in (2,3,4):
                ratio = rows['mt3',k]['median_ms']/t1
                breakeven.append(dict(tag='derived',context=ctx,k=k,verify_over_plain=ratio,
                                      accepted_only_min_mean_strictly_greater_than=ratio,
                                      mandatory_anchor_plus_bonus_min_accepted_drafts_strictly_greater_than=ratio-1,
                                      anchor_draft_capacity=k-1,
                                      anchor_min_fraction_strictly_greater_than=(ratio-1)/(k-1),
                                      max_speedup_all_accepted=k/ratio,
                                      assumption='CPU n-gram/prompt-lookup drafting cost approximately 0; mandatory anchor consumes one input column, and rejection/all-accepted bonus commits one target token. No host rollout/KV rollback overhead measured.'))
    calibration = []
    implementation = hashlib.sha256((ROOT/'gate/layers.py').read_bytes()).hexdigest()
    for path in sorted((ROOT/'gate/cache').glob('layers-calibration-mt-m4-causal-*.json')):
        value = json.loads(path.read_text())
        if (value['signature']['library'].endswith('/megakernel_mt2/libmt4.so')
                and value['implementation_sha256'] == implementation):
            calibration.append(value)
    if {v['probe'] for v in calibration} != {'full','local'}:
        raise RuntimeError('missing trusted mt2 contract calibration evidence')
    (HERE/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    gates = [json.loads(line) for path in ('tiers.jsonl','contract-tiers.jsonl')
             for line in (HERE/path).read_text().splitlines()]
    latest = {}
    for row in gates:
        latest[Path(row['library']).parent.name,row.get('m',1),row.get('mode','batch')] = row
    gate_summary = [dict(library=v['library'],m=v.get('m',1),mode=v.get('mode','batch'),
                         passed=v['pass'],tiers={k:r['pass'] for k,r in v['tiers'].items()})
                    for v in latest.values()]
    result = dict(builds=builds,matrix=matrix,ratios=ratios,breakeven=breakeven,proofs=proofs,gates=gate_summary)
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(ratios=ratios,breakeven=breakeven),indent=2))


if __name__ == '__main__':
    main()
