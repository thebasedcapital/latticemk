"""Interleaved >=25-sample kill checks, compacted decode, and copy events.

Decode timing uses deterministic varied synthetic cache rows, as in v2's
microbench. It measures scan length, not transcript quality or prefill.
"""
import argparse
import ctypes
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
import kvc
from reference import transcript

HERE = Path(__file__).resolve().parent


def timing(engine, logical, physical, steps=8):
    out = (ctypes.c_float * 1)()
    if isinstance(engine, kvc.Engine):
        rc = kvc.LIB.kvc_time_mega(steps,1,logical,physical,out)
    else:
        rc = kvc.lm03b._lib2.mk2_time_mega(steps,1,logical,out)
    assert rc == 0, rc
    return float(out[0])


def summary(name, values, logical, physical):
    ordered = sorted(values)
    ms = statistics.median(values)
    return dict(name=name,logical=logical,physical=physical,runs=len(values),
                median_ms=ms,p10_ms=ordered[len(values)//10],
                p90_ms=ordered[len(values)*9//10],tokens_per_s=1000/ms,
                samples_ms=values)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=int,default=25)
    args = parser.parse_args()
    if args.runs < 25:
        raise ValueError('at least 25 samples required')
    packed, shared = kvc.resources()
    baseline = kvc.lm03b.Engine2(8192,packed,*shared)
    candidate = kvc.Engine(packed,shared,retain_cap=4096)
    generator = torch.Generator(device='cuda').manual_seed(24)
    for name in ('kc','vc'):
        baseline.bufs[name].normal_(0,0.1,generator=generator)
        candidate.bufs[name].copy_(baseline.bufs[name])
    torch.cuda.synchronize()
    rows, bits, ratios = [], [], []
    for ctx in (128,2048):
        samples = [[],[]]
        for i in range(3+args.runs):
            engines = [baseline,candidate] if i%2 == 0 else [candidate,baseline]
            for eng in engines:
                eng.bufs['tok'].fill_(9707)
                torch.cuda.synchronize()
                ms = timing(eng,ctx,ctx)
                if i >= 3:
                    samples[int(eng is candidate)].append(ms)
            a = baseline.bufs['logits'].cpu().view(torch.int32)
            b = candidate.bufs['logits'].cpu().view(torch.int32)
            bits.append(bool(torch.equal(a,b)))
        r0 = summary('v2',samples[0],ctx,ctx)
        r1 = summary('kvc-no-events',samples[1],ctx,ctx)
        ratio = r1['tokens_per_s']/r0['tokens_per_s']
        ratios.append(ratio)
        rows.extend([r0,r1])
        print(json.dumps({'ctx':ctx,'no_event_speed_ratio':ratio,
                          'bits':all(bits)}),flush=True)
    # A complete deterministic chat spans logical 8192. Timing rows are copied
    # from v2, then the real API gathers whole turns: first + last seven.
    ids, spans, _, _ = transcript(32,256)
    candidate.tokens = ids
    candidate.rope_pos = candidate.kv_len = len(ids)
    assert kvc.LIB.kvc_state_set(8192,8192) == 0
    event_ms = candidate.compact(candidate.keep_first_last(7))
    assert candidate.kv_len == 2048 and candidate.rope_pos == 8192
    configs = [('v2-long',baseline,8192,8192),
               ('kvc-compacted',candidate,8192,2048),
               ('v2-short',baseline,2048,2048)]
    samples = {name:[] for name,_,_,_ in configs}
    for i in range(3+args.runs):
        order = configs[i%3:]+configs[:i%3]
        for name,eng,logical,physical in order:
            eng.bufs['tok'].fill_(9707)
            torch.cuda.synchronize()
            ms = timing(eng,logical,physical)
            if i >= 3:
                samples[name].append(ms)
    trio = [summary(name,samples[name],logical,physical)
            for name,_,logical,physical in configs]
    compact_ratio = trio[1]['tokens_per_s']/trio[2]['tokens_per_s']
    rows.extend(trio)
    copy_rows = []
    for physical in (2048,4096,8192):
        retained = 1024 if physical == 2048 else 2048
        times, walls = [], []
        source_ids, _, _, _ = transcript(physical//256,256)
        for i in range(3+args.runs):
            candidate.tokens = source_ids
            candidate.rope_pos = candidate.kv_len = physical
            assert kvc.LIB.kvc_state_set(physical,physical) == 0
            began = time.perf_counter()
            ms = candidate.compact(candidate.keep_first_last(retained//256-1))
            wall = (time.perf_counter()-began)*1000
            if i >= 3:
                times.append(ms)
                walls.append(wall)
        copy_rows.append(dict(physical=physical,retained=retained,
                              gpu_ms=statistics.median(times),
                              wall_ms=statistics.median(walls),samples_ms=times))
    result = dict(tag='measured',script='bench/lm24/bench.py',decode=rows,
                  copies=copy_rows,first_event_ms=event_ms,
                  bitwise_no_compaction=all(bits),no_event_ratios=ratios,
                  compact_ratio=compact_ratio,
                  passed=all(bits) and min(ratios)>=0.97 and compact_ratio>=0.95,
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,clocks.sm',
                                               '--format=csv,noheader']).decode().strip())
    (HERE/'timing.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
