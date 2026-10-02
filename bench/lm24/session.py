"""Actual teacher-forced 16k-token session, 8 whole-turn compactions.

Baseline is an owned copy of v2 with only MAXPOS enlarged to 16896. Compare
it against stock v2 first at positions that fit, then alternate whole-turn
segments in one process. Wall time includes Python token dispatch and copies.
"""
import json
import time
from pathlib import Path

import torch
import kvc
import baseline16k
from reference import transcript

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    packed,shared = kvc.resources()
    # Verify extending the stride does not alter uncompacted logits.
    stock = kvc.lm03b.Engine2(2048,packed,*shared)
    extended = baseline16k.Engine2(16384,packed,*shared)
    generator = torch.Generator(device='cuda').manual_seed(2401)
    for name in ('kc','vc'):
        source = stock.bufs[name].reshape(28,kvc.MAXPOS,1024)
        source.normal_(0,0.1,generator=generator)
        dest = extended.bufs[name].reshape(28,baseline16k.MAXPOS,1024)
        dest[:,:kvc.MAXPOS].copy_(source)
    bits = True
    for ctx in (128,2048,8192):
        for eng,lib in ((stock,kvc.lm03b._lib2),(extended,baseline16k._lib2)):
            eng.set_tok(9707)
            torch.cuda.synchronize()
            out = (kvc.ctypes.c_float*1)()
            assert lib.mk2_time_mega(8,1,ctx,out)==0
        bits &= torch.equal(stock.logits().cpu().view(torch.int32),
                            extended.logits().cpu().view(torch.int32))
    assert bits, 'extended baseline changes v2 numeric bits'
    del stock
    torch.cuda.empty_cache()
    candidate = kvc.Engine(packed,shared,retain_cap=2048)
    ids,spans,_,_ = transcript(64,256)
    cuts = [0]+[spans[t-1][1] for t in (12,18,24,30,36,42,48,54)]+[len(ids)]
    extended.pos_set(0)
    candidate.reset()
    totals = {'v2-extended':0.,'kvc':0.}
    segments, events = [], []
    for segment,(begin,end) in enumerate(zip(cuts,cuts[1:])):
        if segment:
            began = time.perf_counter()
            gpu_ms = candidate.compact(candidate.keep_first_last(7))
            wall = time.perf_counter()-began
            totals['kvc'] += wall
            events.append(dict(logical=begin,physical=candidate.kv_len,
                               gpu_ms=gpu_ms,wall_s=wall))
        order = [('v2-extended',extended),('kvc',candidate)]
        if segment%2:
            order.reverse()
        record = dict(begin=begin,end=end)
        for name,engine in order:
            began = time.perf_counter()
            for token in ids[begin:end]:
                if name == 'kvc':
                    engine.step(token)
                else:
                    engine.set_tok(token)
                    torch.cuda.synchronize()
                    engine.mega(1)
            elapsed = time.perf_counter()-began
            totals[name] += elapsed
            record[name+'_s'] = elapsed
        segments.append(record)
        print(json.dumps(record),flush=True)
    assert candidate.rope_pos == len(ids)
    result = dict(tag='measured',script='bench/lm24/session.py',tokens=len(ids),
                  compactions=len(events),totals_wall_s=totals,
                  speedup=totals['v2-extended']/totals['kvc'],events=events,
                  segments=segments,extended_baseline_bitwise=bool(bits),
                  final_logical=candidate.rope_pos,final_physical=candidate.kv_len,
                  baseline_physical=int(extended.bufs['pos'].cpu()[0]),
                  finite=bool(torch.isfinite(candidate.bufs['logits']).all()
                              and torch.isfinite(extended.bufs['logits']).all()))
    (HERE/'session.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert result['finite'] and result['compactions']==8


if __name__ == '__main__':
    main()
