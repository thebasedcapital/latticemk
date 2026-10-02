"""Scoped gate: full-history eviction reference, v2 bitwise, repeated compaction."""
import json
from pathlib import Path

import torch
import kvc

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    ref = torch.load(HERE/'reference.pt',map_location='cpu')
    ids, events = ref['ids'], ref['events']
    first = min(events)
    packed, shared = kvc.resources()
    base = kvc.lm03b.Engine2(len(ids),packed,*shared)
    control = []
    base.pos_set(0)
    for token in ids:
        base.set_tok(token)
        torch.cuda.synchronize()
        base.mega(1)
        control.append(base.logits().cpu())
    control = torch.stack(control)
    base_error = float((control[first:]-ref['control'][first:]).abs().max())
    del base
    torch.cuda.empty_cache()
    engine = kvc.Engine(packed,shared,retain_cap=2048,
                        im_start=ref['im_start'],im_end=ref['im_end'])
    bitwise = True
    for i,token in enumerate(ids):
        actual = engine.step(token).cpu()
        bitwise &= bool(torch.equal(actual.view(torch.int32),control[i].view(torch.int32)))
    bound = min(0.5,1.25*base_error)
    saved, cases, copies_equal, invalid_rejected = None, [], True, False
    for repeat in range(2):
        engine.reset()
        logits = []
        event_results = []
        for i,token in enumerate(ids):
            if i in events:
                # Validate every layer and both caches against an indexed copy.
                rows = [r for a,b in events[i] for r in range(a,b)]
                before = {name:engine.bufs[name].reshape(28,kvc.MAXPOS,1024)[:,rows].clone()
                          for name in ('kc','vc')}
                rope_before = int(engine.bufs['rope_pos'].cpu()[0])
                try:
                    engine.compact([(1,engine.kv_len)])
                except ValueError:
                    invalid_rejected = True
                else:
                    raise AssertionError('partial-turn compaction accepted')
                ms = engine.compact(events[i])
                assert engine.rope_pos == i == rope_before
                assert int(engine.bufs['rope_pos'].cpu()[0]) == i
                assert int(engine.bufs['kv_len'].cpu()[0]) == len(rows)
                for name in ('kc','vc'):
                    after = engine.bufs[name].reshape(28,kvc.MAXPOS,1024)[:,:len(rows)]
                    copies_equal &= bool(torch.equal(after.view(torch.int16),before[name].view(torch.int16)))
                event_results.append(dict(logical=i,physical=len(rows),gpu_ms=ms))
                del before
            if i >= first:
                logits.append(engine.step(token).cpu())
            else:
                engine.step(token)
        actual = torch.stack(logits)
        expected = ref['compact'][first:]
        errors = (actual-expected).abs().amax(dim=1)
        flips = []
        for j in range(actual.shape[0]):
            top = expected[j].topk(2)
            margin = float(top.values[0]-top.values[1])
            if int(actual[j].argmax()) != int(top.indices[0]):
                flips.append(dict(logical=first+j,margin=margin,error=float(errors[j]),
                                  kind='near' if margin<2*float(errors[j]) else 'hard'))
        same = saved is None or torch.equal(actual.view(torch.int32),saved.view(torch.int32))
        cases.append(dict(repeat=repeat,max_diff=float(errors.max()),bound=bound,
                          steps=len(logits),flips=flips,bitwise_repeat=bool(same),
                          finite=bool(torch.isfinite(actual).all()),events=event_results))
        saved = actual
    passed = (bitwise and copies_equal and invalid_rejected and
              all(c['finite'] and c['max_diff']<=bound and c['bitwise_repeat']
                  and not any(f['kind']=='hard' for f in c['flips']) for c in cases))
    result = dict(tag='measured',script='bench/lm24/check_correctness.py',passed=passed,
                  no_events_bitwise=bitwise,no_events_steps=len(ids),
                  control_max_diff=base_error,bound=bound,
                  cache_rows_bitwise=copies_equal,partial_turn_rejected=invalid_rejected,
                  logical_positions=list(range(len(ids))),cases=cases)
    (HERE/'correctness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
