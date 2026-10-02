"""Independent-trace controls spend exactly the PPT generated-token budget.
Incomplete final traces consume budget but do not enter selection or voting.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import math
import random
import time

from sampler import Record, accept, local_ratio


def independent(engine, prompt, budget, alpha, extract, checkpoint=None, state=None, deadline=None):
    state=state or dict(spent=0,candidates=[],truncated=0)
    while state['spent']<budget:
        remaining=budget-state['spent']
        # Allocate disjoint caps so a batched request can never exceed the budget.
        caps=[]
        while remaining and len(caps)<6:
            cap=min(3072,remaining); caps.append(cap); remaining-=cap
        before=engine.generated
        records=engine.generate(prompt,[(Record(),alpha,cap) for cap in caps])
        state['spent']+=engine.generated-before
        for record,cap in zip(records,caps):
            text=engine.text(record)
            if record.terminal or cap==3072:
                state['candidates'].append(dict(answer=extract(text),text=text,
                    logp=sum(record.logp),tokens=len(record.tokens)))
            else:
                state['truncated']+=len(record.tokens)
        if checkpoint:
            checkpoint(state)
        if deadline and time.monotonic()>=deadline:
            raise TimeoutError('Independent-control checkpoint deadline')
    candidates=state['candidates']
    if not candidates:
        return dict(last=None,best=None,vote=None,**state)
    best=max(candidates,key=lambda r:r['logp'])
    # Invalid answers never combine into a majority.
    valid=[r for r in candidates if r['answer'] is not None]
    counts=Counter(r['answer'] for r in valid)
    scores=defaultdict(float)
    for r in valid: scores[r['answer']]+=r['logp']
    winner=max(counts,key=lambda a:(counts[a],scores[a])) if counts else None
    return dict(last=candidates[-1],best=best,vote=winner,**state)


def single_chain(engine,prompt,budget,extract,checkpoint=None,state=None,deadline=None,seed=719):
    """Fixed-horizon power chain; budget-final incomplete proposals are rejected.

    One initial draw at alpha=4, followed by uniform full-horizon suffix MH.
    This control holds the target fixed, rather than the progressive warm start.
    """
    rng=random.Random(seed)
    state=state or dict(spent=0,record=None,attempts=0,accepts=0,self_moves=0,truncated=0)
    if state.get('rng'):
        def tup(x): return tuple(tup(v) for v in x) if isinstance(x,list) else x
        rng.setstate(tup(state['rng']))
    # Scores remain six-power arrays; the target power 4 is at index 5.
    record=Record(**state['record']) if state['record'] else None
    while state['spent']<budget:
        remaining=budget-state['spent']
        if record is None:
            prefix=Record(); restart=0
        else:
            restart=rng.randrange(3072)
            state['attempts']+=1
            if record.terminal and restart>=len(record.tokens):
                state['self_moves']+=1; state['accepts']+=1
                continue
            prefix=record.prefix(restart)
        cap=min(3072,len(prefix.tokens)+remaining)
        before=engine.generated
        proposed=engine.generate(prompt,[(prefix,4.0,cap)])[0]
        state['spent']+=engine.generated-before
        complete=proposed.terminal or cap==3072
        if not complete:
            state['truncated']+=len(proposed.tokens)-len(prefix.tokens)
        elif record is None:
            record=proposed
        elif accept(local_ratio(record,proposed,restart,5),rng):
            record=proposed; state['accepts']+=1
        state['record']=asdict(record) if record else None
        state['rng']=rng.getstate()
        if checkpoint:
            checkpoint(state)
        if deadline and time.monotonic()>=deadline:
            raise TimeoutError('Single-chain checkpoint deadline')
    text=engine.text(record) if record else ''
    return dict(answer=extract(text),text=text,**state)
