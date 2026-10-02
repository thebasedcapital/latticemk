"""Greedy target verification with a CPU longest-suffix prompt lookup drafter."""
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench/lm23'))
from engine import Engine, shared, lm03b  # noqa: E402


class Lookup:
    """Index previous suffixes, longest first; prefer their latest continuation."""
    def __init__(self, tokens, max_ngram=24):
        self.tokens = []
        self.index = {}
        self.max_ngram = max_ngram
        self.extend(tokens)

    def extend(self, tokens):
        for token in tokens:
            # Index suffixes ending before the new token, which is their continuation.
            end = len(self.tokens)
            for n in range(1, min(end, self.max_ngram) + 1):
                self.index[tuple(self.tokens[end-n:end])] = end
            self.tokens.append(int(token))

    def draft(self, count):
        for n in range(min(len(self.tokens), self.max_ngram), 0, -1):
            end = self.index.get(tuple(self.tokens[-n:]))
            if end is not None:
                return self.tokens[end:end+count], n
        return [], 0


class Target:
    def __init__(self, cap=2304):
        data = shared()
        self.v2 = lm03b.Engine2(cap, *data)
        self.cap = cap
        cache = {name: torch.zeros(28*cap*1024, device='cuda', dtype=torch.half)
                 for name in ('kc', 'vc')}
        self.mt = {m: Engine(m, 'causal', *data, cap=cap, cache_buffers=cache)
                   for m in range(1, 6)}
        self.cache = cache
        self.host = {m: torch.empty(m, dtype=torch.int32, pin_memory=True)
                     for m in range(1, 6)}

    def pass_mt(self, tokens, position):
        m = len(tokens)
        if not 0 <= position <= self.cap-m:
            raise ValueError('verify exceeds KV capacity')
        e = self.mt[m]
        host = self.host[m]
        for i, token in enumerate(tokens):
            host[i] = token
        e.bufs['tok'].copy_(host, non_blocking=True)
        # Every pass explicitly restores the logical length. Rows at or above
        # position are not historical KV; the causal kernel materializes them locally.
        e.bufs['pos'].fill_(position)
        torch.cuda.synchronize()
        rc = e.lib.mt_run()
        if rc:
            raise RuntimeError(f'mt_run CUDA error {rc}')
        # Kernel sampler already computes each logit's argmax, including tie order.
        return e.bufs['tok'].cpu().tolist()

    def prefill(self, prompt, variant):
        begin = time.perf_counter_ns()
        prefix = prompt[:-1]
        if variant == 'v2':
            self.v2.prefill(prefix)
            state = {name: self.v2.bufs[name].view(28, lm03b.MAXPOS, 1024)[:, :len(prefix)].clone()
                     for name in ('kc', 'vc')}
        else:
            for position in range(0, len(prefix), 5):
                self.pass_mt(prefix[position:position+5], position)
            state = {name: self.cache[name].view(28, self.cap, 1024)[:, :len(prefix)].clone()
                     for name in ('kc', 'vc')}
        torch.cuda.synchronize()
        return state, (time.perf_counter_ns()-begin)/1e9

    def restore(self, state, variant):
        for name, value in state.items():
            buffer = self.v2.bufs[name] if variant == 'v2' else self.cache[name]
            cap = lm03b.MAXPOS if variant == 'v2' else self.cap
            buffer.view(28, cap, 1024)[:, :value.shape[1]].copy_(value)
        torch.cuda.synchronize()

    def decode(self, prompt, count, variant):
        position = len(prompt)-1
        begin = time.perf_counter_ns()
        if variant == 'v2':
            self.v2.pos_set(position)
            self.v2.set_tok(prompt[-1])
            torch.cuda.synchronize()
            self.v2.mega(count)
            output = self.v2.bufs['tok_hist'][position:position+count].cpu().tolist()
            return output, {'wall_s': (time.perf_counter_ns()-begin)/1e9}
        history = list(prompt)
        output = []
        lookup_begin = time.perf_counter_ns()
        lookup = Lookup(history) if variant == 'spec' else None
        setup_ns = time.perf_counter_ns()-lookup_begin
        limit = 4
        trace = []
        while len(output) < count:
            draft_begin = time.perf_counter_ns()
            drafts, matched = lookup.draft(min(limit-1, count-len(output)-1)) if lookup else ([], 0)
            draft_ns = time.perf_counter_ns()-draft_begin
            before = time.perf_counter_ns()
            predictions = self.pass_mt([history[-1]]+drafts, position)
            verify_ns = time.perf_counter_ns()-before
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == predictions[accepted]:
                accepted += 1
            committed = drafts[:accepted]+[predictions[accepted]]
            rollback_begin = time.perf_counter_ns()
            # The target token is known but has no KV yet. It becomes the next
            # anchor at this new logical position, overwriting a rejected row.
            position += accepted+1
            history.extend(committed)
            output.extend(committed)
            rollback_ns = time.perf_counter_ns()-rollback_begin
            update_begin = time.perf_counter_ns()
            if lookup:
                lookup.extend(committed)
                if drafts:
                    limit = min(5, limit+1) if accepted == len(drafts) else max(2, limit-1)
            update_ns = time.perf_counter_ns()-update_begin
            trace.append({'k': len(drafts)+1, 'accepted': accepted, 'match_ngram': matched,
                          'draft_ns': draft_ns, 'index_update_ns': update_ns,
                          'verify_ns': verify_ns, 'rollback_ns': rollback_ns})
        wall_s = (time.perf_counter_ns()-begin)/1e9
        return output, {'wall_s': wall_s, 'lookup_setup_ns': setup_ns, 'passes': len(trace),
                        'accepted_histogram': dict(Counter(t['accepted'] for t in trace)),
                        'k_distribution': dict(Counter(t['k'] for t in trace)), 'trace': trace}
