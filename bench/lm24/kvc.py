"""V2 fork with logical RoPE positions and whole-turn physical KV compaction."""
import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench/lm03'))
sys.path.insert(0, str(ROOT / 'bench/lm03b'))
import lm03
import lm03b

MAXPOS = lm03.MAXPOS
KVROWS = lm03.KVROWS
LIB = ctypes.CDLL(str(ROOT / 'kernels/megakernel_kvc/libkvc.so'))
LIB.mk2_init.argtypes = [lm03._i64p] * 3 + [ctypes.c_int64] * 5
LIB.kvc_state_set.argtypes = [ctypes.c_int] * 2
LIB.mk2_mega.argtypes = [ctypes.c_int]
LIB.kvc_time_mega.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_float)]
LIB.kvc_compact.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_float)]


class Engine:
    """One active instance per library, matching v2's process-global descriptor.

    compact takes sorted half-open physical ranges. Each endpoint must be a
    complete chat-turn boundary; a turn includes im_start and im_end. No rows
    are recomputed or re-rotated. Scratch is preallocated for retain_cap rows.
    """
    def __init__(self, packed, shared, logical_cap=16896, retain_cap=4096,
                 im_start=151644, im_end=151645):
        emb, norms, rope = shared
        if rope.shape[0] < logical_cap:
            raise ValueError('RoPE table shorter than logical capacity')
        self.logical_cap, self.retain_cap = logical_cap, retain_cap
        self.im_start, self.im_end = im_start, im_end
        self.codes = [packed[n]['codes'] for n in lm03.matrix_names()]
        self.metas = [packed[n]['meta'] for n in lm03.matrix_names()]
        self.keep = [shared, packed]
        self.bufs = {}
        for name, size, dtype in (
            ('xn',1024,torch.half), ('qkv',4096,torch.half),
            ('oo',1024,torch.half), ('gu',6144,torch.half),
            ('dout',1024,torch.half), ('kc',28*MAXPOS*1024,torch.half),
            ('vc',28*MAXPOS*1024,torch.half), ('part',32*130,torch.float32),
            ('logits',151936,torch.float32), ('argp',72,torch.float32),
            ('rope_pos',1,torch.int32), ('bar',1,torch.int32),
            ('xpad',2048,torch.half), ('kv_len',1,torch.int32),
            ('tok',1,torch.int32), ('tok_hist',logical_cap,torch.int32),
            ('scratch',2*28*retain_cap*1024,torch.half),
            ('rows',retain_cap,torch.int32)):
            self.bufs[name] = torch.zeros(size, dtype=dtype, device='cuda')
        order = ('xn','qkv','oo','gu','dout','kc','vc','part','logits','argp',
                 'rope_pos','bar','xpad','kv_len')
        torch.cuda.synchronize()
        rc = LIB.mk2_init(lm03._i64([self.bufs[n].data_ptr() for n in order]),
                         lm03._i64([t.data_ptr() for t in self.codes]),
                         lm03._i64([t.data_ptr() for t in self.metas]),
                         emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                         self.bufs['tok'].data_ptr(), self.bufs['tok_hist'].data_ptr())
        assert rc == 0, rc
        self.rope_pos = self.kv_len = 0
        self.tokens = []

    def reset(self):
        assert LIB.kvc_state_set(0, 0) == 0
        self.rope_pos = self.kv_len = 0
        self.tokens = []

    def step(self, token):
        if self.rope_pos >= self.logical_cap or self.kv_len >= MAXPOS:
            raise ValueError('decode capacity exceeded')
        self.bufs['tok'].fill_(int(token))
        torch.cuda.synchronize()
        assert LIB.mk2_mega(1) == 0
        assert LIB.mk2_sync() == 0
        self.tokens.append(int(token))
        self.rope_pos += 1
        self.kv_len += 1
        return self.bufs['logits'].clone()

    def turns(self):
        """Include im_start, im_end, and trailing chat-template whitespace."""
        spans, start, closed = [], None, False
        for i, token in enumerate(self.tokens):
            if token == self.im_start:
                if start is not None:
                    if not closed:
                        raise ValueError('nested im_start')
                    spans.append((start, i))
                start, closed = i, False
            elif token == self.im_end:
                if start is None or closed:
                    raise ValueError('im_end without an open turn')
                closed = True
            elif start is None:
                raise ValueError('tokens outside complete turns')
        if start is not None:
            if not closed:
                raise ValueError('cannot compact an unfinished turn')
            spans.append((start, len(self.tokens)))
        return spans

    def compact(self, keep_ranges):
        turns = self.turns()
        starts, ends = {a for a,b in turns}, {b for a,b in turns}
        ranges = list(keep_ranges)
        prior, rows = 0, []
        for a,b in ranges:
            if a < prior or a >= b or a not in starts or b not in ends:
                raise ValueError('ranges must be ordered, disjoint whole turns')
            rows.extend(range(a,b))
            prior = b
        if len(rows) > self.retain_cap:
            raise ValueError('retained rows exceed preallocated scratch')
        mapping = torch.tensor(rows, dtype=torch.int32)
        self.bufs['rows'][:len(rows)].copy_(mapping)
        torch.cuda.synchronize()
        elapsed = ctypes.c_float()
        rc = LIB.kvc_compact(self.bufs['scratch'].data_ptr(),
                             self.bufs['rows'].data_ptr(), len(rows),
                             ctypes.byref(elapsed))
        assert rc == 0, rc
        self.tokens = [self.tokens[i] for i in rows]
        self.kv_len = len(rows)
        return elapsed.value

    def keep_first_last(self, n):
        if n < 0:
            raise ValueError('negative last-turn count')
        turns = self.turns()
        selected = sorted(set(([turns[0]] if turns else []) +
                              (turns[-n:] if n else [])))
        return selected


def resources(logical_cap=16896):
    packed, _ = lm03.pack_weights(ROOT / 'bench/lm11/weights_int4_gptq.pt', False)
    shared = (lm03.load('model.embed_tokens.weight').half().contiguous(),
              lm03.norm_table(), lm03.make_rope(logical_cap).cuda())
    return packed, shared
