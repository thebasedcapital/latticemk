"""LM-03b harness: ctypes bridge to libmega2.so + buffers.

Shares weight packing / norm table / rope with bench/lm03/lm03.py (imported).
The v2 engine keeps its own buffer set (x lives per-CTA in SMEM, so there is
no x/xn/attn buffer; qkv/oo/gu/dout/kc/vc/part/logits/argp/tok/pos/bar only).
"""

import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))

import lm03  # noqa: E402  (pack_weights, norm_table, make_rope, load, ...)

_lib2 = ctypes.CDLL(str(ROOT / "kernels/megakernel_kvc/libbaseline16k.so"))
_i64p = ctypes.POINTER(ctypes.c_int64)
_lib2.mk2_pos_set.argtypes = [ctypes.c_int]
_lib2.mk2_sync.restype = ctypes.c_int
_lib2.mk2_mega.argtypes = [ctypes.c_int]
_lib2.mk2_init.argtypes = [_i64p, _i64p, _i64p] + [ctypes.c_int64] * 5
_lib2.mk2_time_mega.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float)]
_lib2.mk2_time_mega.restype = ctypes.c_int
_lib2.mk2_time_matvec.argtypes = [ctypes.c_int, ctypes.c_int]
_lib2.mk2_time_matvec.restype = ctypes.c_float
_lib2.mk2_matvec_run.restype = ctypes.c_int

MAXPOS = 16896
HID, QKV, QROWS, KVROWS, GU, INTER, VOCAB = (lm03.HID, lm03.QKV, lm03.QROWS,
                                           lm03.KVROWS, lm03.GU, lm03.INTER,
                                           lm03.VOCAB)
NATTN, NARGP, NCTA = 32, 36, 36


class Engine2:
    """v2 megakernel engine."""

    def __init__(self, ctx_cap, packed, emb, norms, rope):
        self.ctx_cap = ctx_cap
        names = lm03.matrix_names()
        self.codes = [packed[n]["codes"] for n in names]
        self.metas = [packed[n]["meta"] for n in names]
        self.keep = [emb, norms, rope, self.codes, self.metas]
        dev = torch.device("cuda")
        self.bufs = {
            "xn": torch.zeros(HID, dtype=torch.half, device=dev),   # stage in
            "qkv": torch.zeros(QKV, dtype=torch.half, device=dev),
            "oo": torch.zeros(HID, dtype=torch.half, device=dev),
            "gu": torch.zeros(GU, dtype=torch.half, device=dev),
            "dout": torch.zeros(HID, dtype=torch.half, device=dev),
            "kc": torch.zeros(lm03.N_LAYERS * MAXPOS * KVROWS,
                              dtype=torch.half, device=dev),
            "vc": torch.zeros(lm03.N_LAYERS * MAXPOS * KVROWS,
                              dtype=torch.half, device=dev),
            "part": torch.zeros(NATTN * 130, dtype=torch.float32, device=dev),
            "logits": torch.zeros(VOCAB, dtype=torch.float32, device=dev),
            "argp": torch.zeros(NARGP * 2, dtype=torch.float32, device=dev),
            "tok": torch.zeros(1, dtype=torch.int32, device=dev),
            "pos": torch.zeros(1, dtype=torch.int32, device=dev),
            "bar": torch.zeros(1, dtype=torch.int32, device=dev),
            "xpad": torch.zeros(QROWS, dtype=torch.half, device=dev),
            "tok_hist": torch.zeros(MAXPOS, dtype=torch.int32, device=dev),
        }
        order = [self.bufs[k].data_ptr() for k in
                 ("xn", "qkv", "oo", "gu", "dout", "kc", "vc", "part",
                  "logits", "argp", "pos", "bar", "xpad")]
        codes = lm03._i64([t.data_ptr() for t in self.codes])
        meta = lm03._i64([t.data_ptr() for t in self.metas])
        rc = _lib2.mk2_init(lm03._i64(order), codes, meta,
                            emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                            self.bufs["tok"].data_ptr(),
                            self.bufs["tok_hist"].data_ptr())
        assert rc == 0, f"mk2_init {rc}"
        self.pos0 = 0

    def set_tok(self, t):
        self.bufs["tok"].fill_(t)

    def pos_set(self, v):
        assert _lib2.mk2_pos_set(v) == 0

    def mega(self, steps=1):
        assert _lib2.mk2_mega(steps) == 0, "mk2_mega launch failed"
        assert _lib2.mk2_sync() == 0, f"mk2_sync {_lib2.mk2_sync()}"

    def prefill(self, ids):
        self.pos_set(0)
        self.pos0 = len(ids)
        for t in ids:
            self.set_tok(int(t))
            self.mega(1)

    def decode(self, n):
        self.mega(n)
        self.pos0 += n
        return self.bufs["tok_hist"][: self.pos0].cpu()

    def logits(self):
        return self.bufs["logits"].clone()
