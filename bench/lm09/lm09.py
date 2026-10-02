"""LM-09 harness: ctypes bridge to libmegakv.so + quantized-KV buffers.

KV format compiled into the kernel (megakv.cu macros):
  KMODE_T=1: per-token INT{KBITS} K, KGRP-dim groups along the head
             (codes u8, meta half2 scale+off per (tok,kvh,grp))
  V: per-token INT{VBITS}, VGRP-dim groups, same meta layout.
Buffers differ from lm03b: kq/kmeta/kring/vq/vmeta/vring replace kc/vc.

The kernel was compiled with the defaults in megakv.cu; the constants below
must match it.
"""

import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))

import lm03  # noqa: E402  (pack_weights, norm_table, make_rope, load, ...)

_lib = ctypes.CDLL(str(ROOT / "kernels" / "megakernel_kv" / "libmegakv.so"))
_i64p = ctypes.POINTER(ctypes.c_int64)
_lib.mkv_pos_set.argtypes = [ctypes.c_int]
_lib.mkv_sync.restype = ctypes.c_int
_lib.mkv_mega.argtypes = [ctypes.c_int]
_lib.mkv_init.argtypes = [_i64p, _i64p, _i64p] + [ctypes.c_int64] * 5
_lib.mkv_time_mega.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float)]
_lib.mkv_time_mega.restype = ctypes.c_int
_lib.mkv_time_matvec.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.mkv_time_matvec.restype = ctypes.c_float
_lib.mkv_matvec_run.restype = ctypes.c_int

MAXPOS = 16640                       # megakv.cu
HID, QKV, QROWS, KVROWS, GU, INTER, VOCAB = (lm03.HID, lm03.QKV, lm03.QROWS,
                                           lm03.KVROWS, lm03.GU, lm03.INTER,
                                           lm03.VOCAB)
NATTN, NARGP, NCTA, NKVH = 32, 36, 36, 8

# must match megakv.cu defines
KBITS, KMODE_T, KGRP, KRES = 8, 1, 128, 0
VBITS, VGRP, VRES = 4, 128, 0
KQROWB = KVROWS * KBITS // 8
VQROWB = KVROWS * VBITS // 8
RCAP = KRES + KGRP if not KMODE_T else 1
KMETA_PER_TOK = KVROWS // KGRP if KMODE_T else 0      # half2 count
KMETA_C = NKVH * 128 if not KMODE_T else 0            # per group, KMODE_C
VMETA_PER_TOK = KVROWS // VGRP
CODEC = f"lm09-k{KBITS}{'t' if KMODE_T else 'c'}{KGRP}r{KRES}-v{VBITS}t{VGRP}r{VRES}"


def kv_bytes_per_tok_layer():
    """Packed bytes touched per token per layer (8 kvh x 128 dims)."""
    return NKVH * (128 * KBITS // 8 + 128 // KGRP * 4 +
                   128 * VBITS // 8 + 128 // VGRP * 4)


class EngineKV:
    """Compressed-KV megakernel engine."""

    def __init__(self, ctx_cap, packed, emb, norms, rope):
        assert ctx_cap <= MAXPOS - 64, f"ctx_cap {ctx_cap} > MAXPOS-64"
        self.ctx_cap = ctx_cap
        names = lm03.matrix_names()
        self.codes = [packed[n]["codes"] for n in names]
        self.metas = [packed[n]["meta"] for n in names]
        self.keep = [emb, norms, rope, self.codes, self.metas]
        dev = torch.device("cuda")
        NLAY = lm03.N_LAYERS
        if KMODE_T:
            kq_bytes = NLAY * MAXPOS * KQROWB
            kmeta_n = NLAY * MAXPOS * KMETA_PER_TOK
            kring_n = 1
        else:
            kq_bytes = NLAY * MAXPOS * KQROWB
            kmeta_n = NLAY * (MAXPOS // KGRP) * KMETA_C
            kring_n = NLAY * RCAP * KVROWS
        self.bufs = {
            "xn": torch.zeros(HID, dtype=torch.half, device=dev),
            "qkv": torch.zeros(QKV, dtype=torch.half, device=dev),
            "oo": torch.zeros(HID, dtype=torch.half, device=dev),
            "gu": torch.zeros(GU, dtype=torch.half, device=dev),
            "dout": torch.zeros(HID, dtype=torch.half, device=dev),
            "kq": torch.zeros(kq_bytes, dtype=torch.uint8, device=dev),
            "kmeta": torch.zeros(2 * kmeta_n, dtype=torch.half, device=dev),
            "kring": torch.zeros(max(1, kring_n), dtype=torch.half, device=dev),
            "vq": torch.zeros(NLAY * MAXPOS * VQROWB, dtype=torch.uint8,
                              device=dev),
            "vmeta": torch.zeros(2 * NLAY * MAXPOS * VMETA_PER_TOK,
                                 dtype=torch.half, device=dev),
            "vring": torch.zeros(
                max(1, NLAY * RCAP * KVROWS if VRES else 1),
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
                 ("xn", "qkv", "oo", "gu", "dout", "kq", "kmeta", "kring",
                  "vq", "vmeta", "vring", "part", "logits", "argp", "pos",
                  "bar", "xpad")]
        codes = lm03._i64([t.data_ptr() for t in self.codes])
        meta = lm03._i64([t.data_ptr() for t in self.metas])
        rc = _lib.mkv_init(lm03._i64(order), codes, meta,
                           emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                           self.bufs["tok"].data_ptr(),
                           self.bufs["tok_hist"].data_ptr())
        assert rc == 0, f"mkv_init {rc}"
        self.pos0 = 0

    def set_tok(self, t):
        self.bufs["tok"].fill_(t)

    def pos_set(self, v):
        assert _lib.mkv_pos_set(v) == 0

    def mega(self, steps=1):
        assert _lib.mkv_mega(steps) == 0, "mkv_mega launch failed"
        assert _lib.mkv_sync() == 0, f"mkv_sync {_lib.mkv_sync()}"

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
