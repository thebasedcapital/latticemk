"""LM-08 harness: ctypes bridge to libfused.so + weight/buffer setup.

Fused separate-kernel engine: one CUDA graph of ~141 nodes per decode step
(vs ~285 in LM-03). pos is advanced inside the lm_head argmax epilogue, so
fx_pos_set(v) both resets the position and seeds the step loop.
"""

import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))

import lm03  # reuse pack_weights/norm_table/make_rope/matrix_names
from lmk.model import N_LAYERS

_lib = ctypes.CDLL(str(ROOT / "build" / "libfused.so"))
_i64p = ctypes.POINTER(ctypes.c_int64)
_lib.fx_init.argtypes = [_i64p, _i64p, _i64p] + [ctypes.c_int64] * 7
_lib.fx_pos_set.argtypes = [ctypes.c_int]
_lib.fx_set_attn.argtypes = [ctypes.c_int] * 4
_lib.fx_step.restype = ctypes.c_int
_lib.fx_graph_build.restype = ctypes.c_int
_lib.fx_graph_launch.restype = ctypes.c_int
_lib.fx_sync.restype = ctypes.c_int
_lib.fx_time_graph.argtypes = [ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float)]
_lib.fx_time_breakdown.argtypes = [ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_float)]
_lib.fx_time_gemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.fx_time_gemv.restype = ctypes.c_float
_lib.fx_time_attn.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.fx_time_attn.restype = ctypes.c_float
_lib.fx_shutdown.restype = ctypes.c_int
_lib.fx_time_gemvseq.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.fx_time_gemvseq.restype = ctypes.c_float
_lib.fx_set_plain_gemv.argtypes = [ctypes.c_int]

CACHE = Path(__file__).resolve().parent / "weights_int4.pt"
MAXPOS = 8704
NSLICE_MAX = 144
HID, QKV, QROWS, KVROWS, GU, INTER, VOCAB = (lm03.HID, lm03.QKV, lm03.QROWS,
                                           lm03.KVROWS, lm03.GU, lm03.INTER,
                                           lm03.VOCAB)

_f32 = lm03._f32
_i64 = lm03._i64


class Engine:
    """Fused engine: buffers + INT4 weights + graph for ctx_cap.

    nslice caps the attention split (runtime slice count = ceil(ctx/per)).
    """

    def __init__(self, ctx_cap, packed, emb, norms, rope, nslice=0):
        self.ctx_cap = ctx_cap
        self.nslice = nslice
        names = lm03.matrix_names()
        self.codes = [packed[n]["codes"] for n in names]
        self.metas = [packed[n]["meta"] for n in names]
        self.keep = [emb, norms, rope, self.codes, self.metas]
        dev = torch.device("cuda")
        self.bufs = {
            "x": torch.zeros(HID, dtype=torch.half, device=dev),
            "qkv": torch.zeros(QKV, dtype=torch.half, device=dev),
            "attn": torch.zeros(QROWS, dtype=torch.half, device=dev),
            "gu": torch.zeros(GU, dtype=torch.half, device=dev),
            "kc": torch.zeros(N_LAYERS * MAXPOS * KVROWS, dtype=torch.half,
                              device=dev),
            "vc": torch.zeros(N_LAYERS * MAXPOS * KVROWS, dtype=torch.half,
                              device=dev),
            "part": torch.zeros(NSLICE_MAX * 16 * 130, dtype=torch.float32,
                                device=dev),
            "logits": torch.zeros(VOCAB, dtype=torch.float32, device=dev),
            "tok": torch.zeros(1, dtype=torch.int32, device=dev),
            "tok_hist": torch.zeros(MAXPOS, dtype=torch.int32, device=dev),
        }
        order = [self.bufs[k].data_ptr() for k in
                 ("x", "qkv", "attn", "gu", "kc", "vc", "part", "logits")]
        codes = [t.data_ptr() for t in self.codes]
        meta = [t.data_ptr() for t in self.metas]
        rc = _lib.fx_init(_i64(order), _i64(codes), _i64(meta),
                          emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                          self.bufs["tok"].data_ptr(),
                          self.bufs["tok_hist"].data_ptr(),
                          nslice, ctx_cap + 64)
        assert rc == 0, f"fx_init {rc}"
        self.pos0 = 0

    def set_tok(self, t):
        self.bufs["tok"].fill_(t)
        torch.cuda.synchronize()  # legacy-stream fill vs nonblocking g_stream

    def set_attn(self, prep=1, split_attnc=1, nt=512, attnc_grid=8):
        _lib.fx_set_attn(prep, split_attnc, nt, attnc_grid)

    def step(self):          # separate kernels, stream-ordered (no graph)
        assert _lib.fx_step() == 0, "fx_step failed"

    def graph_build(self):
        assert _lib.fx_graph_build() == 0, "fx_graph_build failed"

    def graph_launch(self):
        assert _lib.fx_graph_launch() == 0, "fx_graph_launch failed"
        assert _lib.fx_sync() == 0, "graph replay failed"

    def pos_set(self, v):
        assert _lib.fx_pos_set(v) == 0

    def prefill(self, ids, engine="graph"):
        """Feed prompt tokens one per decode step (teacher-forced)."""
        self.pos_set(0)
        self.pos0 = len(ids)
        for t in ids:
            self.set_tok(int(t))
            if engine == "graph":
                self.graph_launch()
            else:
                self.step()
        # tok now holds argmax(h[-1]) = first generated token; pos == len(ids)

    def decode(self, n, engine="graph"):
        """Run n decode steps; return all emitted tokens incl. prefill's."""
        for _ in range(n):
            self.graph_launch() if engine == "graph" else self.step()
        self.pos0 += n
        return self.bufs["tok_hist"][: self.pos0].cpu()

    def logits(self):
        return self.bufs["logits"].clone()


def setup(ctx_cap=2048, fresh_weights=False, nslice=72, want_deq=True):
    cache = CACHE
    if fresh_weights and cache.exists():
        cache.unlink()
    packed, deq = lm03.pack_weights(cache=cache, want_deq=want_deq)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    eng = Engine(ctx_cap, packed, emb, norms, rope, nslice)
    return eng, deq
