"""LM-03 harness: ctypes bridge to libmega.so + weight/buffer setup.

Both engines share one device state:
  * "separate": step_kernels launched op-by-op, captured as one CUDA graph;
  * "mega": one persistent launch executing the sched_gen task list over flag
    counters; mk_mega(n) runs n decode steps inside a single launch.
"""

import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels" / "megakernel"))

import sched_gen
from lmk import pack, quant
from lmk.model import N_LAYERS, SNAPSHOT, load

_lib = ctypes.CDLL(str(ROOT / "build" / "libmega.so"))
_i64p = ctypes.POINTER(ctypes.c_int64)
_lib.mk_init.argtypes = [_i64p, _i64p, _i64p] + [ctypes.c_int64] * 7
_lib.mk_load_sched.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.mk_flags_alloc.argtypes = [ctypes.c_int]
_lib.mk_reset_flags.restype = ctypes.c_int
_lib.mk_pos_set.argtypes = [ctypes.c_int]
_lib.mk_step.restype = ctypes.c_int
_lib.mk_graph_build.restype = ctypes.c_int
_lib.mk_graph_launch.restype = ctypes.c_int
_lib.mk_mega.argtypes = [ctypes.c_int]
_lib.mk_mega.restype = ctypes.c_int
_lib.mk_sync.restype = ctypes.c_int
_lib.mk_time_graph.argtypes = [ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_float)]
_lib.mk_time_mega.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_float)]

CACHE = Path(__file__).resolve().parent / "weights_int4.pt"
MAXPOS, NSLICE, NARGP = sched_gen.MAXPOS, sched_gen.NSLICE, sched_gen.NARGP
HID, QKV, QROWS, KVROWS, GU, INTER, VOCAB = (sched_gen.HID, sched_gen.QKV,
                                           sched_gen.QROWS, sched_gen.KVROWS,
                                           sched_gen.GU, sched_gen.INTER,
                                           sched_gen.VOCAB)
MAX_STEPS = 512   # flags slab is allocated once for this many steps


def _i64(vals):
    return (ctypes.c_int64 * len(vals))(*vals)


def _f32(vals):
    return (ctypes.c_float * len(vals))(*vals)


def make_rope(maxpos=MAXPOS, theta=1e6):
    """rope[pos, 256] = [cos(64) x2, sin(64) x2] fp32; pairs (d, d+64)."""
    inv = theta ** (-torch.arange(0, 128, 2).float() / 128)  # [64]
    ang = torch.arange(maxpos).float()[:, None] * inv[None]  # [p, 64]
    row = torch.zeros(maxpos, 256)
    row[:, :64] = row[:, 64:128] = ang.cos()
    row[:, 128:192] = row[:, 192:256] = ang.sin()
    return row.contiguous()


def matrix_names():
    out = []
    for i in range(N_LAYERS):
        out += [f"L{i}.qkv", f"L{i}.o", f"L{i}.gu", f"L{i}.down"]
    return out + ["lm_head"]


def _load_matrix(name):
    if name == "lm_head":
        return load("lm_head.weight")
    i = int(name[1:name.index(".")])
    p = f"model.layers.{i}."
    kind = name.split(".")[1]
    if kind == "qkv":
        return torch.cat([load(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
    if kind == "o":
        return load(p + "self_attn.o_proj.weight")
    if kind == "gu":
        return torch.cat([load(p + f"mlp.{n}_proj.weight")
                          for n in ("gate", "up")])
    if kind == "down":
        return load(p + "mlp.down_proj.weight")
    return load("lm_head.weight")


def pack_weights(cache=CACHE, want_deq=True):
    """-> (packed {name: {codes, meta} cuda}, deq {name: fp32 cpu} | None).

    INT4-g128 RTN on the fused-qkv / fused-gate_up views, same bytes as
    scripts/bench_gemv.py. Packed tensors are cached on disk.
    """
    names = matrix_names()
    packed = {}
    if cache.exists():
        packed = {k: {f: t.cuda() for f, t in v.items()}
                  for k, v in torch.load(cache).items()}
    else:
        for n in names:
            w = _load_matrix(n)
            q = quant.quant_int(w, 4)
            p = pack.pack_int4(q)
            packed[n] = dict(codes=p.codes.cpu(), meta=p.meta.cpu())
            del w, q, p
            torch.cuda.empty_cache()
        torch.save(packed, cache)
        packed = {k: {f: t.cuda() for f, t in v.items()}
                  for k, v in packed.items()}
    if not want_deq:
        return packed, None
    deq = {}
    for n in names:
        deq[n] = quant.quant_int(_load_matrix(n), 4).deq.cpu()
        torch.cuda.empty_cache()
    return packed, deq


def norm_table():
    """[113][1024] half: ln1 x28, ln2 x28, q_norm x28, k_norm x28, final."""
    from safetensors import safe_open
    rows = []
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt",
                   device="cuda") as f:
        for i in range(N_LAYERS):
            rows.append(f.get_tensor(
                f"model.layers.{i}.input_layernorm.weight"))
        for i in range(N_LAYERS):
            rows.append(f.get_tensor(
                f"model.layers.{i}.post_attention_layernorm.weight"))
        for i in range(N_LAYERS):
            qn = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
            rows.append(torch.nn.functional.pad(qn, (0, HID - qn.numel())))
        for i in range(N_LAYERS):
            kn = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
            rows.append(torch.nn.functional.pad(kn, (0, HID - kn.numel())))
        rows.append(f.get_tensor("model.norm.weight"))
    return torch.stack(rows).half().contiguous()


class Engine:
    """One engine context: buffers + INT4 weights + schedule for ctx_cap."""

    def __init__(self, ctx_cap, packed, emb, norms, rope, nslice=None, nblocks=None):
        self.ctx_cap = ctx_cap
        g = sched_gen.build(ctx_cap + 64, nblocks=nblocks or sched_gen.NB,
                        nslice=nslice)  # decode headroom
        self.gen = g
        self.blob = sched_gen.blob(g)
        names = matrix_names()
        self.codes = [packed[n]["codes"] for n in names]
        self.metas = [packed[n]["meta"] for n in names]
        self.keep = [emb, norms, rope, self.codes, self.metas]
        dev = torch.device("cuda")
        self.bufs = {
            "x": torch.zeros(HID, dtype=torch.half, device=dev),
            "xn": torch.zeros(HID, dtype=torch.half, device=dev),
            "qkv": torch.zeros(QKV, dtype=torch.half, device=dev),
            "attn": torch.zeros(QROWS, dtype=torch.half, device=dev),
            "oo": torch.zeros(HID, dtype=torch.half, device=dev),
            "gu": torch.zeros(GU, dtype=torch.half, device=dev),
            "dout": torch.zeros(HID, dtype=torch.half, device=dev),
            "kc": torch.zeros(N_LAYERS * MAXPOS * KVROWS, dtype=torch.half,
                              device=dev),
            "vc": torch.zeros(N_LAYERS * MAXPOS * KVROWS, dtype=torch.half,
                              device=dev),
            "part": torch.zeros(NSLICE * 16 * 130, dtype=torch.float32,
                                device=dev),
            "logits": torch.zeros(VOCAB, dtype=torch.float32, device=dev),
            "argp": torch.zeros(NARGP * 2, dtype=torch.float32, device=dev),
            "tok": torch.zeros(1, dtype=torch.int32, device=dev),
            "pos": torch.zeros(1, dtype=torch.int32, device=dev),
            "tok_hist": torch.zeros(MAXPOS, dtype=torch.int32, device=dev),
        }
        order = [self.bufs[k].data_ptr() for k in
                 ("x", "xn", "qkv", "attn", "oo", "gu", "dout", "kc", "vc",
                  "part", "logits", "argp", "pos", "tok")]
        codes = [t.data_ptr() for t in self.codes]
        meta = [t.data_ptr() for t in self.metas]
        ns = nslice if nslice else g.nslice
        rc = _lib.mk_init(_i64(order), _i64(codes), _i64(meta),
                          emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                          self.bufs["tok"].data_ptr(),
                          self.bufs["tok_hist"].data_ptr(), ns, ctx_cap + 64)
        assert rc == 0, f"mk_init {rc}"
        rc = _lib.mk_load_sched(self.blob, len(self.blob))
        assert rc == 0, f"mk_load_sched {rc}"
        _lib.mk_flags_alloc(MAX_STEPS)
        self.pos0 = 0

    def set_tok(self, t):
        self.bufs["tok"].fill_(t)

    def step(self):  # separate kernels, stream-ordered (no graph)
        assert _lib.mk_step() == 0, "mk_step failed"

    def mega(self, steps=1):
        assert steps <= MAX_STEPS
        assert _lib.mk_reset_flags() == 0
        assert _lib.mk_mega(steps) == 0, "mk_mega launch failed"
        assert _lib.mk_sync() == 0, f"mk_mega failed { _lib.mk_sync() }"

    def graph_build(self):
        assert _lib.mk_graph_build() == 0, "mk_graph_build failed"

    def graph_launch(self):
        assert _lib.mk_graph_launch() == 0, "mk_graph_launch failed"
        assert _lib.mk_sync() == 0, "graph replay failed"

    def pos_set(self, v):
        assert _lib.mk_pos_set(v) == 0

    def prefill(self, ids, engine="graph"):
        """Feed prompt tokens one per decode step (teacher-forced)."""
        self.pos_set(0)
        self.pos0 = len(ids)
        for t in ids:
            self.set_tok(int(t))
            if engine == "graph":
                self.graph_launch()
            elif engine == "step":
                self.step()
            else:
                self.mega(1)
        # tok_buf now holds argmax(h[-1]) = first generated token

    def decode(self, n, engine="graph"):
        """Run n decode steps; return all emitted tokens incl. prefill's."""
        if engine == "graph":
            for _ in range(n):
                self.graph_launch()
        else:
            self.mega(n)
        self.pos0 += n
        return self.bufs["tok_hist"][: self.pos0].cpu()

    def logits(self):
        return self.bufs["logits"].clone()


def setup(ctx_cap=2048, fresh_weights=False, nslice=None, want_deq=True):
    if fresh_weights and CACHE.exists():
        CACHE.unlink()
    packed, deq = pack_weights(want_deq=want_deq)
    emb = load("model.embed_tokens.weight").half().contiguous()
    norms = norm_table()
    rope = make_rope().cuda()
    eng = Engine(ctx_cap, packed, emb, norms, rope, nslice)
    return eng, deq
