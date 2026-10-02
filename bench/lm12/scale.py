"""Qwen3-1.7B megakernel-v2 fork: model tensors, packed weights and GPU buffers."""

import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
import lm03
from safetensors import safe_open

SNAPSHOT = next((Path.home() / ".cache/huggingface/hub/"
                 "models--Qwen--Qwen3-1.7B-Base/snapshots").iterdir())
N_LAYERS = 28
MAXPOS = 8704
HID, QKV, QROWS, KVROWS, GU, INTER, VOCAB = (
    2048, 4096, 2048, 1024, 12288, 6144, 151936)
NATTN, NARGP, NCTA = 32, 36, 36


def load(key, device="cpu"):
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt",
                   device="cpu") as f:
        if key == "lm_head.weight":
            key = "model.embed_tokens.weight"
        return f.get_tensor(key).to(device=device, dtype=torch.float32)


def matrix_names():
    return lm03.matrix_names()


def load_matrix(name, device="cuda"):
    if name == "lm_head":
        return load("lm_head.weight", device)
    layer = int(name[1:name.index(".")])
    prefix = f"model.layers.{layer}."
    kind = name.split(".")[1]
    if kind == "qkv":
        return torch.cat([load(prefix + f"self_attn.{n}_proj.weight", device)
                          for n in "qkv"])
    if kind == "gu":
        return torch.cat([load(prefix + f"mlp.{n}_proj.weight", device)
                          for n in ("gate", "up")])
    return load(prefix + ("self_attn.o_proj.weight" if kind == "o"
                          else "mlp.down_proj.weight"), device)


def norm_table():
    rows = []
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt",
                   device="cpu") as f:
        for stem in ("input_layernorm", "post_attention_layernorm"):
            rows.extend(f.get_tensor(f"model.layers.{i}.{stem}.weight")
                        for i in range(N_LAYERS))
        for stem in ("q_norm", "k_norm"):
            rows.extend(torch.nn.functional.pad(
                f.get_tensor(f"model.layers.{i}.self_attn.{stem}.weight"),
                (0, HID - 128)) for i in range(N_LAYERS))
        rows.append(f.get_tensor("model.norm.weight"))
    return torch.stack(rows).half().cuda().contiguous()


def make_rope():
    return lm03.make_rope(MAXPOS).cuda()


_i64p = ctypes.POINTER(ctypes.c_int64)


def use_library(filename="libmega_scale.so"):
    global _lib2
    lib = ctypes.CDLL(str(ROOT / "kernels/megakernel_scale" / filename))
    lib.mk2_pos_set.argtypes = [ctypes.c_int]
    lib.mk2_sync.restype = ctypes.c_int
    lib.mk2_mega.argtypes = [ctypes.c_int]
    lib.mk2_init.argtypes = [_i64p, _i64p, _i64p] + [ctypes.c_int64] * 5
    lib.mk2_time_mega.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_float)]
    lib.mk2_time_mega.restype = ctypes.c_int
    lib.mk2_matvec_run.restype = ctypes.c_int
    _lib2 = lib


use_library()



class Engine2:
    """One scaled decoder instance; weight tensors remain owned by the caller."""

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
            "kc": torch.zeros(N_LAYERS * MAXPOS * KVROWS,
                              dtype=torch.half, device=dev),
            "vc": torch.zeros(N_LAYERS * MAXPOS * KVROWS,
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
