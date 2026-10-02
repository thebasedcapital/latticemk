"""KV-cache codec plug-in interface + baselines (LM-07).

Tensor convention: x is [T, H, D] float32/float16 — T tokens, H kv heads (8), head_dim D (128) —
exactly the layout the paged KV cache stores (K is post-k_norm, post-RoPE). `kind` is 'k' or 'v'.

`Packed.bits` is the packed-format bit count: codes at their true bit width plus every byte of
scale/zero/metadata. It is not torch storage size (codes sit in wider dtypes here). Codecs that
pad (per-channel token blocks) charge the padded bits to the real coordinate count.

All codecs are offline/prefill-style (encode sees the whole [T,H,D] window), matching how the
harness fake-quantizes cache writes on a 2048-token window.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import codebooks, quant  # noqa: E402

GROUP = 128  # per-token group = one head vector; per-channel block = 128 tokens (KIVI-style)
DIM = 128


@dataclass
class Packed:
    n: int               # real coordinates encoded
    bits: int            # packed-format payload bits, scales/metadata included
    shape: tuple         # original [T, H, D]
    kind: str            # 'k' | 'v'
    payload: dict = field(default_factory=dict)


class Codec(Protocol):
    name: str

    def encode(self, x: torch.Tensor, kind: str) -> Packed: ...
    def decode(self, p: Packed) -> torch.Tensor: ...
    def bits_per_coord(self, p: Packed) -> float:
        return p.bits / p.n


class CodecBase:
    name = "?"

    def bits_per_coord(self, p: Packed) -> float:
        return p.bits / p.n


# ---------------------------------------------------------------- baselines


class Fp16(CodecBase):
    name = "fp16"

    def encode(self, x, kind):
        h = x.half()
        return Packed(x.numel(), 16 * x.numel(), tuple(x.shape), kind, {"h": h})

    def decode(self, p):
        return p.payload["h"].float()


class IntToken(CodecBase):
    """Asymmetric INT{bits}, per-token group-128 (= one head vector), fp16 scale + fp16 zero.

    Reuses lmk.quant.quant_int, which does an MSE-optimal clip-shrink search per group — the same
    quantizer family that scored INT4 weights in wave 1, so the baseline is not weakened.
    """

    def __init__(self, bits: int):
        self.bits = bits
        self.name = f"int{bits}"

    def encode(self, x, kind):
        g = x.reshape(-1, GROUP).float()
        q = quant.quant_int(g, self.bits)
        b = q.codes.numel() * self.bits + 16 * q.scale.numel() + 16 * q.zero.numel()
        return Packed(x.numel(), b, tuple(x.shape), kind,
                      {"codes": q.codes, "scale": q.scale, "zero": q.zero})

    def decode(self, p):
        c = p.payload["codes"].float().reshape(-1, GROUP)
        rec = c * p.payload["scale"].float() + p.payload["zero"].float()
        return rec.reshape(p.shape)


class IntChannel(CodecBase):
    """Asymmetric INT{bits}, per-channel: blocks of `g` tokens along T, fp16 scale + fp16 zero.

    KIVI-style K quantization. T is zero-padded to a block multiple; padded bits are charged.
    """

    def __init__(self, bits: int, g: int = 128):
        self.bits, self.g = bits, g
        self.name = f"int{bits}-ch{g}"

    def encode(self, x, kind):
        T, H, C = x.shape
        tb = math.ceil(T / self.g)
        xp = x.new_zeros(tb * self.g, H, C)
        xp[:T] = x
        rows = xp.reshape(tb, self.g, H, C).permute(0, 2, 3, 1).reshape(-1, self.g).float()
        q = quant.quant_int(rows, self.bits)
        b = q.codes.numel() * self.bits + 16 * q.scale.numel() + 16 * q.zero.numel()
        return Packed(x.numel(), b, tuple(x.shape), kind,
                      {"codes": q.codes, "scale": q.scale, "zero": q.zero, "tb": tb})

    def decode(self, p):
        T, H, C = p.shape
        c = p.payload["codes"].float().reshape(-1, self.g)
        rec = c * p.payload["scale"].float() + p.payload["zero"].float()
        rec = rec.reshape(p.payload["tb"], H, C, self.g).permute(0, 3, 1, 2).reshape(-1, H, C)
        return rec[:T]


class Kivi(CodecBase):
    """KIVI-style: K per-channel (128-token blocks), V per-token (group = head vector)."""

    def __init__(self, bits: int):
        self.bits = bits
        self.name = f"kivi-int{bits}"
        self.kc, self.vc = IntChannel(bits), IntToken(bits)

    def encode(self, x, kind):
        p = (self.kc if kind == "k" else self.vc).encode(x, kind)
        p.payload["mode"] = kind
        return p

    def decode(self, p):
        return (self.kc if p.payload["mode"] == "k" else self.vc).decode(p)


# ------------------------------------------------------------- A_n stand-in
# NOT D-02. This is the wave-1 A_n truncated-ball codebook (lmk/codebooks.py) applied to KV
# vectors: per-head-vector fp16 scale x unit-RMS codebook, optional RHT over the 128-dim head.


class An(CodecBase):
    def __init__(self, d: int, k: int, rht: bool = False, seed: int = 0,
                 alphas=np.linspace(0.6, 1.5, 10)):
        self.d, self.k, self.rht, self.seed = d, k, rht, seed
        self.alphas = alphas
        self.cb = torch.from_numpy(codebooks.an_codebook(d, k)).cuda()
        self.had = quant.hadamard(DIM, "cuda")
        self.name = f"A{d}-k{k}" + ("+rht" if rht else "")

    def _signs(self, kind: str, H: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(977 * self.seed + (1 if kind == "v" else 0))
        return (torch.randint(0, 2, (H, DIM), generator=g) * 2 - 1).float().cuda()

    def _fwd(self, x, kind):
        """[T,H,128] -> [T*H,128] rotated (if rht)."""
        T, H, C = x.shape
        if not self.rht:
            return x.reshape(-1, C).float()
        s = self._signs(kind, H)
        return ((x * s.unsqueeze(0)).reshape(-1, C) @ self.had).float()

    def _inv(self, g, p):
        T, H, C = p.shape
        if not self.rht:
            return g.reshape(p.shape)
        s = self._signs(p.kind, H)
        return ((g @ self.had).reshape(T, H, C) * s.unsqueeze(0)).reshape(p.shape)

    def encode(self, x, kind):
        g = self._fwd(x, kind)
        q = quant.quant_vq(g, self.cb, self.alphas)
        b = q.codes.numel() * self.k + 16 * q.scale.numel()
        return Packed(x.numel(), b, tuple(x.shape), kind,
                      {"idx": q.codes, "scale": q.scale})

    def decode(self, p):
        rec = self.cb[p.payload["idx"].long()].reshape(-1, DIM) * p.payload["scale"].float()
        return self._inv(rec, p)


class D02(CodecBase):
    """Stub for the real D-02 simplex A_n quantizer (2.41 b/coord claim on KV vectors).

    PLUG-IN STEPS (see reports/wave-2/LM-07.md):
      1. Vendor the D-02 encoder/decoder (machine-proof-builds/ codec) into kvcodec/d02/ or import it.
      2. Fill in encode(): map [T,H,128] -> Packed with the D-02 packed payload and honest `bits`
         (codes + every scale/side-info byte). decode(): the exact inverse.
      3. Add round-trip + format documentation; run kvcodec/eval.py --codecs d02.
    """

    name = "d02"

    def encode(self, x, kind):
        raise NotImplementedError(
            "D-02 codec is not vendored on this machine. Wire it here: see D02 docstring and "
            "reports/wave-2/LM-07.md for the plug-in steps.")

    def decode(self, p):
        raise NotImplementedError("D-02 codec not vendored; see encode().")


# ------------------------------------------------------------------ registry


def make(name: str, seed: int = 0) -> CodecBase:
    if name == "fp16":
        return Fp16()
    if m := re.fullmatch(r"int(\d+)", name):
        return IntToken(int(m[1]))
    if m := re.fullmatch(r"kivi-int(\d+)", name):
        return Kivi(int(m[1]))
    if m := re.fullmatch(r"A(\d)-k(\d+)(\+rht)?", name):
        return An(int(m[1]), int(m[2]), rht=bool(m[3]), seed=seed)
    if name == "d02":
        return D02()
    raise ValueError(f"unknown codec {name!r}; choices: fp16, intN, kivi-intN, A{{2,4}}-kK[+rht], d02")
