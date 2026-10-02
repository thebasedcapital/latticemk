"""LM-09 KV codecs: kernel-faithful sliding-residual KIVI variants.

Semantics mirrored by kernels/megakernel_kv (megakv.cu):

- K ("k"): per-channel asymmetric INT{kb} over gk-token groups. With T tokens in
  cache the quantization boundary is
      nq = max(0, floor((T - rk) / gk) * gk)
  positions [0, nq) are packed, [nq, T) stay exact fp16 (the kernel's residual
  ring). rk is a multiple of gk so groups finalize on group boundaries.
- V ("v"): per-token asymmetric INT{vb}, gv-dim groups along the head vector
  (gv | 128), fp16 scale + fp16 zero-fold per group. Positions [T-rv, T) stay
  exact fp16 (rv=0 -> everything packed).

quantizer `rtn`: per-group min/max, scale=(hi-lo)/qmax fp16,
zp=round(-lo/scale) clamped to [0,qmax], off=-zp*scale fp16, codes
round-half-even — the exact in-kernel quantizer. `clip`: lmk.quant.quant_int
MSE clip-shrink search (eval-only probe; if it wins the same search goes in the
kernel so the shipped pair still matches).

Name syntax:  lm09-k{kb}{c|t}{gk}r{rk}-v{vb}t{gv}r{rv}[-clip]
  c = per-channel K (gk tokens/group), t = per-token K (gk dims/group, gk=128 only)
Examples: lm09-k4c64r64-v4t128r0   lm09-k8t128r0-v4t128r0-clip

Encode/decode operate on [T, H, D] float tensors like kvcodec.codec.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import quant  # noqa: E402

from .codec import CodecBase, Packed  # noqa: E402

DIM = 128


def _rtn_rows(g: torch.Tensor, bits: int):
    """Min/max asymmetric RTN on rows [n, m] -> (deq, codes, scale fp16, off fp16).

    Mirrors megakv.cu: fp32 lo/hi, fp16 scale and offset, rintf codes."""
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-10).half().float()
    zp = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale) + zp, 0, qmax)
    off = (-zp * scale).half().float()
    rec = q * scale + off
    return rec, q.to(torch.uint8), scale.squeeze(-1).half(), off.squeeze(-1).half()


import numpy as np

def _clip_rows(g: torch.Tensor, bits: int, shrinks=np.linspace(1.0, 0.6, 9)):
    """MSE-optimal clip-shrink asymmetric RTN per row of length m (arbitrary m —
    same math as lmk.quant._quant_int but not tied to GROUP=128)."""
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    best_err = torch.full(lo.shape, float("inf"), device=g.device)
    best = None
    for s in shrinks:
        scale = ((hi - lo) * s / qmax).clamp_min(1e-10).half().float()
        zp = torch.round(-lo * s / scale).clamp(0, qmax)
        q = torch.clamp(torch.round(g / scale) + zp, 0, qmax)
        off = (-zp * scale).half().float()
        rec = q * scale + off
        err = (rec - g).pow(2).sum(-1, keepdim=True)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        if best is None:
            best = [q, scale, off, rec]
        else:
            for i, t in enumerate((q, scale, off, rec)):
                best[i] = torch.where(take, t, best[i])
    q, scale, off, rec = best
    return rec, q.to(torch.uint8), scale.squeeze(-1).half(), off.squeeze(-1).half()


class LM09K(CodecBase):
    """Per-channel INT{kb} K cache, gk-token groups, rk-token exact tail."""

    def __init__(self, bits=4, gk=64, rk=64, quantizer="rtn"):
        assert rk % gk == 0, "residual must be a multiple of the group"
        self.kb, self.gk, self.rk, self.quantizer = bits, gk, rk, quantizer
        self.name = f"lm09k-{bits}c{gk}r{rk}-{quantizer}"

    def _nq(self, T: int) -> int:
        return max(0, (T - self.rk) // self.gk * self.gk)

    def encode(self, x: torch.Tensor, kind: str) -> Packed:
        assert kind == "k"
        T, H, C = x.shape
        nq = self._nq(T)
        rows = x[:nq].reshape(nq // self.gk, self.gk, H, C) if nq else None
        meta_bits = qmax_bits = 0
        if nq:
            rows = rows.permute(0, 2, 3, 1).reshape(-1, self.gk).float()
            fn = _rtn_rows if self.quantizer == "rtn" else _clip_rows
            deq, codes, sc, off = fn(rows, self.kb)
            qmax_bits = codes.numel() * self.kb
            meta_bits = 16 * sc.numel() + 16 * off.numel()
            tail = x[nq:].half()
            payload = {"codes": codes, "scale": sc, "zero": off, "tail": tail}
        else:
            deq = x.new_zeros(0)
            payload = {"tail": x.half()}
        bits = qmax_bits + meta_bits + (T - nq) * H * C * 16
        return Packed(x.numel(), bits, tuple(x.shape), kind,
                      {"nq": nq, "deq_packed": deq, **payload})

    def decode(self, p: Packed) -> torch.Tensor:
        T, H, C = p.shape
        nq = p.payload["nq"]
        ref = p.payload["deq_packed"] if nq else p.payload["tail"]
        out = torch.zeros(T, H, C, device=ref.device)
        if nq:
            deq = p.payload["deq_packed"]  # [gk, ...] rows already dequantized
            G = nq // self.gk
            deq = deq.reshape(G, H, C, self.gk).permute(0, 3, 1, 2).reshape(nq, H, C)
            out[:nq] = deq
        out[nq:] = p.payload["tail"].float()
        return out


class LM09V(CodecBase):
    """Per-token INT{vb} V cache, gv-dim groups along the head, rv exact tail."""

    def __init__(self, bits=4, gv=128, rv=0, quantizer="rtn"):
        assert DIM % gv == 0
        self.vb, self.gv, self.rv, self.quantizer = bits, gv, rv, quantizer
        self.name = f"lm09v-{bits}t{gv}r{rv}-{quantizer}"

    def encode(self, x: torch.Tensor, kind: str) -> Packed:
        assert kind == "v"
        T, H, C = x.shape
        nq = T - self.rv
        payload = {"nq": nq}
        bits = 0
        if nq > 0:
            g = x[:nq].reshape(nq * H, C // self.gv, self.gv)
            rows = g.reshape(-1, self.gv).float()
            fn = _rtn_rows if self.quantizer == "rtn" else _clip_rows
            deq, codes, sc, off = fn(rows, self.vb)
            bits += codes.numel() * self.vb + 16 * sc.numel() + 16 * off.numel()
            payload.update(deq_packed=deq, codes=codes, scale=sc, zero=off)

        bits += self.rv * H * C * 16
        if self.rv:
            payload["tail"] = x[nq:].half()
        return Packed(x.numel(), bits, tuple(x.shape), kind, payload)

    def decode(self, p: Packed) -> torch.Tensor:
        T, H, C = p.shape
        nq = p.payload["nq"]
        ref = p.payload["deq_packed"] if nq else p.payload["tail"]
        out = torch.zeros(T, H, C, device=ref.device)
        if nq:
            deq = p.payload["deq_packed"].reshape(nq, H, C)
            out[:nq] = deq
        if self.rv:
            out[nq:] = p.payload["tail"].float()
        return out


class LM09KV(CodecBase):
    """Combined K + V codec (one name covers both kinds)."""

    def __init__(self, kb=4, kmode="c", gk=64, rk=64, vb=4, gv=128, rv=0,
                 quantizer="rtn"):
        if kmode == "c":
            self.kc = LM09K(kb, gk, rk, quantizer)
        else:  # per-token K
            self.kc = _LM09KTok(kb, gk, rk, quantizer)
        self.vc = LM09V(vb, gv, rv, quantizer)
        self.name = (f"lm09-k{kb}{kmode}{gk}r{rk}-v{vb}t{gv}r{rv}"
                     + ("-clip" if quantizer == "clip" else ""))

    def encode(self, x, kind):
        return (self.kc if kind == "k" else self.vc).encode(x, kind)

    def decode(self, p):
        return (self.kc if p.kind == "k" else self.vc).decode(p)


class _LM09KTok(CodecBase):
    """Per-token INT{kb} K, gk-dim groups along the head (gk=128 => one scale/token),
    rk-token exact tail."""

    def __init__(self, bits=8, gk=128, rk=0, quantizer="rtn"):
        self.kb, self.gk, self.rk, self.quantizer = bits, gk, rk, quantizer
        self.name = f"lm09k-{bits}t{gk}r{rk}-{quantizer}"

    def encode(self, x, kind):
        T, H, C = x.shape
        nq = T - self.rk
        bits = self.rk * H * C * 16
        payload = {"nq": nq}
        if nq > 0:
            g = x[:nq].reshape(nq * H, C // self.gk, self.gk)
            rows = g.reshape(-1, self.gk).float()
            fn = _rtn_rows if self.quantizer == "rtn" else _clip_rows
            deq, codes, sc, off = fn(rows, self.kb)
            bits += codes.numel() * self.kb + 16 * sc.numel() + 16 * off.numel()
            payload.update(deq_packed=deq, codes=codes, scale=sc, zero=off)
        if self.rk:
            payload["tail"] = x[nq:].half()
        return Packed(x.numel(), bits, tuple(x.shape), kind, payload)

    def decode(self, p):
        T, H, C = p.shape
        nq = p.payload["nq"]
        ref = p.payload["deq_packed"] if nq else p.payload["tail"]
        out = torch.zeros(T, H, C, device=ref.device)
        if nq:
            out[:nq] = p.payload["deq_packed"].reshape(nq, H, C)
        if self.rk:
            out[nq:] = p.payload["tail"].float()
        return out


_RE = re.compile(
    r"lm09-k(\d+)([ct])(\d+)r(\d+)-v(\d+)t(\d+)r(\d+)(-clip)?")


def make_lm09(name: str, seed: int = 0):
    """Factory for LM-09 codec names; returns None for non-LM-09 names."""
    m = _RE.fullmatch(name)
    if not m:
        return None
    kb, kmode, gk, rk, vb, gv, rv, clip = m.groups()
    return LM09KV(int(kb), kmode, int(gk), int(rk), int(vb), int(gv), int(rv),
                  "clip" if clip else "rtn")


def kv_bytes_per_tok_layer(kb, kmode, gk, vb, gv):
    """Packed bytes per token per layer (8 heads x 128 dims), codes+meta."""
    meta_k = 4 * 128 / gk if kmode == "c" else 4 * 128 / gk   # fp16 s+z per group
    per_head_k = 128 * kb / 8 + meta_k
    per_head_v = 128 * vb / 8 + 4 * 128 / gv
    return 8 * (per_head_k + per_head_v)
