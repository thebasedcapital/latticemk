"""Group-wise weight quantizers (fake-quant for quality, packed codes for kernels).

Layout convention everywhere: W is [out, in]; groups of GROUP consecutive input columns share one scale.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from lmk.codebooks import nearest as vq_nearest

GROUP = 128
RHT_BLOCK = 1024  # all Qwen3-0.6B input dims (1024, 2048, 3072) are multiples


def hadamard(n: int, device="cuda") -> torch.Tensor:
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def rht_signs(n_in: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n_in,), generator=g) * 2 - 1).float().cuda()


def rht_apply(w: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """W -> W Q^T with Q = blockdiag(H D) (orthogonal). Online the kernel consumes Q x instead of x."""
    h = hadamard(RHT_BLOCK, w.device)
    out, n_in = w.shape
    wb = w.reshape(out, n_in // RHT_BLOCK, RHT_BLOCK)
    sb = signs.reshape(n_in // RHT_BLOCK, RHT_BLOCK)
    if inverse:
        return ((wb @ h) * sb).reshape(out, n_in)
    return ((wb * sb) @ h).reshape(out, n_in)


def rht_x(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """x -> Q x, the activation side of rht_apply (what the kernel's fused FWHT prologue computes)."""
    return ((x * signs).reshape(-1, RHT_BLOCK) @ hadamard(RHT_BLOCK, x.device)).reshape(-1)


@dataclass
class Quantized:
    deq: torch.Tensor  # [out, in] float32 reconstruction (in the rotated basis if RHT was used)
    codes: torch.Tensor  # INT: uint8 levels [out, in]; VQ: int32 indices [out, in/d]
    scale: torch.Tensor  # [out, in/GROUP] float16
    zero: torch.Tensor | None  # INT only: [out, in/GROUP] float16, the dequant offset (w = q*scale + zero)


ROW_CHUNK = 16384  # bounds temporaries: lm_head (151936 rows) would otherwise not fit in 8 GB


def _by_rows(fn, w: torch.Tensor, *args) -> Quantized:
    if len(w) <= ROW_CHUNK:
        return fn(w, *args)
    parts = [fn(w[i : i + ROW_CHUNK], *args) for i in range(0, len(w), ROW_CHUNK)]
    cat = lambda f: None if getattr(parts[0], f) is None else torch.cat([getattr(p, f) for p in parts])
    return Quantized(cat("deq"), cat("codes"), cat("scale"), cat("zero"))


def quant_int(w: torch.Tensor, bits: int, shrinks=np.linspace(1.0, 0.6, 9)) -> Quantized:
    """Asymmetric RTN, per-group min/max with MSE-optimal clip shrink; integer zero-point."""
    return _by_rows(_quant_int, w, bits, shrinks)


def quant_vq(w: torch.Tensor, cb: torch.Tensor, alphas=np.linspace(0.6, 1.5, 19)) -> Quantized:
    """Per-group scale (fp16) x unit-RMS codebook; scale = rms(group) * alpha with alpha searched per group."""
    return _by_rows(_quant_vq, w, cb, alphas)


def _quant_int(w, bits, shrinks) -> Quantized:
    out, n_in = w.shape
    g = w.reshape(out, n_in // GROUP, GROUP)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    best_err = torch.full(lo.shape, float("inf"), device=w.device)
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
    return Quantized(rec.reshape(out, n_in), q.to(torch.uint8).reshape(out, n_in),
                     scale.squeeze(-1).half(), off.squeeze(-1).half())


def _quant_vq(w, cb, alphas) -> Quantized:
    out, n_in = w.shape
    d = cb.shape[1]
    g = w.reshape(out, n_in // GROUP, GROUP)
    rms = g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-10)
    best_err = torch.full(rms.shape, float("inf"), device=w.device)
    best_idx = best_scale = None
    for a in alphas:
        scale = (rms * a).half().float()
        idx = vq_nearest((g / scale).reshape(-1, d), cb).reshape(out, n_in // GROUP, GROUP // d)
        rec = cb[idx.long()].reshape(g.shape) * scale
        err = (rec - g).pow(2).sum(-1, keepdim=True)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_idx = idx if best_idx is None else torch.where(take, idx, best_idx)
        best_scale = scale if best_scale is None else torch.where(take, scale, best_scale)
    rec = (cb[best_idx.long()].reshape(g.shape) * best_scale).reshape(out, n_in)
    return Quantized(rec, best_idx.reshape(out, n_in // d), best_scale.squeeze(-1).half(), None)


def int_bits(bits: int) -> float:
    return bits + 32 / GROUP  # fp16 scale + fp16 offset per group (kernel storage)


def vq_bits(cb: torch.Tensor) -> float:
    return math.log2(cb.shape[0]) / cb.shape[1] + 16 / GROUP
