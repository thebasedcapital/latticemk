"""Lower bound on bits/weight any codec needs to match a target SQNR on Qwen3-0.6B weights.

Shannon lower bound for a memoryless source: R(D) >= h(X) - 1/2 log2(2 pi e D). X = weights divided by their
group-128 RMS (the per-group scale is side information, counted separately as 16/128 bits). h(X) is estimated
with a fine histogram over the same 22-matrix sample as sweep_mse.py. Assumes weights are i.i.d. given the scale;
cross-weight dependence could lower the true bound (see report).
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import quant
from lmk.model import PROJ, load

names = [f"model.layers.{i}.{p}.weight" for i in (0, 13, 27) for p in PROJ] + ["lm_head.weight"]


def normalized(rht: bool) -> torch.Tensor:
    xs = []
    for i, n in enumerate(names):
        w = load(n)
        if n == "lm_head.weight":
            w = w[:16384].contiguous()
        if rht:
            w = quant.rht_apply(w, quant.rht_signs(w.shape[1], i))
        g = w.reshape(-1, quant.GROUP)
        xs.append((g / g.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)).flatten())
    return torch.cat(xs)


def diff_entropy_bits(x: torch.Tensor, bins: int = 1 << 14) -> float:
    lo, hi = x.min().item(), x.max().item()
    h = torch.histc(x, bins=bins, min=lo, max=hi).double()
    p = h / h.sum()
    width = (hi - lo) / bins
    p = p[p > 0]
    return float(-(p * p.log2()).sum()) + math.log2(width)


gauss = 0.5 * math.log2(2 * math.pi * math.e)
print(f"Gaussian h(X) for unit variance: {gauss:.4f} bits")
for rht in (False, True):
    x = normalized(rht)
    h = diff_entropy_bits(x)
    kurt = (x.pow(4).mean() / x.pow(2).mean().pow(2)).item() - 3
    print(f"rht={rht}: n={x.numel()}  h(X)={h:.4f} bits  (Gaussian gap {gauss - h:.4f})  excess kurtosis {kurt:.3f}")
    for label, sqnr in (("int4 RTN g128", 20.27 if not rht else 20.41), ("int3 RTN g128", 14.45 if not rht else 14.58)):
        d = 10 ** (-sqnr / 10)  # distortion relative to unit variance
        r = h - 0.5 * math.log2(2 * math.pi * math.e * d)
        print(f"   match {label} ({sqnr:.2f} dB): R >= {r:.3f} bits/weight + 0.125 scale = {r + 0.125:.3f}")
