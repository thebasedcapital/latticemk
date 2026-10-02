"""KV statistics + Shannon lower bound (LM-07 task 2), on kvcodec/data/{k,v}.pt.

Per layer/head (28 x 8): per-channel scale spread, outlier channels (max/median channel RMS),
excess kurtosis of scale-normalized values. Then the Shannon lower bound on bits/coordinate to
match the measured INT4 SQNR, under three scaling regimes:
  - pertok:   each head vector RMS-normalized (group = 128 dims)
  - perchan:  each channel RMS-normalized over tokens
  - rht:      per-token after a Hadamard rotation of the head vector
Method = scripts/rate_bound.py: histogram differential entropy; R(D) >= h - 1/2 log2(2 pi e D)
with D = 10^(-SQNR/10) relative to unit variance. fp16 scale overhead +16/128 = 0.125 b/coord
(per-token group) or +32/128 = 0.25 (per-channel, 128-token block: scale + zero, INT).

usage: .venv/bin/python -m kvcodec.stats          (needs kvcodec/data/{k,v}.pt)
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import quant  # noqa: E402

from .codec import IntChannel, IntToken  # noqa: E402
from .common import DATA, pooled_sqnr  # noqa: E402

GAUSS_H = 0.5 * math.log2(2 * math.pi * math.e)


def diff_entropy_bits(x: torch.Tensor, bins: int = 1 << 14) -> float:
    lo, hi = float(x.min()), float(x.max())
    h = torch.histc(x.float().cpu(), bins=bins, min=lo, max=hi).double()
    p = h / h.sum()
    width = (hi - lo) / bins
    p = p[p > 0]
    return float(-(p * p.log2()).sum()) + math.log2(width)


def kurtosis(x: torch.Tensor) -> float:
    x = x.float()
    return float(x.pow(4).mean() / x.pow(2).mean().pow(2)) - 3.0


def norm_token(x):  # [T,H,D] -> unit-RMS per head vector
    g = x.float().reshape(-1, x.shape[-1])
    return g / g.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)


def norm_channel(x):  # [T,H,D] -> unit-RMS per channel over tokens
    x = x.float()
    return x / x.pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-12)


def norm_rht(x, seed=0):
    T, H, D = x.shape
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (H, D), generator=g) * 2 - 1).float()
    had = quant.hadamard(D, "cpu")
    xr = (x.float() * s.unsqueeze(0)) @ had
    return norm_token(xr)


def per_head_stats(x):  # x: [L, T, H, D] -> dict
    rms = x.float().pow(2).mean(1).sqrt()          # [L,H,D] per-channel RMS over tokens
    r = rms.reshape(-1, rms.shape[-1])             # [L*H, D]
    stats = {
        "chan_rms_max_over_med": float(r.max() / r.median()),
        "chan_spread_p99_over_p50": float(
            torch.quantile(r.reshape(-1), 0.99) / torch.quantile(r.reshape(-1), 0.50)),
    }
    per_lh = r.max(1).values / r.median(1).values  # outlier ratio per (layer, head)
    stats["outlier_ratio_per_head_min_med_max"] = [
        float(per_lh.min()), float(per_lh.median()), float(per_lh.max())]
    flat = per_lh.flatten()
    stats["worst_heads_L_H_ratio"] = [
        (int(i // 8), int(i % 8), float(flat[i])) for i in flat.topk(3).indices]
    return stats


def main():
    K = torch.load(DATA / "k.pt", map_location="cpu").float()  # [W,28,T,8,128]
    V = torch.load(DATA / "v.pt", map_location="cpu").float()
    W, L, T, H, D = K.shape
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # -> [L, W*T, H, D]
    Kc = K.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)
    Vc = V.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)

    out = {"sample": {"windows": W, "layers": L, "tokens_per_layer": W * T,
                      "heads": H, "dim": D, "coords": L * W * T * H * D}}

    # SQNR of the INT4 baselines = the distortion targets for the bound
    tok4, ch4 = IntToken(4), IntChannel(4)
    sqnr = {}
    for nm, x in (("k", Kc), ("v", Vc)):
        sqnr[f"{nm}_int4_pertok"] = pooled_sqnr(lambda s: tok4, x, nm, dev)
        sqnr[f"{nm}_int4_perchan"] = pooled_sqnr(lambda s: ch4, x, nm, dev)
    out["int4_sqnr_db"] = sqnr

    # per layer/head geometry
    for nm, x in (("k", Kc), ("v", Vc)):
        out[f"{nm}_per_head"] = per_head_stats(x)

    # entropy + kurtosis + SLB under each scaling (per-layer, then averaged)
    norms = {"pertok": norm_token, "perchan": norm_channel, "rht": norm_rht}
    ent, slb = {}, {}
    for nm, x in (("k", Kc), ("v", Vc)):
        ent[nm], slb[nm] = {}, {}
        for tag, fn in norms.items():
            xs = [fn(x[l]) for l in range(L)]
            h = sum(diff_entropy_bits(t.flatten()) for t in xs) / L
            kt = sum(kurtosis(t) for t in xs) / L
            ent[nm][tag] = {"h_bits": h, "excess_kurt": kt, "h_gauss_gap": GAUSS_H - h}
            tgt = sqnr[f"{nm}_int4_{'perchan' if tag == 'perchan' else 'pertok'}"]
            d = 10 ** (-tgt / 10)
            r = h - 0.5 * math.log2(2 * math.pi * math.e * d)
            slb[nm][tag] = {"target_sqnr_db": tgt, "slb_bits_excl_meta": r,
                            "slb_bits": r + (0.25 if tag == "perchan" else 0.125)}
    out["entropy"] = ent
    out["slb_at_int4_sqnr"] = slb

    dest = Path(__file__).resolve().parent / "stats.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
