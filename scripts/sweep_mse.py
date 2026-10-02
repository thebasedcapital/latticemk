"""Relative weight MSE (dB) vs bits/weight for INT RTN, A_n truncated lattices, and Lloyd VQ tables.

Sample: layers 0, 13, 27 (all 7 projections) + first 16384 rows of lm_head. Energy-weighted across matrices.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import codebooks, quant
from lmk.model import PROJ, load

names = [f"model.layers.{i}.{p}.weight" for i in (0, 13, 27) for p in PROJ] + ["lm_head.weight"]
mats = {n: load(n) for n in names}
mats["lm_head.weight"] = mats["lm_head.weight"][:16384].contiguous()

configs = [(f"int{b}", "int", b) for b in (2, 3, 4)]
for n, k in [(2, 4), (2, 5), (4, 8), (4, 9), (4, 10), (4, 11)]:
    configs.append((f"A{n}-k{k}", "vq", torch.from_numpy(codebooks.an_codebook(n, k)).cuda()))
for d, k in [(2, 5), (4, 9), (4, 10)]:
    configs.append((f"lloyd{d}-k{k}", "vq", torch.from_numpy(codebooks.lloyd_codebook(d, k)).cuda()))

print(f"{'config':<14}{'bits':>7}{'SQNR raw dB':>13}{'SQNR RHT dB':>13}")
for name, kind, arg in configs:
    row = []
    for rht in (False, True):
        err = tot = 0.0
        for i, (mn, w) in enumerate(mats.items()):
            wq = quant.rht_apply(w, quant.rht_signs(w.shape[1], i)) if rht else w
            q = quant.quant_int(wq, arg) if kind == "int" else quant.quant_vq(wq, arg)
            err += (q.deq - wq).pow(2).sum().item()
            tot += wq.pow(2).sum().item()
        row.append(10 * math.log10(tot / err))
    bits = quant.int_bits(arg) if kind == "int" else quant.vq_bits(arg)
    print(f"{name:<14}{bits:>7.3f}{row[0]:>13.2f}{row[1]:>13.2f}", flush=True)
