"""Correctness gate for kernels/gemv.cu: every format x input width (NC 1/2/3) x fused RHT on/off, including a
row count that is not a multiple of the warp tiling, against a torch fp32 reference. Exit code 1 on failure."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import codebooks, pack, quant

TOL = 5e-3  # fp16 activations/LUT/partial sums; observed worst case ~1.3e-3

torch.manual_seed(0)
cbs = {(d, k): torch.from_numpy(codebooks.an_codebook(d, k)).cuda() for d, k in [(4, 8), (4, 9), (4, 10), (2, 4), (2, 5)]}
worst, failed = 0.0, False
for n_in in (1024, 2048, 3072):
    for out in (1000, 4096):
        w = torch.randn(out, n_in, device="cuda") * 0.02
        x = torch.randn(n_in, device="cuda").half()
        y = torch.empty(out, dtype=torch.half, device="cuda")
        for rht in (False, True):
            s = quant.rht_signs(n_in, 7) if rht else None
            wr = quant.rht_apply(w, s) if rht else w
            xr = quant.rht_x(x.float(), s) if rht else x.float()
            cases = [] if rht else [("f16", pack.pack_f16(wr), wr.half().float())]
            q = quant.quant_int(wr, 4)
            cases.append(("int4", pack.pack_int4(q, s), q.deq))
            for (d, k), cb in cbs.items():
                q = quant.quant_vq(wr, cb)
                cases.append((f"A{d}-k{k}", pack.pack_vq(q, cb, s), q.deq))
            for name, p, deq in cases:
                pack.run([p.desc(x, y)])
                ref = deq @ xr
                e = ((y.float() - ref).norm() / ref.norm()).item()
                worst = max(worst, e)
                if e > TOL:
                    failed = True
                    print(f"FAIL {name} out={out} in={n_in} rht={rht} rel_err={e:.2e}")
print(f"{'FAILED' if failed else 'ok'}: worst rel err {worst:.2e}")
sys.exit(1 if failed else 0)
