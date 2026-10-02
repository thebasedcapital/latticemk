"""Batch-1 decode matvec benchmark on real Qwen3-0.6B-Base weights.

Per token: 28 x [qkv 4096x1024, o 1024x2048, gate_up 6144x1024, down 1024x3072] + lm_head 151936x1024,
captured as one CUDA graph (113 kernels). Every format is checked against a torch reference first.
usage: bench_gemv.py [CONFIG ...]   (default: all)
"""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # several configs stay resident
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import codebooks, pack, quant
from lmk.model import N_LAYERS, load

ROUNDS = 7
CHECK_ROWS = 16384

CONFIGS = {  # timing does not depend on +gptq (same codes/layout), only on format and fused RHT
    "f16": None,
    "int4": ("int", 4, False),
    "int4+rht": ("int", 4, True),
    "A2-k4": ("A", 2, 4, False),
    "A2-k5": ("A", 2, 5, False),
    "A2-k5+rht": ("A", 2, 5, True),
    "A4-k8": ("A", 4, 8, False),
    "A4-k9": ("A", 4, 9, False),
    "A4-k9+rht": ("A", 4, 9, True),
    "A4-k10": ("A", 4, 10, False),
    "A4-k10+rht": ("A", 4, 10, True),
    "lloyd4-k10": ("lloyd", 4, 10, False),
}


def matrices():
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        yield f"L{i}.qkv", lambda p=p: torch.cat([load(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
        yield f"L{i}.o", lambda p=p: load(p + "self_attn.o_proj.weight")
        yield f"L{i}.gate_up", lambda p=p: torch.cat([load(p + "mlp.gate_proj.weight"), load(p + "mlp.up_proj.weight")])
        yield f"L{i}.down", lambda p=p: load(p + "mlp.down_proj.weight")
    yield "lm_head", lambda: load("lm_head.weight")


def build(cfg):
    """-> list of (name, Packed, deq-or-None); deq kept only for the correctness sample."""
    spec = CONFIGS[cfg]
    cb = None
    if spec and spec[0] != "int":
        make = codebooks.an_codebook if spec[0] == "A" else codebooks.lloyd_codebook
        cb = torch.from_numpy(make(spec[1], spec[2])).cuda()
    rht = bool(spec and spec[-1])
    out = []
    for j, (name, get) in enumerate(matrices()):
        w = get()
        check = name.startswith("L0.") or name == "lm_head"
        ref = lambda deq: deq[:CHECK_ROWS].clone() if check else None  # full fp32 lm_head ref would cost 622 MB
        if spec is None:
            out.append((name, pack.pack_f16(w), ref(w.half().float())))
            continue
        signs = quant.rht_signs(w.shape[1], j) if rht else None
        if rht:
            w = quant.rht_apply(w, signs)
        if spec[0] == "int":
            q = quant.quant_int(w, spec[1])
            p = pack.pack_int4(q, signs)
        else:
            q = quant.quant_vq(w, cb)
            p = pack.pack_vq(q, cb, signs)
        out.append((name, p, ref(q.deq)))
        del w, q
    torch.cuda.empty_cache()
    return out


def main():
    cfgs = sys.argv[1:] or list(CONFIGS)
    g = torch.Generator(device="cuda").manual_seed(0)
    xs = {n: torch.randn(n, device="cuda", generator=g).half() for n in (1024, 2048, 3072)}
    runs = {}
    for cfg in cfgs:  # build + verify, then keep only packed weights
        mats = build(cfg)
        ys = [torch.empty(p.out, dtype=torch.half, device="cuda") for _, p, _ in mats]
        descs = [p.desc(xs[p.n_in], y) for (_, p, _), y in zip(mats, ys)]
        pack.run(descs)
        err = 0.0
        for (_, p, deq), y in zip(mats, ys):
            if deq is not None:
                x = quant.rht_x(xs[p.n_in].float(), p.signs) if p.signs is not None else xs[p.n_in].float()
                ref = deq @ x
                err = max(err, ((y[: len(ref)].float() - ref).norm() / ref.norm()).item())
        packed = [p for _, p, _ in mats]
        del mats
        torch.cuda.empty_cache()
        runs[cfg] = dict(packed=packed, ys=ys, descs=descs, err=err, t=[], lm=[], blk=[])
    roofs = []
    for _ in range(ROUNDS):  # interleave configs so clock/thermal drift hits all of them alike
        roofs.append(pack.read_gbps())
        for r in runs.values():
            r["t"].append(pack.time_ms(r["descs"]))
            r["lm"].append(pack.time_ms(r["descs"][-1:]))
            r["blk"].append(pack.time_ms(r["descs"][:-1]))
    roof = statistics.median(roofs)
    print(f"read roofline (256 MiB, uint4 loads): median {roof:.1f} GB/s, range {min(roofs):.1f}-{max(roofs):.1f}"
          f"  [{ROUNDS} interleaved rounds]")
    print(f"{'config':<12}{'bits/w':>7}{'MB/tok':>8}{'rel err':>9}{'tok ms':>8}{'(min-max)':>14}{'GB/s':>7}{'%roof':>6}"
          f"{'tok/s':>7}{'lm_head %roof':>14}{'blocks %roof':>13}")
    for cfg, r in runs.items():
        packed = r["packed"]
        nbytes = sum(p.nbytes for p in packed)
        nparams = sum(p.out * p.n_in for p in packed)
        lm_bytes = packed[-1].nbytes
        ms, lm_ms, blk_ms = (statistics.median(r[k]) for k in ("t", "lm", "blk"))
        gbps = nbytes / ms / 1e6
        print(f"{cfg:<12}{8 * nbytes / nparams:>7.3f}{nbytes / 1e6:>8.1f}{r['err']:>9.1e}{ms:>8.3f}"
              f"{f'({min(r['t']):.3f}-{max(r['t']):.3f})':>14}{gbps:>7.1f}{100 * gbps / roof:>6.1f}{1e3 / ms:>7.0f}"
              f"{100 * lm_bytes / lm_ms / 1e6 / roof:>14.1f}{100 * (nbytes - lm_bytes) / blk_ms / 1e6 / roof:>13.1f}")


if __name__ == "__main__":
    main()
