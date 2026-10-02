"""LM-12: recorded GPTQ INT4-g128, quantized one layer per GPU job.

The quantization core matches LM-11. Calibration is in calibrate.py; 128
WikiText-2 train windows of length 2048, fp32 Hessians from a fp16 model
because the 1.7B fp32 model exceeds the available GPU memory.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale
from lmk import gptq, pack, quant
from lmk.quant import GROUP  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "weights_int4_gptq.pt"
DEQ = HERE / "weights_int4_gptq_deq.pt"
BITS = 4
QMAX = (1 << BITS) - 1
SHRINKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)  # gptq_int default
DAMP = 0.01



def gptq_int_recorded(w: torch.Tensor, h: torch.Tensor, prepared=None):
    """Instrumented _ldlq copy: returns deq, Quantized(codes,scale,zero)."""
    w = w.clone()
    out, n = w.shape
    ng = n // GROUP
    u = gptq._prep(h, w, DAMP) if prepared is None else prepared
    q = torch.zeros_like(w)
    codes = torch.empty(out, n, dtype=torch.float32, device=w.device)
    scale = torch.empty(out, ng, device=w.device)
    off = torch.empty(out, ng, device=w.device)
    for g0 in range(0, n, GROUP):
        g1 = g0 + GROUP
        gi = g0 // GROUP
        wg = w[:, g0:g1]
        # --- start_group (same as gptq_int) ---
        lo, hi = wg.amin(1, keepdim=True), wg.amax(1, keepdim=True)
        best = None
        for s in SHRINKS:
            sc = ((hi - lo) * s / QMAX).clamp_min(1e-10).half().float()
            zp = torch.round(-lo * s / sc).clamp(0, QMAX)
            of = (-zp * sc).half().float()
            rec = torch.clamp(torch.round(wg / sc) + zp, 0, QMAX) * sc + of
            err = (rec - wg).pow(2).sum(1, keepdim=True)
            if best is None:
                best = [err, sc, zp, of]
            else:
                take = err < best[0]
                best = [torch.where(take, a, b)
                        for a, b in zip((err, sc, zp, of), best)]
        sc, zp, of = best[1], best[2], best[3]
        scale[:, gi:gi + 1], off[:, gi:gi + 1] = sc, of
        # --- quant blocks d=1 ---
        scaled = torch.zeros(out, GROUP, device=w.device)
        for b0 in range(g0, g1):
            b1 = b0 + 1
            wb = w[:, b0:b1]
            qb = torch.clamp(torch.round(wb / sc) + zp, 0, QMAX) * sc + of
            q[:, b0:b1] = qb
            codes[:, b0:b1] = torch.round((qb - of) / sc).clamp(0, QMAX)
            ubb_inv = torch.linalg.inv(u[b0:b1, b0:b1])
            s = (w[:, b0:b1] - qb) @ ubb_inv
            scaled[:, b0 - g0:b1 - g0] = s
            w[:, b1:g1] -= s @ u[b0:b1, b1:g1]
        w[:, g1:] -= scaled @ u[g0:g1, g1:]
    return q, quant.Quantized(deq=q, codes=codes,
                              scale=scale.half(), zero=off.half())


def hess_key_for(name: str) -> str:
    """packed name (L{i}.{qkv,o,gu,down}, lm_head) -> calib_hessians key."""
    if name == "lm_head":
        return "lm_head"
    i, kind = name[1:].split(".")
    p = f"model.layers.{i}."
    return {"qkv": p + "self_attn.q_proj", "o": p + "self_attn.o_proj",
            "gu": p + "mlp.gate_proj", "down": p + "mlp.down_proj"}[kind]


def main():
    mode = sys.argv[1]
    if mode == "merge":
        packed, deq = {}, {}
        for layer in range(scale.N_LAYERS):
            packed.update(torch.load(HERE / f"packed.{layer}.pt", map_location="cpu"))
            deq.update(torch.load(HERE / f"deq.{layer}.pt", map_location="cpu"))
        packed.update(torch.load(HERE / "packed.head.pt", map_location="cpu"))
        deq.update(torch.load(HERE / "deq.head.pt", map_location="cpu"))
        assert set(packed) == set(scale.matrix_names())
        torch.save(packed, OUT)
        torch.save(deq, DEQ)
        print(f"saved {OUT} {DEQ}", flush=True)
        return

    layer = int(mode) if mode != "head" else None
    if layer is not None:
        assert 0 <= layer < scale.N_LAYERS
        names = [f"L{layer}.{kind}" for kind in ("qkv", "o", "gu", "down")]
        lo = 4 * (layer // 4)
        hess_file = HERE / f"hessians.{lo}-{lo+4}.pt"
    else:
        names = ["lm_head"]
        hess_file = HERE / "hessians.24-28.pt"
    hess = torch.load(hess_file, map_location="cpu")
    packed, deq = {}, {}
    for name in names:
        h = hess[hess_key_for(name)].cuda()
        if name == "lm_head":
            # The tied embedding has 151936 rows. Never hold its fp32
            # source, clone, output and codes simultaneously on the GPU.
            whole = scale.load("lm_head.weight", device="cpu")
            prepared = gptq._prep(h, torch.zeros(1, whole.shape[1],
                                                device="cuda"), DAMP)
            blocks = (whole[s:s + 2048].cuda() for s in range(0, len(whole), 2048))
        else:
            prepared = None
            blocks = (scale.load_matrix(name),)
        p_codes, p_meta, d_blocks = [], [], []
        for w in blocks:
            d, qtz = gptq_int_recorded(w, h, prepared)
            rec = qtz.codes * qtz.scale.float().repeat_interleave(GROUP, 1) \
                + qtz.zero.float().repeat_interleave(GROUP, 1)
            assert torch.equal(rec, d), f"{name}: packed deq mismatch"
            p = pack.pack_int4(qtz)
            p_codes.append(p.codes.cpu())
            p_meta.append(p.meta.cpu())
            d_blocks.append(d.cpu())
            del w, d, qtz, p, rec
            torch.cuda.empty_cache()
        packed[name] = {"codes": torch.cat(p_codes), "meta": torch.cat(p_meta)}
        deq[name] = torch.cat(d_blocks)
        print(f"{name} packed {tuple(deq[name].shape)}", flush=True)
        del h, prepared, blocks, p_codes, p_meta, d_blocks
        torch.cuda.empty_cache()
    suffix = str(layer) if layer is not None else "head"
    torch.save(packed, HERE / f"packed.{suffix}.pt")
    torch.save(deq, HERE / f"deq.{suffix}.pt")
    print(f"saved {suffix}", flush=True)


if __name__ == "__main__":
    main()
