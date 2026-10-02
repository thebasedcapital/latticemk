"""LM-11: GPTQ (block LDLQ, d=1) INT4-g128 weights in the bench/lm03b packed format.

Copied driver logic from lmk/gptq.py (_ldlq/gptq_int) with instrumentation that
captures the per-group fp16 (scale, offset) chosen by the error-updated group
pass, so the packed codes + meta reconstruct the SAME deq weights byte-for-byte
as gptq_int returns (w = code*scale + off).

Hessians: lmk.gptq.calib_hessians on CALIB x 2048 wikitext-2 train windows —
identical to scripts/eval_ppl.py int4+gptq, so the resulting deq weights should
reproduce its 14.08 ppl.

Output: bench/lm11/weights_int4_gptq.pt = {name: {codes int32, meta fp16}}
plus bench/lm11/weights_int4_gptq_deq.pt = {name: fp32 cpu} (reference for the
HF correctness/ppl harnesses). Cached hessians in bench/lm11/hessians.pt.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(ROOT / "bench" / "lm03b"))

import lm03  # noqa: E402  (matrix_names, _load_matrix)
from lmk import gptq, pack, quant  # noqa: E402
from lmk.quant import GROUP  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "weights_int4_gptq.pt"
DEQ = HERE / "weights_int4_gptq_deq.pt"
HESS_CACHE = HERE / "hessians.pt"
BITS = 4
QMAX = (1 << BITS) - 1
SHRINKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)  # gptq_int default
DAMP = 0.01



def gptq_int_recorded(w: torch.Tensor, h: torch.Tensor):
    """Instrumented _ldlq copy: returns deq, Quantized(codes,scale,zero)."""
    w = w.clone()
    out, n = w.shape
    ng = n // GROUP
    u = gptq._prep(h, w, DAMP)
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
    names = lm03.matrix_names()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode.startswith("hess"):
        half = int(mode[4:]) if len(mode) > 4 else -1
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import pyarrow.parquet as pq
        from lmk.model import SNAPSHOT
        SEQ, CALIB = 2048, 128
        WIKI = next((Path.home() / ".cache/huggingface/hub/"
                     "datasets--Salesforce--wikitext/snapshots").iterdir())
        tok = AutoTokenizer.from_pretrained(SNAPSHOT)
        text = "\n\n".join(pq.read_table(
            WIKI / "wikitext-2-raw-v1/train-00000-of-00001.parquet")
            ["text"].to_pylist())
        ids = tok(text, return_tensors="pt").input_ids[0]
        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, len(ids) - SEQ, (CALIB,), generator=g)
        windows = torch.stack([ids[s:s + SEQ] for s in starts])
        if half >= 0:
            windows = windows[half * CALIB // 2:(half + 1) * CALIB // 2]
        model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT, dtype=torch.float32).cuda().eval()
        hess = gptq.calib_hessians(model, windows)
        # store sums (scaled by token count) so halves merge by averaging
        torch.save({k: v * windows.numel() for k, v in hess.items()},
                   HERE / f"hessians.{half}.pt" if half >= 0 else HESS_CACHE)
        print(f"wrote hessians half={half} windows={len(windows)}")
        return
    if HESS_CACHE.exists():
        hess = torch.load(HESS_CACHE, map_location="cpu")
    else:
        h0 = torch.load(HERE / "hessians.0.pt", map_location="cpu")
        h1 = torch.load(HERE / "hessians.1.pt", map_location="cpu")
        hess = {k: (h0[k] + h1[k]) / (2 * 64 * 2048) for k in h0}
        torch.save(hess, HESS_CACHE)

    packed, deq = {}, {}
    lo, hi = 0, len(names)
    if mode == "quantA":
        lo, hi = 0, 56     # layers 0..13
    elif mode == "quantB":
        lo, hi = 56, len(names)
    elif mode == "merge":
        packed = torch.load(HERE / "weights_gptq.partA.pt") | \
            torch.load(HERE / "weights_gptq.partB.pt")
        deq = torch.load(HERE / "weights_gptq_deq.partA.pt") | \
            torch.load(HERE / "weights_gptq_deq.partB.pt")
        assert set(packed) == set(names), set(names) - set(packed)
        torch.save(packed, OUT)
        torch.save(deq, DEQ)
        print(f"wrote {OUT} + {DEQ}")
        return
    for name in names[lo:hi]:
        w = lm03._load_matrix(name)          # fp32 cuda
        h = hess[hess_key_for(name)].cuda()
        d, qtz = gptq_int_recorded(w, h)
        # self-check: packed deq == LDLQ deq
        rec = qtz.codes * qtz.scale.float().repeat_interleave(GROUP, 1) \
            + qtz.zero.float().repeat_interleave(GROUP, 1)
        assert torch.equal(rec, d), f"{name}: packed deq mismatch"
        p = pack.pack_int4(qtz)
        packed[name] = dict(codes=p.codes.cpu(), meta=p.meta.cpu())
        deq[name] = d.cpu()
        print(f"{name:12s} {tuple(w.shape)}  max|deq-w| "
              f"{(d - w).abs().max().item():.4f}", flush=True)
        del w, h, d, qtz, p
        torch.cuda.empty_cache()
    if mode == "quantA":
        torch.save(packed, HERE / "weights_gptq.partA.pt")
        torch.save(deq, HERE / "weights_gptq_deq.partA.pt")
    elif mode == "quantB":
        torch.save(packed, HERE / "weights_gptq.partB.pt")
        torch.save(deq, HERE / "weights_gptq_deq.partB.pt")
    else:
        torch.save(packed, OUT)
        torch.save(deq, DEQ)
        print(f"wrote {OUT} + {DEQ}")


if __name__ == "__main__":
    main()
