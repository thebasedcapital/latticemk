"""LM-07 evaluation harness: SQNR + bits, wikitext-2 perplexity with fake-quantized KV cache,
attention-output error vs fp16 at context 2048. One row per codec appended to results.jsonl.

usage (one GPU job per codec or codec pair to keep jobs short):
  scripts/gpu.sh .venv/bin/python -m kvcodec.eval --codecs int4 int8 kivi-int4 A4-k9+rht
  scripts/gpu.sh .venv/bin/python -m kvcodec.eval --codecs fp16 --parts ppl --ppl-windows 0:40
  .venv/bin/python -m kvcodec.eval --merge      # collapse chunked rows -> one row per codec

parts: sqnr (captured sample), attn (error vs fp16 @ ctx 2048), ppl (fake-quantized cache writes;
query and current-token K/V stay exact — see common.kv_attn). Perplexity chunks via
--ppl-windows A:B; `nll_sum`/`nll_tok` are merged exactly by --merge (`ppl` in a partial row is
the partial-window value).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk.model import SNAPSHOT  # noqa: E402

from .codec import make  # noqa: E402
from .common import (DATA, SEQ, kv_attn, patch_attention, pooled_sqnr,  # noqa: E402
                     unheadify, wiki_test_ids)

RESULTS = Path(__file__).resolve().parent / "results.jsonl"
N_WIN_TEST = 146  # wikitext-2 test: 146 x 2048 tokens (same convention as scripts/eval_ppl.py)
SCALING = 0.08838834764831845  # head_dim**-0.5


def hw_state() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,driver_version",
             "--format=csv,noheader"], text=True).strip()
        clk, drv = out.split(", ")
        return {"clocks_sm": clk, "driver": drv}
    except Exception:
        return {"clocks_sm": "?", "driver": "?"}


def eval_sqnr(name: str, Kc, Vc) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    row, bits = {}, {}
    for kind, x in (("k", Kc), ("v", Vc)):
        row[f"sqnr_db_{kind}"] = pooled_sqnr(lambda s: make(name, s), x, kind, dev)
        c = make(name, 0)
        p = c.encode(x[0].reshape(-1, x.shape[2], x.shape[3]).to(dev).float(), kind)
        bits[kind] = p.bits / p.n
    row["bits_per_coord_k"], row["bits_per_coord_v"] = bits["k"], bits["v"]
    row["bits_per_coord"] = (bits["k"] + bits["v"]) / 2
    return row


def eval_attn(name: str) -> float:
    """RMS-relative attention output error, exact KV vs fake-quantized KV, context 2048.

    Same diagonal-exact semantics as the ppl run (common.kv_attn). Pooled over all 28 layers on
    captured window 0."""
    K = torch.load(DATA / "k.pt", map_location="cpu")[0].float()  # [28,T,8,128]
    V = torch.load(DATA / "v.pt", map_location="cpu")[0].float()
    Q = torch.load(DATA / "q.pt", map_location="cpu").float()
    mod = SimpleNamespace(num_key_value_groups=2)
    num = den = 0.0
    for l in range(28):
        c = make(name, l)
        q = unheadify(Q[l]).cuda()
        k, v = unheadify(K[l]).cuda(), unheadify(V[l]).cuda()
        kq = unheadify(c.decode(c.encode(K[l].cuda(), "k")))
        vq = unheadify(c.decode(c.encode(V[l].cuda(), "v")))
        o_ref, _ = kv_attn(mod, q, k, k, v, v, None, SCALING)
        o_q, _ = kv_attn(mod, q, k, kq, v, vq, None, SCALING)
        num += float((o_q - o_ref).double().pow(2).sum())
        den += float(o_ref.double().pow(2).sum())
    return math.sqrt(num / den)


@torch.no_grad()
def eval_ppl(name: str, w0: int, w1: int) -> dict:
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids = wiki_test_ids(tok)
    n_win = len(ids) // SEQ
    w1 = min(w1, n_win)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.float32, attn_implementation="eager").cuda().eval()

    # patch every codec incl. fp16: all rows share the eager + diagonal-exact semantics
    for l, am in enumerate(model.model.layers):
        orig = am.self_attn.forward
        am.self_attn.forward = patch_attention(
            am.self_attn, orig, state=None, codec=make(name, l))

    nll, t0 = 0.0, time.time()
    for j in range(w0, w1):
        x = ids[j * SEQ:(j + 1) * SEQ].cuda()
        h = model.model(x.unsqueeze(0)).last_hidden_state[0]
        for s in range(0, SEQ - 1, 512):
            e = min(s + 512, SEQ - 1)
            nll += F.cross_entropy(model.lm_head(h[s:e]), x[s + 1:e + 1],
                                   reduction="sum").item()
        if (j - w0) % 10 == 0:
            print(f"  {name} win {j}/{w1}  {time.time() - t0:.0f}s", flush=True)
    n_tok = (w1 - w0) * (SEQ - 1)
    return {"ppl": math.exp(nll / n_tok), "nll_sum": nll, "nll_tok": n_tok,
            "ppl_windows": f"{w0}:{w1}", "ppl_s": time.time() - t0}


def merge_results():
    """Rewrite results.jsonl as one row per codec: fields unioned (later wins).

    ppl chunks merge by nll_sum, but overlapping window ranges are deduplicated by keeping only
    the FIRST row covering each window interval (assumes chunks are written disjoint or that an
    earlier chunk is superseded by a later full-range one — we keep the largest coverage)."""
    rows = {}
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cur = rows.setdefault(r["codec"], {"_chunks": []})
        if "nll_sum" in r:
            locked = "+" in r["ppl_windows"]  # already-merged row: keep nll as-is
            cur["_chunks"].append((r["nll_sum"], r["nll_tok"], r["ppl_windows"], locked))
            r = {k: v for k, v in r.items()
                 if k not in ("ppl", "nll_sum", "nll_tok", "ppl_windows")}
        cur.update(r)
    for r in rows.values():
        chunks = r.pop("_chunks")
        if not chunks:
            continue
        def span(c):
            a, b = c[2].split("+")[0].split(":")
            return int(b) - int(a)
        locked = [c for c in chunks if c[3]]
        free = [c for c in chunks if not c[3]]
        if locked:  # merged row is authoritative; drop superseded raw chunks
            keep = locked
        else:
            best = max(free, key=span)
            keep = [c for c in free if not (
                c is not best and c[2] in _subsumed(best[2], [x[2] for x in free]))]
        nll = sum(c[0] for c in keep)
        ntok = sum(c[1] for c in keep)
        r["nll_sum"], r["nll_tok"] = nll, ntok
        r["ppl"] = math.exp(nll / ntok)
        r["ppl_windows"] = "+".join(c[2] for c in keep)
    with RESULTS.open("w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
    print(f"merged {len(rows)} codec rows -> {RESULTS}")


def _subsumed(big: str, others):
    """Return other ranges fully inside big's [a:b)."""
    a, b = (int(t) for t in big.split(":"))
    out = []
    for o in others:
        x, y = (int(t) for t in o.split(":"))
        if o != big and a <= x and y <= b:
            out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", nargs="*", default=[])
    ap.add_argument("--parts", default="sqnr,attn,ppl")
    ap.add_argument("--ppl-windows", default=f"0:{N_WIN_TEST}")
    ap.add_argument("--merge", action="store_true",
                    help="merge chunked rows into one row per codec and exit")
    args = ap.parse_args()
    if args.merge:
        merge_results()
        return
    w0, w1 = (int(t) for t in args.ppl_windows.split(":"))
    parts = set(args.parts.split(","))

    Kc = Vc = None
    if "sqnr" in parts:
        K = torch.load(DATA / "k.pt", map_location="cpu").float()
        V = torch.load(DATA / "v.pt", map_location="cpu").float()
        W, L, T, H, D = K.shape
        Kc = K.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)
        Vc = V.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)

    hw = hw_state()
    for name in args.codecs:
        row = {"codec": name, "script": "kvcodec/eval.py", "measured": True, **hw}
        try:
            if "sqnr" in parts:
                row.update(eval_sqnr(name, Kc, Vc))
            if "attn" in parts:
                row["attn_rel_err"] = eval_attn(name)
            if "ppl" in parts:
                row.update(eval_ppl(name, w0, w1))
        except NotImplementedError as e:
            row["error"] = str(e)
        print(json.dumps(row), flush=True)
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
