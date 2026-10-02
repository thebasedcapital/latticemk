"""Interleave RTN and GPTQ in the same v2 kernel at one context.

Run: scripts/gpu.sh --timing .venv/bin/python bench/lm11/compare_gptq.py 128
"""
import argparse
import ctypes
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "bench/lm03"),
                str(ROOT / "bench/lm03b"), str(Path(__file__).parent)]
import lm03
import lm03b
from interleave import OUT, row


def load(path):
    return {k: {f: t.cuda() for f, t in v.items()}
            for k, v in torch.load(path, map_location="cpu").items()}


def bind(eng, emb, norms, rope):
    """Engine2's library has a single active descriptor; select the engine."""
    order = [eng.bufs[k].data_ptr() for k in
             ("xn", "qkv", "oo", "gu", "dout", "kc", "vc", "part",
              "logits", "argp", "pos", "bar", "xpad")]
    rc = lm03b._lib2.mk2_init(lm03._i64(order),
                              lm03._i64([t.data_ptr() for t in eng.codes]),
                              lm03._i64([t.data_ptr() for t in eng.metas]),
                              emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                              eng.bufs["tok"].data_ptr(),
                              eng.bufs["tok_hist"].data_ptr())
    assert rc == 0, f"mk2_init: {rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int)
    ap.add_argument("--runs", type=int, default=25)
    a = ap.parse_args()
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    paths = [ROOT / "bench/lm03b/weights_int4.pt",
             ROOT / "bench/lm11/weights_int4_gptq.pt"]
    engines = [lm03b.Engine2(a.ctx, load(p), emb, norms, rope) for p in paths]
    ms = [[], []]
    out = lm03._f32([0.0])
    for r in range(a.runs + 3):
        for j in (r % 2, 1 - r % 2):
            e = engines[j]
            bind(e, emb, norms, rope)
            e.set_tok(9707)
            assert lm03b._lib2.mk2_time_mega(8, 1, a.ctx, out) == 0
            if r >= 3:
                ms[j].append(float(out[0]))
    kv = 2 * lm03.N_LAYERS * a.ctx * lm03.KVROWS * 2
    rows = []
    for j, e in enumerate(engines):
        wb = sum(t.numel() * t.element_size() for t in e.codes + e.metas)
        rows.append(row("megakernel-v2-" + ("rtn" if j == 0 else "gptq"),
                        a.ctx, ms[j], wb, kv,
                        {"steps_per_launch": 8, "timing": "interleaved"}))
    with OUT.open("a") as f:
        for rr in rows:
            f.write(json.dumps(rr) + "\n")
    print("RTN", statistics.median(ms[0]), "GPTQ", statistics.median(ms[1]),
          "ratio RTN/GPTQ", statistics.median(ms[0]) / statistics.median(ms[1]))


if __name__ == "__main__":
    main()
