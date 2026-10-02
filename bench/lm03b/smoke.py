"""LM-03b smoke: v2 megakernel vs LM-03 graph engine, same packed weights.
Prefill 8 tokens (teacher-forced), decode 16; compare tokens + max|logit|.

usage: scripts/gpu.sh .venv/bin/python bench/lm03b/smoke.py [steps]
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402

NPRE, NDEC = 8, int(sys.argv[1]) if len(sys.argv) > 1 else 16
CTX_CAP = 256


def main():
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()

    e1 = lm03.Engine(CTX_CAP, packed, emb, norms, rope)
    e1.graph_build()
    e2 = lm03b.Engine2(CTX_CAP, packed, emb, norms, rope)

    ids = [9707, 11, 1879, 330, 1207, 274, 318, 1378]  # fixed fake prompt
    e1.pos_set(0)
    for t in ids:
        e1.set_tok(t)
        e1.graph_launch()
    e1.pos0 = len(ids)

    e2.pos_set(0)
    for t in ids:
        e2.set_tok(t)
        e2.mega(1)
    e2.pos0 = len(ids)

    lg1, lg2 = [], []
    t1, t2 = [], []
    for s in range(NDEC):
        e1.graph_launch()
        lg1.append(e1.logits().cpu())
        t1.append(int(e1.bufs["tok"][0].item()))
    e2.mega(NDEC)
    # mega2 only exposes final logits + hist; re-run stepwise for logits
    e2.pos_set(0)
    for t in ids:
        e2.set_tok(t)
        e2.mega(1)
    e2.pos0 = len(ids)
    for s in range(NDEC):
        e2.mega(1)
        lg2.append(e2.logits().cpu())
        t2.append(int(e2.bufs["tok"][0].item()))

    md = max((a - b).abs().max().item() for a, b in zip(lg1, lg2))
    match = sum(int(a == b) for a, b in zip(t1, t2))
    print(f"tokens match {match}/{NDEC}; max|dlogit| = {md:.4f}")
    print("graph:", t1)
    print("mega2:", t2)


if __name__ == "__main__":
    main()
