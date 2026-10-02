"""LM-09 codec eval: runs kvcodec.eval on lm09-* codec names (patches
kvcodec.eval.make), writes rows to bench/lm09/codec_results.jsonl.

usage:
  scripts/gpu.sh .venv/bin/python bench/lm09/eval_codec.py \
      --codecs lm09-k4c64r64-v4t128r0 --parts sqnr,attn,ppl \
      --ppl-windows 0:48
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import kvcodec.codec_lm09 as c9  # noqa: E402
import kvcodec.eval as kev  # noqa: E402

OUT = Path(__file__).resolve().parent / "codec_results.jsonl"

_orig_make = kev.make


def make(name: str, seed: int = 0):
    c = c9.make_lm09(name, seed)
    return c if c is not None else _orig_make(name, seed)


kev.make = make  # patch module-global used by eval_sqnr/eval_attn/eval_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", nargs="*", required=True)
    ap.add_argument("--parts", default="sqnr,attn,ppl")
    ap.add_argument("--ppl-windows", default="0:146")
    args = ap.parse_args()
    w0, w1 = (int(t) for t in args.ppl_windows.split(":"))
    parts = set(args.parts.split(","))

    Kc = Vc = None
    if "sqnr" in parts:
        from kvcodec.common import DATA
        K = torch.load(DATA / "k.pt", map_location="cpu").float()
        V = torch.load(DATA / "v.pt", map_location="cpu").float()
        W, L, T, H, D = K.shape
        Kc = K.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)
        Vc = V.permute(1, 0, 2, 3, 4).reshape(L, -1, H, D)

    hw = kev.hw_state()
    for name in args.codecs:
        row = {"codec": name, "script": "bench/lm09/eval_codec.py",
               "measured": True, **hw}
        try:
            if "sqnr" in parts:
                row.update(kev.eval_sqnr(name, Kc, Vc))
            if "attn" in parts:
                row["attn_rel_err"] = kev.eval_attn(name)
            if "ppl" in parts:
                row.update(kev.eval_ppl(name, w0, w1))
        except NotImplementedError as e:
            row["error"] = str(e)
        print(json.dumps(row), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
