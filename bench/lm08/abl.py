"""LM-08 ablation/tuning: per-kernel fusion costs and nslice sweep.

  abl.py gemv      -- fused vs plain prologue/epilogue per GEMV kind
  abl.py attn      -- attention kernel time vs pos, prep/split_attnc, nslice
  abl.py graph     -- whole-step graph time under attention ablations, ctx 128
"""

import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03
import lm08


def med(f, n=15):
    return statistics.median(f() for _ in range(n))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    packed, _ = lm03.pack_weights(cache=lm08.CACHE, want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()

    if mode in ("gemv", "all"):
        eng = lm08.Engine(128, packed, emb, norms, rope)
        eng.pos_set(64)
        eng.set_tok(9707)
        names = ["qkv(norm)", "o(res)", "gu(norm)", "down(silu+res)",
                 "lm_head(norm+amax)"]
        for kind in range(5):
            fused = med(lambda: lm08._lib.fx_time_gemv(kind, 0, 20))
            plain = med(lambda: lm08._lib.fx_time_gemv(kind, 1, 20))
            print(f"gemv {names[kind]:20s} fused {fused:.4f} ms | "
                  f"plain {plain:.4f} ms | fusion cost "
                  f"{1000 * (fused - plain):+.1f} us", flush=True)
        del eng
        torch.cuda.empty_cache()

    if mode in ("attn", "all"):
        for ctx, nsl in [(128, 72), (128, 36), (2048, 72), (2048, 144),
                         (8192, 72), (8192, 144), (8192, 288)]:
            if nsl > lm08.NSLICE_MAX:
                continue
            eng = lm08.Engine(ctx, packed, emb, norms, rope, nslice=nsl)
            eng.pos_set(ctx)
            for prep, sc, tag in [(1, 0, "prep+incomb"), (1, 1, "prep+sep"),
                                  (0, 0, "qkra+incomb"), (0, 1, "qkra+sep")]:
                eng.set_attn(prep, sc)
                t = med(lambda: lm08._lib.fx_time_attn(3, 15))
                print(f"attn ctx {ctx:5d} nsl {nsl:3d} {tag:12s} "
                      f"{t:.4f} ms", flush=True)
            del eng
            torch.cuda.empty_cache()

    if mode in ("graph", "all"):
        eng = lm08.Engine(128, packed, emb, norms, rope)
        out = lm08._f32([0.0] * 3)
        for prep, sc, tag in [(1, 0, "fused-attn"), (0, 1, "qkra+attnc"),
                              (1, 1, "prep+sep-attnc")]:
            eng.set_attn(prep, sc)
            eng.graph_build()
            o = lm08._f32([0.0])
            lm08._lib.fx_time_graph(5, 128, o)
            ms = []
            for _ in range(15):
                lm08._lib.fx_time_graph(1, 128, o)
                ms.append(o[0])
            print(f"graph ctx 128 {tag:14s} {statistics.median(ms):.4f} ms",
                  flush=True)


if __name__ == "__main__":
    main()
