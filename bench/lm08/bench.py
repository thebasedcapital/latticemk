"""LM-08 benchmark: fused separate-kernel graph vs LM-03 separate-kernel
graph baseline, interleaved in one process (unlocked clocks make any
non-interleaved comparison meaningless).

usage: bench.py [ctx ...] [--nslice N]   default ctx: 128 2048 8192
"""

import ctypes
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03
import lm03ref
import lm08

ROOF = 406.7          # GB/s, measured read roofline (hw/hw.json)
RUNS = 25
WARMUP = 3
CTXS = [128, 2048, 8192]
NSLICE = 0   # 0 = auto per ctx
args = sys.argv[1:]
if "--nslice" in args:
    i = args.index("--nslice")
    NSLICE = int(args[i + 1])
    del args[i:i + 2]
CTXS = [int(a) for a in args] or CTXS
OUT = Path(__file__).resolve().parent / "results.jsonl"


def sm_clock():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
        return int(out.decode().strip().split()[0])
    except Exception:
        return -1


def driver():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"])
        return out.decode().strip()
    except Exception:
        return "unknown"


def weight_bytes(eng):
    return sum(t.numel() * t.element_size()
               for t in eng.codes + eng.metas)


def row(kernel, ctx, ms, wb):
    kv = 2 * lm03.N_LAYERS * ctx * lm03.KVROWS * 2   # k+v fp16 per token
    ab = 60_000  # activations/partials/logits traffic per step [derived est]
    tot = kv + wb + ab
    med, p10, p90 = (statistics.median(ms),
                     statistics.quantiles(ms, n=10)[0],
                     statistics.quantiles(ms, n=10)[8])
    tps = 1e3 / med
    return {
        "commit": "nogit", "work_package": "LM-08", "model": "qwen3-0.6b",
        "kernel": kernel, "context": ctx, "batch": 1,
        "tokens_per_s": round(tps, 2), "gbps": round(tot / med / 1e6, 1),
        "pct_roofline": round(100 * tot / med / 1e6 / ROOF, 1),
        "sm_clock_mhz": sm_clock(), "driver": driver(), "runs": len(ms),
        "median_ms": round(med, 4), "p10_ms": round(p10, 4),
        "p90_ms": round(p90, 4), "weight_bytes": wb, "kv_bytes": kv,
    }


def main():
    print("packing/loading weights (cached) ...", flush=True)
    packed, _ = lm03.pack_weights(cache=lm08.CACHE, want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    rows = []
    f32p = ctypes.POINTER(ctypes.c_float)

    for ctx in CTXS:
        eng3 = lm03ref.Engine(ctx, packed, emb, norms, rope)
        eng3.bufs["kc"].zero_()   # explicit "prefill": decode reads ctx+1 rows
        eng3.bufs["vc"].zero_()
        eng3.set_tok(9707)
        eng3.graph_build()
        eng8 = lm08.Engine(ctx, packed, emb, norms, rope, nslice=NSLICE)
        eng8.bufs["kc"].zero_()
        eng8.bufs["vc"].zero_()
        eng8.set_tok(9707)
        eng8.set_attn(1, 1, 512, 8)
        eng8.graph_build()
        wb = weight_bytes(eng3)

        out = lm03._f32([0.0])
        out2 = lm03._f32([0.0])
        lm03ref._lib.mx_time_graph(WARMUP, ctx, out)
        lm08._lib.fx_time_graph(WARMUP, ctx, out2)
        gm, fm = [], []
        for _ in range(RUNS):
            lm03ref._lib.mx_time_graph(1, ctx, out)
            gm.append(out[0])
            lm08._lib.fx_time_graph(1, ctx, out2)
            fm.append(out2[0])
        r = row("separate-graph", ctx, gm, wb)
        rows.append(r)
        print(f"ctx {ctx:5d} separate-graph  {r['tokens_per_s']:8.1f} tok/s "
              f"{r['median_ms']:.3f} ms  {r['pct_roofline']:.1f}% roof",
              flush=True)
        r2 = row("fused-graph", ctx, fm, wb)
        rows.append(r2)
        print(f"ctx {ctx:5d} fused-graph     {r2['tokens_per_s']:8.1f} tok/s "
              f"{r2['median_ms']:.3f} ms  {r2['pct_roofline']:.1f}% roof "
              f"(x{r2['tokens_per_s'] / r['tokens_per_s']:.3f})", flush=True)

        # per-class breakdown of the fused step (event-timed, non-graph)
        bd = lm08._f32([0.0] * 3)
        lm08._lib.fx_time_breakdown(ctx, bd)
        print(f"         fused split: gemv {bd[0]:.3f} ms | attn "
              f"{bd[1]:.3f} ms | sum {bd[0] + bd[1]:.3f} ms", flush=True)

        # destroy both graph execs BEFORE freeing the buffers they reference
        lm03ref._lib.mx_shutdown()
        lm08._lib.fx_shutdown()
        del eng3, eng8
        torch.cuda.empty_cache()

    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")
    import os
    lm03ref._lib.mx_shutdown()
    lm08._lib.fx_shutdown()
    os._exit(0)   # two statically-linked cudart runtimes crash at exit


if __name__ == "__main__":
    main()
