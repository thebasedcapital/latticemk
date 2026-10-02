"""LM-03b benchmark: LM-03 separate-kernel+graph baseline vs megakernel-v2
(36 CTAs x 1024 threads, barrier-phased), decode tokens/s.

Measurement protocol (per spec): both engines timed in the same process,
interleaved run-by-run so unlocked clocks (1005-1845 MHz) hit both equally;
>=25 runs each; median/p10/p90; SM clock + driver recorded.

usage: bench.py [ctx ...]      # default 128 2048 8192
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

import lm03   # baseline engine + packed weights helpers
import lm03b  # v2 megakernel engine

ROOF = 406.7          # GB/s, measured read roofline (hw/hw.json)
RUNS = 25
WARMUP = 3
MEGA_STEPS = 8        # decode steps inside one mega launch per timed run
CTXS = [int(a) for a in sys.argv[1:]] or [128, 2048, 8192]
OUT = Path(__file__).resolve().parent / "results.jsonl"


def sm_clock():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
        return int(o.decode().split()[0])
    except Exception:
        return -1


def driver():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"])
        return o.decode().strip()
    except Exception:
        return "unknown"


def weight_bytes(eng):
    return sum(t.numel() * t.element_size()
               for t in eng.codes + eng.metas)


def row(wp, kernel, ctx, ms, eng):
    kv = 2 * lm03.N_LAYERS * ctx * lm03.KVROWS * 2   # k+v fp16 per token
    wb = weight_bytes(eng)
    med = statistics.median(ms)
    ms_sorted = sorted(ms)
    return {
        "commit": "nogit", "work_package": wp, "model": "qwen3-0.6b-int4",
        "kernel": kernel, "context": ctx, "batch": 1,
        "tokens_per_s": 1000.0 / med,
        "gbps": (wb + kv) / med * 1e-6,
        "pct_roofline": (wb + kv) / med * 1e-6 / ROOF * 100,
        "sm_clock_mhz": sm_clock(), "driver": driver(),
        "runs": len(ms), "median_ms": med,
        "p10_ms": ms_sorted[len(ms) // 10],
        "p90_ms": ms_sorted[(9 * len(ms)) // 10],
        "weight_bytes": wb, "kv_bytes": kv,
    }


def main():
    print("packing/loading weights (cached) ...", flush=True)
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    rows = []
    lib = lm03._lib
    lib2 = lm03b._lib2

    for ctx in CTXS:
        eng = lm03.Engine(ctx, packed, emb, norms, rope)
        eng.bufs["kc"].zero_()
        eng.bufs["vc"].zero_()
        eng.set_tok(9707)
        e2 = lm03b.Engine2(ctx, packed, emb, norms, rope)
        e2.bufs["kc"].zero_()
        e2.bufs["vc"].zero_()
        e2.set_tok(9707)

        # interleaved A/B: both engines see the same clock mix
        eng.graph_build()
        gm, om, mm = [], [], []
        out = lm03._f32([0.0])
        out3 = lm03._f32([0.0])
        out2 = lm03._f32([0.0])
        lib.mk_time_graph(WARMUP, ctx, out)
        lib.mk_time_mega(MEGA_STEPS, WARMUP, ctx, out3)
        lib2.mk2_time_mega(MEGA_STEPS, WARMUP, ctx, out2)
        for _ in range(RUNS):
            lib.mk_time_graph(1, ctx, out)
            gm.append(out[0])
            lib.mk_time_mega(MEGA_STEPS, 1, ctx, out3)
            om.append(out3[0])
            lib2.mk2_time_mega(MEGA_STEPS, 1, ctx, out2)
            mm.append(out2[0])
        r = row("LM-03", "separate-graph", ctx, gm, eng)
        ro = row("LM-03", "megakernel-v1", ctx, om, eng)
        rows.append(ro)
        print(f"ctx {ctx:5d} megakernel-v1   {ro['tokens_per_s']:8.1f} tok/s "
              f"{ro['median_ms']:.3f} ms  {ro['pct_roofline']:.1f}% roof "
              f"(x{ro['tokens_per_s'] / r['tokens_per_s']:.2f})", flush=True)
        rows.append(r)
        print(f"ctx {ctx:5d} separate-graph  {r['tokens_per_s']:8.1f} tok/s "
              f"{r['median_ms']:.3f} ms  {r['pct_roofline']:.1f}% roof",
              flush=True)
        r2 = row("LM-03b", "megakernel-v2", ctx, mm, e2)
        rows.append(r2)
        print(f"ctx {ctx:5d} megakernel-v2   {r2['tokens_per_s']:8.1f} tok/s "
              f"{r2['median_ms']:.3f} ms  {r2['pct_roofline']:.1f}% roof "
              f"(x{r2['tokens_per_s'] / r['tokens_per_s']:.2f})", flush=True)

        del eng, e2
        torch.cuda.empty_cache()

    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
