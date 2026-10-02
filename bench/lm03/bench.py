"""LM-03 benchmark: separate-kernel+graph vs megakernel, decode tokens/s.

Contexts 128 / 2048 / 8192, batch 1. KV cache is pre-filled with zeros once
(decode-step bytes are identical for any cache content); each timed run resets
pos to ctx so every decode step reads ctx+1 KV positions. Only decode steps
are timed (CUDA events bracket the graph replay / mega launch; flag and pos
resets happen outside).

Row: commit, work_package, model, kernel, context, batch, tokens_per_s, gbps,
pct_roofline, sm_clock_mhz, driver, runs, median_ms, p10_ms, p90_ms,
weight_bytes, kv_bytes.

usage (through the GPU lock): scripts/gpu.sh --timing .venv/bin/python \
    bench/lm03/bench.py [ctx ...]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03

ROOF = 406.7          # GB/s, measured read roofline (hw/hw.json)
RUNS = 25
WARMUP = 3
MEGA_STEPS = 8        # decode steps inside one mega launch per timed run
CTXS = [int(a) for a in sys.argv[1:]] or [128, 2048, 8192]
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


def row(kernel, ctx, ms, eng):
    kv = 2 * lm03.N_LAYERS * ctx * lm03.KVROWS * 2   # k+v fp16 per token
    wb = weight_bytes(eng)
    ab = 60_000  # activations/partials/logits traffic per step [derived est]
    tot = kv + wb + ab
    med, p10, p90 = (statistics.median(ms),
                     statistics.quantiles(ms, n=10)[0],
                     statistics.quantiles(ms, n=10)[8])
    tps = 1e3 / med
    return {
        "commit": "nogit", "work_package": "LM-03", "model": "qwen3-0.6b",
        "kernel": kernel, "context": ctx, "batch": 1,
        "tokens_per_s": round(tps, 2), "gbps": round(tot / med / 1e6, 1),
        "pct_roofline": round(100 * tot / med / 1e6 / ROOF, 1),
        "sm_clock_mhz": sm_clock(), "driver": driver(), "runs": len(ms),
        "median_ms": round(med, 4), "p10_ms": round(p10, 4),
        "p90_ms": round(p90, 4), "weight_bytes": wb, "kv_bytes": kv,
    }


def main():
    print("packing/loading weights (cached) ...", flush=True)
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    rows = []
    f32p = ctypes.POINTER(ctypes.c_float)
    lib = lm03._lib

    for ctx in CTXS:
        eng = lm03.Engine(ctx, packed, emb, norms, rope)
        eng.bufs["kc"].zero_()   # explicit "prefill": decode reads ctx+1 rows
        eng.bufs["vc"].zero_()
        eng.set_tok(9707)

        # -- interleaved A/B: clocks drift 1005-1845 MHz (unlocked); alternate
        # graph/mega one step-batch at a time so both see the same clock mix --
        eng.graph_build()
        gm, mm = [], []
        out = lm03._f32([0.0] * 1)
        out2 = lm03._f32([0.0] * 1)
        lib.mk_time_graph(WARMUP, ctx, out)          # warm both paths
        lib.mk_time_mega(MEGA_STEPS, WARMUP, ctx, out2)
        for _ in range(RUNS):
            lib.mk_time_graph(1, ctx, out)
            gm.append(out[0])
            lib.mk_time_mega(MEGA_STEPS, 1, ctx, out2)
            mm.append(out2[0])
        r = row("separate-graph", ctx, gm, eng)
        rows.append(r)
        print(f"ctx {ctx:5d} separate-graph  {r['tokens_per_s']:8.1f} tok/s "
              f"{r['median_ms']:.3f} ms  {r['pct_roofline']:.1f}% roof", flush=True)
        r2 = row("megakernel", ctx, mm, eng)
        rows.append(r2)
        print(f"ctx {ctx:5d} megakernel      {r2['tokens_per_s']:8.1f} tok/s "
              f"{r2['median_ms']:.3f} ms  {r2['pct_roofline']:.1f}% roof "
              f"(x{r2['tokens_per_s'] / r['tokens_per_s']:.2f})", flush=True)

        del eng
        torch.cuda.empty_cache()

    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
