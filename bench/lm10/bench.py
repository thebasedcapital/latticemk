"""Interleaved LM-10 vs megakernel-v2, 25 runs/context with unlocked clocks.

Identical INT4 weights, fp16 KV layout, initial token, and position; each
8-step launch resets the position to the requested context for both engines.
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

import lm03   # shared packed INT4 weight and norm/rope helpers
import lm03b  # v2 baseline and LM-10 fork

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
    for ctx in CTXS:
        e2 = lm03b.Engine2(ctx + MEGA_STEPS, packed, emb, norms, rope)
        e3 = lm03b.Engine3(ctx + MEGA_STEPS, packed, emb, norms, rope)
        for eng in (e2, e3):
            eng.bufs["kc"].zero_()
            eng.bufs["vc"].zero_()
            eng.set_tok(9707)
        samples = {e2: [], e3: []}
        out2 = lm03._f32([0.0])
        out3 = lm03._f32([0.0])
        for i in range(WARMUP + RUNS):
            # Reverse AB order every run to distribute clock drift fairly.
            pair = ((e2, out2), (e3, out3))
            for eng, out in (pair if i % 2 == 0 else pair[::-1]):
                rc = eng.lib.__getattr__(f"{eng.prefix}_time_mega")(
                    MEGA_STEPS, 1, ctx, out)
                if rc:
                    raise RuntimeError(f"{eng.prefix}_time_mega failed: {rc}")
                if i >= WARMUP:
                    samples[eng].append(out[0])
        for eng, label, wp in ((e2, "megakernel-v2", "LM-03b"),
                               (e3, "megakernel-sync", "LM-10")):
            r = row(wp, label, ctx, samples[eng], eng)
            rows.append(r)
            print(f"ctx {ctx:5d} {label:18} {r['tokens_per_s']:8.1f} tok/s "
                  f"{r['median_ms']:.4f} ms, p10/p90 "
                  f"{r['p10_ms']:.4f}/{r['p90_ms']:.4f}, "
                  f"SM {r['sm_clock_mhz']} MHz", flush=True)
        print(f"ctx {ctx}: v3 speedup "
              f"{rows[-2]['median_ms'] / rows[-1]['median_ms']:.4f}x",
              flush=True)
        del e2, e3
        torch.cuda.empty_cache()
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
