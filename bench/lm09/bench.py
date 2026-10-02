"""LM-09 benchmark: megakernel-v2 (fp16 KV, read-only import of bench/lm03b)
vs megakernel-kv (quantized KV). Interleaved A/B in one process, 25 runs,
8 decode steps per launch — same protocol as bench/lm03b/bench.py.

usage: scripts/gpu.sh --timing .venv/bin/python bench/lm09/bench.py [ctxs...]
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
sys.path.insert(0, str(ROOT / "bench" / "lm03b"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03   # baseline engine + packed weights helpers
import lm03b  # v2 megakernel engine (engine of record, unmodified)
import lm09   # kv megakernel engine

ROOF = 406.7          # GB/s, measured read roofline (hw/hw.json)
RUNS = 25
WARMUP = 3
MEGA_STEPS = 8        # decode steps inside one mega launch per timed run
CTXS = [int(a) for a in sys.argv[1:]] or [128, 2048, 8192, 16384]
OUT = Path(__file__).resolve().parent / "results.jsonl"


def sm_clock():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
            text=True).strip()
        return int(out.split()[0])
    except Exception:
        return -1


def driver():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "unknown"


def weight_bytes(eng):
    return sum(t.numel() * t.element_size()
               for t in eng.codes + eng.metas)


def row(wp, kernel, ctx, ms, eng, kv_bpt):
    kv = lm03.N_LAYERS * ctx * kv_bpt
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
        "kv_format": lm09.CODEC if kernel == "megakernel-kv" else "fp16",
        "diagnostic": True,
        "correctness": "FAIL" if kernel == "megakernel-kv" else "PASS",
    }


def main():
    print("packing/loading weights (cached) ...", flush=True)
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope(maxpos=lm09.MAXPOS).cuda()
    OUT.write_text("")
    rows = []
    lib2 = lm03b._lib2
    libkv = lm09._lib
    kv_fp16 = lm03.KVROWS * 2 * 2   # K and V fp16 bytes/token/layer
    kv_pq = lm09.kv_bytes_per_tok_layer()

    for ctx in CTXS:
        assert ctx <= lm09.MAXPOS - 2 * MEGA_STEPS, "ctx exceeds megakv MAXPOS"
        e2 = lm03b.Engine2(ctx, packed, emb, norms, rope)
        e2.set_tok(9707)
        ekv = lm09.EngineKV(ctx, packed, emb, norms, rope)
        for k in ("kq", "vq"):
            ekv.bufs[k].zero_()
        ekv.set_tok(9707)

        # interleaved A/B: both engines see the same clock mix
        mm, vm = [], []
        out2 = lm03._f32([0.0])
        outk = lm03._f32([0.0])
        lib2.mk2_time_mega(MEGA_STEPS, WARMUP, ctx, out2)
        libkv.mkv_time_mega(MEGA_STEPS, WARMUP, ctx, outk)
        for i in range(RUNS):
            pair = ((lib2.mk2_time_mega, out2, mm),
                    (libkv.mkv_time_mega, outk, vm))
            for fn, out, samples in (pair if i % 2 == 0 else pair[::-1]):
                assert fn(MEGA_STEPS, 1, ctx, out) == 0
                samples.append(out[0])
        r2 = row("LM-03b", "megakernel-v2", ctx, mm, e2, kv_fp16)
        rv = row("LM-09", "megakernel-kv", ctx, vm, ekv, kv_pq)
        rows += [r2, rv]
        ratio = rv["tokens_per_s"] / r2["tokens_per_s"]
        print(f"ctx {ctx:5d} v2   {r2['median_ms']:7.3f} ms  "
              f"{r2['tokens_per_s']:6.1f} tok/s  {r2['pct_roofline']:5.1f}% |"
              f" kv {rv['median_ms']:7.3f} ms  {rv['tokens_per_s']:6.1f} tok/s"
              f"  {rv['pct_roofline']:5.1f}%  ratio {ratio:.2f}x", flush=True)
        with OUT.open("a") as f:
            for r in (r2, rv):
                f.write(json.dumps(r) + "\n")

        del e2, ekv
        torch.cuda.empty_cache()
        if ctx == 8192 and ratio < 1.3:
            print("KILL: 8192 speedup below 1.3x; skipping 16384", flush=True)
            break

    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
