"""LM-11: re-time megakernel-v2 in this session (clock parity with llama.cpp runs).

Same engine + timing path as bench/lm03b/bench.py, but times ONLY v2 and can
point at an alternate weights file (--weights bench/lm11/weights_int4_gptq.pt)
for the RTN-vs-GPTQ identical-speed check.

usage: v2_time.py CTX [--weights PATH] [--runs N] [--label NAME]
writes JSONL rows to bench/lm11/results.jsonl
"""
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

import lm03  # noqa: E402
import lm03b  # noqa: E402

ROOF = 406.7
MEGA_STEPS = 8
WARMUP = 3
OUT = Path(__file__).resolve().parent / "results.jsonl"


def sm_clock():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
        return int(o.decode().split()[0])
    except Exception:
        return -1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int)
    ap.add_argument("--weights", default=str(ROOT / "bench" / "lm03b"
                                             / "weights_int4.pt"))
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--label", default="megakernel-v2")
    ap.add_argument("--tag", default="")  # extra label suffix
    a = ap.parse_args()

    packed = {k: {f: t.cuda() for f, t in v.items()}
              for k, v in torch.load(a.weights).items()}
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    eng = lm03b.Engine2(a.ctx, packed, emb, norms, rope)
    eng.bufs["kc"].zero_()
    eng.bufs["vc"].zero_()
    eng.set_tok(9707)

    lib2 = lm03b._lib2
    out = lm03._f32([0.0] * a.runs)
    lib2.mk2_time_mega(MEGA_STEPS, WARMUP, a.ctx, out)
    ms = []
    for _ in range(a.runs):
        lib2.mk2_time_mega(MEGA_STEPS, 1, a.ctx, out)
        ms.append(out[0])
    med = statistics.median(ms)
    ms_sorted = sorted(ms)
    wb = sum(t.numel() * t.element_size() for t in eng.codes + eng.metas)
    kv = 2 * lm03.N_LAYERS * a.ctx * lm03.KVROWS * 2
    row = {
        "commit": "nogit", "work_package": "LM-11",
        "model": "qwen3-0.6b", "kernel": a.label + a.tag,
        "context": a.ctx, "batch": 1,
        "tokens_per_s": 1000.0 / med,
        "gbps": (wb + kv) / med * 1e-6,
        "pct_roofline": (wb + kv) / med * 1e-6 / ROOF * 100,
        "sm_clock_mhz": sm_clock(), "driver": "610.57.04",
        "runs": a.runs, "median_ms": med,
        "p10_ms": ms_sorted[len(ms) // 10],
        "p90_ms": ms_sorted[(9 * len(ms)) // 10],
        "weight_bytes": wb, "kv_bytes": kv,
        "steps_per_launch": MEGA_STEPS,
    }
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"v2 {a.label}{a.tag} ctx {a.ctx}: {med:.3f} ms "
          f"{row['tokens_per_s']:.1f} tok/s clk {row['sm_clock_mhz']}",
          flush=True)


if __name__ == "__main__":
    main()
