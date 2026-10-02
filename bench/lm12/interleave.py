"""Paired Qwen3-1.7B GPTQ vs llama.cpp Q4_0/Q4_K_M at one context per job."""
import argparse
import json
import statistics
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(HERE))
import lm03
import scale

ROOF = 406.7
LLAMA_BENCH = str(Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))
                  / "build-cuda" / "bin" / "llama-bench")
OUT = HERE / "results.jsonl"


def sm_clock():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
        return int(o.decode().split()[0])
    except Exception:
        return -1


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def row(kernel, ctx, ms, wb, kv, extra=None):
    med = statistics.median(ms)
    r = {
        "commit": "nogit", "work_package": "LM-12",
        "model": "qwen3-1.7b", "kernel": kernel, "context": ctx,
        "batch": 1, "tokens_per_s": 1000.0 / med,
        "gbps": (wb + kv) / med * 1e-6,
        "pct_roofline": (wb + kv) / med * 1e-6 / ROOF * 100,
        "sm_clock_mhz": sm_clock(), "driver": "610.57.04",
        "runs": len(ms), "median_ms": med,
        "p10_ms": pct(ms, 0.1), "p90_ms": pct(ms, 0.9),
        "weight_bytes": wb, "kv_bytes": kv,
    }
    r.update(extra or {})
    quality = HERE / "ppl.json"
    if quality.exists():
        scores = json.loads(quality.read_text())["ppl"]
        if kernel.startswith("llamacpp-"):
            quant = kernel.removeprefix("llamacpp-").split("-fa")[0].lower()
            score = scores.get(f"{quant}/f16")
        elif kernel.startswith("megakernel-scale"):
            score = scores.get("int4gptq-fake/f16")
        elif kernel.startswith("hf-fp16"):
            score = scores.get("hf-fp16/f16")
        else:
            score = None
        if score is not None:
            r["ppl"] = score
            r["ppl_protocol"] = "wikitext-2 llama-perplexity -c 2048"
    if "ppl" not in r:
        r["ppl"] = None
    return r



def llamabench(gguf, ctx, n_gen, reps, fa, ctk, ctv):
    """One llama-bench call -> list of per-rep ms."""
    args = [LLAMA_BENCH, "-m", gguf, "-p", "0", "-n", str(n_gen),
            "-d", str(ctx), "-r", str(reps), "-o", "json", "-ngl", "99"]
    args += ["-fa", "1" if fa else "0"]
    if ctk:
        args += ["-ctk", ctk]
    if ctv:
        args += ["-ctv", ctv]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("llama-bench failed: " + r.stderr[-2000:])
    rec = [t for t in json.loads(r.stdout) if t.get("n_gen", 0) > 0]
    if not rec:
        raise RuntimeError("no tg rows: " + r.stdout[:1000])
    t = rec[0]
    ms = [1000.0 / ts for ts in t["samples_ts"]]
    return ms, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int)
    ap.add_argument("--llama", action="append", required=True,
                    help="each GGUF path, repeat for both Q4_0 and Q4_K_M")
    ap.add_argument("--weights", default=str(HERE / "weights_int4_gptq.pt"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--v2-reps", type=int, default=9)
    ap.add_argument("--llama-n", type=int, default=32)
    ap.add_argument("--llama-r", type=int, default=10)
    a = ap.parse_args()
    assert a.ctx in (128, 2048, 8192)
    gate = HERE / "gate_v21.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("pass"):
        raise SystemExit("LM-12 correctness gate v2.1 must PASS before timing")
    cpu_packed = torch.load(a.weights, map_location="cpu")
    cpu_emb = scale.load("model.embed_tokens.weight").half().contiguous()
    weight_bytes = sum(t.numel() * t.element_size()
                       for weight in cpu_packed.values() for t in weight.values())
    kv_bytes = 2 * scale.N_LAYERS * a.ctx * scale.KVROWS * 2
    out = lm03._f32([0.0])
    lib = scale._lib2

    def make_engine():
        gpu = {name: {field: tensor.cuda() for field, tensor in weight.items()}
               for name, weight in cpu_packed.items()}
        engine = scale.Engine2(a.ctx, gpu, cpu_emb.cuda(),
                               scale.norm_table(), scale.make_rope())
        engine.bufs["kc"].zero_()
        engine.bufs["vc"].zero_()
        engine.set_tok(9707)
        assert lib.mk2_time_mega(8, 3, a.ctx, out) == 0
        return engine

    eng = make_engine()

    v2_ms = {gguf: [] for gguf in a.llama}
    llama_ms = {gguf: [] for gguf in a.llama}
    llama_records = {}
    clocks = {gguf: [] for gguf in a.llama}
    for iteration in range(a.rounds):
        for gguf in (a.llama if iteration % 2 == 0 else a.llama[::-1]):
            if eng is None:
                eng = make_engine()
            for _ in range(a.v2_reps):
                assert lib.mk2_time_mega(8, 1, a.ctx, out) == 0
                v2_ms[gguf].append(out[0])
            if a.ctx == 8192:
                # Both full fp16 KV allocations plus a desktop process do
                # not coexist with llama's context graph on this 8 GB card.
                # Warm/reload outside all measured intervals, each round.
                del eng
                eng = None
                torch.cuda.empty_cache()
            ms, rec = llamabench(gguf, a.ctx, a.llama_n, a.llama_r,
                                 True, "f16", "f16")
            llama_ms[gguf].extend(ms)
            llama_records[gguf] = rec
            clocks[gguf].append(sm_clock())
            print(f"round {iteration} {Path(gguf).name}: v2 "
                  f"{statistics.median(v2_ms[gguf]):.3f} ms llama "
                  f"{statistics.median(ms):.3f} ms clock {clocks[gguf][-1]}",
                  flush=True)
    rows = []
    for gguf in a.llama:
        quant = Path(gguf).stem.rsplit("-", 1)[-1]
        rec = llama_records[gguf]
        rows.append(row("megakernel-scale-gptq", a.ctx, v2_ms[gguf],
                        weight_bytes, kv_bytes, {
                            "paired_with": quant,
                            "memory_mode": ("unload-megakernel-between-rounds"
                                            if a.ctx == 8192 else "resident"),
                            "sm_clock_mhz": statistics.median(clocks[gguf]),
                            "clock_samples_mhz": clocks[gguf]}))
        rows.append(row(f"llamacpp-{quant}-faon", a.ctx, llama_ms[gguf],
                        int(rec["model_size"]), kv_bytes,
                        {"paired_with": "megakernel-scale-gptq",
                         "memory_mode": ("unload-megakernel-between-rounds"
                                         if a.ctx == 8192 else "resident"),
                         "sm_clock_mhz": statistics.median(clocks[gguf]),
                         "clock_samples_mhz": clocks[gguf],
                         "avg_ts": rec["avg_ts"],
                         "stddev_ts": rec["stddev_ts"],
                         "n_gen": a.llama_n,
                         "llama_commit": rec["build_commit"],
                         "type_k": rec["type_k"],
                         "type_v": rec["type_v"],
                         "flash_attn": rec["flash_attn"]}))
    with OUT.open("a") as f:
        for record in rows:
            f.write(json.dumps(record) + "\n")
    for record in rows:
        print(f"{record['kernel']:28s} {record['context']:5d} "
              f"{record['median_ms']:8.3f} ms "
              f"{record['tokens_per_s']:8.1f} tok/s "
              f"{record['pct_roofline']:5.1f}% roof "
              f"clock {record['sm_clock_mhz']}", flush=True)



if __name__ == "__main__":
    main()
