"""LM-11 interleaved comparison driver: v2 megakernel vs llama.cpp (and HF),
one process hosting v2 + round-robin subprocesses so unlocked clocks hit all
engines equally.

usage (run under scripts/gpu.sh --timing):
  python interleave.py CTX --llama GGUF [--fa on|off] [--ctk T] [--ctv T]
        [--rounds R] [--hf] [--hf-compile] [--llama-n 64] [--llama-r 4]

Per round: v2 times RUNS_V2 steps (8/launch), llama-bench --repetitions R_LAMA
runs its own tg test, HF worker (if enabled) runs DECODE_N steps.
Rows appended to results.jsonl.
"""
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
sys.path.insert(0, str(ROOT / "bench" / "lm03b"))

import lm03  # noqa: E402
import lm03b  # noqa: E402

ROOF = 406.7
LLAMA_BENCH = str(Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))
                  / "build-cuda" / "bin" / "llama-bench")
PY = str(ROOT / ".venv" / "bin" / "python")
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
        "commit": "nogit", "work_package": "LM-11",
        "model": "qwen3-0.6b", "kernel": kernel, "context": ctx,
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
            quant = kernel.removeprefix("llamacpp-").split("-fa")[0]
            k, v = r.get("type_k", "f16"), r.get("type_v", "f16")
            score = scores.get(f"{quant}/{k}+{v}") if k != "f16" or v != "f16" else scores.get(f"{quant}/f16")
        elif kernel.startswith("megakernel-v2"):
            score = scores.get("int4gptq-fake/f16" if kernel.endswith("-gptq")
                               else "int4rtn-fake/f16")
        elif kernel.startswith("hf-fp16"):
            score = scores.get("hf-fp16/f16")
        else:
            score = None
        if score is not None:
            r["ppl"] = score
            r["ppl_protocol"] = "wikitext-2 llama-perplexity -c 2048"
    return r


class HFWorker:
    def __init__(self, ctx, do_compile):
        self.p = subprocess.Popen([PY, str(HERE / "hf_worker.py")],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        self.cmd({"cmd": "init", "ctx": ctx, "compile": do_compile})
        self.cmd({"cmd": "fill"})

    def cmd(self, c):
        self.p.stdin.write(json.dumps(c) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "error" in r:
            raise RuntimeError(r)
        return r

    def decode(self, n):
        return self.cmd({"cmd": "decode", "n": n})["ms"]

    def close(self):
        try:
            self.cmd({"cmd": "quit"})
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


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
    ap.add_argument("--llama", default=None, help="gguf path")
    ap.add_argument("--weights", default=str(ROOT / "bench/lm03b/weights_int4.pt"))
    ap.add_argument("--fa", default="on")
    ap.add_argument("--ctk", default=None)
    ap.add_argument("--ctv", default=None)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--v2-reps", type=int, default=4)
    ap.add_argument("--llama-n", type=int, default=64)
    ap.add_argument("--llama-r", type=int, default=4)
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--hf-compile", action="store_true")
    ap.add_argument("--hf-decode-n", type=int, default=32)
    a = ap.parse_args()

    packed = {k: {f: t.cuda() for f, t in v.items()}
              for k, v in torch.load(a.weights, map_location="cpu").items()}
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    eng = lm03b.Engine2(a.ctx, packed, emb, norms, rope)
    eng.bufs["kc"].zero_()
    eng.bufs["vc"].zero_()
    eng.set_tok(9707)
    out = lm03._f32([0.0])
    lib2 = lm03b._lib2
    lib2.mk2_time_mega(8, 3, a.ctx, out)  # warmup

    wb_v2 = sum(t.numel() * t.element_size()
                for t in eng.codes + eng.metas)
    kv_f16 = 2 * lm03.N_LAYERS * a.ctx * lm03.KVROWS * 2

    hf = HFWorker(a.ctx, a.hf_compile) if (a.hf or a.hf_compile) else None

    v2_ms, llama_ms, hf_ms, llama_rec = [], [], [], None
    llama_label = None
    for r in range(a.rounds):
        # v2 slice
        for _ in range(a.v2_reps):
            lib2.mk2_time_mega(8, 1, a.ctx, out)
            v2_ms.append(out[0])
        # llama.cpp slice
        if a.llama:
            ms, rec = llamabench(a.llama, a.ctx, a.llama_n, a.llama_r,
                                 a.fa == "on", a.ctk, a.ctv)
            llama_ms += ms
            llama_rec = rec
            llama_label = (
                f"llamacpp-{Path(a.llama).stem.split('-')[-1]}"
                f"-fa{a.fa}"
                f"{'-k' + a.ctk if a.ctk else ''}"
                f"{'-v' + a.ctv if a.ctv else ''}")
        # HF slice
        if hf:
            hf_ms += hf.decode(a.hf_decode_n)
        print(f"round {r}: v2 {v2_ms[-1]:.3f} ms"
              + (f" llama {statistics.median(ms):.3f} ms" if a.llama else "")
              + (f" hf {statistics.median(hf_ms[-a.hf_decode_n:]):.3f} ms"
                 if hf else "") + f" clk {sm_clock()}", flush=True)

    rows = []
    v2_label = ("megakernel-v2-gptq" if Path(a.weights).name == "weights_int4_gptq.pt"
                else "megakernel-v2")
    rows.append(row(v2_label, a.ctx, v2_ms, wb_v2, kv_f16))
    if a.llama:
        bpe = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}  # ggml block bytes/element
        kv = int(lm03.N_LAYERS * a.ctx * 1024
                 * (bpe.get(a.ctk or "f16", 2.0)
                    + bpe.get(a.ctv or "f16", 2.0)))
        rows.append(row(llama_label, a.ctx, llama_ms,
                        int(llama_rec["model_size"]), kv,
                        {"avg_ts": llama_rec["avg_ts"],
                         "stddev_ts": llama_rec["stddev_ts"],
                         "n_gen": a.llama_n,
                         "llama_commit": llama_rec["build_commit"],
                         "type_k": llama_rec["type_k"],
                         "type_v": llama_rec["type_v"],
                         "flash_attn": llama_rec["flash_attn"]}))
    if hf:
        wb = 2 * int(llama_rec["model_n_params"]) if llama_rec else 2 * 596_049_920
        rows.append(row(
            "hf-fp16" + ("-compile" if a.hf_compile else "-eager"),
            a.ctx, hf_ms, wb, kv_f16))
    if hf:
        hf.close()

    with OUT.open("a") as f:
        for rr in rows:
            f.write(json.dumps(rr) + "\n")
    for rr in rows:
        print(f"{rr['kernel']:28s} ctx {rr['context']:5d} "
              f"{rr['median_ms']:8.3f} ms {rr['tokens_per_s']:8.1f} tok/s "
              f"{rr['pct_roofline']:5.1f}% roof clk {rr['sm_clock_mhz']}",
              flush=True)


if __name__ == "__main__":
    main()
