"""Globaltimer phase traces (ns) for instrumented v2 and LM-10 at two contexts.

Profiles the first step of a one-step launch with identical INT4 weights, fp16
KV buffers, and token. The KV cache is initialized to zero, as in bench.py;
this is a scheduler/position-length measurement, not a prefill benchmark.
"""
import json
import sys
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm03
import lm03b

OUT = Path(__file__).with_name("phases.json")


def sm_clock():
    data = subprocess.check_output([
        "nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
    return int(data.decode().split()[0])


def summary(eng, ctx):
    clock_before = sm_clock()
    eng.set_tok(9707)
    eng.pos_set(ctx)
    eng.mega(1)
    stamps = eng.bufs["prof"].cpu().reshape(36, -1)
    nph = 143 if eng.prefix == "mk2p" else 142
    data = {"sm_clock_before_mhz": clock_before, "sm_clock_after_mhz": sm_clock(),
            "trace_span_us": (stamps[:, -1].max() - stamps[:, 0].min()).item() / 1000}
    for kind, indices in (("qkv", range(0, 140, 5)),
                          ("attn", range(1, 140, 5)),
                          ("o", range(2, 140, 5)),
                          ("gu", range(3, 140, 5)),
                          ("down", range(4, 140, 5)),
                          ("lm", [140]), ("argp", [141] if nph == 143 else []),
                          ("argc", [142] if nph == 143 else [141])):
        work, wait, span, imbalance, release = [], [], [], [], []
        for ph in indices:
            if eng.prefix == "mk2p":
                start, arrived, end = [stamps[:, ph * 3 + i] for i in range(3)]
                work.append((arrived - start).max().item())
                wait.append((end - arrived).max().item())
                span.append((end.max() - start.min()).item())
                release.append((end.max() - arrived.max()).item())
                if kind == "attn":
                    imbalance.append([(arrived[c] - start[c]).item() for c in range(32)])
            else:
                ready, arrived = [stamps[:, ph * 2 + i] for i in range(2)]
                work.append((arrived - ready).max().item())
                span.append((arrived.max() - ready.min()).item())
                if ph:
                    prev = stamps[:, (ph - 1) * 2 + 1]
                    wait.append((ready - prev).max().item())
                if kind == "attn":
                    imbalance.append([(arrived[c] - ready[c]).item() for c in range(32)])
        if not indices:
            continue
        data[kind] = {"max_cta_work_us_sum": sum(work) / 1000,
                      "max_cta_wait_us_sum": sum(wait) / 1000,
                      "phase_wall_us_sum": sum(span) / 1000,
                      "barrier_release_tail_us_sum": sum(release) / 1000
                                                    if release else None,
                      "layers": len(indices)}
        if imbalance:
            half0 = [v for layer in imbalance for v in layer[::2]]
            half1 = [v for layer in imbalance for v in layer[1::2]]
            data[kind]["half0_mean_us"] = sum(half0) / len(half0) / 1000
            data[kind]["half1_mean_us"] = sum(half1) / len(half1) / 1000
            data[kind]["half0_max_us"] = max(half0) / 1000
            data[kind]["half1_max_us"] = max(half1) / 1000
    return data


def main():
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    result = {}
    for ctx in (128, 8192):
        result[str(ctx)] = {}
        for cls in (lm03b.Engine2p, lm03b.Engine3p):
            e = cls(ctx + 1, packed, emb, norms, rope)
            key = e.prefix
            result[str(ctx)][key] = summary(e, ctx)
            print(ctx, key, json.dumps(result[str(ctx)][key]), flush=True)
            del e
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
