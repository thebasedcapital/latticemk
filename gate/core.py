"""V2.1 reference checks and copied engine bridges shared by the permanent gate.

Numerical rules are copied from the wave-6 gate without tolerance changes.
Fail-fast prefixes establish a failure only, never a pass or equivalence.
"""
import importlib.util
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "gate"
sys.path[:0] = [str(ROOT), str(ROOT / "bench/lm03"), str(ROOT / "bench/lm03b"), str(ROOT / "bench/lm11"), str(ROOT / "bench/lm12"), str(HERE)]
import lm03
from extra_tests import sampler_mismatch



def bridge(model, library=None, folder=None):
    original = ROOT / ("bench/lm03b/lm03b.py" if model == "0.6B" else "bench/lm12/scale.py")
    text = original.read_text()
    text = text.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
    if model == "0.6B" and library:
        text = text.replace('str(ROOT / "kernels" / "megakernel_v2" / "libmega2.so")', repr(str(library)))
    if model == "1.7B" and library:
        text = text.replace("use_library()\n", f"use_library({str(library)!r})\n")
    folder = folder or HERE / "build" / f"original-{model}"
    folder.mkdir(parents=True, exist_ok=True)
    copy = folder / "harness.py"
    copy.write_text(text)
    spec = importlib.util.spec_from_file_location("mutation_harness", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_model(model, deq):
    if model == "1.7B":
        import scale
        snapshot, inter = scale.SNAPSHOT, scale.INTER
    else:
        from lmk.model import SNAPSHOT
        snapshot, inter = SNAPSHOT, 3072
    net = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).eval()
    lin = {n: m for n, m in net.named_modules() if isinstance(m, torch.nn.Linear)}
    device = "cuda" if model == "0.6B" else "cpu"
    for layer in range(28):
        p = f"model.layers.{layer}."
        qkv = deq[f"L{layer}.qkv"].to(device)
        for name, lo, hi in (("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)):
            lin[p + f"self_attn.{name}_proj"].weight.data = qkv[lo:hi].clone()
        lin[p + "self_attn.o_proj"].weight.data = deq[f"L{layer}.o"].to(device)
        gu = deq[f"L{layer}.gu"].to(device)
        lin[p + "mlp.gate_proj"].weight.data = gu[:inter].clone()
        lin[p + "mlp.up_proj"].weight.data = gu[inter:].clone()
        lin[p + "mlp.down_proj"].weight.data = deq[f"L{layer}.down"].to(device)
    net.lm_head.weight = torch.nn.Parameter(deq["lm_head"].to(device))
    return net.to(device).eval(), snapshot


def _resources(model, module, packed_file=None):
    bench = ROOT / ("bench/lm11" if model == "0.6B" else "bench/lm12")
    pk = torch.load(packed_file or bench / "weights_int4_gptq.pt", map_location="cpu")
    gpu = {n: {f: t.cuda() for f, t in v.items()} for n, v in pk.items()}
    if model == "0.6B":
        shared = (lm03.load("model.embed_tokens.weight").half().contiguous(), lm03.norm_table(), lm03.make_rope().cuda())
    else:
        shared = (module.load("model.embed_tokens.weight", "cuda").half().contiguous(), module.norm_table(), module.make_rope())
    return gpu, shared


@torch.no_grad()
def pass_cases(module, gpu, shared, cases, prior=None, bound=None, fail_fast=False, check_sampling=False):
    started = time.perf_counter()
    engine = module.Engine2(512, gpu, *shared)
    saved, events, sample_errors = [], [], []
    max_diff, same, original_equal = 0., True, True
    count, killed = 0, None
    kl_sum = sq_sum = orig_max = 0.
    for ci, case in enumerate(cases):
        engine.prefill(case["prompt"])
        lines = []
        for step in range(case["steps"]):
            actual = engine.logits().cpu()
            expected = case["reference"][step]
            error = float((actual - expected).abs().max())
            max_diff = max(max_diff, error) if math.isfinite(error) else float("inf")
            top = expected.topk(2)
            margin = float(top.values[0] - top.values[1])
            if int(actual.argmax()) != int(top.indices[0]):
                events.append({"case": case["name"], "step": step, "margin": margin, "diff": error,
                               "kind": "near" if math.isfinite(error) and margin < 2 * error else "hard"})
            if prior is not None:
                same &= torch.equal(actual, prior[ci][step])
            if "original" in case:
                original_equal &= torch.equal(actual, case["original"][step])
                orig_max = max(orig_max, float((actual - case["original"][step]).abs().max()))
            if not fail_fast or error <= bound:  # statistics only matter for runs that complete
                lp, lq = torch.log_softmax(expected.double(), 0), torch.log_softmax(actual.double(), 0)
                kl_sum += float((lp.exp() * (lp - lq)).sum())
                sq_sum += float(((actual - expected).double() ** 2).mean())
            if check_sampling:
                mismatch = sampler_mismatch(engine, actual)
                if mismatch:
                    sample_errors.append({"case": case["name"], "step": step, **mismatch})
            lines.append(actual)
            count += 1
            if fail_fast and (not math.isfinite(error) or error > bound):
                killed = "logit_bound"
                break
            if fail_fast and events and events[-1]["kind"] == "hard":
                killed = "argmax"
                break
            if step + 1 < case["steps"]:
                engine.set_tok(int(case["forced"][step]))
                engine.mega(1)
        saved.append(torch.stack(lines))
        if killed:
            break
    del engine
    torch.cuda.synchronize()
    bad = not math.isfinite(max_diff)
    return {"max_diff": None if bad else max_diff, "nonfinite": bad, "events": events, "bitwise_repeat": same,
            "original_equal": original_equal, "sample_errors": sample_errors, "steps": count, "kill": killed,
            "over_bound": bool(bound is not None and (bad or max_diff > bound)),
            "mean_kl": kl_sum / count if count else None, "rms_diff": math.sqrt(sq_sum / count) if count else None,
            "orig_max_diff": orig_max,
            "wall_s": time.perf_counter() - started}, saved


MODEL = {"v2": "0.6B", "scale": "1.7B", "mt": "0.6B"}
LIBRARIES = {"v2": ROOT / "kernels/megakernel_v2/libmega2.so",
             "scale": ROOT / "kernels/megakernel_scale/libmega_scale.so"}


def make_module(engine, library=None, folder=None):
    return bridge(MODEL[engine], Path(library or LIBRARIES[engine]).resolve(), folder)


def resources(engine, module, packed_file=None):
    return _resources(MODEL.get(engine, engine), module, packed_file)


def load_cache(engine):
    path = ROOT / "mutation" / f"reference-{MODEL[engine]}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path}: generate the wave-6 reference using mutation/gate.py prepare")
    return torch.load(path, map_location="cpu", weights_only=False)
