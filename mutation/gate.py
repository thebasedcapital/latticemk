"""Copied-harness mutation gate: the v2.1 rules, a persistent HF reference cache, plus extra tests.

  prepare   (once per model, GPU+CPU)  HF fp32 fake-quant reference, original-kernel logits, gate timings
  evaluate  (batched, GPU)             run the gate against built mutant libraries, append to results.jsonl

Stages per mutant (all wall times are host wall seconds, not CUDA-event performance timings):
  build -> schedule (validator, sync mutants only) -> logit_bound -> argmax/near-tie -> determinism
  -> greedy (informational) -> extras (sampling, context, adversarial, head_perm, multistep, repeat).
Fail-fast numerical prefixes may establish a kill; they never establish a pass or an equivalence.
Equivalent = bitwise-identical logits to the original on every case (base + extras) plus a correct sampler.
That is an observation on these inputs, not a proof of equivalence.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "mutation"
sys.path[:0] = [str(ROOT), str(ROOT / "bench/lm03"), str(ROOT / "bench/lm03b"), str(ROOT / "bench/lm11"), str(ROOT / "bench/lm12"), str(HERE)]
import lm03
import extra_tests
from extra_tests import scenarios, sampler_mismatch

PROMPTS = ["The capital of France is", "def quicksort(arr):", "In a shocking finding, scientists discovered a herd of unicorns living in"]
PHASE_FLAGS = ("qkv", "attn", "o", "gu", "down")
DEVICE_ERRORS = ("mk2_sync", "mk2_mega", "illegal memory", "device-side", "launch failure", "unspecified launch", "an illegal")


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


@torch.no_grad()
def hf_case(net, case, forced=None):
    device = next(net.parameters()).device
    out = net(input_ids=case["prompt"][None].to(device), use_cache=True)
    past = out.past_key_values
    logits, tokens = [], []
    for step in range(case["steps"]):
        line = out.logits[0, -1].cpu().clone()
        logits.append(line)
        tokens.append(int(line.argmax()))
        if step + 1 < case["steps"]:
            token = tokens[-1] if forced is None else int(forced[step])
            out = net(input_ids=torch.tensor([[token]], device=device), past_key_values=past, use_cache=True)
            past = out.past_key_values
    return {**case, "reference": torch.stack(logits), "forced": torch.tensor(tokens if forced is None else forced)}


def resources(model, module, packed_file=None):
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


@torch.no_grad()
def greedy(module, gpu, shared, cases):
    stamp = time.perf_counter()
    engine = module.Engine2(512, gpu, *shared)
    matches, sampler_errors = 0, []
    for case in cases:
        engine.prefill(case["prompt"])
        for step in range(case["steps"]):
            token = int(engine.bufs["tok"].cpu()[0])
            matches += token == int(case["forced"][step])
            err = sampler_mismatch(engine, engine.logits().cpu())
            if err:
                sampler_errors.append({"case": case["name"], "step": step, **err})
            if step + 1 < case["steps"]:
                engine.mega(1)
    del engine
    return {"matches": matches, "total": sum(c["steps"] for c in cases), "sampler_errors": sampler_errors,
            "wall_s": time.perf_counter() - stamp}


def head_setup(cache, gpu):
    tokens = torch.cat([c["forced"] for c in cache["cases"]])
    return [{"perm": perm, "gpu": extra_tests.permuted_head(gpu, perm), "cases": extra_tests.permuted_cases(cache["cases"], perm)}
            for perm in extra_tests.head_permutations(tokens)]


def verdict(stage, seconds, **detail):
    return {"status": stage, "wall_s": seconds, **detail}


def run_gate(module, gpu, shared, cache, head, jit_module=None):
    """Run every stage and return (stages, summary). Never raises on a numerical failure."""
    bound = cache["bound"]
    stages = {name: verdict("NOT_RUN", 0.) for name in ("determinism", "logit_bound", "argmax", "greedy", *extra_tests.EXTRA_ORDER)}
    first, stored = pass_cases(module, gpu, shared, cache["cases"], bound=bound, fail_fast=True, check_sampling=True)
    out = {"max_diff": first["max_diff"], "teacher_steps": first["steps"], "gpu_stage": "SURVIVED", "extra_stage": None,
           "extra_fails": [], "equivalent": False}
    out.update({"mean_kl": first["mean_kl"], "rms_diff": first["rms_diff"], "orig_max_diff": first["orig_max_diff"]})
    stages["logit_bound"] = verdict("FAIL" if first["kill"] == "logit_bound" else "PASS", first["wall_s"], bound=bound)
    stages["argmax"] = verdict("FAIL" if first["kill"] == "argmax" else "PASS", 0., events=first["events"])
    if first["kill"]:
        out["gpu_stage"] = first["kill"]
        return stages, out
    second, _ = pass_cases(module, gpu, shared, cache["cases"], prior=stored)
    stages["determinism"] = verdict("PASS" if second["bitwise_repeat"] else "FAIL", second["wall_s"])
    if not second["bitwise_repeat"]:
        out["gpu_stage"] = "determinism"
        return stages, out
    free = greedy(module, gpu, shared, cache["cases"])
    stages["greedy"] = verdict("MATCH" if free["matches"] == free["total"] else "MISMATCH", free["wall_s"],
                               matches=free["matches"], total=free["total"], binding=False)
    sample_errors = first["sample_errors"] + free["sampler_errors"]
    equal = [first["original_equal"]]
    fails = {}

    def passes(cases, gpu_, name):
        res, logs = pass_cases(module, gpu_, shared, cases, bound=bound, fail_fast=True, check_sampling=True)
        equal.append(res["original_equal"] and not res["kill"])
        sample_errors.extend(res["sample_errors"])
        fails[name] = res["kill"]
        return res, logs

    longctx = [c for c in cache["extra"] if c["name"].startswith("context")]
    adverse = [c for c in cache["extra"] if c["name"].startswith("adversarial")]
    ctx_res, logs = passes(longctx, gpu, "context")
    stages["context"] = verdict("FAIL" if ctx_res["kill"] else "PASS", ctx_res["wall_s"], max_diff=ctx_res["max_diff"], steps=ctx_res["steps"])
    res, _ = passes(adverse, gpu, "adversarial")
    stages["adversarial"] = verdict("FAIL" if res["kill"] else "PASS", res["wall_s"], max_diff=res["max_diff"], steps=res["steps"])
    killed_perm, wall, worst = False, 0., 0.
    for variant in head:
        res, _ = passes(variant["cases"], variant["gpu"], "head_perm")
        killed_perm |= bool(res["kill"])
        wall += res["wall_s"]
        worst = max(worst, res["max_diff"] if res["max_diff"] is not None else float("inf"))
    stages["head_perm"] = verdict("FAIL" if killed_perm else "PASS", wall, max_diff=worst, variants=len(head))
    stamp = time.perf_counter()
    ms = extra_tests.multistep(module, gpu, shared, cache["cases"][2])
    stages["multistep"] = verdict("PASS" if ms["pass"] else "FAIL", time.perf_counter() - stamp, **ms)
    if ctx_res["kill"]:
        stages["repeat"] = verdict("NOT_RUN", 0., reason="context pass was cut short by a kill")
    else:
        rep, _ = pass_cases(module, gpu, shared, longctx, prior=logs)
        stages["repeat"] = verdict("PASS" if rep["bitwise_repeat"] else "FAIL", rep["wall_s"])
    if jit_module is None:
        stages["jitter"] = verdict("NOT_RUN", 0., reason="no jitter build for this mutant")
    else:
        jit_equal, jit_wall = True, 0.
        try:
            for _ in range(2):
                jit, _ = pass_cases(jit_module, gpu, shared, cache["cases"], prior=stored)
                jit_equal &= jit["bitwise_repeat"]
                jit_wall += jit["wall_s"]
        except (AssertionError, RuntimeError):
            jit_equal = False
            jit_wall = -1.
        stages["jitter"] = verdict("PASS" if jit_equal else "FAIL", jit_wall)
    kl_limit = cache["kl_bound"]
    stages["distribution"] = verdict("FAIL" if first["mean_kl"] > kl_limit else "PASS", 0., mean_kl=first["mean_kl"], limit=kl_limit)
    equal.append(stages["jitter"]["status"] != "FAIL")
    stages["sampling"] = verdict("FAIL" if sample_errors else "PASS", 0., errors=sample_errors[:6], count=len(sample_errors))
    out["extra_fails"] = [n for n in extra_tests.EXTRA_ORDER if stages[n]["status"] == "FAIL"]
    out["extra_stage"] = out["extra_fails"][0] if out["extra_fails"] else "SURVIVED"
    out["equivalent"] = all(equal) and not out["extra_fails"]
    out["equivalence_scope"] = "bitwise-identical logits on base + extra cases + sampler; not a global proof"
    return stages, out


def schedule_stage(spec):
    if spec["family"] != "synchronization" or spec["operator"] not in ("remove_grid", "wait_one_less", "wait_previous"):
        return verdict("NOT_APPLICABLE", 0., reason="IR v1 expresses grid-wait edges only; CTA-local syncs and early release are not events")
    path = ROOT / ("kernels/megakernel_v2/schedules/mk2_ctx128.json" if spec["model"] == "0.6B" else "kernels/megakernel_scale/schedules/scale_ctx128.json")
    ir = json.loads(path.read_text())
    layer, site = spec["site"]["layer"], spec["site"]["projection"]
    flag = f"L{layer}.{PHASE_FLAGS[site]}_done" if site < 5 else ("lm_done", "argp_done", "argc_done")[site - 5]
    waits = 0
    for task in ir["tasks"]:
        for wait in task["waits"]:
            if wait["flag"] == flag:
                waits += 1
                if spec["operator"] == "wait_one_less":
                    wait["value"] -= 1
                elif spec["operator"] == "wait_previous":
                    wait["value"] = 0
        if spec["operator"] == "remove_grid":
            task["waits"] = [w for w in task["waits"] if w["flag"] != flag]
    folder = (ROOT / spec["library"]).parent
    (folder / "schedule.json").write_text(json.dumps(ir))
    stamp = time.perf_counter()
    proc = subprocess.run([str(ROOT / "validator/target/release/schedcheck"), str(folder / "schedule.json")], capture_output=True, text=True, timeout=60)
    (folder / "schedule.log").write_text(proc.stdout + proc.stderr)
    return verdict("PASS" if proc.returncode == 0 else "FAIL", time.perf_counter() - stamp, waits_edited=waits,
                   file=str((folder / "schedule.json").relative_to(ROOT)), verdict_line=(proc.stdout.strip().splitlines() or [""])[0][:200])


def prepare_hf(model):
    """HF fp32 fake-quant reference for the base + extra cases (CPU for 1.7B: run WITHOUT the GPU lock)."""
    stamp = time.perf_counter()
    torch.set_num_threads(8 if model == "1.7B" else 12)
    bench = ROOT / ("bench/lm11" if model == "0.6B" else "bench/lm12")
    deq = torch.load(bench / "weights_int4_gptq_deq.pt", map_location="cpu")
    net, snapshot = reference_model(model, deq)
    tok = AutoTokenizer.from_pretrained(snapshot)
    cases = [{"name": f"prompt-{i}", "prompt": tok(t, return_tensors="pt").input_ids[0], "steps": 64} for i, t in enumerate(PROMPTS)]
    refs = [hf_case(net, c) for c in cases]
    extra = [hf_case(net, c) for c in scenarios(tok)]
    del net, deq
    torch.cuda.empty_cache()
    return cases, refs, extra, time.perf_counter() - stamp


def prepare(model, phase="all"):
    stamp = time.perf_counter()
    torch.set_num_threads(12)
    bench = ROOT / ("bench/lm11" if model == "0.6B" else "bench/lm12")
    hf_file = HERE / f"reference-{model}-hf.pt"
    if phase == "hf":
        cases, refs, extra, hf_s = prepare_hf(model)
        torch.save({"cases": cases, "refs": refs, "extra": extra, "hf_s": hf_s}, hf_file)
        print(json.dumps({"hf_wall_s": hf_s}), flush=True)
        return
    if phase == "gpu":
        saved = torch.load(hf_file, weights_only=False, map_location="cpu")
        cases, refs, extra, hf_s = saved["cases"], saved["refs"], saved["extra"], saved["hf_s"]
    else:
        cases, refs, extra, hf_s = prepare_hf(model)
    module = bridge(model)
    gpu, shared = resources(model, module)
    first, stored = pass_cases(module, gpu, shared, refs, check_sampling=True)
    repeat, _ = pass_cases(module, gpu, shared, refs, prior=stored)
    control_diff = first["max_diff"]
    # The 0.6B v2.1 control is RTN vs its own fake-quant reference, forced with GPTQ's HF sequence
    # (bound = min(0.5, 1.25 x control)). The 1.7B gate has no relative term: bound = 0.5.
    if model == "0.6B":
        del gpu, shared
        torch.cuda.empty_cache()
        from ppl_llama_protocol import deq_int4
        pk = torch.load(ROOT / "bench/lm03b/weights_int4.pt", map_location="cpu")
        deq = {n: deq_int4(v["codes"], v["meta"]) for n, v in pk.items()}
        net, _ = reference_model(model, deq)
        control_refs = [hf_case(net, c, refs[i]["forced"]) for i, c in enumerate(cases)]
        del net, deq, pk
        torch.cuda.empty_cache()
        gpu, shared = resources(model, module, ROOT / "bench/lm03b/weights_int4.pt")
        control, _ = pass_cases(module, gpu, shared, control_refs)
        control_diff = control["max_diff"]
        control_kl = control["mean_kl"]
        del gpu, shared
        gpu, shared = resources(model, module)
    else:
        control_kl = None
    bound = min(.5, 1.25 * control_diff) if model == "0.6B" else .5
    if first["nonfinite"] or first["max_diff"] > bound or any(e["kind"] == "hard" for e in first["events"]) or not repeat["bitwise_repeat"] or first["sample_errors"]:
        raise RuntimeError("known-correct original failed cache preparation")
    for c, logs in zip(refs, stored):
        c["original"] = logs
    for c in extra:
        res, logs = pass_cases(module, gpu, shared, [c], check_sampling=True)
        c["original"] = logs[0]
        if res["sample_errors"]:
            raise RuntimeError(f"original sampler mismatch on {c['name']}")
    kl_bound = 1.25 * (control_kl if control_kl is not None else first["mean_kl"])
    cache = {"model": model, "cases": refs, "extra": extra, "bound": bound, "control_diff": control_diff, "hf_wall_s": hf_s,
             "control_kl": control_kl, "original_kl": first["mean_kl"], "kl_bound": kl_bound}
    head = head_setup(cache, gpu)
    jit_lib = HERE / "build" / f"original-{model}" / "original_jit.so"
    jit_module = bridge(model, jit_lib, HERE / "build" / f"original-{model}-jit") if jit_lib.exists() else None
    stages, out = run_gate(module, gpu, shared, cache, head, jit_module)  # the original must pass every stage and be self-equivalent
    if out["gpu_stage"] != "SURVIVED" or out["extra_stage"] != "SURVIVED" or not out["equivalent"]:
        raise RuntimeError(f"known-correct original fails its own gate: {out} {stages}")
    cache.update({"original_gate": {"stages": stages, **out}, "prepare_wall_s": time.perf_counter() - stamp,
                  "original_first": first, "original_repeat": repeat})
    torch.save(cache, HERE / f"reference-{model}.pt")
    summary = {k: v for k, v in cache.items() if k not in ("cases", "extra")}
    (HERE / f"reference-{model}.json").write_text(json.dumps(summary, indent=2, default=lambda o: None))
    print(json.dumps({"bound": bound, "control_diff": control_diff, "orig_max_diff": first["max_diff"], "hf_wall_s": hf_s,
                      "kl": [first["mean_kl"], control_kl, kl_bound], "stages": {k: v["wall_s"] for k, v in stages.items()}}), flush=True)


def finite(value):
    """JSON cannot carry nan/inf; keep them as strings so a non-finite logit stays visible."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def append(row):
    with (HERE / "results.jsonl").open("a") as out:
        out.write(json.dumps(finite(row), allow_nan=False) + "\n")


def arm(spec, seconds):
    def fire():
        append({**spec, "stage": "execution_timeout", "extra_stage": None, "gpu_stage": "execution_timeout", "max_diff": None,
                "equivalent": False, "measurement": "measured", "producing_script": "mutation/gate.py",
                "runtime_s": seconds, "stages": {"execution": {"status": "HANG", "deadline_s": seconds}}})
        print(spec["mutant_id"], "execution_timeout", flush=True)
        os._exit(124)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def evaluate(spec_files, deadline):
    specs = [json.loads(f.read_text()) for f in spec_files]
    model = specs[0]["model"]
    load_stamp = time.perf_counter()
    torch.set_num_threads(4)
    cache = torch.load(HERE / f"reference-{model}.pt", weights_only=False, map_location="cpu")
    original = bridge(model)
    gpu, shared = resources(model, original)
    head = head_setup(cache, gpu)
    print(f"session load {time.perf_counter() - load_stamp:.1f}s", flush=True)
    for spec_file, spec in zip(spec_files, specs):
        stamp = time.perf_counter()
        timer = arm(spec, deadline)
        stages = {"build": verdict(spec["build"], spec["build_s"])}
        row = {**spec, "stage": "SURVIVED", "gpu_stage": None, "extra_stage": None, "max_diff": None, "equivalent": False,
               "measurement": "measured", "producing_script": "mutation/gate.py"}
        try:
            stages["schedule"] = schedule_stage(spec)
            module = bridge(model, ROOT / spec["library"], spec_file.parent)
            jit_lib = spec_file.parent / "mutant_jit.so"
            jit_module = bridge(model, jit_lib, spec_file.parent / "jit") if jit_lib.exists() else None
            gate_stages, out = run_gate(module, gpu, shared, cache, head, jit_module)
            stages.update(gate_stages)
            row.update(out)
            row["stage"] = "schedule" if stages["schedule"]["status"] == "FAIL" else out["gpu_stage"]
            if row["stage"] == "SURVIVED" and out["equivalent"]:  # validator PASS/N-A and bitwise-identical everywhere
                row["stage"], row["extra_stage"] = "EQUIVALENT", "EQUIVALENT"
        except Exception as exc:  # a crashed/poisoned device is a gate failure, not an infrastructure failure
            message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            timer.cancel()
            if not any(w in message.lower() for w in DEVICE_ERRORS) and not isinstance(exc, (AssertionError, RuntimeError)):
                raise
            if "out of memory" in message.lower():
                raise
            row.update({"stage": "execution_error", "gpu_stage": "execution_error", "extra_stage": None})
            stages["execution"] = verdict("FAIL", time.perf_counter() - stamp, error=message[:400])
            row.update(stages=stages, runtime_s=time.perf_counter() - stamp)
            append(row)
            print(spec["mutant_id"], "execution_error", message[:120], flush=True)
            sys.exit(3)  # the CUDA context may be poisoned; the driver restarts for the remaining mutants
        timer.cancel()
        row.update(stages=stages, runtime_s=time.perf_counter() - stamp)
        append(row)
        print(spec["mutant_id"], row["stage"], row["extra_stage"], row["max_diff"], f"{row['runtime_s']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--model", choices=("0.6B", "1.7B"), default="0.6B")
    parser.add_argument("--phase", choices=("all", "hf", "gpu"), default="all")
    parser.add_argument("--spec", type=Path, nargs="+")
    parser.add_argument("--deadline", type=int, default=45)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.model, args.phase)
    else:
        evaluate(args.spec, args.deadline)


if __name__ == "__main__":
    main()
