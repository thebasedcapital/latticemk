"""Single-fault CUDA mutants of the INT4 decode megakernel, run through the v2.1 gate rules.

CPU  build:  .venv/bin/python mutation/mutate.py build --model 0.6B
GPU  prepare (HF reference cache, once per model):
             scripts/gpu.sh .venv/bin/python mutation/gate.py prepare --model 0.6B
GPU  run:    .venv/bin/python mutation/mutate.py run --model 0.6B --batch 8
List catalogue: .venv/bin/python mutation/mutate.py list

Each mutant is a text-level copy of the known-correct kernel with ONE operator at ONE site.
GEMV / attention / barrier faults are injected by cloning the original function as *_mut and
dispatching to the clone only at the chosen (layer, projection) site, so every other site runs
the original code. CTA-local __syncthreads faults are guarded by a shared `mut_l` layer tag.
Compile failures are recorded as COMPILE_FAILURE and never count as kills.
Originals are never edited; mutants live under mutation/build/ (git-ignored).
Every GPU subprocess goes through scripts/gpu.sh and carries a hard deadline.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "mutation"
SOURCES = {"0.6B": ROOT / "kernels/megakernel_v2/mega2.cu",
           "1.7B": ROOT / "kernels/megakernel_scale/mega_scale.cu"}
LAYERS = (0, 3, 7, 13, 20, 27)
NVCC = Path.home() / ".local/cuda-13.3/bin/nvcc"
CUDA_LIB = Path.home() / ".local/cuda-13.3/lib"

# Operators applied to a GEMV (dot4 / gemv2 / gemv_run clone). site = (layer, projection); layer -1 = lm_head.
GEMV_OPS = {
    "precision": ("drop_scale", "drop_offset", "fp16_partial", "truncate_reduction", "round_zero", "half_partial", "fp16_dot"),
    "bounds": ("row_tail", "chunk_floor"),
    "dequant": ("swap_nibbles", "group_next", "scale_row", "layout_quarter", "nibble_shift"),
}
ATTN_OPS = {
    "bounds": ("kv_previous", "kv_next", "gqa_head", "append_next", "append_prev", "rope_pos_next", "warp_chunk_off", "slice_gap"),
    "attention": ("softmax_min", "no_head_scale", "skip_last", "skip_slice_last", "half_score", "skip_first",
                  "long_skip_last", "long_kv_previous"),
}
GRID_OPS = ("remove_grid", "wait_one_less", "wait_previous", "release_early", "drop_pre_sync", "drop_post_sync")
LOCAL_SYNC = {  # operator -> (function start marker, function end marker, nth __syncthreads, final-phase only)
    "drop_sync_block_sum0": ("__device__ float block_sum", "// grid barrier", 0, False),
    "drop_sync_block_sum1": ("__device__ float block_sum", "// grid barrier", 1, False),
    "drop_sync_block_sum2": ("__device__ float block_sum", "// grid barrier", 2, False),
    "drop_sync_stage_norm": ("__device__ void stage_norm", "// qkv/gu/lm prologue", 0, False),
    "drop_sync_attnc": ("__device__ void pro_attnc", "// down-proj prologue", 0, False),
    "drop_sync_silu": ("__device__ void pro_silu", "// raw transposed copy", 0, False),
    "drop_sync_attn_pro": ("__device__ void attn2", "// ============================ argmax", 0, False),
    "drop_sync_attn_part": ("__device__ void attn2", "// ============================ argmax", 1, False),
    "drop_sync_argp": ("__device__ void argp2", "__device__ void argc2", 0, True),
    "drop_sync_argc": ("__device__ void argc2", "// ============================ main kernel", 0, True),
}
SAMPLER_OPS = ("sampler_tail", "sampler_warp", "sampler_token", "sampler_reduce")
CONDITIONAL = ("long_skip_last", "long_kv_previous")  # fault only fires at pos >= 128 (unreachable in the base gate)
PHASES = ("qkv", "attn", "o", "gu", "down")


def _family_of(op):
    for table in (GEMV_OPS, ATTN_OPS):
        for family, ops in table.items():
            if op in ops:
                return family
    if op in GRID_OPS or op in LOCAL_SYNC:
        return "synchronization"
    if op in SAMPLER_OPS:
        return "bounds"
    raise KeyError(op)


def catalogue(model):
    rng = random.Random(f"lm16-{model}")
    full = model == "0.6B"
    layers = LAYERS if full else (0, 7, 20)
    rows = []

    def add(op, layer, proj, family=None):
        name = f"L{layer}" if layer >= 0 else "Lhead"
        rows.append({"mutant_id": f"{model}-{family or _family_of(op)}-{op}-{name}-P{proj}", "model": model,
                     "family": family or _family_of(op), "operator": op,
                     "site": {"layer": layer, "projection": proj},
                     "conditional": op in CONDITIONAL})

    # Per-operator layer/projection counts: dense on 0.6B, thinner on 1.7B (build/run budget).
    limit = {"round_zero": 3, "half_partial": 3, "kv_previous": 4, "kv_next": 4, "gqa_head": 4, "append_next": 3,
             "append_prev": 3, "rope_pos_next": 3, "warp_chunk_off": 3, "slice_gap": 3, "skip_first": 3,
             "skip_slice_last": 4, "half_score": 3}
    for opidx, op in enumerate(sum((v for v in GEMV_OPS.values()), ())):
        if op == "fp16_dot" and full:
            continue
        for li, layer in enumerate(layers[:limit.get(op, 6)]):
            add(op, layer, (li + opidx) % 4)
        if op in ("fp16_partial", "truncate_reduction", "half_partial", "row_tail", "chunk_floor", "group_next", "scale_row") and full:
            add(op, -1, 4)
    for op in sum((v for v in ATTN_OPS.values()), ()):
        for li, layer in enumerate(layers[:limit.get(op, 6)]):
            if op.startswith("long_") and not full and li > 1:
                continue
            add(op, layer, 1)
    for op in SAMPLER_OPS:
        add(op, -1, 4)
    sites = [(layer, phase) for phase in range(5) for layer in rng.sample(range(28), 3)]
    sites += [(-1, 5), (-1, 6), (-1, 7)]  # lm_head, argmax partial, argmax combine barriers
    for op in GRID_OPS:
        for layer, phase in rng.sample(sites, 6 if full else 3):
            add(op, layer, phase)
    for op, (_, _, _, final) in LOCAL_SYNC.items():
        if final:
            add(op, -1, 5)
        else:
            for layer in rng.sample(range(28), 2 if full else 1):
                add(op, layer, 0)
    if full:
        # Null-mutation controls: cloned-dispatch scaffolding with NO behavioural change. Must be bitwise equal.
        for op in ("null_gemv", "null_attn", "null_gbar", "null_sync"):
            add(op, 3, 0, family="control")
    return rows


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one mutation anchor, got {text.count(old)}: {old}")
    return text.replace(old, new, 1)


def span(text, start_marker, end_marker):
    start = text.index(start_marker)
    return start, text.index(end_marker, start)


def drop_nth_sync(part, n, cond):
    hits = [m.start() for m in re.finditer(r"__syncthreads\(\);", part)]
    if n >= len(hits):
        raise ValueError(f"function has only {len(hits)} __syncthreads")
    at = hits[n]
    return part[:at] + f"if ({cond}) __syncthreads();" + part[at + len("__syncthreads();"):]


def mutate_gemv(text, op, layer, phase):
    start, end = span(text, "__device__ __forceinline__ float dot4", "// ---- prologues:")
    part = text[start:end]
    for name in ("dot4", "gemv2", "gemv_run"):
        part = re.sub(rf"\b{name}\b", name + "_mut", part)
    if op == "null_gemv":
        pass
    elif op == "drop_scale":
        part = part.replace("* __low2float(meta)", "* 1.f")
    elif op == "drop_offset":
        part = part.replace("+ __high2float(meta) * xsum", "+ 0.f")
    elif op == "half_partial":
        part = replace_once(part, "float v = acc[0];", "float v = __half2float(__float2half(acc[0]));")
    elif op == "fp16_partial":
        part, count = re.subn(r"acc\[r\] \+= (dot4_mut\([^;]+\));", r"acc[r] = __half2float(__float2half(acc[r] + \1));", part)
        if count != 1:
            raise ValueError("missing partial-accumulation site")
    elif op == "fp16_dot":
        part = replace_once(part, "#ifdef FP32_DOT", "#if 0")
    elif op == "truncate_reduction":
        part = replace_once(part, "acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);",
                            "acc[r] = __half2float(__float2half(acc[r] + __shfl_xor_sync(0xffffffffu, acc[r], o)));")
    elif op == "round_zero":
        part = replace_once(part, "__float2half(v)", "__float2half_rz(v)")
    elif op == "row_tail":
        part = replace_once(part, "min(w.n_out, r0 + chunk)", "min(w.n_out, r0 + chunk) - 1")
    elif op == "chunk_floor":
        part = replace_once(part, "(w.n_out + NCTA - 1) / NCTA", "w.n_out / NCTA")
    elif op == "swap_nibbles":
        part = part.replace("(4 * j)", "(4 * (3 - j))")
    elif op == "group_next":
        part = part.replace("(c >> 2)", "(((c >> 2) + 1) % ngrp)")
    elif op == "scale_row":
        part = part.replace("(int64_t)(row0 + r * WARPS) * ngrp", "(int64_t)((row0 + r * WARPS + 1) % w.n_out) * ngrp")
        part = part.replace("(int64_t)row * ngrp", "(int64_t)((row + 1) % w.n_out) * ngrp")
    elif op == "layout_quarter":
        part = part.replace("t2 * cpr + c", "((t2 + 1) % 4) * cpr + c")
    elif op == "nibble_shift":
        part = part.replace("(4 * j)", "(4 * ((j + 1) % 4))")
    else:
        raise ValueError(op)
    text = text[:end] + part + text[end:]
    if layer < 0:
        return replace_once(text, "gemv_run(p.w[W_LM], cta);", "gemv_run_mut(p.w[W_LM], cta);")
    old = f"gemv_run(p.w[l * 4 + {phase}], cta);"
    return replace_once(text, old, f"if (l == {layer}) gemv_run_mut(p.w[l * 4 + {phase}], cta); else {old}")


def mutate_attn(text, op, layer):
    start, end = span(text, "__device__ void attn2", "// ============================ argmax")
    part = text[start:end].replace("void attn2(", "void attn2_mut(")
    if op == "null_attn":
        pass
    elif op in ("kv_previous", "long_kv_previous"):
        part = part.replace("(int64_t)t * KVROWS", "(int64_t)max(0, t - 1) * KVROWS")
    elif op == "kv_next":
        part = part.replace("(int64_t)t * KVROWS", "(int64_t)(t + 1) * KVROWS")
    elif op == "gqa_head":
        part = replace_once(part, "kvh = h >> 1", "kvh = (h + 1) % 8")
    elif op == "append_next":
        part = part.replace("((int64_t)layer * MAXPOS + pos)", "((int64_t)layer * MAXPOS + pos + 1)")
    elif op == "append_prev":
        part = part.replace("((int64_t)layer * MAXPOS + pos)", "((int64_t)layer * MAXPOS + max(pos - 1, 0))")
    elif op == "rope_pos_next":
        part = part.replace("layer, false, pos,", "layer, false, pos + 1,").replace("layer, true, pos,", "layer, true, pos + 1,")
    elif op == "warp_chunk_off":
        part = replace_once(part, "(len + WARPS - 1) / WARPS", "(len + WARPS - 2) / WARPS")
    elif op == "slice_gap":
        part = replace_once(part, "const int a0 = half ? mid : 0,", "const int a0 = half ? min(mid + 1, npos) : 0,")
    elif op == "skip_first":
        part = replace_once(part, "const int a0 = half ? mid : 0,", "const int a0 = half ? mid : (mid > 1 ? 1 : 0),")
    elif op == "softmax_min":
        part = replace_once(part, "fmaxf(m, s)", "fminf(m, s)")
    elif op == "no_head_scale":
        part = replace_once(part, "warp_sum(s) * 0.08838834764831845f", "warp_sum(s)")
    elif op in ("skip_last", "long_skip_last"):
        part = replace_once(part, "const int npos = pos + 1;", "const int npos = max(1, pos);")
    elif op == "skip_slice_last":
        part = replace_once(part, "t < w1;", "t < w1 - 1;")
    elif op == "half_score":
        part = replace_once(part, "const float mn =", "s = __half2float(__float2half(s));\n    const float mn =")
    else:
        raise ValueError(op)
    text = text[:end] + part + text[end:]
    cond = f"l == {layer}" + (" && pos >= 128" if op.startswith("long_") else "")
    return replace_once(text, "if (cta < NATTN) attn2(p, l, pos, cta);",
                        f"if (cta < NATTN) {{ if ({cond}) attn2_mut(p, l, pos, cta); else attn2(p, l, pos, cta); }}")


def mutate_grid(text, op, layer, site):
    start, end = span(text, "__device__ __forceinline__ void gbar", "// ========================== GEMV")
    part = text[start:end].replace("void gbar(", "void gbar_mut(")
    if op == "null_gbar":
        pass
    elif op == "remove_grid":
        part = "__device__ __forceinline__ void gbar_mut(int*, int&) {}\n\n"
    elif op == "wait_one_less":
        part = replace_once(part, "< expect)", "< expect - 1)")
    elif op == "wait_previous":
        part = replace_once(part, "< expect)", "< expect - NCTA)")
    elif op == "release_early":
        part = replace_once(part, "  __syncthreads();\n  if (threadIdx.x == 0) {\n    red_rel(ctr, 1);",
                            "  if (threadIdx.x == 0) red_rel(ctr, 1);\n  __syncthreads();\n  if (threadIdx.x == 0) {")
    elif op == "drop_pre_sync":
        part = drop_nth_sync(part, 0, "false")
        part = part.replace("if (false) __syncthreads();", "")
    elif op == "drop_post_sync":
        part = drop_nth_sync(part, 1, "false")
        part = part.replace("if (false) __syncthreads();", "")
    else:
        raise ValueError(op)
    text = text[:end] + part + text[end:]
    main = text.index("__global__ void __launch_bounds__(THREADS, 1) mega2_kernel")
    stop = text.index("// ================== stage test", main)
    calls = [m for m in re.finditer(r"gbar\(p\.bar, expect\);", text[main:stop])]
    if len(calls) != 8:
        raise ValueError(f"expected 8 grid barriers in the step loop, found {len(calls)}")
    where = main + calls[site].start()
    new = "gbar_mut(p.bar, expect);" if layer < 0 else f"if (l == {layer}) gbar_mut(p.bar, expect); else gbar(p.bar, expect);"
    return text[:where] + new + text[where + len("gbar(p.bar, expect);"):]


def mutate_local_sync(text, op, layer):
    marker, end_marker, n, final = LOCAL_SYNC[op] if op != "null_sync" else ("__device__ float block_sum", "// grid barrier", 0, False)
    cond = "mut_l != -2" if op == "null_sync" else f"mut_l != {layer}"
    start, end = span(text, marker, end_marker)
    text = text[:start] + drop_nth_sync(text[start:end], n, cond) + text[end:]
    text = replace_once(text, "__shared__ float red[WARPS];", "__shared__ float red[WARPS];\n__shared__ int mut_l;")
    text = replace_once(text, "    for (int l = 0; l < NLAY; ++l) {\n      // PH qkv", "    for (int l = 0; l < NLAY; ++l) {\n      mut_l = l;\n      // PH qkv")
    return replace_once(text, "    // lm_head: residual += dout + final norm -> sx; logits (f32)\n",
                        "    // lm_head: residual += dout + final norm -> sx; logits (f32)\n    mut_l = -1;\n")


def mutate_sampler(text, op):
    start, end = span(text, "__device__ void argp2", "// ============================ main kernel")
    part = text[start:end]
    if op == "sampler_tail":
        part = replace_once(part, "i < a1;", "i < a1 - 1;")
    elif op == "sampler_warp":
        part = part.replace("for (int o = 16;", "for (int o = 4;")
    elif op == "sampler_token":
        part = replace_once(part, "p.tok[0] = bi; p.tok_hist[pos] = bi;", "p.tok[0] = (bi + 1) % VOCAB; p.tok_hist[pos] = (bi + 1) % VOCAB;")
    else:
        part = replace_once(part, "threadIdx.x < NARGP", "threadIdx.x < NARGP - 1")
    return text[:start] + part + text[end:]


def mutate(text, spec):
    op = spec["operator"]
    layer, phase = spec["site"]["layer"], spec["site"]["projection"]
    if op in SAMPLER_OPS:
        return mutate_sampler(text, op)
    if op in LOCAL_SYNC or op == "null_sync":
        return mutate_local_sync(text, op, layer)
    if op in GRID_OPS or op == "null_gbar":
        return mutate_grid(text, op, layer, phase)
    if op in sum(ATTN_OPS.values(), ()) or op == "null_attn":
        return mutate_attn(text, op, layer)
    return mutate_gemv(text, op, layer, phase)


JITTER = """
// ---- schedule-skew injection (mutate.py jitter) ------------------------------------------------------------
// mk_jit : pseudo-random __nanosleep (<~1 us) before and after every __syncthreads.
// mk_slow: at GEMV / attention entry ~25% of the warps sleep 2-4 us, so warps finish their global stores at
//          different times; a missing pre-barrier __syncthreads or an early release then exposes stale reads.
__device__ __forceinline__ void mk_jit() {
  unsigned h = ((unsigned)clock() ^ (threadIdx.x * 2654435761u) ^ (blockIdx.x * 40503u)) * 2246822519u;
  h ^= h >> 15;
  if ((h & 3u) == 0) __nanosleep(64 + ((h >> 4) & 1023u));
}
__device__ __forceinline__ void mk_slow() {
  unsigned h = (((unsigned)clock() >> 11) ^ ((threadIdx.x >> 5) * 2654435761u) ^ (blockIdx.x * 40503u)) * 2246822519u;
  h ^= h >> 13;
  if ((h & 3u) == 0) __nanosleep(2000 + ((h >> 4) & 2047u));
}
__device__ __forceinline__ void mk_sync_real() { __syncthreads(); }
#define __syncthreads() do { mk_jit(); mk_sync_real(); mk_jit(); } while (0)
"""


def jitter_text(text):
    text = replace_once(text, "#include <cuda_runtime.h>\n", "#include <cuda_runtime.h>\n" + JITTER)
    text, n_gemv = re.subn(r"(__device__ __forceinline__ void gemv2\w*\(const W2& w, int r0, int r1\) \{)", r"\1\n  mk_slow();", text)
    text, n_attn = re.subn(r"(__device__ void attn2\w*\(const P2& p, int layer, int pos, int cta\) \{)", r"\1\n  mk_slow();", text)
    if n_gemv < 1 or n_attn < 1:
        raise ValueError(f"jitter anchors missing: gemv2 {n_gemv}, attn2 {n_attn}")
    return text


def compile_cu(model, cu, lib):
    cmd = [str(NVCC), "-O3", "-arch=sm_75", "-std=c++17", "-allow-unsupported-compiler", "-L" + str(CUDA_LIB),
           "-Xcompiler", "-fPIC", "-shared", "-cudart", "static", "-Xptxas=-v", "-o", str(lib), str(cu)]
    if model == "1.7B":
        cmd.insert(1, "-DFP32_DOT")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    (lib.parent / (lib.stem + ".build.log")).write_text(proc.stdout + proc.stderr)
    return proc.returncode


def jitter_one(model, folder):
    folder = ROOT / folder
    source = folder / "mutant.cu" if (folder / "mutant.cu").exists() else SOURCES[model]
    (folder / "mutant_jit.cu").write_text(jitter_text(source.read_text()))
    return compile_cu(model, folder / "mutant_jit.cu", folder / "mutant_jit.so")


def build_one(spec):
    stamp = time.perf_counter()
    source = SOURCES[spec["model"]]
    text = source.read_text()
    mutated = mutate(text, spec)
    if text == mutated:
        raise ValueError("mutation did not change source")
    build_root = HERE / "build"
    build_root.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=spec["mutant_id"] + "-", dir=build_root))
    cu, lib = folder / "mutant.cu", folder / "mutant.so"
    cu.write_text(mutated)
    cmd = [str(NVCC), "-O3", "-arch=sm_75", "-std=c++17", "-allow-unsupported-compiler", "-L" + str(CUDA_LIB),
           "-Xcompiler", "-fPIC", "-shared", "-cudart", "static", "-Xptxas=-v", "-o", str(lib), str(cu)]
    if spec["model"] == "1.7B":
        cmd.insert(1, "-DFP32_DOT")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    (folder / "build.log").write_text(proc.stdout + proc.stderr)
    row = {**spec, "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "library": str(lib.relative_to(ROOT)), "build_s": time.perf_counter() - stamp,
           "build": "PASS" if proc.returncode == 0 else "COMPILE_FAILURE"}
    (folder / "spec.json").write_text(json.dumps(row, indent=2))
    return row


def validator_only(model):
    """IR-level 'wait on the wrong flag id' mutants. The kernel has one barrier counter, so this fault has no
    kernel twin; it measures the schedule validator alone (family validator_only, outside the gate kill rates)."""
    ir_path = ROOT / ("kernels/megakernel_v2/schedules/mk2_ctx128.json" if model == "0.6B" else "kernels/megakernel_scale/schedules/scale_ctx128.json")
    base = json.loads(ir_path.read_text())
    flags = [f["id"] for f in base["flags"]]
    rng = random.Random(f"lm16-validator-{model}")
    out = HERE / "results.jsonl"
    done = {r["mutant_id"] for r in load_jsonl(out)}
    for layer in rng.sample(range(1, 28), 8):
        phase = rng.randrange(5)
        right = f"L{layer}.{PHASES[phase]}_done"
        wrong = flags[flags.index(right) - 1 - rng.randrange(2)]
        mid = f"{model}-validator_only-wait_wrong_flag-L{layer}-P{phase}"
        if mid in done:
            continue
        ir = json.loads(ir_path.read_text())
        edited = 0
        for task in ir["tasks"]:
            for wait in task["waits"]:
                if wait["flag"] == right:
                    wait["flag"] = wrong
                    edited += 1
        folder = HERE / "build" / f"validator-{mid}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "schedule.json").write_text(json.dumps(ir))
        stamp = time.perf_counter()
        proc = subprocess.run([str(ROOT / "validator/target/release/schedcheck"), str(folder / "schedule.json")], capture_output=True, text=True, timeout=60)
        wall = time.perf_counter() - stamp
        (folder / "schedule.log").write_text(proc.stdout + proc.stderr)
        row = {"mutant_id": mid, "model": model, "family": "validator_only", "operator": "wait_wrong_flag",
               "site": {"layer": layer, "projection": phase}, "build": "N/A", "conditional": False,
               "wrong_flag": wrong, "right_flag": right, "waits_edited": edited, "equivalent": False,
               "stage": "schedule" if proc.returncode else "SURVIVED", "gpu_stage": None, "extra_stage": None, "max_diff": None,
               "runtime_s": wall, "measurement": "measured", "producing_script": "mutation/mutate.py",
               "stages": {"schedule": {"status": "FAIL" if proc.returncode else "PASS", "wall_s": wall,
                                       "verdict_line": (proc.stdout.strip().splitlines() or [""])[0][:200]}}}
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(mid, row["stage"], row["stages"]["schedule"]["verdict_line"][:100])


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "build", "run", "validator", "jitter"))
    parser.add_argument("--model", choices=tuple(SOURCES), default="0.6B")
    parser.add_argument("--mutant", help="one labeled mutant id")
    parser.add_argument("--source", type=Path, help="known-correct source with the same engine ABI")
    parser.add_argument("--family", help="restrict to one family (or 'control')")
    parser.add_argument("--operator", help="restrict to one operator")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8, help="mutants evaluated per GPU job")
    parser.add_argument("--deadline", type=int, default=45, help="seconds before a mutant is declared hung")
    args = parser.parse_args()
    if args.source:
        SOURCES[args.model] = args.source.resolve()
    rows = catalogue(args.model)
    if args.mutant:
        rows = [r for r in rows if r["mutant_id"] == args.mutant]
        if not rows:
            parser.error("unknown mutant id")
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if args.operator:
        rows = [r for r in rows if r["operator"] == args.operator]
    if args.action == "list":
        print(json.dumps(rows, indent=1))
        return
    if args.action == "jitter":
        results = load_jsonl(HERE / "results.jsonl")
        folders = [str(ROOT / r["library"]).rsplit("/", 1)[0] for r in results
                   if r["model"] == args.model and r.get("build") == "PASS" and r.get("gpu_stage") == "SURVIVED"]
        original = HERE / "build" / f"original-{args.model}"
        original.mkdir(parents=True, exist_ok=True)
        (original / "original_jit.cu").write_text(jitter_text(SOURCES[args.model].read_text()))
        todo = [f for f in folders if not (Path(f) / "mutant_jit.so").exists()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(compile_cu, args.model, original / "original_jit.cu", original / "original_jit.so"): "original"}
            jobs.update({pool.submit(jitter_one, args.model, Path(f).relative_to(ROOT)): f for f in todo})
            for future in concurrent.futures.as_completed(jobs):
                print(Path(jobs[future]).name, "PASS" if future.result() == 0 else "COMPILE_FAILURE", flush=True)
        return
    if args.action == "validator":
        validator_only(args.model)
        return
    manifest = HERE / f"manifest-{args.model}.jsonl"
    prior = load_jsonl(manifest)
    if args.action == "build":
        done = {r["mutant_id"] for r in prior}
        todo = [r for r in rows if r["mutant_id"] not in done]
        if args.limit:
            todo = todo[:args.limit]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(build_one, row) for row in todo]):
                row = future.result()
                with manifest.open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(row["mutant_id"], row["build"], flush=True)
        return
    results = HERE / "results.jsonl"
    completed = {r["mutant_id"] for r in load_jsonl(results)}
    selected = {r["mutant_id"] for r in rows}
    latest = {r["mutant_id"]: r for r in prior}
    todo = [r for r in latest.values() if r["mutant_id"] in selected and r["mutant_id"] not in completed]
    for row in [r for r in todo if r["build"] != "PASS"]:
        with results.open("a") as out:
            out.write(json.dumps({**row, "stage": "COMPILE_FAILURE", "max_diff": None, "runtime_s": 0}) + "\n")
    todo = [r for r in todo if r["build"] == "PASS"]
    if args.limit:
        todo = todo[:args.limit]
    while todo:
        chunk, todo = todo[:args.batch], todo[args.batch:]
        specs = [str(ROOT / r["library"]).replace("mutant.so", "spec.json") for r in chunk]
        cmd = [str(ROOT / "scripts/gpu.sh"), "timeout", "--kill-after=5", str(60 + args.deadline * len(chunk)),
               str(ROOT / ".venv/bin/python"), str(HERE / "gate.py"), "evaluate", "--deadline", str(args.deadline), "--spec", *specs]
        stamp = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT)
        done = {r["mutant_id"] for r in load_jsonl(results)}
        left = [r for r in chunk if r["mutant_id"] not in done]
        print(f"job rc={proc.returncode} wall {time.perf_counter() - stamp:.1f}s, {len(chunk) - len(left)}/{len(chunk)} recorded", flush=True)
        if left and proc.returncode not in (0, 124, 137):
            raise SystemExit(f"evaluation infrastructure failed: {left[0]['mutant_id']} rc={proc.returncode}")
        if left and proc.returncode in (124, 137):
            row = left[0]  # outer timeout: attribute to the first unrecorded mutant
            with results.open("a") as out:
                out.write(json.dumps({**row, "stage": "execution_timeout", "extra_stage": None, "max_diff": None,
                                      "equivalent": False, "measurement": "measured",
                                      "stages": {"execution": {"status": "TIMEOUT"}}}) + "\n")
            left = left[1:]
        todo = left + todo


if __name__ == "__main__":
    main()
