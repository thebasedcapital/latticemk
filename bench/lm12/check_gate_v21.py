"""Qwen3-1.7B gate v2.1: 3 x 64 forced-token logits vs CPU fp32 fake-GPTQ HF.

GPU model plus fp32 weights exceed 8 GB, so HF reference runs on CPU.
Requires packed GPTQ weights and their exact fp32 reconstructed tensors.
"""
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "bench/lm03"), str(HERE)]
import scale

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
]
NGEN = 64


def make_reference(deq):
    model = AutoModelForCausalLM.from_pretrained(
        scale.SNAPSHOT, dtype=torch.float32).eval()
    lin = {name: module for name, module in model.named_modules()
           if isinstance(module, torch.nn.Linear)}
    for layer in range(scale.N_LAYERS):
        p = f"model.layers.{layer}."
        qkv = deq[f"L{layer}.qkv"]
        lin[p + "self_attn.q_proj"].weight.data = qkv[:2048]
        lin[p + "self_attn.k_proj"].weight.data = qkv[2048:3072]
        lin[p + "self_attn.v_proj"].weight.data = qkv[3072:]
        lin[p + "self_attn.o_proj"].weight.data = deq[f"L{layer}.o"]
        gu = deq[f"L{layer}.gu"]
        lin[p + "mlp.gate_proj"].weight.data = gu[:scale.INTER]
        lin[p + "mlp.up_proj"].weight.data = gu[scale.INTER:]
        lin[p + "mlp.down_proj"].weight.data = deq[f"L{layer}.down"]
    model.lm_head.weight = torch.nn.Parameter(deq["lm_head"])
    return model


@torch.no_grad()
def reference(model, prompt, forced=None):
    output = model(input_ids=prompt.unsqueeze(0), use_cache=True)
    past = output.past_key_values
    first = output.logits[0, -1].clone()
    logits = [first]
    predicted = [int(first.argmax())]
    for step in range(1, NGEN):
        token = predicted[-1] if forced is None else int(forced[step - 1])
        output = model(input_ids=torch.tensor([[token]]),
                       past_key_values=past, use_cache=True)
        past = output.past_key_values
        line = output.logits[0, -1].clone()
        logits.append(line)
        predicted.append(int(line.argmax()))
    return predicted, logits


def engine_pass(packed, references, forced, prompts, shared, prior=None):
    gpu = {name: {field: value.cuda() for field, value in weight.items()}
           for name, weight in packed.items()}
    engine = scale.Engine2(256, gpu, *shared)
    stored = [] if prior is None else None
    max_abs, by_prompt, near_ties, hard_flips = 0., [], [], []
    nonfinite = 0
    bitwise = True
    for pi, prompt in enumerate(prompts):
        engine.prefill(prompt)
        prompt_max = 0.
        for step in range(NGEN):
            actual = engine.logits().cpu()
            expected = references[pi][1][step]
            error = float((actual - expected).abs().max())
            if not math.isfinite(error):
                nonfinite += 1
            else:
                prompt_max = max(prompt_max, error)
                max_abs = max(max_abs, error)
            top = expected.topk(2)
            margin = float(top.values[0] - top.values[1])
            if int(actual.argmax()) != int(top.indices[0]):
                event = {"prompt": pi, "step": step, "margin": margin,
                         "step_max_diff": error, "actual": int(actual.argmax()),
                         "reference": int(top.indices[0])}
                (near_ties if math.isfinite(error) and margin < 2 * error
                 else hard_flips).append(event)
            if prior is None:
                stored.append(actual)
            else:
                bitwise &= torch.equal(actual, prior[pi * NGEN + step])
            if step + 1 < NGEN:
                engine.set_tok(int(forced[pi][step]))
                engine.mega(1)
        by_prompt.append(prompt_max)
        print(f"prompt {pi} max|logit|={prompt_max:.6f}", flush=True)
    del engine, gpu
    torch.cuda.empty_cache()
    return {"max_diff": max_abs, "per_prompt": by_prompt,
            "near_ties": near_ties, "hard_flips": hard_flips,
            "nonfinite_steps": nonfinite, "bitwise_repeat": bitwise}, stored


def main():
    torch.set_num_threads(12)
    tokenizer = AutoTokenizer.from_pretrained(scale.SNAPSHOT)
    prompts = [tokenizer(text, return_tensors="pt").input_ids[0]
               for text in PROMPTS]
    packed = torch.load(HERE / "weights_int4_gptq.pt", map_location="cpu")
    deq = torch.load(HERE / "weights_int4_gptq_deq.pt", map_location="cpu")
    model = make_reference(deq)
    refs = [reference(model, prompt) for prompt in prompts]
    del model, deq
    forced = [ref[0] for ref in refs]
    shared = (scale.load("model.embed_tokens.weight", "cuda").half().contiguous(),
              scale.norm_table(), scale.make_rope())
    first, logits = engine_pass(packed, refs, forced, prompts, shared)
    second, _ = engine_pass(packed, refs, forced, prompts, shared, logits)
    scale.use_library("libmega_scale_fp16.so")
    debug, _ = engine_pass(packed, refs, forced, prompts, shared)
    scale.use_library()
    passed = (first["max_diff"] <= .5 and not first["hard_flips"]
              and first["nonfinite_steps"] == 0 and second["bitwise_repeat"])
    result = {"gate": "v2.1", "reference": "CPU HF fp32 fake-GPTQ",
              "forced_sequences": "HF greedy 3x64", "first": first,
              "second": {"max_diff": second["max_diff"],
                         "bitwise_repeat": second["bitwise_repeat"],
                         "nonfinite_steps": second["nonfinite_steps"]},
              "fp16_dot_diagnostic": {
                  "max_finite_diff": debug["max_diff"],
                  "nonfinite_steps": debug["nonfinite_steps"],
                  "near_ties": len(debug["near_ties"]),
                  "hard_flips": len(debug["hard_flips"])},
              "pass": passed}
    (HERE / "gate_v21.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit("LM-12 gate FAIL")


if __name__ == "__main__":
    main()
