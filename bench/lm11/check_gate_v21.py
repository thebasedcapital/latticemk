"""Integrator v2.1 gate: GPTQ vs HF, RTN control, and repeat determinism.

Run: scripts/gpu.sh .venv/bin/python bench/lm11/check_gate_v21.py
Each HF model sees GPTQ's same 64 generated tokens at each of three prompts.
RTN and GPTQ kernel outputs are compared to their own fake-quant HF weights.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "bench/lm03"),
                str(ROOT / "bench/lm03b"), str(HERE)]
import lm03
import lm03b
from check_correctness_gptq import NGEN, PROMPTS, hf_reference
from ppl_llama_protocol import deq_int4
from lmk.model import SNAPSHOT


def packed(path):
    return torch.load(path, map_location="cpu")


def reference(deq, prompts, forced=None):
    return [hf_reference(deq, p, None if forced is None else forced[i])
            for i, p in enumerate(prompts)]


def measure(pk, refs, forced, prompts, shared, previous=None):
    emb, norms, rope = shared
    on_gpu = {k: {field: t.cuda() for field, t in v.items()}
              for k, v in pk.items()}
    eng = lm03b.Engine2(256, on_gpu, emb, norms, rope)
    max_diff, per_prompt, near, hard, repeat_diff = 0., [], [], [], 0.
    stored = [] if previous is None else None
    all_equal = True
    for pi, prompt in enumerate(prompts):
        eng.prefill(prompt)
        prompt_diff = 0.
        for s in range(NGEN):
            logits = eng.logits().cpu()
            ref = refs[pi][1][s]
            err = float((logits - ref).abs().max())
            max_diff = max(max_diff, err)
            prompt_diff = max(prompt_diff, err)
            top = ref.topk(2)
            margin = float(top.values[0] - top.values[1])
            mine, expected = int(logits.argmax()), int(top.indices[0])
            if mine != expected:
                event = {"prompt": pi, "step": s, "margin": margin,
                         "step_max_diff": err, "mine": mine, "reference": expected}
                (near if margin < 2 * err else hard).append(event)
            if previous is not None:
                first = previous[pi * NGEN + s]
                all_equal &= torch.equal(logits, first)
                repeat_diff = max(repeat_diff, float((logits - first).abs().max()))
            else:
                stored.append(logits)
            if s + 1 < NGEN:
                eng.set_tok(int(forced[pi][s]))
                eng.mega(1)
        per_prompt.append(prompt_diff)
    del eng, on_gpu
    torch.cuda.empty_cache()
    return {"max_diff": max_diff, "per_prompt": per_prompt,
            "near_ties": near, "hard_flips": hard,
            "bitwise_repeat": all_equal, "max_repeat_diff": repeat_diff}, stored


def main():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    gptq_pk = packed(HERE / "weights_int4_gptq.pt")
    deq = torch.load(HERE / "weights_int4_gptq_deq.pt", map_location="cpu")
    refs_gptq = reference(deq, prompts)
    del deq
    forced = [r[0] for r in refs_gptq]
    rtn_pk = packed(ROOT / "bench/lm03b/weights_int4.pt")
    deq_rtn = {name: deq_int4(v["codes"], v["meta"])
               for name, v in rtn_pk.items()}
    refs_rtn = reference(deq_rtn, prompts, forced)
    del deq_rtn
    shared = (lm03.load("model.embed_tokens.weight").half().contiguous(),
              lm03.norm_table(), lm03.make_rope().cuda())
    result1, first = measure(gptq_pk, refs_gptq, forced, prompts, shared)
    result2, _ = measure(gptq_pk, refs_gptq, forced, prompts, shared, first)
    baseline, _ = measure(rtn_pk, refs_rtn, forced, prompts, shared)
    passed = (result1["max_diff"] <= 0.5
              and result1["max_diff"] <= 1.25 * baseline["max_diff"]
              and not result1["hard_flips"]
              and result1["bitwise_repeat"] and result2["bitwise_repeat"])
    result = {"gate": "v2.1", "forced_sequences": "GPTQ HF greedy, 3x64",
              "gptq": result1, "gptq_repeat": result2,
              "rtn_baseline": baseline, "pass": passed}
    (HERE / "gate_v21.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit("v2.1 gate FAIL")


if __name__ == "__main__":
    main()
