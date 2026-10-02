"""LM-11 GPTQ gate: teacher-forced v2 logits vs HF fp32 fake-quant.

Records the original free-running 64-token greedy comparison separately.
The binding gate feeds the HF-generated tokens to v2 for 64 steps per prompt.
Both reference tensors and kernel packed codes come from gptq_pack.py.

usage: check_correctness_gptq.py
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(ROOT / "bench" / "lm03b"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402
from lmk.model import N_LAYERS, SNAPSHOT  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
]
NGEN = 64
LOGIT_LIMIT = 0.25
CTX_CAP = 256


@torch.no_grad()
def hf_reference(deq, prompt_ids, forced_tokens=None):
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32)
    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear)}
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        w = deq[f"L{i}.qkv"].cuda()
        lin[p + "self_attn.q_proj"].weight.data = w[:2048].clone()
        lin[p + "self_attn.k_proj"].weight.data = w[2048:3072].clone()
        lin[p + "self_attn.v_proj"].weight.data = w[3072:].clone()
        lin[p + "self_attn.o_proj"].weight.data = deq[f"L{i}.o"].cuda()
        g = deq[f"L{i}.gu"].cuda()
        lin[p + "mlp.gate_proj"].weight.data = g[:3072].clone()
        lin[p + "mlp.up_proj"].weight.data = g[3072:].clone()
        lin[p + "mlp.down_proj"].weight.data = deq[f"L{i}.down"].cuda()
    model.lm_head.weight = torch.nn.Parameter(deq["lm_head"].cuda())
    model.cuda().eval()

    ids = prompt_ids[None].cuda()
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = [out.logits[0, -1].cpu()]
    toks = [int(logits[0].argmax())]
    nxt = int(forced_tokens[0]) if forced_tokens is not None else toks[0]
    for s in range(1, NGEN):
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1].clone()
        predicted = int(lg.argmax())
        toks.append(predicted)
        logits.append(lg.cpu())
        nxt = int(forced_tokens[s]) if forced_tokens is not None else predicted
    del model, past
    torch.cuda.empty_cache()
    return torch.tensor(toks), torch.stack(logits)


def main():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    packed = {k: {f: t.cuda() for f, t in v.items()}
              for k, v in torch.load(HERE / "weights_int4_gptq.pt").items()}
    deq = torch.load(HERE / "weights_int4_gptq_deq.pt", map_location="cpu")

    print("== HF reference (fake-quant GPTQ-INT4, fp32) ==")
    refs = [hf_reference(deq, p) for p in prompts]
    del deq
    torch.cuda.empty_cache()

    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    eng = lm03b.Engine2(CTX_CAP, packed, emb, norms, rope)
    for pi, p in enumerate(prompts):
        ref_gen, _ = refs[pi]
        eng.prefill(p)
        mine = eng.decode(NGEN)[eng.pos0 - 1 - NGEN: eng.pos0 - 1]
        match = int((mine == ref_gen).sum())
        stat = "OK" if match == NGEN else f"FAIL ({match}/{NGEN})"
        print(f"[free-running] prompt {pi}: tokens {match}/{NGEN} {stat}")
        if match != NGEN:
            idx = int((mine != ref_gen).nonzero()[0])
            print(f"   first diff at gen {idx}: mine {mine[idx].item()} "
                  f"ref {ref_gen[idx].item()}")

    del eng
    torch.cuda.empty_cache()
    eng = lm03b.Engine2(CTX_CAP, packed, emb, norms, rope)
    md = 0.0
    near_ties, hard_flips = [], []
    for pi, p in enumerate(prompts):
        ref_gen, ref_logits = refs[pi]
        eng.prefill(p)
        md_p = 0.0
        for s in range(NGEN):
            logits = eng.logits().cpu()
            ref = ref_logits[s]
            diff = (logits - ref).abs().max().item()
            md_p = max(md_p, diff)
            top = ref.topk(2)
            margin = (top.values[0] - top.values[1]).item()
            mine = int(logits.argmax())
            if mine != int(ref_gen[s]):
                event = (pi, s, margin, diff, mine, int(ref_gen[s]))
                if margin < 2 * diff:
                    near_ties.append(event)
                else:
                    hard_flips.append(event)
            if s + 1 < NGEN:
                eng.set_tok(int(ref_gen[s]))
                eng.mega(1)
        md = max(md, md_p)
        print(f"[teacher-forced] prompt {pi}: max|logit diff| "
              f"{md_p:.4f} over {NGEN} steps")
    print(f"near-tie argmax flips ({len(near_ties)}): "
          f"(prompt,step,margin,step_max_diff,mine,reference)")
    for event in near_ties:
        print("  ", event)
    print(f"hard argmax flips ({len(hard_flips)}):", hard_flips)
    print(f"max |logit diff| over all {NGEN * len(prompts)} steps = {md:.4f}")
    print("TEACHER-FORCED RESULT:",
          "PASS" if md <= LOGIT_LIMIT and not hard_flips else "FAIL")


if __name__ == "__main__":
    main()
