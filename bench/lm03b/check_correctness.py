"""LM-03b correctness gate: v2 megakernel vs HF transformers on the SAME
fake-quantized INT4 weights (Quantized.deq, fp32).

For each of 3 fixed prompts: HF greedy-decodes 64 tokens (parallel prefill,
then argmax chain); mega2 must emit the same tokens. Max |logit diff| is
reported on the first 16 decode steps (against the graph engine's logits,
identical math to HF up to fp16 accumulation order).

usage: check_correctness.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402
from lmk.model import N_LAYERS, SNAPSHOT  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
]
NGEN = 64
NLOG = 16
CTX_CAP = 256


@torch.no_grad()
def hf_reference(deq, prompt_ids):
    """HF model with every streamed linear patched to the INT4 deq weights."""
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
    nxt = int(out.logits[0, -1].argmax())
    toks, logits = [nxt], [out.logits[0, -1].cpu()]
    for _ in range(NGEN - 1):
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1].clone()
        nxt = int(lg.argmax())
        toks.append(nxt)
        logits.append(lg.cpu())
    del model, past
    torch.cuda.empty_cache()
    return torch.tensor(toks), torch.stack(logits[:NLOG])


def main():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    packed, deq = lm03.pack_weights()

    print("== HF reference (fake-quant INT4, fp32) ==")
    refs = [hf_reference(deq, p) for p in prompts]
    del deq
    torch.cuda.empty_cache()

    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    ok = True
    eng = lm03b.Engine2(CTX_CAP, packed, emb, norms, rope)
    for pi, p in enumerate(prompts):
        ref_gen, _ = refs[pi]
        eng.prefill(p)
        mine = eng.decode(NGEN)[eng.pos0 - 1 - NGEN: eng.pos0 - 1]
        match = int((mine == ref_gen).sum())
        stat = "OK" if match == NGEN else f"FAIL ({match}/{NGEN})"
        ok &= match == NGEN
        print(f"[mega2] prompt {pi}: tokens {match}/{NGEN} {stat}")
        if match != NGEN:
            idx = int((mine != ref_gen).nonzero()[0])
            print(f"   first diff at gen {idx}: mine {mine[idx].item()} "
                  f"ref {ref_gen[idx].item()}")

    # logit diffs vs HF on the first NLOG steps
    del eng
    torch.cuda.empty_cache()
    eng = lm03b.Engine2(CTX_CAP, packed, emb, norms, rope)
    md = 0.0
    for pi, p in enumerate(prompts):
        _, ref_logits = refs[pi]
        eng.prefill(p)
        md_p = (eng.logits().cpu() - ref_logits[0]).abs().max().item()
        for s in range(1, NLOG):
            eng.mega(1)
            d = (eng.logits().cpu() - ref_logits[s]).abs().max().item()
            md_p = max(md_p, d)
        md = max(md, md_p)
        print(f"logit|diff| prompt {pi}: {md_p:.4f} over {NLOG} steps")
    print(f"max |logit diff| = {md:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
