"""LM-08 correctness gate: fused engine vs HF transformers on the SAME
fake-quantized INT4 weights (Quantized.deq, fp32) — same protocol as LM-03.

For each of 3 fixed prompts: HF greedy-decodes 64 tokens (parallel prefill,
then argmax chain); the fused engine must emit the same tokens.
Max |logit diff| is reported on the first 16 decode steps.

usage: check_correctness.py [graph|step|both]
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03
import lm08
from lmk.model import N_LAYERS, SNAPSHOT

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
        lin[p + "self_attn.q_proj"].weight.data = deq[f"L{i}.qkv"][:2048].cuda()
        lin[p + "self_attn.k_proj"].weight.data = \
            deq[f"L{i}.qkv"][2048:3072].cuda()
        lin[p + "self_attn.v_proj"].weight.data = \
            deq[f"L{i}.qkv"][3072:].cuda()
        lin[p + "self_attn.o_proj"].weight.data = deq[f"L{i}.o"].cuda()
        lin[p + "mlp.gate_proj"].weight.data = deq[f"L{i}.gu"][:3072].cuda()
        lin[p + "mlp.up_proj"].weight.data = deq[f"L{i}.gu"][3072:].cuda()
        lin[p + "mlp.down_proj"].weight.data = deq[f"L{i}.down"].cuda()
    model.lm_head.weight = torch.nn.Parameter(deq["lm_head"].cuda())  # untie
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
        nxt = int(out.logits[0, -1].argmax())
        lg = out.logits[0, -1]
        toks.append(nxt)
        logits.append(lg.cpu())
    del model, past
    torch.cuda.empty_cache()
    # logits[s] is the distribution that produced generated token s and is
    # aligned with the engine's logits after decode step s.
    return torch.tensor(toks), torch.stack(logits[:NLOG])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "graph"
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    packed, deq = lm03.pack_weights(cache=lm08.CACHE)

    print("== HF reference (fake-quant INT4, fp32) ==")
    refs = [hf_reference(deq, p) for p in prompts]
    del deq
    torch.cuda.empty_cache()

    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    ok = True
    engines = ["graph", "step"] if which == "both" else [which]
    for eng_name in engines:
        eng = lm08.Engine(CTX_CAP, packed, emb, norms, rope)
        if eng_name == "graph":
            eng.graph_build()
        for pi, p in enumerate(prompts):
            ref_gen, _ = refs[pi]
            eng.prefill(p, engine=eng_name)
            mine = eng.decode(NGEN, engine=eng_name)[
                eng.pos0 - 1 - NGEN: eng.pos0 - 1]
            match = int((mine == ref_gen).sum())
            stat = "OK" if match == NGEN else f"FAIL ({match}/{NGEN})"
            ok &= match == NGEN
            print(f"[{eng_name}] prompt {pi}: tokens {match}/{NGEN} {stat}")
            if match != NGEN:
                idx = int((mine != ref_gen).nonzero()[0])
                print(f"   first diff at gen {idx}: mine {mine[idx].item()} "
                      f"ref {ref_gen[idx].item()}")
        del eng
        torch.cuda.empty_cache()

    # logit diffs: engine logits after decode step s vs HF logits that
    # produced generated token s (same alignment).
    eng = lm08.Engine(CTX_CAP, packed, emb, norms, rope)
    eng.graph_build()
    md = 0.0
    for pi, p in enumerate(prompts):
        _, ref_logits = refs[pi]
        eng.prefill(p, engine="graph")
        md_p = (eng.logits().cpu() - ref_logits[0]).abs().max().item()
        for s in range(1, NLOG):
            eng.graph_launch()
            d = (eng.logits().cpu() - ref_logits[s]).abs().max().item()
            md_p = max(md_p, d)
        md = max(md, md_p)
        print(f"logit|diff| prompt {pi}: {md_p:.4f} over {NLOG} steps")
    print(f"max |logit diff| = {md:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
