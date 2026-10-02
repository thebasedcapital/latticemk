"""LM-10 quality gate: HF fp32 fake-quant INT4, v2, and sync fork.

Run 64 teacher-forced decode steps on each of three prompts. Require <=0.5
absolute logit error, <=1.25x the matched v2 error, deterministic repeat,
and argmax agreement except reference near-ties. Also show free-running counts.
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
    return torch.tensor(toks), torch.stack(logits)


def forced_pass(engine_class, prompts, refs, packed, emb, norms, rope):
    """Return CPU logits for all 192 forced steps from a fresh engine state."""
    eng = engine_class(CTX_CAP, packed, emb, norms, rope)
    eng.bufs["kc"].zero_()
    eng.bufs["vc"].zero_()
    observed = []
    for p, (ref_tokens, _) in zip(prompts, refs):
        eng.prefill(p)
        log = [eng.logits().cpu()]
        for s in range(1, NGEN):
            eng.set_tok(int(ref_tokens[s - 1]))
            eng.mega(1)
            log.append(eng.logits().cpu())
        observed.append(torch.stack(log))
    return observed


def compare(observed, refs):
    max_diff = 0.0
    near_ties, bad_argmax, per_prompt = [], [], []
    for pi, (got, (ref_tokens, expected)) in enumerate(zip(observed, refs)):
        diff = (got - expected).abs().amax(dim=1)
        per_prompt.append(float(diff.max()))
        max_diff = max(max_diff, per_prompt[-1])
        for s in range(NGEN):
            top = expected[s].topk(2)
            margin = float(top.values[0] - top.values[1])
            mine, gold = int(got[s].argmax()), int(ref_tokens[s])
            if margin < 2 * float(diff[s]):
                near_ties.append((pi, s, margin, float(diff[s]), mine, gold))
            elif mine != gold:
                bad_argmax.append((pi, s, margin, float(diff[s]), mine, gold))
    return max_diff, per_prompt, near_ties, bad_argmax


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
    eng = lm03b.Engine3(CTX_CAP, packed, emb, norms, rope)
    for pi, p in enumerate(prompts):
        ref_gen, _ = refs[pi]
        eng.prefill(p)
        mine = eng.decode(NGEN)[eng.pos0 - 1 - NGEN: eng.pos0 - 1]
        match = int((mine == ref_gen).sum())
        print(f"[mega3] free-running prompt {pi}: tokens {match}/{NGEN}")
        if match != NGEN:
            idx = int((mine != ref_gen).nonzero()[0])
            print(f"   first diff at gen {idx}: mine {mine[idx].item()} "
                  f"ref {ref_gen[idx].item()}")

    del eng
    torch.cuda.empty_cache()
    baseline = forced_pass(lm03b.Engine2, prompts, refs, packed, emb, norms, rope)
    v2diff, v2prompt, _, _ = compare(baseline, refs)
    del baseline
    observed = forced_pass(lm03b.Engine3, prompts, refs, packed, emb, norms, rope)
    max_diff, per_prompt, near_ties, bad_argmax = compare(observed, refs)
    repeat = forced_pass(lm03b.Engine3, prompts, refs, packed, emb, norms, rope)
    run_diff = max(float((a - b).abs().max()) for a, b in zip(observed, repeat))
    identical = all(torch.equal(a, b) for a, b in zip(observed, repeat))
    for pi, value in enumerate(per_prompt):
        print(f"[mega3] teacher-forced prompt {pi}: max |logit diff| "
              f"{value:.6f} across {NGEN} steps", flush=True)
    print(f"v2 max |logit diff| = {v2diff:.6f}, prompt maxima = {v2prompt}")
    print(f"mega3 max |logit diff| = {max_diff:.6f}")
    print(f"deterministic repeat: {identical}; max run-to-run diff = {run_diff:.6f}")
    print("near ties (prompt, step, margin, step_max_diff, mine, ref):", near_ties)
    print("unexcused argmax errors:", bad_argmax)
    ok = max_diff <= 0.5 and max_diff <= 1.25 * v2diff and not bad_argmax and identical
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
