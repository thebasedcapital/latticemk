"""Extra tests that run only after a mutant survives the unchanged v2.1 gate.

No tolerance is relaxed or changed: every logit comparison reuses the gate bound (0.5 absolute,
1.25x the control engine for 0.6B) and the near-tie rule. The extras only widen WHAT is exercised:

  sampling     the kernel's own sampled token must equal argmax(its own logits) (lowest id on exact ties).
               v2.1 teacher-forces tokens from the host, so the device sampler is never consumed.
  context      prompts of 129 / 257 / 1100 tokens: positions cross the 128 mark, attention-slice
               boundaries (npos/2 split), per-warp ranges longer than one element, and many KV rows.
  adversarial  number-reversal and code-completion prompts (low-margin, repetitive, indentation).
  head_perm    lm_head rows are swapped so the reference top-1 tokens sit in the kernel's tail rows
               (last 20 vocab rows, last/first row of every CTA row-chunk and argmax slice). The row
               permutation is an involution, so kernel logits must equal reference[perm] under the same bound.
  multistep    one mega(12) launch must reproduce twelve mega(1) launches (tokens + final logits bitwise);
               exercises the barrier at the end of a step and the device-side token feedback.
  repeat       a second bitwise pass over the long-context cases (stress for rare races).
  jitter       the same mutant rebuilt with pseudo-random __nanosleep skew around every __syncthreads
               (mutate.py jitter); two passes must reproduce the un-jittered logits bit for bit.
  distribution mean KL(reference || engine) over the 192 teacher-forced steps must stay <= 1.25x the
               control engine's value (the v2.1 relative rule, applied to a low-noise mean statistic).

"""

import torch

TEXTS = [
    "Repeat this exact list of numbers in reverse order: 0 1 2 3 5 8 13 21 34 55 89 144. Answer:",
    "Complete the Python code, including indentation:\ndef f(x):\n    if x < 0:\n        return -x\n    else:\n",
]
CONTEXT_LENGTHS = ((129, 8), (257, 8), (1100, 4))
VOCAB, NCTA = 151936, 36
MULTISTEP = 12
EXTRA_ORDER = ("sampling", "context", "adversarial", "head_perm", "multistep", "repeat", "jitter", "distribution")


def scenarios(tokenizer, vocab=VOCAB):
    seed = tokenizer("The capital of France is Paris. The capital of Japan is Tokyo. ", return_tensors="pt").input_ids[0]
    cases = []
    for length, steps in CONTEXT_LENGTHS:
        ids = seed.repeat((length + len(seed) - 1) // len(seed))[:length]
        cases.append({"name": f"context-{length}", "prompt": ids, "steps": steps})
    for i, text in enumerate(TEXTS):
        cases.append({"name": f"adversarial-{i}", "prompt": tokenizer(text, return_tensors="pt").input_ids[0], "steps": 16})
    return cases


def sampler_mismatch(engine, logits):
    actual = int(engine.bufs["tok"].cpu()[0])
    expected = int(logits.argmax())
    return None if actual == expected else {"actual": actual, "expected": expected}


def tail_rows(vocab=VOCAB):
    """Rows where row-tail / chunk / argmax-slice off-by-one faults of the kernel act."""
    chunk = (vocab + NCTA - 1) // NCTA
    rows = set(range(vocab - 20, vocab))
    for b in range(NCTA):
        lo, hi = b * chunk, min(vocab, (b + 1) * chunk)
        rows.update({lo, lo + 1, hi - 1, hi - 2})
    return sorted(rows)


def head_permutations(tokens, vocab=VOCAB):
    """Involutions moving the reference's top-1 tokens onto tail rows: logits_new[i] = logits_old[perm[i]].

    There are fewer distinct top-1 tokens than tail rows, so the rows are split over several variants and
    every tail row receives a real top-1 token in exactly one variant."""
    distinct = list(dict.fromkeys(int(x) for x in tokens))
    targets = [r for r in tail_rows(vocab) if r not in set(distinct)]
    variants = max(1, -(-len(targets) // len(distinct)))
    perms = []
    for v in range(variants):
        perm = torch.arange(vocab)
        for row, token in zip(targets[v::variants], distinct):
            perm[row], perm[token] = token, row
        perms.append(perm)
    return perms


def permuted_cases(cases, perm):
    out = []
    for case in cases:
        new = {**case, "name": "headperm-" + case["name"], "reference": case["reference"].index_select(1, perm)}
        if "original" in case:
            new["original"] = case["original"].index_select(1, perm)
        out.append(new)
    return out


def permuted_head(gpu, perm):
    device = gpu["lm_head"]["codes"].device
    index = perm.to(device)
    return {**gpu, "lm_head": {k: v.index_select(0, index).contiguous() for k, v in gpu["lm_head"].items()}}


@torch.no_grad()
def multistep(module, gpu, shared, case, steps=MULTISTEP):
    """One mega(steps) launch vs `steps` mega(1) launches from the same prefill."""
    outs = []
    for chunked in (False, True):
        engine = module.Engine2(512, gpu, *shared)
        engine.prefill(case["prompt"])
        base = len(case["prompt"])
        if chunked:
            engine.mega(steps)
        else:
            for _ in range(steps):
                engine.mega(1)
        torch.cuda.synchronize()
        outs.append((engine.bufs["tok_hist"][base - 1: base + steps - 1].cpu().clone(), engine.logits().cpu()))
        del engine
    same_tokens = bool(torch.equal(outs[0][0], outs[1][0]))
    same_logits = bool(torch.equal(outs[0][1], outs[1][1]))
    return {"pass": same_tokens and same_logits, "tokens_equal": same_tokens, "logits_equal": same_logits}

