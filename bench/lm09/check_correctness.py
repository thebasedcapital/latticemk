"""LM-09 teacher-forced and free-running correctness vs an fp32 HF reference
with identical INT4 weights and KV fake quantization.

The shared codec hook handles prompt prefill. Single-query decoding needs a
separate attention path because kvcodec.common.kv_attn assumes as many queries
as cache entries.
"""

import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as q3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(ROOT / "bench" / "lm03b"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402
import lm09  # noqa: E402
from lmk.model import N_LAYERS, SNAPSHOT  # noqa: E402
from kvcodec.codec_lm09 import make_lm09  # noqa: E402
from kvcodec.common import (headify, kv_hook_state, patch_attention,
                            unheadify)

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
]
NGEN = 64
NLOG = NGEN
CTX_CAP = 256


class HalfInputCodec:
    """The CUDA appender quantizes fp16 matvec outputs, not fp32 HF tensors."""

    def __init__(self, codec):
        self.codec = codec

    def encode(self, x, kind):
        return self.codec.encode(x.half().float(), kind)

    def decode(self, packed):
        return self.codec.decode(packed)


def decode_attention(attn, prefill, codec, capture):
    """Use the shared prefill codec hook, with a single-query past-cache path."""
    def forward(hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kw):
        if hidden_states.shape[1] != 1 or past_key_values is None:
            return prefill(hidden_states, position_embeddings, attention_mask,
                           past_key_values=past_key_values, **kw)
        shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = q3.apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = past_key_values.update(k, v, attn.layer_idx)
        capture.update(k=headify(k), v=headify(v))
        kq = unheadify(codec.decode(codec.encode(headify(k), "k")))
        vq = unheadify(codec.decode(codec.encode(headify(v), "v")))
        # The current row is exact in the kernel. Prior rows come from cache.
        kq[:, :, -1:] = k[:, :, -1:]
        vq[:, :, -1:] = v[:, :, -1:]
        score = torch.matmul(q, q3.repeat_kv(kq, attn.num_key_value_groups)
                             .transpose(2, 3)) * attn.scaling
        if attention_mask is not None:
            score += attention_mask
        prob = torch.softmax(score, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(prob, q3.repeat_kv(vq, attn.num_key_value_groups))
        return attn.o_proj(out.transpose(1, 2).reshape(
            *hidden_states.shape[:-1], -1)), None
    return forward


@torch.no_grad()
def hf_reference(deq, prompt_ids, codec, forced_tokens=None):
    """HF model on its own greedy path or an externally fixed token sequence.

    When codec is None, this is the fp16-KV reference for megakernel-v2.
    """
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

    state = kv_hook_state()
    origs = []
    for l, am in enumerate(model.model.layers):
        orig = am.self_attn.forward
        origs.append(orig)
        c = HalfInputCodec(make_lm09(codec, l)) if codec else None
        pf = patch_attention(am.self_attn, orig, state=state, codec=c)
        am.self_attn.forward = (decode_attention(am.self_attn, pf, c, state[l])
                                if c else pf)

    ids = prompt_ids[None].cuda()
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    tokens, logits = [nxt], [out.logits[0, -1].cpu()]
    for s in range(1, NGEN):
        fed = int(forced_tokens[s - 1]) if forced_tokens is not None else nxt
        out = model(input_ids=torch.tensor([[fed]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        tokens.append(nxt)
        if len(logits) < NLOG:
            logits.append(out.logits[0, -1].cpu())

    kv = {l: (d["k"].float().cpu(), d["v"].float().cpu())
          for l, d in state.items()}
    for am, orig in zip(model.model.layers, origs):
        am.self_attn.forward = orig
    del model, past, out, lin, am, orig, origs, pf, c, state
    gc.collect()
    torch.cuda.empty_cache()
    return torch.tensor(tokens), torch.stack(logits[:NLOG]), kv


def kv_parity(eng, kv, codec, prefix):
    """Compare packed cache with Python reconstruction on the shared prefix."""
    KQROWB, VQROWB = lm09.KQROWB, lm09.VQROWB
    worst = {}
    co = make_lm09(codec, 0)
    for l in range(N_LAYERS):
        T = min(kv[l][0].shape[0], prefix)
        # ---- K: per-token INT{KBITS} g{KGRP}
        if lm09.KMODE_T:
            codes = eng.bufs["kq"][
                l * lm09.MAXPOS * KQROWB:(l * lm09.MAXPOS + T) * KQROWB]
            codes = codes.reshape(T, lm09.NKVH, 128 * lm09.KBITS // 8).cuda()
            # kmeta: [layer][tok][kvh][128/KGRP] half2 — slice in half units
            meta = eng.bufs["kmeta"][
                2 * (l * lm09.MAXPOS) * lm09.KMETA_PER_TOK:
                2 * (l * lm09.MAXPOS + T) * lm09.KMETA_PER_TOK]
            meta = meta.reshape(T, lm09.NKVH, 128 // lm09.KGRP, 2).cuda().float()
            if lm09.KBITS == 8:
                q = codes.view(T, lm09.NKVH, 128).float()
            else:
                q = torch.stack([codes & 0xF, codes >> 4], -1)
                q = q.reshape(T, lm09.NKVH, 128).float()
            # meta group g covers dims g*KGRP.. ; broadcast
            sc = meta[..., 0].repeat_interleave(lm09.KGRP, -1)
            off = meta[..., 1].repeat_interleave(lm09.KGRP, -1)
            k_recon = q * sc + off
            # kernel quantized the fp16 row -> feed the codec the same input
            ref = co.kc.decode(
                co.kc.encode(kv[l][0][:T].cuda().half().float(), "k"))
            rel = ((k_recon - ref).pow(2).sum() /
                   ref.pow(2).sum()).sqrt().item()
            worst["k"] = max(worst.get("k", 0.0), rel)
        # ---- V: per-token INT{VBITS} g{VGRP}
        codes = eng.bufs["vq"][
            l * lm09.MAXPOS * VQROWB:(l * lm09.MAXPOS + T) * VQROWB]
        codes = codes.reshape(T, lm09.NKVH, 128 * lm09.VBITS // 8).cuda()
        meta = eng.bufs["vmeta"][
            2 * (l * lm09.MAXPOS) * lm09.VMETA_PER_TOK:
            2 * (l * lm09.MAXPOS + T) * lm09.VMETA_PER_TOK]
        meta = meta.reshape(T, lm09.NKVH, 128 // lm09.VGRP, 2).cuda().float()
        if lm09.VBITS == 8:
            q = codes.view(T, lm09.NKVH, 128).float()
        else:
            q = torch.stack([codes & 0xF, codes >> 4], -1)
            q = q.reshape(T, lm09.NKVH, 128).float()
        sc = meta[..., 0].repeat_interleave(lm09.VGRP, -1)
        off = meta[..., 1].repeat_interleave(lm09.VGRP, -1)
        v_recon = q * sc + off
        ref = co.vc.decode(co.vc.encode(kv[l][1][:T].cuda().half().float(), "v"))
        rel = ((v_recon - ref).pow(2).sum() / ref.pow(2).sum()).sqrt().item()
        worst["v"] = max(worst.get("v", 0.0), rel)
    return worst


def run_forced(engine, prompts, refs):
    """Replay reference tokens, retaining every full-vocab logit row."""
    rows = []
    for pi, p in enumerate(prompts):
        engine.prefill(p)
        ref_gen = refs[pi][0]
        steps = [engine.logits().cpu()]
        for s in range(1, NGEN):
            engine.set_tok(int(ref_gen[s - 1]))
            engine.mega(1)
            steps.append(engine.logits().cpu())
        rows.append(torch.stack(steps))
    return rows


def main():

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    prompts = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    packed, deq = lm03.pack_weights()

    print(f"== HF reference (INT4 weights + {lm09.CODEC} KV, fp32) ==")
    refs = [hf_reference(deq, p, lm09.CODEC) for p in prompts]
    base_refs = [hf_reference(deq, p, None, forced_tokens=refs[i][0])
                 for i, p in enumerate(prompts)]
    del deq
    torch.cuda.empty_cache()

    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope(maxpos=lm09.MAXPOS).cuda()
    free_matches = []
    eng = lm09.EngineKV(CTX_CAP, packed, emb, norms, rope)
    for pi, p in enumerate(prompts):
        ref_gen, _, kv_ref = refs[pi]
        eng.prefill(p)
        mine = eng.decode(NGEN)[eng.pos0 - 1 - NGEN: eng.pos0 - 1]
        match = int((mine == ref_gen).sum())
        stat = "OK" if match == NGEN else f"FAIL ({match}/{NGEN})"
        free_matches.append(match)
        print(f"[megakv] free-running prompt {pi}: tokens {match}/{NGEN} {stat}")
        if match != NGEN:
            idx = int((mine != ref_gen).nonzero()[0])
            print(f"   first diff at gen {idx}: mine {mine[idx].item()} "
                  f"ref {ref_gen[idx].item()}")
        if pi == 0:
            worst = kv_parity(eng, kv_ref, lm09.CODEC, len(p) + 1)
            print(f"[megakv] packed-cache decode rel err vs Python codec: "
                  f"K {worst.get('k', -1):.5f}  V {worst.get('v', -1):.5f}")

    # Gate v2.1: identical reference tokens fed to both engines; replay the
    # KV engine from fresh state to detect races in the implementation.
    del eng
    torch.cuda.empty_cache()
    first = run_forced(lm09.EngineKV(CTX_CAP, packed, emb, norms, rope),
                       prompts, refs)
    second = run_forced(lm09.EngineKV(CTX_CAP, packed, emb, norms, rope),
                        prompts, refs)
    drift = max((a - b).abs().max().item()
                for a, b in zip(first, second))
    deterministic = all(torch.equal(a, b) for a, b in zip(first, second))
    base = run_forced(lm03b.Engine2(CTX_CAP, packed, emb, norms, rope),
                      prompts, refs)
    base_md = max((got - base_refs[pi][1]).abs().max().item()
                  for pi, got in enumerate(base))
    md = 0.0
    mismatches = []
    near_ties = []
    for pi, got in enumerate(first):
        ref_gen, ref_logits, _ = refs[pi]
        md_p = (got - ref_logits).abs().max().item()
        md = max(md, md_p)
        for s, logits in enumerate(got):
            step_diff = (logits - ref_logits[s]).abs().max().item()
            top = ref_logits[s].topk(2).values
            margin = (top[0] - top[1]).item()
            if int(logits.argmax()) != int(ref_gen[s]):
                row = (pi, s, round(margin, 5), round(step_diff, 5))
                (near_ties if margin < 2 * step_diff else mismatches).append(row)
        print(f"teacher-forced prompt {pi}: {NGEN} logits, "
              f"max |diff| {md_p:.4f}")
    ok = md <= 0.5 and md <= 1.25 * base_md and not mismatches and deterministic
    print(f"max |logit diff| KV {md:.4f} vs v2 {base_md:.4f} "
          f"(limit {min(0.5, 1.25 * base_md):.4f})")
    print(f"run-to-run deterministic: {deterministic}, max diff {drift:.6f}")
    print(f"near-tie steps (prompt, step, margin, max_diff): {near_ties}")
    print(f"other argmax mismatches: {mismatches}")
    print(f"free-running matches: {free_matches}")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
