"""Shared pieces for kvcodec scripts: constants, the attention patch, and wikitext windows.

patch_attention replaces Qwen3Attention.forward (transformers 5.17 signature). It reproduces the
upstream body verbatim, inserts codec hooks where the cache would write K/V, and — in quant mode —
dispatches to `kv_attn`, an eager attention that honours the spec rule "query and current token
stay exact": each position's attention output uses the quantized K/V for all context positions but
the exact K/V for its own (diagonal) position.
"""
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers.models.qwen3 import modeling_qwen3 as q3  # noqa: E402

SEQ = 2048
DATA = Path(__file__).resolve().parent / "data"
WIKI = next((Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots").iterdir())


def headify(t: torch.Tensor) -> torch.Tensor:
    """[1, H, T, D] -> [T, H, D] (codec convention)."""
    return t[0].transpose(0, 1).contiguous()


def unheadify(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(0, 1).unsqueeze(0).contiguous()


def kv_hook_state() -> dict:
    return {i: {} for i in range(28)}


def wiki_test_ids(tok) -> torch.Tensor:
    text = "\n\n".join(pq.read_table(WIKI / "wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist())
    return tok(text, return_tensors="pt").input_ids[0]


def kv_attn(module, query, k_exact, k_q, v_exact, v_q, attention_mask, scaling):
    """Eager attention: quantized K/V everywhere except the query position's own (diagonal) entry.

    w[:, i, i] is recomputed with k_exact and the output row i gets p_ii * v_exact[:, i] — i.e. the
    diagonal behaves as if the current token's cache entry were exact, matching decode semantics
    where the just-produced K/V is used unquantized.
    """
    g = module.num_key_value_groups
    kk_q = q3.repeat_kv(k_q, g)
    vv_q = q3.repeat_kv(v_q, g)
    kk_e = q3.repeat_kv(k_exact, g)
    vv_e = q3.repeat_kv(v_exact, g)

    T = query.shape[-2]
    idx = torch.arange(T, device=query.device)
    w = torch.matmul(query, kk_q.transpose(2, 3)) * scaling
    w[..., idx, idx] = (query * kk_e).sum(-1)[..., idx] * scaling
    if attention_mask is not None:
        w = w + attention_mask
    else:
        causal = torch.triu(torch.full((T, T), torch.finfo(w.dtype).min, device=w.device), 1)
        w = w + causal
    p = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    p_ii = p[..., idx, idx].clone()
    p[..., idx, idx] = 0
    out = torch.matmul(p, vv_q) + p_ii.unsqueeze(-1) * vv_e
    return out.transpose(1, 2).contiguous(), None


def patch_attention(attn, orig, state=None, codec=None, stash_half=False):
    """Return a replacement forward for attn (a Qwen3Attention).

    state: kv_hook_state() dict -> stash post-RoPE Q,K and V as [T,H,D] per layer
           (fp16 when stash_half, to bound GPU memory during capture).
    codec: kvcodec Codec; K/V are fake-quantized on write and attention runs kv_attn so the
           diagonal stays exact.
    """
    layer_idx = attn.layer_idx


    def forward(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = q3.apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        if state is not None:
            if stash_half:
                state[layer_idx].update(q=headify(q).half(), k=headify(k).half(),
                                        v=headify(v).half())
            else:
                state[layer_idx].update(q=headify(q), k=headify(k), v=headify(v))

        if codec is None:
            fn = q3.ALL_ATTENTION_FUNCTIONS.get_interface(
                attn.config._attn_implementation, q3.eager_attention_forward)
            out, aw = fn(attn, q, k, v, attention_mask,
                         dropout=0.0 if not attn.training else attn.attention_dropout,
                         scaling=attn.scaling, sliding_window=attn.sliding_window, **kw)
        else:
            kq = unheadify(codec.decode(codec.encode(headify(k), "k")))
            vq = unheadify(codec.decode(codec.encode(headify(v), "v")))
            out, aw = kv_attn(attn, q, k, kq, v, vq, attention_mask, attn.scaling)

        out = out.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(out), aw

    return forward


def pooled_sqnr(codec_factory, x: torch.Tensor, kind: str, dev="cuda") -> float:
    """SQNR (dB) of codec on x [L, T, H, D], pooled over layers, computed in per-layer chunks so
    quantizer temporaries stay small. codec_factory(seed) -> Codec (A_n sign seeds are per-layer).
    Returns energy-weighted pooled SQNR."""
    sig = err = 0.0
    for l in range(x.shape[0]):
        c = codec_factory(l)
        flat = x[l].reshape(-1, x.shape[2], x.shape[3]).to(dev).float()
        rec = c.decode(c.encode(flat, kind))
        sig += float(flat.double().pow(2).sum())
        err += float((rec - flat).double().pow(2).sum())
        del flat, rec
    return float("inf") if err == 0 else float(10 * math.log10(sig / err))
