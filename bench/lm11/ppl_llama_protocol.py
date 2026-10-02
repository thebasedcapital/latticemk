"""LM-11: wikitext-2 ppl under the exact llama-perplexity protocol, in HF.

Protocol (tools/perplexity/perplexity.cpp, ppl_stride<=0 path):
  text = "\n\n".join(test parquet rows); tokenize once (Qwen add_bos=false)
  n_ctx = 2048; chunks = floor(len/2048) non-overlapping
  per chunk: KV cleared; decode all 2048 tokens; score logits at positions
  j in [n_ctx/2, n_ctx-1) against token j+1 -> n_ctx/2 - 1 = 1023 terms/chunk.

This lets OUR packed weights (and plain HF weights) sit in the same table as
`llama-perplexity` GGUF numbers.

usage:
  ppl_llama_protocol.py fp
  ppl_llama_protocol.py fp16 --half-only
  ppl_llama_protocol.py weights PATH.pt        # packed {name:{codes,meta}}
"""
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lmk.model import N_LAYERS, SNAPSHOT  # noqa: E402

N_CTX = 2048
WIKI = next((Path.home() / ".cache/huggingface/hub/"
             "datasets--Salesforce--wikitext/snapshots").iterdir())
HERE = Path(__file__).resolve().parent

def deq_int4(codes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Unpack [out, in/8] int32 codes in pack.words nibble order.

    Each word stores columns 0,2,4,6 in its lower 16 bits and columns
    1,3,5,7 in its upper 16 bits; metadata stores an fp16 scale and offset
    per row and 128-column group.
    """
    out, nw = codes.shape
    sh = torch.arange(4, dtype=torch.int64).view(1, 1, 4)
    c = codes.to(torch.int64).unsqueeze(-1)
    lo = (c >> (4 * sh)) & 0xF           # codes 0,2,4,6 of each 8-group
    hi = (c >> (16 + 4 * sh)) & 0xF      # codes 1,3,5,7
    pair = torch.stack([lo, hi], -1).reshape(out, nw * 8)  # [out, in]
    meta_r = meta.float().repeat_interleave(128, 1)        # [out, in, 2]
    return pair.float() * meta_r[..., 0] + meta_r[..., 1]


@torch.no_grad()
def run(model, ids):
    n_chunk = len(ids) // N_CTX
    nll, count = 0.0, 0
    model.cuda().eval()
    for i in range(n_chunk):
        x = ids[i * N_CTX:(i + 1) * N_CTX].cuda()
        h = model.model(x.unsqueeze(0)).last_hidden_state[0]
        # logits at j predict token j+1; score j in [1024, 2047)
        for s in range(1024, N_CTX - 1, 512):
            e = min(s + 512, N_CTX - 1)
            nll += torch.nn.functional.cross_entropy(
                model.lm_head(h[s:e]).float(), x[s + 1:e + 1],
                reduction="sum").item()
        count += N_CTX - 1 - 1024
    return math.exp(nll / count)


def main():
    which = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = "\n\n".join(pq.read_table(
        WIKI / "wikitext-2-raw-v1/test-00000-of-00001.parquet")
        ["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    print(f"tokens {len(ids)} -> {len(ids) // N_CTX} chunks x {N_CTX}")

    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.float16 if which == "fp16" else torch.float32)
    if which in ("fp", "fp16"):
        pass
    elif which == "weights":
        pk = torch.load(sys.argv[2], map_location="cpu")
        deq = {n: deq_int4(v["codes"], v["meta"]) for n, v in pk.items()}
        lin = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}
        for i in range(N_LAYERS):
            p = f"model.layers.{i}."
            w = deq[f"L{i}.qkv"]
            lin[p + "self_attn.q_proj"].weight.data = w[:2048]
            lin[p + "self_attn.k_proj"].weight.data = w[2048:3072]
            lin[p + "self_attn.v_proj"].weight.data = w[3072:]
            lin[p + "self_attn.o_proj"].weight.data = deq[f"L{i}.o"]
            g = deq[f"L{i}.gu"]
            lin[p + "mlp.gate_proj"].weight.data = g[:3072]
            lin[p + "mlp.up_proj"].weight.data = g[3072:]
            lin[p + "mlp.down_proj"].weight.data = deq[f"L{i}.down"]
        model.lm_head.weight = torch.nn.Parameter(deq["lm_head"])
        del pk, deq
    else:
        raise SystemExit(f"unknown mode {which}")

    ppl = run(model, ids)
    print(f"{which} llama-protocol ppl = {ppl:.4f}  (-c 2048, half-window)")
    if "--half-only" in sys.argv:
        return

    # also emit the eval_ppl protocol number for the same weights: full-window
    # scoring, non-overlapping (tokens 1..2047 scored per window)
    n_win = len(ids) // N_CTX
    nll = 0.0
    model.cuda().eval()
    for j in range(n_win):
        x = ids[j * N_CTX:(j + 1) * N_CTX].cuda()
        h = model.model(x.unsqueeze(0)).last_hidden_state[0]
        for s in range(0, N_CTX - 1, 512):
            e = min(s + 512, N_CTX - 1)
            nll += torch.nn.functional.cross_entropy(
                model.lm_head(h[s:e]).float(), x[s + 1:e + 1],
                reduction="sum").item()
    print(f"{which} eval_ppl-protocol ppl = "
          f"{math.exp(nll / (n_win * (N_CTX - 1))):.4f}  (full window)")


if __name__ == "__main__":
    main()
