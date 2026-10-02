"""wikitext-2 (test) perplexity of Qwen3-0.6B-Base with every streamed linear fake-quantized (incl. an untied lm_head).

usage: eval_ppl.py CONFIG [CONFIG ...]
  CONFIG = fp | BASE[+rht][+gptq]   BASE = int{2,3,4} | A{2,4}-k{K} | lloyd{2,4}-k{K}
  +rht: block randomized Hadamard on the input dim; +gptq: Hessian-aware rounding (block LDLQ, 128x2048 wikitext-2 train tokens)
"""

import gc
import math
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk import codebooks, gptq, quant
from lmk.model import SNAPSHOT

SEQ = 2048
CALIB = 128
WIKI = next((Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots").iterdir())


def quantizer(cfg: str):
    """-> (fn(w, seed, h) returning the dequantized weight in the original basis, or None for fp; needs_hessian)."""
    parts = cfg.split("+")
    base, rht, use_h = parts[0], "rht" in parts, "gptq" in parts
    if base == "fp":
        return None, False
    if m := re.fullmatch(r"int(\d)", base):
        bits = int(m[1])
        fn = (lambda w, h: gptq.gptq_int(w, h, bits)) if use_h else (lambda w, h: quant.quant_int(w, bits).deq)
    elif m := re.fullmatch(r"(A|lloyd)(\d)-k(\d+)", base):
        d, k = int(m[2]), int(m[3])
        cb = codebooks.an_codebook(d, k) if m[1] == "A" else codebooks.lloyd_codebook(d, k)
        cb = torch.from_numpy(cb).cuda()
        fn = (lambda w, h: gptq.gptq_vq(w, h, cb)) if use_h else (lambda w, h: quant.quant_vq(w, cb).deq)
    else:
        raise ValueError(cfg)

    def run(w, seed, h):
        if not rht:
            return fn(w, h)
        s = quant.rht_signs(w.shape[1], seed)
        if h is not None:  # H' = Q H Q^T for W' = W Q^T
            h = quant.rht_apply(quant.rht_apply(h, s).T.contiguous(), s).T.contiguous()
        return quant.rht_apply(fn(quant.rht_apply(w, s), h), s, inverse=True)

    return run, use_h


def calib_windows(tok) -> torch.Tensor:
    text = "\n\n".join(pq.read_table(WIKI / "wikitext-2-raw-v1/train-00000-of-00001.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - SEQ, (CALIB,), generator=g)
    return torch.stack([ids[s : s + SEQ] for s in starts])


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = "\n\n".join(pq.read_table(WIKI / "wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_win = len(ids) // SEQ
    hess = None
    for cfg in sys.argv[1:]:
        model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32)
        q, use_h = quantizer(cfg)
        if use_h and hess is None:
            hess = gptq.calib_hessians(model.cuda().eval(), calib_windows(tok))
            model.cpu()
            torch.cuda.empty_cache()
        if q is not None:
            lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
            for i, (n, m) in enumerate(lins):
                w = m.weight.data.cuda()
                h = hess[gptq.h_key(n)].cuda() if use_h else None
                deq = q(w, i, h)
                m.weight = torch.nn.Parameter(deq.cpu())  # lm_head gets its own tensor: embedding stays exact
                del w, h, deq
                torch.cuda.empty_cache()
        model.cuda().eval()
        nll = 0.0
        for j in range(n_win):
            x = ids[j * SEQ : (j + 1) * SEQ].cuda()
            h = model.model(x.unsqueeze(0)).last_hidden_state[0]
            for s in range(0, SEQ - 1, 512):  # chunked logits: full [2048, 151936] fp32 does not fit next to the model
                e = min(s + 512, SEQ - 1)
                nll += torch.nn.functional.cross_entropy(model.lm_head(h[s:e]), x[s + 1 : e + 1], reduction="sum").item()
        print(f"{cfg:<16} ppl {math.exp(nll / (n_win * (SEQ - 1))):8.3f}   ({n_win} x {SEQ} tokens)", flush=True)
        del model, h
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
