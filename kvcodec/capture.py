"""Capture the K/V tensors Qwen3-0.6B-Base writes to its KV cache, on wikitext-2 test windows.

K is taken post-k_norm, post-RoPE (exactly what the cache stores); V pre-attention, post-v_proj.
Both are [1, 8, T, 128] inside attention; saved transposed to [T, 8, 128] fp16 (the codec
convention). Q (post-q_norm, post-RoPE) is saved for window 0 only, for the attn-error eval.

usage: scripts/gpu.sh .venv/bin/python kvcodec/capture.py [n_windows=3]
Output: kvcodec/data/{k,v}.pt  [n_windows, T, 8, 128] fp16; kvcodec/data/q.pt [T, 8, 128] fp16
        (window 0). K+V at n_windows=3 ~= 1.4 GB on disk.
"""

import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lmk.model import SNAPSHOT


from .common import DATA, SEQ, WIKI, kv_hook_state, patch_attention


@torch.no_grad()
def main():
    n_win = int(sys.argv[1]) if len(sys.argv) > 1 else N_WIN
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = "\n\n".join(pq.read_table(WIKI / "wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32).cuda().eval()

    state = kv_hook_state()  # per-layer capture lists
    for am in model.model.layers:
        orig = am.self_attn.forward
        am.self_attn.forward = patch_attention(am.self_attn, orig, state=state, codec=None,
                                               stash_half=True)

    ks, vs = [], []
    t0 = time.time()
    for w in range(n_win):
        for s in state.values():
            s.clear()
        x = ids[w * SEQ:(w + 1) * SEQ].cuda()
        model.model(x.unsqueeze(0))
        ks.append(torch.stack([state[l]["k"] for l in range(28)]))  # [28, T, 8, 128]
        vs.append(torch.stack([state[l]["v"] for l in range(28)]))
        if w == 0:
            DATA.mkdir(exist_ok=True)
            torch.save(torch.stack([state[l]["q"] for l in range(28)]).half().cpu(), DATA / "q.pt")
        print(f"window {w}: {time.time() - t0:.1f}s total", flush=True)

    DATA.mkdir(exist_ok=True)
    torch.save(torch.stack(ks).half().cpu(), DATA / "k.pt")  # [n_win, 28, T, 8, 128]
    torch.save(torch.stack(vs).half().cpu(), DATA / "v.pt")
    mb = sum(f.stat().st_size for f in DATA.glob("*.pt")) / 1e6
    print(f"saved {mb:.0f} MB to {DATA}")


if __name__ == "__main__":
    main()
