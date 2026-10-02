"""Reproducible sm_75 ExLlamaV2 fp16 attempt on original Qwen3 checkpoint.

Run: scripts/gpu.sh --timing bash bench/lm11/run_exllama.sh 128
The launcher supplies CUDA 13.3 and the cuBLAS wheel to ExLlama's JIT build.
"""
import json
import time
import sys
from pathlib import Path

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lmk.model import SNAPSHOT
from interleave import OUT, row


def main():
    ctx = int(sys.argv[1])
    config = ExLlamaV2Config()
    config.model_dir = str(SNAPSHOT)
    config.prepare()
    config.max_seq_len = ctx + 128
    config.no_flash_attn = True  # Triton FA-2 doesn't target sm_75
    model = ExLlamaV2(config)
    model.load()
    cache = ExLlamaV2Cache(model, max_seq_len=ctx + 128, lazy=False)
    for i in range(0, ctx, 256):
        ids = torch.full((1, min(256, ctx - i)), 9707, device="cpu",
                         dtype=torch.long)
        model.forward(ids, cache=cache, preprocess_only=True)
    ms = []
    cur = torch.tensor([[9707]], device="cpu")
    for i in range(30):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model.forward(cur, cache=cache, last_id_only=True)
        cur = logits[:, -1].argmax(-1).reshape(1, 1).cpu()
        torch.cuda.synchronize()
        if i >= 5:
            ms.append((time.perf_counter() - start) * 1000)
    wb = 2 * 596_049_920  # [assumed] upper-bound total F16 parameters
    kv = 2 * 28 * ctx * 1024 * 2
    result = row("exllamav2-fp16-eager", ctx, ms, wb, kv,
                 {"weight_bytes_note": "2 x parameter count upper bound"})
    with OUT.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(result)


if __name__ == "__main__":
    main()
