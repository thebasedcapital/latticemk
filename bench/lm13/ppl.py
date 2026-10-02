"""EXL2 native forward, WikiText-2 test; llama-perplexity -c 2048 half-window protocol.

Run in bounded slices: scripts/gpu.sh bash bench/lm13/run.sh bench/lm13/ppl.py --start 0 --count 12
The saved partial sums allow restarts without rounding or silently skipping windows.
"""
import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config

ROOT = Path(__file__).resolve().parents[2]
WIKI = next((Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots").iterdir())
SOURCE = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots").iterdir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "baselines/exl2/qwen3-0.6b-4.125bpw-head4-causal"))
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--headnorm-corrected", action="store_true")
    ap.add_argument("--no-sdpa", action="store_true",
                    help="Use ExLlama's explicitly causal matmul attention for multi-token prefill")
    args = ap.parse_args()
    text = "\n\n".join(pq.read_table(WIKI / "wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].to_pylist())
    ids = AutoTokenizer.from_pretrained(SOURCE)(text, return_tensors="pt").input_ids[0]
    windows = len(ids) // 2048
    assert 0 <= args.start < windows and args.start + args.count <= windows
    config = ExLlamaV2Config()
    config.model_dir = args.model
    config.prepare()
    if args.headnorm_corrected:
        config.arch.lm.headnorm = "rmsnorm"
    config.max_seq_len = 2048
    config.max_input_len = 512
    config.max_output_len = 128
    config.no_flash_attn = True
    config.no_sdpa = args.no_sdpa
    model = ExLlamaV2(config)
    model.load()
    cache = ExLlamaV2Cache(model, max_seq_len=2048, lazy=False)
    nll = 0.0
    count = 0
    with torch.inference_mode():
        for wi in range(args.start, args.start + args.count):
            x = ids[wi * 2048:(wi + 1) * 2048]
            cache.current_seq_len = 0
            model.forward(x[:1024].reshape(1, -1), cache=cache, preprocess_only=True)
            for j in range(1024, 2048, 128):
                e = min(j + 128, 2048)
                logits = model.forward(x[j:e].reshape(1, -1), cache=cache)[0]
                end = min(e, 2047)
                nll += torch.nn.functional.cross_entropy(
                    logits[:end-j].float(), x[j+1:end+1].to(logits.device), reduction="sum").item()
                count += end - j
            assert cache.current_seq_len == 2048
            print(f"window {wi + 1}/{windows}: ppl {math.exp(nll / count):.5f}", flush=True)
    path = ROOT / "bench/lm13/ppl_parts.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({"model": args.model, "headnorm": config.arch.lm.headnorm,
                            "no_sdpa": config.no_sdpa, "start": args.start,
                            "count_windows": args.count, "nll": nll,
                            "tokens_scored": count, "all_windows": windows,
                            "ppl": math.exp(nll/count)}) + "\n")
    print(f"Saved {count} scored tokens: {math.exp(nll/count):.6f}", flush=True)


if __name__ == "__main__":
    main()
