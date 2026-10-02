"""Collect Qwen3-1.7B GPTQ input Hessians on WikiText-2 training windows.

Usage: scripts/gpu.sh .venv/bin/python bench/lm12/calibrate.py
       START STOP WINDOW_START WINDOW_STOP
Each GPU job processes 32 of the same seeded 128 x 2048 WikiText-2 train
windows as LM-11. `calibrate.py merge START STOP` averages four disjoint
32-window fp32 Hessian files, with all accumulators kept on CPU.
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import scale
from lmk.gptq import h_key


def main():
    here = Path(__file__).resolve().parent
    if sys.argv[1] == "merge":
        lo, hi = map(int, sys.argv[2:4])
        parts = [torch.load(here / f"hessians.{lo}-{hi}.{i}-{i+32}.pt",
                            map_location="cpu") for i in range(0, 128, 32)]
        assert all(part.keys() == parts[0].keys() for part in parts)
        merged = {key: sum((part[key] for part in parts)) / 4
                  for key in parts[0]}
        outpath = here / f"hessians.{lo}-{hi}.pt"
        torch.save(merged, outpath)
        print(f"merged {outpath} keys={len(merged)}", flush=True)
        return
    lo, hi, begin, end = map(int, sys.argv[1:5])
    assert 0 <= lo < hi <= scale.N_LAYERS
    assert 0 <= begin < end <= 128 and begin % 32 == end % 32 == 0
    wiki = next((Path.home() / ".cache/huggingface/hub/"
                 "datasets--Salesforce--wikitext/snapshots").iterdir())
    text = "\n\n".join(pq.read_table(
        wiki / "wikitext-2-raw-v1/train-00000-of-00001.parquet")
        ["text"].to_pylist())
    ids = AutoTokenizer.from_pretrained(scale.SNAPSHOT)(
        text, return_tensors="pt").input_ids[0]
    gen = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - 2048, (128,), generator=gen)
    model = AutoModelForCausalLM.from_pretrained(
        scale.SNAPSHOT, dtype=torch.float16).cuda().eval()
    sums, hooks = {}, []

    def collect(name, inp):
        x = inp.reshape(-1, inp.shape[-1]).float()
        cur = (x.T @ x).cpu()  # keep the 4-layer accumulators off the 8 GB GPU
        if name in sums:
            sums[name].add_(cur)
        else:
            sums[name] = cur

    for name, module in model.model.named_modules():
        if (isinstance(module, torch.nn.Linear) and h_key(name) == name
                and lo <= int(name.split(".")[1]) < hi):
            hooks.append(module.register_forward_hook(
                lambda _m, inp, _out, key=f"model.{name}":
                collect(key, inp[0])))
    if hi == scale.N_LAYERS:
        hooks.append(model.model.norm.register_forward_hook(
            lambda _m, _inp, out: collect("lm_head", out)))
    with torch.inference_mode():
        for i in range(begin, end):
            start = starts[i]
            model.model(ids[start:start + 2048].unsqueeze(0).cuda())
            if (i + 1) % 16 == 0:
                print(f"range {lo}:{hi} windows {i+1}/128", flush=True)
    for hook in hooks:
        hook.remove()
    out = {key: value / ((end - begin) * 2048) for key, value in sums.items()}
    path = here / f"hessians.{lo}-{hi}.{begin}-{end}.pt"
    torch.save(out, path)
    print(f"saved {path} keys={len(out)}", flush=True)


if __name__ == "__main__":
    main()
