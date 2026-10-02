"""Score one Qwen3-1.7B GGUF on the LM-11 WikiText-2 perplexity protocol.

Run one model per GPU job, e.g. scripts/gpu.sh .venv/bin/python
bench/lm12/measure_ppl.py q4_0. A completed run updates ppl.json; an
interrupted run never records a partial PPL as a full-corpus result.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LABELS = {
    "f16": ("f16/f16", "qwen3-1.7b-F16.gguf"),
    "q4_0": ("q4_0/f16", "qwen3-1.7b-Q4_0.gguf"),
    "q4_k_m": ("q4_k_m/f16", "qwen3-1.7b-Q4_K_M.gguf"),
    "gptq": ("int4gptq-fake/f16", "qwen3-1.7b-gptq-fake.gguf"),
}


def main():
    harvest = sys.argv[1] == "harvest"
    kind = sys.argv[2] if harvest else sys.argv[1]
    label, filename = LABELS[kind]
    if harvest:
        output = (HERE / f"ppl.{kind}.log").read_text()
        status = 0
    else:
        cmd = [str(Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))
                   / "build-cuda/bin/llama-perplexity"),
               "-m", str(HERE / filename), "-f",
               str(ROOT / "baselines/wikitext-2-test.txt"),
               "-c", "2048", "-ngl", "99", "-b", "2048", "-ub", "512",
               "--no-warmup", "-lv", "3", "-fa", "on", "-ctk", "f16",
               "-ctv", "f16"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        status, output = result.returncode, result.stdout
        (HERE / f"ppl.{kind}.log").write_text(output)
    match = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    progress = re.findall(r"\[(\d+)\]([0-9.]+)", output)
    if match is None and progress and int(progress[-1][0]) == 146:
        value = progress[-1][1]
    else:
        value = match.group(1) if match is not None else None
    if status != 0 or value is None:
        print(output[-3000:])
        raise SystemExit(f"perplexity {kind} did not finish all 146 windows")
    path = HERE / "ppl.json"
    payload = (json.loads(path.read_text()) if path.exists() else {
        "protocol": "llama-perplexity -c 2048 -ngl 99 -b 2048 -ub 512 --no-warmup -fa on; 146 WikiText-2 test windows, second half scored",
        "llama_commit": "6011c34c", "ppl": {}})
    payload["ppl"][label] = float(value)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{label} PPL {value}; {HERE / f'ppl.{kind}.log'}", flush=True)


if __name__ == "__main__":
    main()
