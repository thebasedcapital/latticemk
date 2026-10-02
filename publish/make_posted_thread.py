"""Build the posted version of the thread: posts 1-11 of x_thread.md, lowercased in the account's voice.

Identifiers whose case carries meaning stay as written. Output: publish/x_thread_posted.json
([{"n": 1, "text": ..., "image": path-or-null}, ...]); exit 1 if any post exceeds 280 chars.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
KEEP = ["Qwen3-0.6B", "Qwen3-1.7B", "0.6B", "1.7B", "RTX 4000", "CUDA", "INT4", "GPTQ", "KV", "SM", "MB", "Q4_0",
        "Q4_K_M", "ExLlamaV2", "EXL2", "RMSNorm", "LayerNorm", "SDPA", "HF", "NaNs"]
LAST = 11

text = (HERE / "x_thread.md").read_text()
parts = re.split(r"^## (\d+)\s*$", text, flags=re.M)[1:]
posts = []
for num, body in zip(parts[::2], parts[1::2]):
    n = int(num)
    if n > LAST:
        continue
    lines = body.strip().splitlines()
    chart = next((re.search(r"\[chart: (\S+)\]", l)[1] for l in lines if l.startswith("[chart:")), None)
    out = "\n".join(l for l in lines if not l.startswith("[")).strip().lower()
    for word in KEEP:
        out = re.sub(rf"(?<![\w.]){re.escape(word.lower())}(?![\w])", word, out)
    posts.append({"n": n, "text": out, "image": chart})

(HERE / "x_thread_posted.json").write_text(json.dumps(posts, indent=1, ensure_ascii=False))
bad = [p["n"] for p in posts if len(p["text"]) > 280]
for p in posts:
    print(f"--- {p['n']} ({len(p['text'])} chars){'  image: ' + p['image'] if p['image'] else ''}\n{p['text']}")
sys.exit(1 if bad else 0)
