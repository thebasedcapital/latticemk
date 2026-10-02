"""Character count per post of publish/x_thread.md (marker lines excluded); exit 1 if any post exceeds 280."""

import re
import sys
from pathlib import Path

text = (Path(__file__).parent / "x_thread.md").read_text()
posts = re.split(r"^## (\d+)\s*$", text, flags=re.M)[1:]
bad = False
for num, body in zip(posts[::2], posts[1::2]):
    lines = [l for l in body.strip().splitlines() if not l.startswith("[")]
    post = "\n".join(lines).strip()
    n = len(post)
    bad |= n > 280
    print(f"post {num:>2}: {n:3d} chars{'  OVER' if n > 280 else ''}")
sys.exit(1 if bad else 0)
