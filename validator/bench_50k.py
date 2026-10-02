#!/usr/bin/env python3
"""Reproduce the LM-05 50k-task timing row.

  cargo build --release --offline
  python3 bench_50k.py            # 21 runs, prints median/p10/p90

Generates a valid schedule with `fuzz --emit` (198 layers x 36 blocks =
49,932 tasks, ~10.8 MB JSON) into /tmp, then times `schedcheck` wall time
per invocation (parse + all checks included).
"""
import math
import statistics
import subprocess
import sys
import tempfile
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCHED = os.path.join(tempfile.gettempdir(), "lm05_sched50k.json")

subprocess.run(
    [f"{HERE}/target/release/fuzz", "--emit", SCHED, "--layers", "198", "--blocks", "36"],
    check=True,
)

ts = []
for _ in range(21):
    t0 = time.perf_counter()
    r = subprocess.run([f"{HERE}/target/release/schedcheck", SCHED], capture_output=True, text=True)
    ts.append(time.perf_counter() - t0)
    assert r.returncode == 0 and "ACCEPT" in r.stdout, r.stdout + r.stderr

ts.sort()
n = len(ts)
pct = lambda p: ts[min(n - 1, math.ceil(p * n) - 1)]
print(f"n={n} median={statistics.median(ts)*1e3:.1f}ms "
      f"p10={pct(0.10)*1e3:.1f}ms p90={pct(0.90)*1e3:.1f}ms "
      f"min={ts[0]*1e3:.1f}ms max={ts[-1]*1e3:.1f}ms")
sys.exit(0)
