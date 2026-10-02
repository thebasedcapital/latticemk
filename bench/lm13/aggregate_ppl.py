"""Check exact coverage of 146 independent WikiText-2 windows, then combine NLL."""
import json
import math
from pathlib import Path

parts = [json.loads(s) for s in (Path(__file__).parent / "ppl_parts.jsonl").read_text().splitlines()]
by_model = {}
for p in parts:
    by_model.setdefault((p["model"], p.get("headnorm", "layernorm"), p.get("no_sdpa", False)), []).append(p)
for (model, headnorm, no_sdpa), rows in by_model.items():
    windows = rows[0]["all_windows"]
    seen = set()
    for row in rows:
        assert row["all_windows"] == windows
        segment = set(range(row["start"], row["start"] + row["count_windows"]))
        assert not seen.intersection(segment), f"duplicate windows in {model}: {seen.intersection(segment)}"
        assert row["tokens_scored"] == 1023 * row["count_windows"]
        seen.update(segment)
    if seen != set(range(windows)):
        missing = sorted(set(range(windows)) - seen)
        print(f"{model} headnorm={headnorm} no_sdpa={no_sdpa}: {len(seen)}/{windows} windows; missing {len(missing)} (first {missing[:8]})")
        continue
    ppl = math.exp(sum(p["nll"] for p in rows) / sum(p["tokens_scored"] for p in rows))
    print(f"{model} headnorm={headnorm} no_sdpa={no_sdpa}: {windows} windows, {windows * 1023} scored tokens, PPL {ppl:.6f}")
