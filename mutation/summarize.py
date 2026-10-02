"""Kill matrix summary for mutation/results.jsonl: per model x family, killing stage, survivors.

usage: .venv/bin/python mutation/summarize.py [--survivors]
Status of a mutant: COMPILE_FAILURE (not counted), EQUIVALENT (identical logits to the original,
not counted), SURVIVED, or KILLED. `gate` = the v2.1 gate alone (gpu_stage, plus the schedule
validator); `+extra` = after mutation/extra_tests.py stages (`stage`). Control rows are excluded.
"""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def status(r: dict, key: str = "stage") -> str:
    if r.get("build") not in ("PASS", "N/A"):  # N/A: schedule-only mutants, never compiled
        return "COMPILE_FAILURE"
    if r.get("equivalent"):
        return "EQUIVALENT"
    s = r.get(key)
    if key == "gpu_stage" and s is None:
        s = r.get("stage")  # killed before any GPU stage (schedule validator)
    if key == "stage" and r.get("extra_fails"):
        return "KILLED"  # survived the v2.1 gate, failed an extra test (recorded in extra_fails)
    return "SURVIVED" if s in (None, "SURVIVED") else "KILLED"


def rate(killed: int, survived: int) -> str:
    n = killed + survived
    return f"{100 * killed / n:.1f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", action="store_true", help="list every surviving mutant")
    a = ap.parse_args()
    rows = {}
    for line in (ROOT / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["mutant_id"]] = r  # last record wins (reruns)
    manifests = {m: sum(1 for _ in open(ROOT / f"manifest-{m}.jsonl")) for m in ("0.6B", "1.7B")}

    for model in ("0.6B", "1.7B"):
        rs = [r for r in rows.values() if r["model"] == model and r["family"] != "control"]
        print(f"\n== {model}: {len(rs)}/{manifests[model]} mutants recorded")
        by_fam = collections.defaultdict(collections.Counter)
        stages = collections.Counter()
        for r in rs:
            g, x = status(r, "gpu_stage"), status(r)
            by_fam[r["family"]][x] += 1
            by_fam[r["family"]]["gate_" + g] += 1
            if x == "KILLED":
                stages[r["stage"]] += 1
        print(f"{'family':<16}{'killed':>7}{'survived':>9}{'equiv':>7}{'compile':>9}{'gate rate':>11}{'+extra rate':>13}")
        tot = collections.Counter()
        for fam, c in sorted(by_fam.items()):
            tot.update(c)
            print(f"{fam:<16}{c['KILLED']:>7}{c['SURVIVED']:>9}{c['EQUIVALENT']:>7}{c['COMPILE_FAILURE']:>9}"
                  f"{rate(c['gate_KILLED'], c['gate_SURVIVED']):>11}{rate(c['KILLED'], c['SURVIVED']):>13}")
        print(f"{'all':<16}{tot['KILLED']:>7}{tot['SURVIVED']:>9}{tot['EQUIVALENT']:>7}{tot['COMPILE_FAILURE']:>9}"
              f"{rate(tot['gate_KILLED'], tot['gate_SURVIVED']):>11}{rate(tot['KILLED'], tot['SURVIVED']):>13}")
        print("killing stage:", dict(stages.most_common()))
        if a.survivors:
            for r in sorted((r for r in rs if status(r) == "SURVIVED"), key=lambda r: (r["family"], r["operator"])):
                print(f"   SURVIVED {r['mutant_id']:<48} max_diff={r.get('max_diff')}")


if __name__ == "__main__":
    main()
