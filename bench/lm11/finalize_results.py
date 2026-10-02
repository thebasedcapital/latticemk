"""Attach measured perplexity to LM-11 rows; repair early llama-bench total-ms rows.

The first interleave.py pass accidentally converted samples_ts to n_gen / tok/s
rather than 1 / tok/s. Correct this algebraically using the stored n_gen.
Run only after all timing jobs have completed.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results.jsonl"
PPL = json.loads((HERE / "ppl.json").read_text())["ppl"]


def score(r):
    name = r["kernel"]
    if name.startswith("llamacpp-"):
        quant = name.removeprefix("llamacpp-").split("-fa")[0]
        k, v = r["type_k"], r["type_v"]
        key = f"{quant}/f16" if (k, v) == ("f16", "f16") else f"{quant}/{k}+{v}"
    elif name.startswith("megakernel-v2"):
        key = "int4gptq-fake/f16" if name.endswith("-gptq") else "int4rtn-fake/f16"
    elif name.startswith("hf-fp16") or name.startswith("exllamav2-fp16"):
        key = "hf-fp16/f16"
    else:
        return None
    return PPL.get(key)


def main():
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    fixed = 0
    for r in rows:
        if r["kernel"].startswith("llamacpp-") and "n_gen" in r:
            n = r["n_gen"]
            # avg_ts is reported by llama-bench independently of our conversion.
            if abs(r["tokens_per_s"] * n - r["avg_ts"]) < 0.15 * r["avg_ts"]:
                r["median_ms"] /= n
                r["p10_ms"] /= n
                r["p90_ms"] /= n
                r["tokens_per_s"] *= n
                r["gbps"] *= n
                r["pct_roofline"] *= n
                fixed += 1
        ppl = score(r)
        if ppl is not None:
            r["ppl"] = ppl
            r["ppl_protocol"] = ("wikitext-2 2048 half-window, HF fp16"
                                 if r["kernel"].startswith(("hf-fp16", "exllamav2-fp16"))
                                 else "wikitext-2 llama-perplexity -c 2048")
            if r["kernel"].startswith("exllamav2-fp16"):
                r["ppl_source"] = "HF fp16 weight-quality proxy; ExLlama arithmetic not scored"
            elif r["kernel"] == "hf-fp16-compile":
                r["ppl_source"] = "HF fp16 eager evaluation; compiled arithmetic not separately scored"
            target = PPL["int4gptq-fake/f16"]
            r["quality_matched_to_v2_gptq"] = target <= ppl <= 1.05 * target
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"normalized {len(rows)} rows, corrected {fixed} early total-ms rows")


if __name__ == "__main__":
    main()
