"""Byte model [derived]: decode-step DRAM bytes and roofline-bound tokens/s vs context length.

Weight bytes/token: 316.6 MB (INT4, wave-1 measurement). KV bytes/token: 114,688 B/ctx-token at
fp16 (2 x 28 L x 8 H x 128 D x 2 B), scaled by codec bits_per_coord / 16. Roofline: 406.7 GB/s
measured read bandwidth (hw/hw.json); tok/s = roofline / bytes-per-token, an upper bound ignoring
attention FLOPs and any non-memory work. usage: .venv/bin/python -m kvcodec.byte_model
"""

import json
import sys
from pathlib import Path

WEIGHT_MB = 316.6          # INT4 weights per token [measured, wave-1]
ROOFLINE_GBS = 406.7       # measured read roofline
KV_FP16_BYTES = 114_688    # bytes per context token per decode step at fp16
CTX = [128, 2048, 8192, 32768]
DEFAULT_BITS = {"fp16": 16.0, "int8": 8.25, "int4": 4.25, "kivi-int4": 4.25,
                "kivi-int2": 2.25, "A2-k10": 2.625, "A4-k9": 2.375}


def table(bits_map: dict) -> list[dict]:
    rows = []
    for codec, bpc in bits_map.items():
        kv_b = KV_FP16_BYTES * bpc / 16.0
        for c in CTX:
            mb = WEIGHT_MB + kv_b * c / 1e6
            rows.append({"codec": codec, "bits_per_coord": bpc, "ctx": c,
                         "kv_mb_per_tok": kv_b * c / 1e6, "total_mb_per_tok": mb,
                         "kv_share": round(kv_b * c / 1e6 / mb, 4),
                         "toks_at_roofline": round(ROOFLINE_GBS * 1e3 / mb)})
    return rows


def main():
    bits = dict(DEFAULT_BITS)
    res = Path(__file__).resolve().parent / "results.jsonl"
    if res.exists():  # replace defaults with measured bits_per_coord where available
        for line in res.read_text().splitlines():
            r = json.loads(line)
            if "bits_per_coord" in r:
                bits[r["codec"]] = r["bits_per_coord"]
    for r in table(bits):
        print(f"{r['codec']:<10} {r['bits_per_coord']:5.3f} b/c  ctx {r['ctx']:>6}: "
              f"KV {r['kv_mb_per_tok']:8.1f} MB ({r['kv_share'] * 100:5.1f}%)  "
              f"total {r['total_mb_per_tok']:8.1f} MB  -> <= {r['toks_at_roofline']:5d} tok/s")


if __name__ == "__main__":
    main()
