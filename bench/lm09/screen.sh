#!/bin/bash
# LM-09 ppl screen: 48 wikitext windows per codec, one gpu.sh job each.
cd "$(dirname "$0")/../.."
for c in "$@"; do
  scripts/gpu.sh .venv/bin/python bench/lm09/eval_codec.py \
    --codecs "$c" --parts sqnr,attn,ppl --ppl-windows 0:48
done
