#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for model in f16 q4_0 q4_k_m gptq; do
  timeout 300 scripts/gpu.sh .venv/bin/python bench/lm12/measure_ppl.py "$model"
done
