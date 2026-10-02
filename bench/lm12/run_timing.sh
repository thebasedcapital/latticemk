#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for ctx in 128 2048 8192; do
  timeout 300 scripts/gpu.sh --timing .venv/bin/python bench/lm12/interleave.py "$ctx" \
    --llama bench/lm12/qwen3-1.7b-Q4_0.gguf \
    --llama bench/lm12/qwen3-1.7b-Q4_K_M.gguf
done
