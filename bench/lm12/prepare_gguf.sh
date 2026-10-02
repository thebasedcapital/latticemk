#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=$(.venv/bin/python -c 'from huggingface_hub import snapshot_download;print(snapshot_download("Qwen/Qwen3-1.7B-Base", revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"))')
CONVERT="${LLAMA_CPP_DIR:-$HOME/llama.cpp}/convert_hf_to_gguf.py"
QUANT="${LLAMA_CPP_DIR:-$HOME/llama.cpp}/build-cuda/bin/llama-quantize"
.venv/bin/python "$CONVERT" "$BASE" --outfile bench/lm12/qwen3-1.7b-F16.gguf --outtype f16
scripts/gpu.sh "$QUANT" bench/lm12/qwen3-1.7b-F16.gguf \
  bench/lm12/qwen3-1.7b-Q4_0.gguf Q4_0
scripts/gpu.sh "$QUANT" bench/lm12/qwen3-1.7b-F16.gguf \
  bench/lm12/qwen3-1.7b-Q4_K_M.gguf Q4_K_M
.venv/bin/python bench/lm12/weights_to_gguf.py \
  bench/lm12/weights_int4_gptq.pt bench/lm12/qwen3-1.7b-gptq-fake.gguf
