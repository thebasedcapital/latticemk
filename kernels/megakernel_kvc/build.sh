#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
NVCC="${NVCC:-$HOME/.local/cuda-13.3/bin/nvcc}"
CUDA_LIB="${CUDA_LIB:-$HOME/.local/cuda-13.3/lib}"
for name in kvc baseline16k; do
  "$NVCC" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
    -L"$CUDA_LIB" -Xcompiler -fPIC -Xptxas -v -shared -cudart static \
    -o "$ROOT/kernels/megakernel_kvc/lib${name}.so" \
    "$ROOT/kernels/megakernel_kvc/${name}.cu"
done
