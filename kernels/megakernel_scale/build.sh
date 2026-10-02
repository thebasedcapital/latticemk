#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA="$HOME/.local/cuda-13.3"
"$CUDA/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
  -DFP32_DOT -L"$CUDA/lib" -Xptxas=-v -Xcompiler -fPIC -shared \
  -cudart static -o kernels/megakernel_scale/libmega_scale.so \
  kernels/megakernel_scale/mega_scale.cu
"$CUDA/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
  -L"$CUDA/lib" -Xptxas=-v -Xcompiler -fPIC -shared -cudart static \
  -o kernels/megakernel_scale/libmega_scale_fp16.so \
  kernels/megakernel_scale/mega_scale.cu
for ctx in 128 2048 8192; do
  .venv/bin/python kernels/megakernel_scale/sched_scale.py "$ctx" \
    "kernels/megakernel_scale/schedules/scale_ctx$ctx.json"
  validator/target/release/schedcheck \
    "kernels/megakernel_scale/schedules/scale_ctx$ctx.json"
done
