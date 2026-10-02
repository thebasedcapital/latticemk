#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
variant=${1:-mma1}; shift || true
flags=()
if [[ "$variant" == mma* ]]; then flags=(-DMMA -DMMA_SPLIT="${variant#mma}"); fi
if [[ "$variant" == sw* ]]; then flags=(-DMMA -DMMA_SWIZZLE -DMMA_SPLIT="${variant#sw}"); fi
for m in ${*:-1 2 4}; do
  threads=512
  if [[ "$variant" == cuda && "$m" == 1 ]]; then threads=1024; fi
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -L"$HOME/.local/cuda-13.3/lib" --fmad=false -Xcompiler -fPIC -shared -cudart static -Xptxas=-v -DMT="$m" -DTHREADS="$threads" "${flags[@]}" gemm_probe.cu -o "libshape_${variant}_m${m}.so" 2>&1 | tee "build-shape-${variant}-m${m}.log"
done
