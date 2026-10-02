#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
variant=${1:-original}
flags=()
if [[ "$variant" == original ]]; then flags=(-DORIGINAL); fi
if [[ "$variant" == nonvolatile ]]; then flags=(-DNONVOLATILE); fi
if [[ "$variant" == preload ]]; then flags=(-DPRELOAD); fi
for m in 1 2 3 4; do
  threads=1024
  if [[ "$variant" == preload ]] && (( m > 1 )); then threads=512; fi
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v -DMT="$m" -DTHREADS="$threads" "${flags[@]}" shape_probe.cu -o "libshape_${variant}_m${m}.so" 2>&1 | tee "build-shape-${variant}-m${m}.log"
done
