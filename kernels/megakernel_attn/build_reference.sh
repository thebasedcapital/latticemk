#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for m in 1 4; do
  threads=1024
  if (( m > 1 )); then threads=512; fi
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v --fmad=false -DPROFILE -DPROFILE_DETAIL -DMT="$m" -DTHREADS="$threads" ../megakernel_mt2/mega_mt.cu -o "libmt${m}referencedetail.so" 2>&1 | tee "build-reference-detail-m${m}.log"
done
