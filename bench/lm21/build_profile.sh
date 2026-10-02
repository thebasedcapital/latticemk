#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash kernels/megakernel_pro/build.sh detail 1 4
for m in 1 4; do
  threads=512; if (( m == 1 )); then threads=1024; fi
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v --fmad=false -DPROFILE -DPROFILE_DETAIL -DMT="$m" -DTHREADS="$threads" kernels/megakernel_mt2/mega_mt.cu -o "kernels/megakernel_pro/libmt${m}baseline_detail.so" 2>&1 | tee "kernels/megakernel_pro/build-baseline-detail-m${m}.log"
done
