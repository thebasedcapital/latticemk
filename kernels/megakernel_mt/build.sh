#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for m in "${@:-1 2 3 4 5}"; do
  for n in $m; do
    threads=1024
    if (( n == 5 )); then threads=512; fi
    "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
      -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v \
      -DMT="$n" -DTHREADS="$threads" -o "libmt$n.so" mega_mt.cu 2>&1 | tee "build-m$n.log"
  done
done
for n in 1 4; do
  threads=1024
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
    -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v \
    -DMT="$n" -DTHREADS="$threads" -DPROFILE -o "libmt${n}profile.so" mega_mt.cu 2>&1 | tee "build-profile-m$n.log"
done
"$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
  -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v \
  -DMT=4 -DMT_MMA -o libmt4mma.so mega_mt.cu 2>&1 | tee build-mma-m4.log
"$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
  -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v \
  -DMT=4 -DTHREADS=512 -o libmt4initial.so initial/mega_mt.cu 2>&1 | tee build-initial-m4.log
"$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
  -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v \
  -DMT=4 -DMT_FP32 -o libmt4fp32.so mega_mt.cu 2>&1 | tee build-fp32-m4.log
