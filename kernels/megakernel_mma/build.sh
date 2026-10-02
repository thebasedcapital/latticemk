#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
variant=${1:-selected}
shift || true
flags=();suffix=''
case "$variant" in
  original) flags=(-DORIGINAL);suffix=original;;
  nonvolatile) flags=(-DNONVOLATILE);suffix=nonvolatile;;
  interleaved) flags=(-DINTERLEAVED);suffix=interleaved;;
  preload) suffix=preload;;
  selected) flags=(--fmad=false);;
  profile) flags=(--fmad=false -DPROFILE);suffix=profile;;
  nocontract) flags=(--fmad=false);suffix=nocontract;;
  detail) flags=(--fmad=false -DPROFILE -DPROFILE_DETAIL);suffix=detail;;
  mma) flags=(--fmad=false -DMMA -DMMA_SWIZZLE);suffix=mma;;
  *) echo "unknown variant: $variant" >&2;exit 2;;
esac
for n in ${*:-1 2 3 4 5}; do
  threads=1024
  if (( n == 5 )); then threads=512; fi
  if [[ "$variant" == preload || "$variant" == selected || "$variant" == profile || "$variant" == nocontract || "$variant" == detail ]] && (( n > 1 )); then threads=512; fi
  if [[ "$variant" == mma ]]; then threads=512; fi
  "$HOME/.local/cuda-13.3/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -L"$HOME/.local/cuda-13.3/lib" -Xcompiler -fPIC -shared -cudart static -Xptxas=-v -DMT="$n" -DTHREADS="$threads" "${flags[@]}" mega_mt.cu -o "libmt${n}${suffix}.so" 2>&1 | tee "build-${variant}-m${n}.log"
done
