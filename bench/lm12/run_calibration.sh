#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for lo in 0 4 8 12 16 20 24; do
  if [[ ! -f "bench/lm12/hessians.$lo-$((lo + 4)).pt" ]]; then
    for begin in 0 32 64 96; do
      [[ -f "bench/lm12/hessians.$lo-$((lo + 4)).$begin-$((begin + 32)).pt" ]] ||
        timeout 300 scripts/gpu.sh .venv/bin/python bench/lm12/calibrate.py \
          "$lo" "$((lo + 4))" "$begin" "$((begin + 32))"
    done
    .venv/bin/python bench/lm12/calibrate.py merge "$lo" "$((lo + 4))"
  fi
done
for layer in $(seq 0 27); do
  [[ -f "bench/lm12/packed.$layer.pt" ]] ||
    timeout 300 scripts/gpu.sh .venv/bin/python bench/lm12/gptq_pack.py "$layer"
done
[[ -f bench/lm12/packed.head.pt ]] ||
  timeout 300 scripts/gpu.sh .venv/bin/python bench/lm12/gptq_pack.py head
.venv/bin/python bench/lm12/gptq_pack.py merge
