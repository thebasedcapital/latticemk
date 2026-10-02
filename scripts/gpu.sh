#!/bin/bash
# Serialize GPU use across agents: one 8 GB card, ~2 GB already held by the mpvpaper wallpaper.
#   scripts/gpu.sh CMD...           run CMD holding the GPU lock (any CUDA work: tests, PPL, encoding)
#   scripts/gpu.sh --timing CMD...  same, and pause the wallpaper video for the duration (benchmarks)
set -euo pipefail
LOCK=/tmp/latticemk-gpu.lock
SOCK=/run/user/$(id -u)/mpvpaper.sock
timing=0
if [[ "${1:-}" == "--timing" ]]; then timing=1; shift; fi
exec 9>"$LOCK"
flock 9
if (( timing )) && [[ -S "$SOCK" ]]; then
  echo '{"command":["set_property","pause",true]}' | socat - "$SOCK" >/dev/null || true
  trap 'echo "{\"command\":[\"set_property\",\"pause\",false]}" | socat - "'"$SOCK"'" >/dev/null || true' EXIT
fi
"$@"
