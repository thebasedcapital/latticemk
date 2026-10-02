#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
SP="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
CB="$ROOT/baselines/deps/cublas13/nvidia/cu13"
export CUDA_HOME="$HOME/.local/cuda-13.3"
export NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
export CPATH="$SP/cusparse/include:$CB/include:$SP/cusolver/include:$SP/cufft/include:$SP/curand/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CB/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CB/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=4
exec "$ROOT/.venv/bin/python" "$ROOT/bench/lm11/exllama_try.py" "$@"
