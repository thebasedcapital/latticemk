#!/usr/bin/env bash
# Usage: ./reproduce.sh gate|validate|bench|ppl|gate-tiers|multitoken|compaction|spec|schedules|all
# GPU commands run singly through scripts/gpu.sh; timings additionally pause mpvpaper.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
export CUDA_HOME="${CUDA_HOME:-$HOME/.local/cuda-13.3}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
LLAMA_COMMIT=6011c34ce6099646ccdf0d39a61c6e681477c178
stage=${1:-all}
case "$stage" in gate|validate|bench|ppl|gate-tiers|multitoken|compaction|spec|schedules|all) ;; *) echo 'Usage: ./reproduce.sh gate|validate|bench|ppl|gate-tiers|multitoken|compaction|spec|schedules|all' >&2; exit 2;; esac

# Cached artifacts are reused. Every missing artifact is built with the original driver.
setup() {
  if [[ ! -x "$PY" ]]; then python3 -m venv .venv; fi
  if ! "$PY" -m pip --version >/dev/null 2>&1; then "$PY" -m ensurepip --upgrade; fi
  if ! "$PY" -c 'import torch, transformers, safetensors, pyarrow, huggingface_hub, gguf' >/dev/null 2>&1; then
    "$PY" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install 'transformers==5.17.0' 'safetensors==0.8.0' \
      'pyarrow==25.0.1' 'huggingface_hub==1.33.0' 'gguf==0.19.0'
  fi
  if ! "$PY" -c 'import matplotlib' >/dev/null 2>&1; then
    "$PY" -m pip install 'matplotlib==3.11.2'
  fi
  make
  cargo build --release --manifest-path validator/Cargo.toml
  if [[ ! -f kernels/megakernel_v2/libmega2.so ]]; then
    "$CUDA_HOME/bin/nvcc" -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler \
      -L"$CUDA_HOME/lib" -Xptxas=-v -Xcompiler -fPIC -shared -cudart static \
      -o kernels/megakernel_v2/libmega2.so kernels/megakernel_v2/mega2.cu
  fi
  bash kernels/megakernel_scale/build.sh
  "$PY" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"); snapshot_download("Qwen/Qwen3-1.7B-Base", revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"); snapshot_download("Salesforce/wikitext", revision="b08601e04326c79dfdd32d625aee71d232d685c3", repo_type="dataset", allow_patterns="wikitext-2-raw-v1/*")'
  if [[ ! -f baselines/wikitext-2-test.txt ]]; then
    "$PY" -c 'from pathlib import Path; import pyarrow.parquet as pq; p=next((Path.home()/".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots").iterdir())/"wikitext-2-raw-v1/test-00000-of-00001.parquet"; Path("baselines/wikitext-2-test.txt").write_text("\n\n".join(pq.read_table(p)["text"].to_pylist()))'
  fi
}

llama_build() {
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
    git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_COMMIT"
  fi
  local actual
  actual=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
  if [[ "$actual" != "$LLAMA_COMMIT" ]]; then
    echo "llama.cpp must be at $LLAMA_COMMIT (found $actual at $LLAMA_CPP_DIR). Check out that commit, then rerun." >&2
    exit 1
  fi
  local cublas="$ROOT/baselines/deps/cublas13/nvidia/cu13"
  if [[ ! -f "$cublas/lib/libcublas.so.13" ]]; then
    mkdir -p baselines/deps/cublas13
    "$PY" -m pip download --no-deps 'nvidia-cublas==13.3.0.5' -d baselines/deps
    "$PY" -c 'from pathlib import Path; import zipfile; wheel=next(Path("baselines/deps").glob("nvidia_cublas-13.3.0.5*.whl")); zipfile.ZipFile(wheel).extractall("baselines/deps/cublas13")'
    [[ -e "$cublas/lib/libcublas.so" ]] || ln -s libcublas.so.13 "$cublas/lib/libcublas.so"
    [[ -e "$cublas/lib/libcublasLt.so" ]] || ln -s libcublasLt.so.13 "$cublas/lib/libcublasLt.so"
  fi
  export LD_LIBRARY_PATH="$cublas/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  if [[ ! -x "$LLAMA_CPP_DIR/build-cuda/bin/llama-bench" || ! -x "$LLAMA_CPP_DIR/build-cuda/bin/llama-perplexity" || ! -x "$LLAMA_CPP_DIR/build-cuda/bin/llama-quantize" ]]; then
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build-cuda" \
      -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
      '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDA_HOME;$cublas" \
      -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -I$cublas/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$CUDA_HOME/lib -L$cublas/lib -Wl,-rpath,$CUDA_HOME/lib -Wl,-rpath,$cublas/lib -Wl,-rpath-link,$CUDA_HOME/lib -Wl,-rpath-link,$cublas/lib" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L$CUDA_HOME/lib -L$cublas/lib -Wl,-rpath,$CUDA_HOME/lib -Wl,-rpath,$cublas/lib -Wl,-rpath-link,$CUDA_HOME/lib -Wl,-rpath-link,$cublas/lib"
    cmake --build "$LLAMA_CPP_DIR/build-cuda" -j 4 \
      --target llama-bench llama-perplexity llama-quantize
  fi
}

weights() {
  if [[ ! -f bench/lm03b/weights_int4.pt ]]; then
    scripts/gpu.sh "$PY" -c 'import sys;sys.path.insert(0,"bench/lm03");import lm03;lm03.pack_weights(want_deq=False)'
    cp bench/lm03/weights_int4.pt bench/lm03b/weights_int4.pt
  fi
  if [[ ! -f bench/lm11/weights_int4_gptq.pt || ! -f bench/lm11/weights_int4_gptq_deq.pt ]]; then
    for half in 0 1; do
      [[ -f "bench/lm11/hessians.$half.pt" ]] || scripts/gpu.sh "$PY" bench/lm11/gptq_pack.py "hess$half"
    done
    for part in A B; do
      [[ -f "bench/lm11/weights_gptq.part$part.pt" ]] || scripts/gpu.sh "$PY" bench/lm11/gptq_pack.py "quant$part"
    done
    "$PY" bench/lm11/gptq_pack.py merge
  fi
  if [[ ! -f bench/lm12/weights_int4_gptq.pt || ! -f bench/lm12/weights_int4_gptq_deq.pt ]]; then
    bash bench/lm12/run_calibration.sh
  fi
}

validate() {
  for schedule in kernels/*/schedules/*.json bench/lm23/schedules/*.json; do
    [[ -f "$schedule" && "$(basename "$schedule")" != MANIFEST.json ]] || continue
    validator/target/release/schedcheck "$schedule"
  done
}

gate() {
  weights
  scripts/gpu.sh "$PY" bench/lm11/check_gate_v21.py
  scripts/gpu.sh "$PY" bench/lm12/check_gate_v21.py
}

prepare_gguf() {
  llama_build
  if [[ ! -f baselines/qwen3-0.6b-f16.gguf ]]; then
    "$PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
      "$($PY -c 'from lmk.model import SNAPSHOT;print(SNAPSHOT)')" \
      --outfile baselines/qwen3-0.6b-f16.gguf --outtype f16
  fi
  if [[ ! -f baselines/qwen3-0.6b-q4_0.gguf ]]; then
    scripts/gpu.sh "$LLAMA_CPP_DIR/build-cuda/bin/llama-quantize" \
      baselines/qwen3-0.6b-f16.gguf baselines/qwen3-0.6b-q4_0.gguf Q4_0
  fi
  if [[ ! -f baselines/qwen3-0.6b-q4_k_m.gguf ]]; then
    scripts/gpu.sh "$LLAMA_CPP_DIR/build-cuda/bin/llama-quantize" \
      baselines/qwen3-0.6b-f16.gguf baselines/qwen3-0.6b-q4_k_m.gguf Q4_K_M
  fi
  if [[ ! -f baselines/qwen3-0.6b-int4gptq-fake.gguf ]]; then
    "$PY" bench/lm11/weights_to_gguf.py bench/lm11/weights_int4_gptq.pt \
      baselines/qwen3-0.6b-int4gptq-fake.gguf
  fi
  for kind in F16 Q4_0 Q4_K_M gptq-fake; do
    [[ -f "bench/lm12/qwen3-1.7b-$kind.gguf" ]] || { bash bench/lm12/prepare_gguf.sh; break; }
  done
}

bench() {
  weights
  prepare_gguf
  [[ $("$PY" -c 'import json; print(json.load(open("bench/lm11/gate_v21.json"))["pass"] and json.load(open("bench/lm12/gate_v21.json"))["pass"])' 2>/dev/null) == True ]] || gate
  for ctx in 128 2048 8192; do
    scripts/gpu.sh --timing "$PY" bench/lm11/interleave.py "$ctx" \
      --weights bench/lm11/weights_int4_gptq.pt --llama baselines/qwen3-0.6b-q4_0.gguf \
      --rounds 3 --v2-reps 9 --llama-n 32 --llama-r 10
    scripts/gpu.sh --timing "$PY" bench/lm12/interleave.py "$ctx" \
      --llama bench/lm12/qwen3-1.7b-Q4_0.gguf
  done
}

ppl() {
  weights
  prepare_gguf
  bash bench/lm12/run_ppl.sh
  # The 0.6B GGUF quality files are evaluated using the same second-half protocol.
  for kind in q4_0 q4_k_m int4gptq-fake; do
    local log="bench/lm11/ppl.$kind.log"
    scripts/gpu.sh "$LLAMA_CPP_DIR/build-cuda/bin/llama-perplexity" \
      -m "baselines/qwen3-0.6b-$kind.gguf" -f baselines/wikitext-2-test.txt \
      -c 2048 -ngl 99 -b 2048 -ub 512 --no-warmup -lv 3 -fa on -ctk f16 -ctv f16 \
      >"$log" 2>&1
    "$PY" -c 'import json,re,sys;from pathlib import Path;name,filename=sys.argv[1:];text=Path(filename).read_text();ppl=re.search(r"Final estimate: PPL = ([0-9.]+)",text);progress=re.findall(r"\[(\d+)\]([0-9.]+)",text);assert ppl or (progress and int(progress[-1][0])==146), "PPL did not complete all windows";value=float(ppl.group(1) if ppl else progress[-1][1]);dest=Path("bench/lm11/ppl.json");payload=json.loads(dest.read_text()) if dest.exists() else {"protocol":"llama-perplexity -c 2048 -ngl 99 -b 2048 -ub 512 --no-warmup -fa on; 146 WikiText-2 test windows, second half scored","ppl":{}};payload["ppl"][{"int4gptq-fake":"int4gptq-fake/f16"}.get(name,name+"/f16")]=value;dest.write_text(json.dumps(payload,indent=2)+"\n");print(name,value)' "$kind" "$log"
  done
}

# Package drivers retain their original output conventions. Archive just this
# run's outputs, then restore the published evidence even when a command fails.
capture_results() {
  "$PY" scripts/reproduce_capture.py "$@"
}

mt_build() {
  bash kernels/megakernel_mt3/build.sh selected
  for m in 1 2 3 4 5; do
    [[ -f "kernels/megakernel_mt2/libmt$m.so" ]] || bash kernels/megakernel_mt2/build.sh selected "$m"
  done
}

references() {
  [[ -f mutation/reference-0.6B.pt ]] || scripts/gpu.sh timeout 285 "$PY" mutation/gate.py prepare --model 0.6B
  for prompt in 0 1 2; do
    [[ -f "bench/lm14/reference-$prompt.pt" ]] || scripts/gpu.sh timeout 285 "$PY" bench/lm14/reference.py "$prompt"
  done
}

gate_tiers() {
  weights
  mt_build
  references
  # Tier 3 is cumulative: both invocations execute and report tiers 1, 2 and 3.
  scripts/gpu.sh timeout 285 "$PY" gate/run.py --engine v2 --tier 3 \
    --output publish/reproduction/gate-tiers.jsonl
  scripts/gpu.sh timeout 285 "$PY" gate/run.py --engine mt --m 4 --mode causal --tier 3 \
    --lib kernels/megakernel_mt3/libmt4.so --output publish/reproduction/gate-tiers.jsonl
}

multitoken() {
  weights
  mt_build
  references
  capture_results bench/lm23/gate.json -- \
    scripts/gpu.sh timeout 285 "$PY" bench/lm23/check_gate.py
  capture_results bench/lm23/causal-128.json bench/lm23/results.jsonl -- \
    scripts/gpu.sh --timing timeout 285 "$PY" bench/lm23/bench.py --ctx 128 --mode causal --samples 25
}

compaction() {
  weights
  bash kernels/megakernel_kvc/build.sh
  [[ -f bench/lm24/reference.pt ]] || "$PY" bench/lm24/reference.py
  capture_results bench/lm24/correctness.json -- \
    scripts/gpu.sh timeout 285 "$PY" bench/lm24/check_correctness.py
  capture_results bench/lm24/timing.json -- \
    scripts/gpu.sh --timing timeout 285 "$PY" bench/lm24/bench.py --runs 25
  capture_results bench/lm24/session.json -- \
    scripts/gpu.sh --timing timeout 285 "$PY" bench/lm24/session.py
}

spec() {
  weights
  mt_build
  # Fixed subset: code-00, rag-00, summarization-00, chat-00, full lengths,
  # three rotating/reversing repeats. Never append this subset to the full corpus.
  for category in code rag summarization chat; do
    scripts/gpu.sh --timing timeout 285 "$PY" spec/run.py --category "$category" \
      --start 0 --count 1 --repeats 3 --output publish/reproduction/spec.jsonl
  done
  "$PY" spec/analyze.py
}

schedules() {
  capture_results bench/lm20/schedules.json -- "$PY" kernels/megakernel_attn/sched_mt.py
  capture_results bench/lm23/schedules.json -- "$PY" kernels/megakernel_mt3/sched_mt.py
  "$PY" kernels/megakernel_kvc/sched_gen.py
  "$PY" scripts/schedule_manifest.py --check kernels/megakernel_attn/schedules \
    bench/lm23/schedules kernels/megakernel_kvc/schedules
}

record_stage() (
  local name=$1 begin=$SECONDS
  shift
  trap 'status=$?; elapsed=$((SECONDS-begin)); printf "{\"stage\":\"%s\",\"wall_seconds\":%s,\"exit_code\":%s,\"recorded_at\":\"%s\"}\n" "$name" "$elapsed" "$status" "$(date -u +%FT%TZ)" >> publish/reproduction_times.jsonl; echo "reproduce.sh $name wall time: $elapsed s (exit $status)"' EXIT
  "$@"
)

start=$SECONDS
mkdir -p publish/reproduction
setup
case "$stage" in
  validate) record_stage validate validate ;;
  gate) record_stage gate gate ;;
  bench) record_stage bench bench ;;
  ppl) record_stage ppl ppl ;;
  gate-tiers) record_stage gate-tiers gate_tiers ;;
  multitoken) record_stage multitoken multitoken ;;
  compaction) record_stage compaction compaction ;;
  spec) record_stage spec spec ;;
  schedules) record_stage schedules schedules ;;
  all)
    record_stage gate gate
    record_stage schedules schedules
    record_stage validate validate
    record_stage gate-tiers gate_tiers
    record_stage multitoken multitoken
    record_stage compaction compaction
    record_stage spec spec
    record_stage bench bench
    record_stage ppl ppl
    record_stage charts "$PY" publish/make_charts.py
    ;;
esac
elapsed=$((SECONDS-start))
if [[ "$stage" == all ]]; then
  printf '{"stage":"all","wall_seconds":%s}\n' "$elapsed" >> publish/reproduction_times.jsonl
fi
echo "reproduce.sh $stage total wall time: $elapsed s"
