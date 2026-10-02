"""Convert the Qwen3-1.7B GPTQ packed weights to dense fake-quant F16 GGUF."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench/lm11"))
sys.path.insert(0, str(HERE))
from ppl_llama_protocol import deq_int4
import scale

CONVERT = str(Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))
              / "convert_hf_to_gguf.py")
PY = str(ROOT / ".venv" / "bin" / "python")


def main():
    packed = torch.load(sys.argv[1], map_location="cpu")
    out_gguf = sys.argv[2]
    deq = {n: deq_int4(v["codes"], v["meta"]).half()
           for n, v in packed.items()}
    del packed

    model = AutoModelForCausalLM.from_pretrained(scale.SNAPSHOT,
                                               dtype=torch.float16)
    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear)}
    for i in range(scale.N_LAYERS):
        p = f"model.layers.{i}."
        w = deq[f"L{i}.qkv"]
        lin[p + "self_attn.q_proj"].weight.data = w[:2048]
        lin[p + "self_attn.k_proj"].weight.data = w[2048:3072]
        lin[p + "self_attn.v_proj"].weight.data = w[3072:]
        lin[p + "self_attn.o_proj"].weight.data = deq[f"L{i}.o"]
        g = deq[f"L{i}.gu"]
        lin[p + "mlp.gate_proj"].weight.data = g[:scale.INTER]
        lin[p + "mlp.up_proj"].weight.data = g[scale.INTER:]
        lin[p + "mlp.down_proj"].weight.data = deq[f"L{i}.down"]
    model.lm_head.weight = torch.nn.Parameter(deq["lm_head"])
    model.config.tie_word_embeddings = False

    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        (Path(td) / "tokenizer_config.json")
        # copy tokenizer files
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "config.json", "generation_config.json"):
            src = scale.SNAPSHOT / f
            if src.exists() and not (Path(td) / f).exists():
                (Path(td) / f).write_bytes(src.read_bytes())
        subprocess.run([PY, CONVERT, td, "--outfile", out_gguf,
                        "--outtype", "f16"], check=True)
    print("wrote", out_gguf)


if __name__ == "__main__":
    main()
