"""Same-process alternating ExLlamaV2 EXL2 and megakernel-v2 GPTQ, batch-1 decode.

Run: scripts/gpu.sh --timing bash bench/lm13/run.sh bench/lm13/paired.py 128
ExLlama CUDA events bracket one forward including embedding, attention, head and logits;
GPU-idle gaps between host-issued kernels are included, but sampling and CPU logits
copy are not. Both engines reuse the same token and restore cache depth before each
sample, measuring one step at fixed context; ExLlama's internal graphs remain enabled.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "bench/lm03"), str(ROOT / "bench/lm03b")]
import lm03  # noqa: E402
import lm03b  # noqa: E402

ROOF = 406.7
HERE = Path(__file__).resolve().parent


def clock():
    s = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"], text=True)
    return int(s.split()[0])


def row(label, ctx, ms, weight_bytes, kv_bytes, ppl, extra):
    xs = sorted(ms)
    med = statistics.median(xs)
    return dict(commit="nogit", work_package="LM-13", model="qwen3-0.6b", kernel=label,
                context=ctx, batch=1, tokens_per_s=1000/med,
                gbps=(weight_bytes+kv_bytes)/med*1e-6,
                pct_roofline=(weight_bytes+kv_bytes)/med*1e-6/ROOF*100,
                sm_clock_mhz=int(statistics.median(extra["round_clocks_mhz"])),
                driver="610.57.04", runs=len(ms),
                median_ms=med, p10_ms=xs[int(.1*len(xs))], p90_ms=xs[int(.9*len(xs))],
                weight_bytes=weight_bytes, kv_bytes=kv_bytes, ppl=ppl,
                ppl_protocol="wikitext-2 llama-perplexity -c 2048 half-window", **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int, choices=(128, 2048, 8192))
    ap.add_argument("--exl2", default=str(ROOT / "baselines/exl2/qwen3-0.6b-4.125bpw-head4-causal"))
    ap.add_argument("--ppl", type=float, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    cfg = ExLlamaV2Config()
    cfg.model_dir = args.exl2
    cfg.prepare()
    cfg.arch.lm.headnorm = "rmsnorm"
    cfg.no_sdpa = True  # ExLlamaV2 0.3.2 SDPA prefill omits the causal mask
    cfg.max_seq_len = args.ctx + 128
    cfg.max_input_len = 256
    cfg.no_flash_attn = True  # FA-2 requires sm_80; ExLlama's own graph path stays on
    exl = ExLlamaV2(cfg)
    exl.load()
    cache = ExLlamaV2Cache(exl, max_seq_len=cfg.max_seq_len, lazy=False)
    token = torch.tensor([[9707]], dtype=torch.long, device="cpu")
    for _ in range(0, args.ctx, 256):
        exl.forward(torch.full((1, min(256, args.ctx-cache.current_seq_len)), 9707, dtype=torch.long),
                    cache=cache, preprocess_only=True)
    assert cache.current_seq_len == args.ctx
    # Decode q_len=1 can safely use the faster SDPA fallback: all keys
    # precede this token. Prefill must retain the explicit causal mask.
    cfg.no_sdpa = False

    packed = {k: {f: t.cuda() for f, t in v.items()}
              for k, v in torch.load(ROOT / "bench/lm11/weights_int4_gptq.pt", map_location="cpu").items()}
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    eng = lm03b.Engine2(args.ctx, packed, emb, norms, rope)
    eng.bufs["kc"].zero_()
    eng.bufs["vc"].zero_()
    eng.set_tok(9707)
    out = lm03._f32([0.0])
    step = lm03b._lib2.mk2_time_mega
    step(8, 3, args.ctx, out)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(4):
        cache.current_seq_len = args.ctx
        exl.forward(token, cache=cache, last_id_only=True)
    torch.cuda.synchronize()

    v2_ms, ex_ms, clocks = [], [], []
    for r in range(args.rounds):
        # Reverse ordering each round so boost-clock drift cannot always
        # favor the same engine.
        for which in (("v2", "exl2") if r % 2 == 0 else ("exl2", "v2")):
            for _ in range(args.reps):
                if which == "v2":
                    step(8, 1, args.ctx, out)
                    v2_ms.append(float(out[0]))
                else:
                    cache.current_seq_len = args.ctx
                    start.record()
                    logits = exl.forward(token, cache=cache, last_id_only=True)
                    end.record()
                    end.synchronize()
                    assert logits.shape[-1] == cfg.vocab_size
                    ex_ms.append(start.elapsed_time(end))
        clocks.append(clock())
        print(f"round {r}: v2 {statistics.median(v2_ms[-args.reps:]):.4f} ms, "
              f"EXL2 {statistics.median(ex_ms[-args.reps:]):.4f} ms, SM {clocks[-1]} MHz", flush=True)
    wb_v2 = sum(t.numel() * t.element_size() for t in eng.codes + eng.metas)
    kv = 2 * lm03.N_LAYERS * args.ctx * lm03.KVROWS * 2
    # Embedding table lives on CPU in ExLlama and contributes only one
    # selected row per step, not a full GPU DRAM table scan.
    files = sorted(Path(args.exl2).glob("*.safetensors"))
    wb_ex = sum(nbytes for file in files for name, nbytes in _tensors(file)
                if name != "model.embed_tokens.weight")
    wb_ex += cfg.hidden_size * 2
    quant_config = json.loads((Path(args.exl2) / "config.json").read_text())["quantization_config"]
    head_bits = quant_config["head_bits"]
    target_bpw = quant_config["bits"]
    rows = [row("megakernel-v2-gptq", args.ctx, v2_ms, wb_v2, kv, 12.4435,
                {"steps_per_launch": 8, "timing": "CUDA internal 8-step timer / step",
                 "round_clocks_mhz": clocks}),
            row(f"exllamav2-exl2-{target_bpw:g}bpw-head{head_bits}-sdpa-decode", args.ctx, ex_ms, wb_ex, kv, args.ppl,
                {"timing": "CUDA event brackets one forward; includes host dispatch gaps, excludes sampling",
                 "quant_model_dir": args.exl2, "exl2_graphs": not cfg.no_graphs,
                 "exl2_prefill_no_sdpa": True, "exl2_decode_no_sdpa": cfg.no_sdpa,
                 "exl2_headnorm": cfg.arch.lm.headnorm,
                 "round_clocks_mhz": clocks})]
    with (HERE / "results.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['kernel']}: {r['tokens_per_s']:.1f} tok/s, median {r['median_ms']:.4f} ms", flush=True)


def _tensors(path):
    from math import prod
    from safetensors import safe_open
    dtype_size = {"F16": 2, "BF16": 2, "I16": 2, "U16": 2,
                  "F32": 4, "I32": 4, "U32": 4,
                  "I8": 1, "U8": 1, "I64": 8, "U64": 8, "F64": 8}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_slice(name)
            yield name, prod(tensor.get_shape()) * dtype_size[tensor.get_dtype()]


if __name__ == "__main__":
    main()
