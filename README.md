# latticemk — phase 0 gate: lattice-coded weights vs INT4 in a batch-1 matvec

Project goal: a decode megakernel that streams lattice-compressed weights (simplex/A_n codes, LUT-decoded in
registers) and pins hot weights on-chip. This directory holds the gate experiment that decides whether the
compression lever is worth building: a fused batch-1 GEMV, lattice-coded vs INT4, on Qwen3-0.6B-Base.

Hardware (`hw/hw.json`, read from the device): **Quadro RTX 4000, Turing sm_75**, 36 SMs, 64 KB SMEM/SM, 4 MB L2,
416 GB/s spec, **406.7 GB/s measured read roofline** (copy 358–361 GB/s). Clocks cannot be locked without root.
Not Ada: Ada-MK targets Ada (100 KB SMEM) and MPK is evaluated on A100/H100/B200, so neither is a drop-in baseline
on this card. Phase reports against the agent build spec: `reports/wave-1.md` … `reports/wave-7.md`.

## Results

On the Quadro RTX 4000, the GPTQ INT4-g128 decode megakernel outpaced llama.cpp Q4_0 with flash attention at all three tested contexts. Qwen3-0.6B reached 578.6 / 416.6 / 221.6 tokens/s against 352.7 / 284.3 / 187.6 at contexts 128 / 2048 / 8192. Qwen3-1.7B reached 250.0 / 211.4 / 145.7 against 197.5 / 173.7 / 133.2. The paired gain shrinks from 1.64x to 1.09x as the model and context grow. GPTQ perplexity is better than Q4_0 in both models, but worse than Q4_K_M. No speed comparison to Q4_K_M is called quality-matched.

The abandoned ideas matter here. Lattice weight codes could not meet INT4 error at the hoped-for rate, the first persistent kernel lost to CUDA graphs, and compressed KV plus finer synchronization both slowed v2 down. The occupancy fix was the one that stuck. See [sourced tables](publish/results.md) and the [speed](publish/charts/speed_vs_llamacpp.png), [quality/speed](publish/charts/ppl_vs_speed.png), [experiment](publish/charts/waves.png), and [byte-model roofline](publish/charts/roofline.png) charts.

## Quickstart

Use a Turing sm_75 CUDA 13.3 host with Python, Rust, CMake and a CUDA compiler. Set `LLAMA_CPP_DIR` to a llama.cpp checkout at commit `6011c34ce6099646ccdf0d39a61c6e681477c178`, or leave it unset and let the script clone that commit into `~/llama.cpp`. The script builds its CUDA backend with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`, `-allow-unsupported-compiler`, and headers and libraries from the `nvidia-cublas==13.3.0.5` wheel because the local CUDA toolkit lacked cuBLAS. It downloads Qwen3-0.6B-Base, Qwen3-1.7B-Base and WikiText-2, then reuses cached GPTQ weights if available.

```sh
./reproduce.sh all
```

For shorter runs use `./reproduce.sh gate`, `validate`, `bench`, or `ppl`. Each GPU job uses `scripts/gpu.sh`, with `--timing` for benchmarks; the commands can take substantial time when calibration and model downloads are uncached. `all` also regenerates the charts; `.venv/bin/python publish/make_charts.py` regenerates charts alone from the recorded JSONL/JSON data.

## Layout

| path | role |
|---|---|
| `lmk/codebooks.py` | A_n truncated-ball codebooks (2^k lattice points nearest the origin), Lloyd/k-means Gaussian codebooks, exact CUDA nearest-codeword encoder |
| `lmk/quant.py` | group-128 quantizers: asymmetric INT RTN (clip search), VQ with per-group fp16 scale (scale search); block randomized Hadamard (RHT, 1024) |
| `lmk/gptq.py` | Hessian collection + block LDLQ (GPTQ generalised to d-dim codes: `W_R -= E_B U_BB^-1 U_BR`) |
| `lmk/pack.py` | packs codes into the kernel layouts; ctypes bridge to `libgemv.so` |
| `kernels/gemv.cu` | fp16 / INT4-g128 / VQ (A2, A4; k = 4..10) GEMV, optional fused FWHT prologue, CUDA-graph timer, read roofline |
| `kernels/vqenc.cu` | offline exact VQ encoder |
| `scripts/sweep_mse.py` | weight SQNR vs bits on 22 real matrices |
| `scripts/eval_ppl.py` | wikitext-2 test perplexity (146 x 2048 tokens), all linears incl. an untied lm_head fake-quantized |
| `scripts/bench_gemv.py` | per-token matvec sequence on real quantized weights: 28 x [qkv 4096x1024, o 1024x2048, gate_up 6144x1024, down 1024x3072] + lm_head 151936x1024 = 113 kernels in one CUDA graph; 7 interleaved rounds, medians |
| `scripts/check_kernels.py` | kernel correctness gate: every format x NC 1/2/3 x RHT vs torch |
| `scripts/rate_bound.py` | Shannon lower bound on bits/weight to match a target SQNR on the real weights |
| `hw/hwprobe.cu` | LM-00: device facts, copy/read roofline, launch and graph-node overhead -> `hw/hw.json` |

```sh
make                                   # nvcc 13.3; gcc 16 needs -allow-unsupported-compiler (set in Makefile)
.venv/bin/python scripts/check_kernels.py
.venv/bin/python scripts/sweep_mse.py
.venv/bin/python scripts/eval_ppl.py fp int4+rht+gptq A4-k9+rht+gptq
.venv/bin/python scripts/bench_gemv.py int4 A2-k4 A2-k5 A4-k9 A4-k10
```

Bits/weight include group metadata: INT4 = 4 + (fp16 scale + fp16 offset)/128 = 4.25; VQ `Ad-kK` = K/d + 16/128.

## Results (2026-09-29)

### Quality: wikitext-2 perplexity (fp32 baseline 12.66)

| config | bits/w | RTN | +RHT | +LDLQ | +RHT+LDLQ |
|---|---|---|---|---|---|
| int4 | 4.25 | 16.53 | — | 14.08 | **13.43** |
| int3 | 3.25 | 41.40 | — | 21.99 | **16.40** |
| A4-k11 | 2.875 | 279.8 | 102.6 | 29.67 | **17.23** |
| A4-k10 | 2.625 | 1458 | — | 35.06 | **19.97** |
| A2-k5 | 2.625 | — | — | 52.32 | **21.38** |
| A4-k9 | 2.375 | 3756 | 19251 | 59.57 | **25.41** |
| int2 | 2.25 | — | — | — | 138.7 |
| A2-k4 | 2.125 | — | — | 424.6 | — |
| lloyd4-k10 | 2.625 | 348.9 | 193.9 | — | — |

Weight SQNR (energy-weighted, 22 matrices, RTN): int4 20.3 dB, int3 14.5, A4-k11 13.9, A4-k10 12.6, A2-k5 12.1,
A4-k9 11.3, int2 8.5. At equal rate A4 beats scalar by ~2 dB; Lloyd tables beat A4 by ~0.5 dB (a LUT-decoded code
gains nothing from lattice structure); RHT adds ~0.1 dB SQNR but is decisive for perplexity once LDLQ is used.

### Speed: per-token matvec sequence (113 kernels, CUDA graph), median of 7 interleaved rounds

| config | bits/w | MB/tok | ms/tok | matvec tok/s | lm_head % roofline | small matrices % roofline |
|---|---|---|---|---|---|---|
| f16 | 16 | 1192.0 | 3.449 | 290 | 98.3 | 81.0 |
| int4 | 4.25 | 316.6 | 1.318 | 759 | 94.9 | 50.8 |
| A2-k4 | 2.125 | 158.3 | 1.145 | 874 | 66.0 | 28.3 |
| A2-k5 | 2.625 | 195.6 | 1.221 | 819 | 72.4 | 32.1 |
| A4-k9 | 2.375 | 176.9 | 1.777 | 563 | 38.3 | 22.1 |
| A4-k10 | 2.625 | 195.6 | 1.777 | 563 | 41.9 | 23.5 |
| int4 + fused RHT | 4.25 | 316.6 | 1.690 | 592 | 93.7 | 39.0 |
| A2-k5 + fused RHT | 2.625 | 195.6 | 1.686 | 593 | 68.2 | 23.4 |
| A4-k9 + fused RHT | 2.375 | 176.9 | 2.368 | 422 | 33.8 | 16.0 |
| A4-k10 + fused RHT | 2.625 | 195.6 | 2.386 | 419 | 36.9 | 17.4 |

"matvec tok/s" excludes attention, norms and sampling. The VQ kernels are issue-bound and track the SM boost clock,
which varies run to run and cannot be locked here (no power/thermal capping was recorded); a single-config
A2-k5+RHT run gave 1.517 ms vs 1.686 interleaved. INT4 is memory-bound and stable to ±1%. Fused RHT is recomputed by
every block; a megakernel would do it once per activation.

### Why the VQ kernels are slower than their byte count

INT4 reaches the roofline on lm_head, so the baseline is not a strawman. The VQ decode is the limiter:

- same instructions per weight as INT4 (~3.7–4.1 in the SASS loop), but 1.8–2x more weights per DRAM byte;
- each LUT read is a random SMEM access. A 16/32-entry half2 table (A2) puts one entry per bank and is
  conflict-free (66–72% of roofline). A4's 512–2048-entry uint2 tables conflict (~38–42%). Per-lane conflict-free
  replication of an A4-k9 table would need 64 KB, all of Turing's SMEM;
- ruled out: more rows in flight (RPW 2/4/8: no gain, RPW 8 spills), 4 independent HFMA2 chains (accuracy gain
  only), a 256-entry pair LUT for A2-k4 (half the LDS count, 1.7x slower from conflicts).

## Gate verdict

**Fails as posed on this card.** No lattice config beats INT4 on tok/s at INT4-level quality:

- Where it is faster (A2-k4 +15%, A2-k5 +8%, no RHT), perplexity is 30x and 3.7x worse than int4+LDLQ.
- The best lattice point (A4-k11 + RHT + LDLQ, 2.875 b, PPL 17.23) is still 28% worse than int4+RHT+LDLQ (13.43),
  and A4 kernels run at 0.74x INT4 speed.
- At about 2.4 b the best result is A4-k9 + RHT + LDLQ at PPL 25.41, 1.9x INT4. The "2.41 b ≈ INT4" result from
  KV vectors does not transfer to weights. A4 does dominate scalar at equal rate (int2+RHT+LDLQ: 138.7).

Levers that stay open, by evidence:

1. **Megakernel on INT4.** Small matrices stream at 50.8% of roofline vs 94.9% for lm_head. Bringing all
   316.6 MB/token to 95% is 0.82 ms, about 1220 matvec tok/s (+60%); this is a roofline calculation, not measured.
2. **INT3 + RHT + LDLQ** (3.25 b, PPL 16.40) streams 24% fewer bytes than INT4 with scalar, LUT-free decode.
   No kernel exists yet.
3. Lattice codes are worth revisiting only with LUT-free, algebraically decoded codes, where the decode cost
   would move from SMEM to ALU.
