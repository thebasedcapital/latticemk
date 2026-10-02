# Draft X thread (v2)

Draft only, do not post. Chart markers are placement notes, not post text. Character counts are checked by
`publish/check_thread.py`. Every number is mapped to its source in `publish/claims_check.md` (section "Thread v2").

## 1

I got 1.64x over llama.cpp on a 2018 Quadro RTX 4000. Same card, 4-bit weights, lower perplexity. Qwen3-0.6B, batch-1 decode, 579 vs 353 tok/s.

The trick wasn't a new quantizer. It was one CUDA kernel that stays resident for the whole token.

[chart: publish/charts/speed_vs_llamacpp.png]

## 2

Each decode step is one launch. 36 blocks of 1024 threads stay on the GPU and hand work to each other through flags in global memory. Weights are INT4 with GPTQ, the KV cache is fp16. A small Rust checker tests every schedule for races and deadlocks before it ships.

## 3

The win shrinks as the work gets heavier. On 0.6B it's 1.64x at 128 tokens of context, 1.47x at 2k, 1.18x at 8k. On Qwen3-1.7B it's 1.27x, 1.22x, 1.09x.

Small models burn time on kernel launches. Bigger ones mostly stream weights, and nobody skips that.

[chart: publish/charts/roofline.png]

## 4

Quality is measured with llama.cpp's own perplexity tool, WikiText-2 at 2048 context. My INT4 weights score 12.44 vs 12.96 for Q4_0 on 0.6B, and 9.24 vs 9.71 on 1.7B.

Q4_K_M beats my perplexity on both models, so I don't claim that comparison.

## 5

This started as a lattice quantizer project. I wanted simplex-lattice weight codes near 2.4 bits.

Then I measured the weights. After per-group scaling they're Gaussian to within 0.0005 bits of entropy. Shannon says matching INT4's error needs at least 3.49 bits. Dead idea.

[chart: publish/charts/waves.png]

## 6

My first megakernel was slower than plain CUDA graphs, 0.96x.

A persistent kernel gets the registers of its heaviest task. Mine got 214, which left 8 warps per SM to hide memory latency. Rewriting it to fit 64 registers at 1024 threads per block turned it into a 1.42x win.

## 7

Compressing the KV cache didn't pay either. At 8k context my 6.25-bit cache read 573 MB less per token and ran at 0.72x. That kernel also failed my correctness gate, so treat the speed as rough.

llama.cpp's q8_0 cache is slower than f16 on this card too.

## 8

Swapping grid-wide barriers for fine-grained flags lost too, 0.93x.

The first version also had a race. My schedule checker passed it, because the schedule was right and the kernel didn't implement it. Runs that disagreed with each other caught it.

## 9

Porting to 1.7B broke the math. Accumulating dot products in fp16 worked on 0.6B. On 1.7B it drifted to a 24.6 logit error and then produced NaNs.

fp32 accumulation fixed it, with 0.05 max error against an fp32 reference, still 64 registers and no spills.

## 10

I also hit two bugs in ExLlamaV2 0.3.2 with Qwen3. It runs the per-head q/k norm as LayerNorm instead of RMSNorm, and its SDPA prefill path drops the causal mask. First-window perplexity was 59.5 vs 9.64 in HF.

Config overrides fix both. Fixed EXL2 still ran 3-6x slower here.

[chart: publish/charts/ppl_vs_speed.png]

## 11

How I kept this honest. Clocks can't be locked on this card, so every ratio comes from alternating both engines in one process. Correctness is checked token by token against an fp32 reference, and logits must repeat bit for bit across runs.

## 12

Everything reruns with ./reproduce.sh all, about 12 minutes with models cached. Kernels, the schedule checker, raw JSONL and a longer write-up with the dead ends are in the repo.

[link: repo URL]
