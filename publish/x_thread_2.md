# Draft X thread 2 (follow-up)

Draft only, do not post. Lowercase account voice. Chart markers are placement notes. Lengths: `python3 publish/check_thread.py publish/x_thread_2.md`.
Sources per post: compaction `reports/wave-8/LM-24.md` + `bench/lm24/`; speculative `reports/wave-9/LM-25.md` + `spec/analysis.json`; multi-token `reports/wave-8/LM-23.md`; gate `reports/wave-6/LM-16.md`, `reports/wave-7/LM-17.md`; tensor cores `reports/wave-8/LM-22.md`.

## 1

follow-up to the 2018 quadro megakernel thread. two new things that work on the same old card:

an 8k-token agent session decoding at 2k speed: 216 -> 406 tok/s

lossless speculative decoding on code and rag: 1.41x and 1.44x over my own kernel

[chart: publish/charts/compaction.png]

## 2

the compaction trick is from kv-streams (arxiv 2609.35750). drop whole old turns straight out of the live kv cache, keep the surviving keys exactly as rotated, keep the rope position counting.

no re-prefill. each event costs 1.3-2.6 ms.

## 3

in a 16k-token session with 8 compaction events, total time went 79.3 s -> 44.8 s, compaction included.

with no compaction the kernel is bitwise identical to the old one at 99% of its speed. which turns to drop is the agent's call, not the kernel's.

## 4

speculative decoding needed a kernel that checks several tokens per weight read. my first try cost 2.6x for 4 tokens. the fix wasn't the matmul: over half the extra time was attention and small per-token loops. after a 3-agent swarm and a merge, 1.89x.

[chart: publish/charts/multitoken.png]

## 5

drafts come from the cpu, no draft model: find the longest match of the current suffix in prompt + output, propose what followed. the gpu verifies 1-5 tokens per pass. output is token-for-token identical to plain greedy decoding on all 80 test prompts.

## 6

results vs the 1-token kernel, wall time incl. drafting and rollback:

code edits 1.41x [1.25, 1.56]
rag answers 1.44x [1.32, 1.56]
summarization 0.92x, so it's off

it pays when the output copies the input. summaries don't.

[chart: publish/charts/speculative.png]

## 7

tensor cores didn't help. on turing, unpacking int4 into mma fragments costs ~275 instructions per 16 mma ops. it won ~0.09 ms at best and broke bitwise equality with 1-token decoding, so i left them out.

## 8

the most useful finding was about my own correctness checker. i injected 332 single bugs into the kernels. it caught 72%. dropped syncs and early flag releases mostly passed: the race never fired in a normal run.

timing jitter and longer contexts took it to 89%.

## 9

the 41 left, like halving one layer's attention scale, hide inside the engine's own fp16 noise at the output. a per-layer check catches 34.

only a bitwise comparison against the last accepted kernel catches all 41.

## 10

all of it is in the repo with the dead ends, raw jsonl and a one-command reproduce script. built by a swarm of agents with exact verifiers as the gate.

github.com/thebasedcapital/latticemk
