"""LM-11 HF transformers baseline worker: fp16 + StaticCache, batch-1 decode.

Persistent subprocess: reads JSON commands on stdin, writes JSON on stdout.
  {"cmd":"init","ctx":8192,"compile":false}   load model, allocate cache
  {"cmd":"fill"}                            push ctx tokens through the cache
  {"cmd":"decode","n":32}                   -> {"ms":[per-step ms, ...]}
  {"cmd":"quit"}

Greedy argmax feedback each step (same semantics as the v2 megakernel and
llama-bench's tg test).
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lmk.model import SNAPSHOT  # noqa: E402

FILL_CHUNK = 512


class HFEngine:
    def __init__(self, ctx, do_compile):
        from transformers.cache_utils import StaticCache
        self.ctx = ctx
        self.model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT, dtype=torch.float16, attn_implementation="sdpa")
        self.model.cuda().eval()
        self.cache = StaticCache(config=self.model.config,
                                 max_cache_len=ctx + 2048, device="cuda",
                                 dtype=torch.float16)
        self.pos = 0
        self.do_compile = do_compile

        def step(tok, cp):
            out = self.model(input_ids=tok, past_key_values=self.cache,
                             cache_position=cp, use_cache=True)
            return out.logits[0, -1].argmax()

        self.step = torch.compile(step, mode="reduce-overhead",
                                  fullgraph=True) if do_compile else step

    @torch.no_grad()
    def fill(self):
        """Push ctx copies of token 9707 through the cache in chunks."""
        while self.pos < self.ctx:
            n = min(FILL_CHUNK, self.ctx - self.pos)
            tok = torch.full((1, n), 9707, dtype=torch.long, device="cuda")
            cp = torch.arange(self.pos, self.pos + n, device="cuda")
            self.model(input_ids=tok, past_key_values=self.cache,
                       cache_position=cp, use_cache=True)
            self.pos += n
        torch.cuda.synchronize()

    @torch.no_grad()
    def decode(self, n):
        cur = torch.tensor([[9707]], device="cuda")
        # warmup / compile
        for _ in range(4):
            cp = torch.tensor([self.pos], device="cuda")
            nxt = self.step(cur, cp)
            self.pos += 1
            cur = nxt.reshape(1, 1).clone()
        ms = []
        for _ in range(n):
            cp = torch.tensor([self.pos], device="cuda")
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            nxt = self.step(cur, cp)
            cur = nxt.reshape(1, 1).clone()
            e1.record()
            torch.cuda.synchronize()
            ms.append(e0.elapsed_time(e1))
            self.pos += 1
        return ms


def main():
    eng = None
    out = sys.stdout
    for line in sys.stdin:
        c = json.loads(line)
        if c["cmd"] == "quit":
            break
        elif c["cmd"] == "init":
            eng = HFEngine(c["ctx"], c.get("compile", False))
            r = {"ok": True}
        elif c["cmd"] == "fill":
            eng.fill()
            r = {"ok": True}
        elif c["cmd"] == "decode":
            r = {"ms": eng.decode(c["n"])}
        else:
            r = {"error": c["cmd"]}
        out.write(json.dumps(r) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
