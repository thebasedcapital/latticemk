"""Schedule generator for the LM-03b megakernel (v2) — single source of truth
for the phase layout that mega2_kernel hardcodes AND the emitted IR.

Phase schedule per decode step (36 CTAs, one task per CTA per phase; the
kernel barriers after every phase):
    per layer l in 0..27:
        5*l+0  qkv   xloc=emb[tok]|+=dout, rmsnorm ln1(l) -> sx; W_qkv -> qkv
        5*l+1  attn  CTAs 0..31: head=c>>1, half=c&1 position slice ->
                     part[c]; CTAs c%4==0 append kv head c/4 at pos
        5*l+2  o     combine part -> sx; W_o -> oo
        5*l+3  gu    xloc += oo, rmsnorm ln2(l) -> sx; W_gu -> gu
        5*l+4  down  silu(gu)*up -> sx; W_down -> dout
    140  lm     xloc += dout, rmsnorm final -> sx; W_lm -> logits (f32)
    141  argp   argmax slice c -> argp[c]
    142  argc   CTA 0: merge argp -> tok, tok_hist[pos]

Cross-step dependency (not modeled by IR v1, same convention as LM-03): the
first qkv phase of step s reads tok written by step s-1's argc phase; the
143rd barrier per step enforces it on-device.

usage: sched_gen2.py <ctx> <path.json>
"""

import json
import sys
from pathlib import Path

HID = 1024
NLAY = 28
NQH, NKVH, HDIM = 16, 8, 128
QROWS = NQH * HDIM            # 2048
KVROWS = NKVH * HDIM          # 1024
QKV = 4096
INTER = 3072
GU = 6144
VOCAB = 151936
MAXPOS = 8704
KVB = KVROWS * 2              # bytes per position per layer per cache
NCTA = 36
NATTN = 32
NARGP = 36

# buffer ids (IR v1: named buffers with byte sizes)
BUF_BYTES = {
    "emb": VOCAB * HID * 2,
    "norms": 113 * HID * 2,
    "rope": MAXPOS * 256 * 4,
    "qkv": QKV * 2,
    "oo": HID * 2,
    "gu": GU * 2,
    "dout": HID * 2,
    "kcache": NLAY * MAXPOS * KVB,
    "vcache": NLAY * MAXPOS * KVB,
    "part": NATTN * 130 * 4,
    "logits": VOCAB * 4,
    "argp": NARGP * 8,
    "tok": 4,
}
BUFS = list(BUF_BYTES)


class Gen:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tasks = []
        self.flags = []

    def phase(self, name, bodies):
        """bodies: list of per-CTA dicts (reads/writes). Appends one task per
        CTA waiting on the previous phase flag, all setting the new flag."""
        prev = self.flags[-1] if self.flags else None
        flag = f"{name}_done"
        self.flags.append(flag)
        for c in range(NCTA):
            b = bodies[c]
            self.tasks.append({
                "id": f"{name}.c{c}", "block": c,
                "order": len(self.flags) - 1,
                "reads": b.get("reads", []),
                "writes": b.get("writes", []),
                "waits": [{"flag": prev, "value": NCTA}] if prev else [],
                "sets": [{"flag": flag, "add": 1}],
            })


def row_writes(buf, c, n_out, bpp):
    chunk = (n_out + NCTA - 1) // NCTA
    r0, r1 = c * chunk, min(n_out, (c + 1) * chunk)
    return [(buf, r0 * bpp, r1 * bpp)]


def build(ctx):
    g = Gen(ctx)
    pos = ctx - 1
    mid = (pos + 2) // 2  # (npos+1)>>1 with npos=pos+1

    for l in range(NLAY):
        # ---- qkv ----
        qkvb = []
        for c in range(NCTA):
            reads = [("norms", l * HID * 2, (l + 1) * HID * 2)]
            if l == 0:
                reads += [("tok", 0, 4), ("emb", 0, VOCAB * HID * 2)]
            else:
                reads += [("dout", 0, HID * 2)]
            qkvb.append({"reads": reads, "writes": row_writes("qkv", c, QKV, 2)})
        g.phase(f"L{l}.qkv", qkvb)

        # ---- attn ----
        attn = []
        for c in range(NCTA):
            if c >= NATTN:
                attn.append({})
                continue
            h, half, kvh = c >> 1, c & 1, (c >> 1) >> 1
            a0, a1 = (mid, pos + 1) if half else (0, mid)
            # positions read from the caches: [a0, min(a1, pos))
            hi = min(a1, pos)
            kb = l * MAXPOS * KVB
            reads = [("qkv", h * 256, h * 256 + 256),          # q row
                     ("rope", pos * 1024, pos * 1024 + 1024)]
            writes = [("part", c * 520, c * 520 + 520)]
            cov = pos >= mid  # which half owns the pos row
            if half == cov:
                reads += [("qkv", (QROWS + kvh * 128) * 2,
                           (QROWS + kvh * 128) * 2 + 256),
                          ("qkv", (QROWS + KVROWS + kvh * 128) * 2,
                           (QROWS + KVROWS + kvh * 128) * 2 + 256)]
            if c % 4 == 0:    # appender: kv head c/4 row at pos
                kvh_a = c // 4
                reads += [("qkv", (QROWS + kvh_a * 128) * 2,
                           (QROWS + kvh_a * 128) * 2 + 256),
                          ("qkv", (QROWS + KVROWS + kvh_a * 128) * 2,
                           (QROWS + KVROWS + kvh_a * 128) * 2 + 256)]
                writes += [("kcache", kb + pos * KVB + kvh_a * 256,
                            kb + pos * KVB + kvh_a * 256 + 256),
                           ("vcache", kb + pos * KVB + kvh_a * 256,
                            kb + pos * KVB + kvh_a * 256 + 256)]
            if a0 < hi:
                # rows a0..hi-1, head slice kvh*256..+256 within each 2 KB row
                reads += [("kcache", kb + a0 * KVB + kvh * 256,
                           kb + (hi - 1) * KVB + kvh * 256 + 256),
                          ("vcache", kb + a0 * KVB + kvh * 256,
                           kb + (hi - 1) * KVB + kvh * 256 + 256)]
            attn.append({"reads": reads, "writes": writes})
        g.phase(f"L{l}.attn", attn)

        # ---- o ----
        g.phase(f"L{l}.o", [{"reads": [("part", 0, NATTN * 520)],
                             "writes": row_writes("oo", c, HID, 2)}
                            for c in range(NCTA)])
        # ---- gu ----
        g.phase(f"L{l}.gu", [{"reads": [("oo", 0, HID * 2),
                                       ("norms", (NLAY + l) * HID * 2,
                                        (NLAY + l + 1) * HID * 2)],
                              "writes": row_writes("gu", c, GU, 2)}
                             for c in range(NCTA)])
        # ---- down ----
        g.phase(f"L{l}.down", [{"reads": [("gu", 0, GU * 2)],
                                "writes": row_writes("dout", c, HID, 2)}
                               for c in range(NCTA)])

    # ---- lm_head ----
    g.phase("lm", [{"reads": [("dout", 0, HID * 2),
                              ("norms", 112 * HID * 2, 113 * HID * 2)],
                    "writes": row_writes("logits", c, VOCAB, 4)}
                   for c in range(NCTA)])
    # ---- argp ----
    aper = (VOCAB + NARGP - 1) // NARGP
    g.phase("argp", [{"reads": [("logits", c * aper * 4,
                                min(VOCAB, (c + 1) * aper) * 4)],
                      "writes": [("argp", c * 8, c * 8 + 8)]}
                     for c in range(NCTA)])
    # ---- argc ----
    g.phase("argc", [{"reads": [("argp", 0, NARGP * 8)],
                      "writes": [("tok", 0, 4)]} if c == 0 else {}
                     for c in range(NCTA)])
    return g


def ir_json(g):
    bufs = [{"id": b, "bytes": BUF_BYTES[b]} for b in BUFS]
    flags = [{"id": f} for f in g.flags]
    tasks = []
    for t in g.tasks:
        tasks.append({
            "id": t["id"], "block": t["block"], "order": t["order"],
            "reads": [{"buffer": b, "begin": x, "end": y}
                      for b, x, y in t["reads"]],
            "writes": [{"buffer": b, "begin": x, "end": y}
                       for b, x, y in t["writes"]],
            "waits": t["waits"], "sets": t["sets"]})
    return {"version": 1, "model": "qwen3-0.6b", "blocks": NCTA,
            "flags_per_step": len(g.flags),
            "note": "one decode step at pos=ctx-1; phases are grid barriers "
                    "(each CTA has one task per phase, sets phase_done +=1, "
                    "waits on prev phase flag=36). Step s+1 qkv reads tok "
                    "written by step s argc (cross-step, outside IR v1).",
            "buffers": bufs, "flags": flags, "tasks": tasks}


def emit(ctx, path):
    g = build(ctx)
    d = ir_json(g)
    Path(path).write_text(json.dumps(d, indent=1))
    return g, d


if __name__ == "__main__":
    ctx = int(sys.argv[1])
    path = sys.argv[2]
    g, d = emit(ctx, path)
    print(f"ctx {ctx}: {len(g.tasks)} tasks, {len(g.flags)} flags -> {path}")
