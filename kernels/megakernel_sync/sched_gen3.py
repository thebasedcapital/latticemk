"""LM-10 single-step Schedule IR v1, matching mega3.cu's phase/CTA table.

Each qkv producer publishes a per-CTA flag for the attention consumers that
need its q/k/v rows. The remaining vector dependencies use a full phase flag.
IR v1 covers one step; cross-step tok ordering uses done[0] after argc on GPU.
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

    def phase(self, name, bodies, partial_qkv=False):
        prev = self.flags[-1] if self.flags else None
        flag = f"{name}_done"
        self.flags.append(flag)
        if name.endswith(".qkv"):
            self.flags.extend(f"{name}.c{c}_done" for c in range(NCTA))
        for c in range(NCTA):
            b = bodies[c]
            waits = ([{"flag": prev, "value": NCTA}] if prev else [])
            if partial_qkv:
                waits = ([{"flag": f"{name.replace('.attn', '.qkv')}.c{p}_done",
                           "value": 1} for p in qkv_producers(c)]
                         if c < NATTN else [])
            sets = [{"flag": flag, "add": 1}]
            if name.endswith(".qkv"):
                sets.append({"flag": f"{name}.c{c}_done", "add": 1})
            self.tasks.append({
                "id": f"{name}.c{c}", "block": c,
                "order": len(self.tasks) // NCTA,
                "reads": b.get("reads", []), "writes": b.get("writes", []),
                "waits": waits, "sets": sets,
            })


def row_writes(buf, c, n_out, bpp):
    r0, r1 = c * n_out // NCTA, (c + 1) * n_out // NCTA
    return [(buf, r0 * bpp, r1 * bpp)]


def qkv_producers(c):
    """Producer CTA row ranges, identical to mega3.cu::qkv_producers."""
    h, kvh = c >> 1, c >> 2
    rows = [h * 128, QROWS + kvh * 128, QROWS + KVROWS + kvh * 128]
    producers = set()
    for lo in rows:
        producers.update(range(((lo + 1) * NCTA - 1) // QKV,
                               ((lo + 128) * NCTA - 1) // QKV + 1))
    return sorted(producers)


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
        g.phase(f"L{l}.attn", attn, partial_qkv=True)
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

    # ---- lm_head: argmax is folded into the owning CTA's GEMV writeback ----
    g.phase("lm", [{"reads": [("dout", 0, HID * 2),
                              ("norms", 112 * HID * 2, 113 * HID * 2)],
                    "writes": row_writes("logits", c, VOCAB, 4) +
                              [("argp", c * 8, c * 8 + 8)]}
                   for c in range(NCTA)])
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
            "note": "one decode step at pos=ctx; qkv producer flags gate each "
                    "attention head, all other edges use full phase counters. "
                    "The next step waits on CTA 0's argc done flag before tok.",
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
