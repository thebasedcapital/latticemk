"""Schedule generator for the LM-03 megakernel — single source of truth.

One decode step of Qwen3-0.6B-Base, batch 1, INT4-g128 weights. Emits:
  * the packed binary blob uploaded to the kernel (mk_load_schedule), and
  * Schedule IR v1 JSON (shared contract, checked by LM-05).

Per decode step the kernel executes `flags_per_step` flag slots; a multi-step
launch offsets every flag index by step * flags_per_step. The emitted JSON
describes ONE step (the step at pos = ctx-1, the maximal-attention step).
`_prev_argdone` in the binary marks waits on the previous step's argc_done;
it is omitted from the IR JSON (cross-step ordering is outside Schedule IR v1:
"reset between steps is outside the IR").

KV cache layout: layer L's slab starts at L * MAXPOS * KVB
(KVB = 8 kv heads * 128 dims * 2 B = 2048 B per position).
"""

import json
import struct
from pathlib import Path

# ---- model constants -------------------------------------------------------
HID = 1024
NLAY = 28
NQH, NKVH, HDIM = 16, 8, 128
QROWS = NQH * HDIM            # 2048
KVROWS = NKVH * HDIM          # 1024
QKV = QROWS + 2 * KVROWS      # 4096
INTER = 3072
GU = 2 * INTER                # 6144
VOCAB = 151936
KVB = KVROWS * 2              # bytes per position per layer in one cache
MAXPOS = 8704                 # slab stride (ctx 8192 + decode headroom)
NSLICE = 72                   # attention slices per layer (clamped by ctx)
NARGP = 72                    # argmax partial slices
NB = 36                       # persistent CTAs (1/SM: spill-free 255-reg budget)

# ---- op codes (mirrored in mega.cu) ----------------------------------------
OP_TICK, OP_EMBED, OP_NORM, OP_ANORM, OP_QKRA, OP_ATTN, OP_ATTNC = range(7)
OP_GEMV, OP_SILU, OP_ARGP, OP_ARGC = 7, 8, 9, 10

# weight indices into the device weight table
def W_QKV(l):  return l * 4 + 0
def W_O(l):    return l * 4 + 1
def W_GU(l):   return l * 4 + 2
def W_DOWN(l): return l * 4 + 3
W_LM = NLAY * 4                                # 112

# norm-weight selectors (rows of the norms table, HID halfs per row)
def NW_LN1(l): return l
def NW_LN2(l): return NLAY + l
def NW_QN(l):  return 2 * NLAY + l      # per-layer q_norm (first 128 elems)
def NW_KN(l):  return 3 * NLAY + l      # per-layer k_norm
NW_FINAL = 4 * NLAY                      # 112
NNORM = 4 * NLAY + 1                     # 113
# buffer ids
(B_X, B_XN, B_QKV, B_ATT, B_O, B_GU, B_D, B_KC, B_VC, B_PART, B_LOG, B_ARGP,
 B_TOK, B_POS) = range(14)
BUF_NAME = {B_X: "x", B_XN: "xn", B_QKV: "qkv", B_ATT: "attn", B_O: "oout",
            B_GU: "gu", B_D: "dout", B_KC: "kcache", B_VC: "vcache",
            B_PART: "partials", B_LOG: "logits", B_ARGP: "argp", B_TOK: "tok",
            B_POS: "pos"}
BUF_BYTES = {B_X: HID * 2, B_XN: HID * 2, B_QKV: QKV * 2, B_ATT: QROWS * 2,
             B_O: HID * 2, B_GU: GU * 2, B_D: HID * 2,
             B_KC: NLAY * MAXPOS * KVB, B_VC: NLAY * MAXPOS * KVB,
             B_PART: NSLICE * NQH * (HDIM + 2) * 4, B_LOG: VOCAB * 4,
             B_ARGP: NARGP * 8, B_TOK: 4, B_POS: 4}

# cross-step pseudo flag: wait on the previous step's argc_done (binary only)
PREV_ARGDONE = -1


class Gen:
    """Accumulates tasks; flag ids are per-step (0..flags_per_step-1)."""

    def __init__(self, ctx, nblocks=NB, nslice=None):
        self.ctx, self.nb = ctx, nblocks
        self.nslice = nslice or min(NSLICE, max(8, ctx // 4))
        self.flags = []          # flag names in id order
        self.fid = {}
        self.tasks = []
        self.order = [0] * nblocks

    def flag(self, name):
        if name not in self.fid:
            self.fid[name] = len(self.flags)
            self.flags.append(name)
        return self.fid[name]

    def task(self, name, op, warg=0, a0=0, a1=0, s0=0, block=None,
             reads=(), writes=(), waits=(), sets=()):
        b = len(self.tasks) % self.nb if block is None else block
        t = dict(id=name, op=op, warg=warg, a0=a0, a1=a1, s0=s0, block=b,
                 order=self.order[b], reads=list(reads), writes=list(writes),
                 waits=list(waits), sets=list(sets))
        self.order[b] += 1
        self.tasks.append(t)
        return t

    def kv_slab(self, buf, layer):
        base = layer * MAXPOS * KVB
        return (buf, base, base + self.ctx * KVB)

    def gemv(self, name, widx, n_out, xbuf, xbytes, ybuf, ybpp, wait):
        """out-rows split into contiguous per-block chunks (all 72 blocks)."""
        n = min(self.nb, n_out)
        chunk = (n_out + n - 1) // n
        done = self.flag(name + "_done")
        cnt = 0
        for i in range(n):
            rb, re = i * chunk, min((i + 1) * chunk, n_out)
            if rb >= re:
                break
            self.task(f"{name}.b{i}", OP_GEMV, warg=widx, a0=rb, a1=re,
                      block=i,
                      reads=[(xbuf, 0, xbytes)],
                      writes=[(ybuf, rb * ybpp, re * ybpp)],
                      waits=[wait], sets=[(done, 1)])
            cnt += 1
        return (done, cnt)


def build(ctx, nblocks=NB, nslice=None):
    """-> Gen with tasks annotated for the IR; flags in id order."""
    g = Gen(ctx, nblocks, nslice)
    ns = g.nslice
    pos_done = g.flag("pos_done")
    embed_done = g.flag("embed_done")

    g.task("tick", OP_TICK, block=0, writes=[(B_POS, 0, 4)],
           waits=[(PREV_ARGDONE, 1)], sets=[(pos_done, 1)])
    g.task("embed", OP_EMBED, block=1 % nblocks,
           reads=[(B_TOK, 0, 4)], writes=[(B_X, 0, HID * 2)],
           waits=[(PREV_ARGDONE, 1)], sets=[(embed_done, 1)])

    for l in range(NLAY):
        p = f"L{l}"
        xn_f = g.flag(f"{p}.xn_done")
        if l == 0:
            g.task(f"{p}.rms", OP_NORM, warg=NW_LN1(0), block=2 % nblocks,
                   reads=[(B_X, 0, HID * 2)], writes=[(B_XN, 0, HID * 2)],
                   waits=[(embed_done, 1)], sets=[(xn_f, 1)])

        qkv_done = g.gemv(f"{p}.qkv", W_QKV(l), QKV, B_XN, HID * 2, B_QKV, 2,
                          (xn_f, 1))

        qkra_f = g.flag(f"{p}.qkra_done")
        kcb = l * MAXPOS * KVB
        # pos == ctx-1 for the emitted step: append writes one 2048 B slot.
        slot = (ctx - 1) * KVB
        g.task(f"{p}.qkra", OP_QKRA, warg=l, block=(3 + l) % nblocks,
               reads=[(B_QKV, 0, QKV * 2), (B_POS, 0, 4)],
               writes=[(B_QKV, 0, (QROWS + KVROWS) * 2),
                       (B_KC, kcb + slot, kcb + slot + KVB),
                       (B_VC, kcb + slot, kcb + slot + KVB)],
               waits=[(pos_done, 1), qkv_done], sets=[(qkra_f, 1)])

        # attention slices over positions [0, ctx); runtime clamp is pos+1
        attn_f = g.flag(f"{p}.attn_done")
        per = (ctx + ns - 1) // ns
        cnt_attn = 0
        for s in range(ns):
            b0, b1 = s * per, min((s + 1) * per, ctx)
            if b0 >= ctx:
                break
            pr = (B_PART, s * NQH * (HDIM + 2) * 4,
                  (s + 1) * NQH * (HDIM + 2) * 4)
            g.task(f"{p}.attn.s{s}", OP_ATTN, warg=l, a0=b0, a1=b1, s0=s,
                   block=s % nblocks,
                   reads=[(B_QKV, 0, QROWS * 2), (B_POS, 0, 4),
                          (B_KC, kcb + b0 * KVB, kcb + b1 * KVB),
                          (B_VC, kcb + b0 * KVB, kcb + b1 * KVB)],
                   writes=[pr], waits=[(qkra_f, 1)], sets=[(attn_f, 1)])
            cnt_attn += 1

        attnc_f = g.flag(f"{p}.attnc_done")
        g.task(f"{p}.attnc", OP_ATTNC, warg=l, s0=cnt_attn,
               block=(4 + l) % nblocks,
               reads=[(B_PART, 0, cnt_attn * NQH * (HDIM + 2) * 4)],
               writes=[(B_ATT, 0, QROWS * 2)],
               waits=[(attn_f, cnt_attn)], sets=[(attnc_f, 1)])

        o_done = g.gemv(f"{p}.o", W_O(l), HID, B_ATT, QROWS * 2, B_O, 2,
                        (attnc_f, 1))

        an1_f = g.flag(f"{p}.an1_done")
        g.task(f"{p}.addnorm1", OP_ANORM, warg=NW_LN2(l), s0=0,
               block=(5 + l) % nblocks,
               reads=[(B_X, 0, HID * 2), (B_O, 0, HID * 2)],
               writes=[(B_X, 0, HID * 2), (B_XN, 0, HID * 2)],
               waits=[o_done], sets=[(an1_f, 1)])

        gu_done = g.gemv(f"{p}.gu", W_GU(l), GU, B_XN, HID * 2, B_GU, 2,
                         (an1_f, 1))

        silu_f = g.flag(f"{p}.silu_done")
        nsilu = 8
        sper = INTER // nsilu
        for s in range(nsilu):
            b0, b1 = s * sper, (s + 1) * sper
            g.task(f"{p}.silu.s{s}", OP_SILU, a0=b0, a1=b1,
                   block=(6 + s) % nblocks,
                   reads=[(B_GU, b0 * 2, b1 * 2),
                          (B_GU, (INTER + b0) * 2, (INTER + b1) * 2)],
                   writes=[(B_GU, b0 * 2, b1 * 2)],
                   waits=[gu_done], sets=[(silu_f, 1)])

        d_done = g.gemv(f"{p}.down", W_DOWN(l), HID, B_GU, INTER * 2, B_D, 2,
                        (silu_f, nsilu))

        # residual + next norm: layer l+1's ln1, or the final norm
        xn_next = g.flag(f"L{l + 1}.xn_done" if l + 1 < NLAY else "xn_done")
        g.task(f"{p}.addnorm2", OP_ANORM,
               warg=NW_LN1(l + 1) if l + 1 < NLAY else NW_FINAL, s0=1,
               block=(7 + l) % nblocks,
               reads=[(B_X, 0, HID * 2), (B_D, 0, HID * 2)],
               writes=[(B_X, 0, HID * 2), (B_XN, 0, HID * 2)],
               waits=[d_done], sets=[(xn_next, 1)])

    lm_done = g.gemv("lm_head", W_LM, VOCAB, B_XN, HID * 2, B_LOG, 4,
                     (g.fid["xn_done"], 1))

    arg_f = g.flag("argp_done")
    aper = (VOCAB + NARGP - 1) // NARGP
    for s in range(NARGP):
        b0, b1 = s * aper, min((s + 1) * aper, VOCAB)
        g.task(f"argp.s{s}", OP_ARGP, a0=b0, a1=b1, s0=s, block=s % nblocks,
               reads=[(B_LOG, b0 * 4, b1 * 4)],
               writes=[(B_ARGP, s * 8, s * 8 + 8)],
               waits=[lm_done], sets=[(arg_f, 1)])
    argc_f = g.flag("argc_done")
    g.task("argc", OP_ARGC, s0=NARGP, block=0,
           reads=[(B_ARGP, 0, NARGP * 8)], writes=[(B_TOK, 0, 4)],
           waits=[(arg_f, NARGP)], sets=[(argc_f, 1)])

    g.flags_per_step = len(g.flags)
    return g


def ir_json(g):
    """Schedule IR v1 (frozen contract). One step at pos = ctx-1."""
    bufs = [{"id": BUF_NAME[i], "bytes": BUF_BYTES[i]} for i in range(14)]
    flags = [{"id": n} for n in g.flags]
    tasks = []
    for t in g.tasks:
        waits = [{"flag": g.flags[f], "value": v} for f, v in t["waits"]
                 if f >= 0]
        sets = [{"flag": g.flags[f], "add": a} for f, a in t["sets"]]
        tasks.append({"id": t["id"], "block": t["block"], "order": t["order"],
                      "reads": [{"buffer": BUF_NAME[b], "begin": x, "end": y}
                                for b, x, y in t["reads"]],
                      "writes": [{"buffer": BUF_NAME[b], "begin": x, "end": y}
                                 for b, x, y in t["writes"]],
                      "waits": waits, "sets": sets})
    return {"version": 1, "model": "qwen3-0.6b", "blocks": g.nb,
            "flags_per_step": g.flags_per_step,
            "note": "one decode step at pos=ctx-1; tick/embed also wait on the "
                    "previous step's argc_done (cross-step, outside IR v1)",
            "buffers": bufs, "flags": flags, "tasks": tasks}


# ---- binary blob -------------------------------------------------------------
# header (i32): magic, nt, nb, fper, f_argc, reserved
# then per block b: bcnt[b] (nb ints) and boff[b] (nb ints, int offsets into
# the record stream from its start); then records:
#   op,warg,s0,a0,a1,nwait,nset,order, nwait*(flag,val), nset*(flag,add)
# flag == -1 waits on (step-1)*fper + f_argc.
def blob(g):
    per_block = [[] for _ in range(g.nb)]
    for t in g.tasks:
        per_block[t["block"]].append(t)
    for b in per_block:
        b.sort(key=lambda t: t["order"])
    counts, offsets, recs = [], [], []
    off = 0
    for b in per_block:
        counts.append(len(b))
        offsets.append(off)
        for t in b:
            ws, ss = t["waits"], t["sets"]
            parts = [struct.pack("<8i", t["op"], t["warg"], t["s0"],
                                 t["a0"], t["a1"], len(ws), len(ss),
                                 t["order"])]
            parts += [struct.pack("<2i", f, v) for f, v in ws]
            parts += [struct.pack("<2i", f, a) for f, a in ss]
            recs.append(b"".join(parts))
            off += len(recs[-1]) // 4
    hdr = struct.pack("<6i", 0x4C4D3033, len(g.tasks), g.nb, g.flags_per_step,
                      g.fid["argc_done"], 0)
    return (hdr + struct.pack(f"<{g.nb}i", *counts)
            + struct.pack(f"<{g.nb}i", *offsets) + b"".join(recs))


def emit(ctx, path, nblocks=NB, nslice=None):
    g = build(ctx, nblocks, nslice)
    d = ir_json(g)
    Path(path).write_text(json.dumps(d, indent=1))
    return g, d


if __name__ == "__main__":
    import sys
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else NB
    out = Path(__file__).resolve().parent / "schedules"
    out.mkdir(exist_ok=True)
    for ctx in (128, 2048, 8192):
        g, d = emit(ctx, out / f"mk_ctx{ctx}{'_nb%d' % nb if nb != NB else ''}.json",
                    nblocks=nb)
        print(f"ctx {ctx}: {len(g.tasks)} tasks, {g.flags_per_step} flags, "
              f"{len(d['buffers'])} buffers, nslice={g.nslice}")
