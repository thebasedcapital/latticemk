"""LM-03b stage kill: persistent matvec-only loop (36x1024, <=64 regs) vs the
same 113 GEMVs as separate kernels (LM-03 wave-1 engine, mk_time_gemvs).

Interleaved rounds in one process; medians + SM clock. Kill if
mega2_matvec > (1/0.9) * separate_gemvs. Reports barrier-free and
barrier-per-GEMV variants.

usage: scripts/gpu.sh --timing .venv/bin/python bench/lm03b/stage_gemv.py
"""
import ctypes
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402

lm03._lib.mk_time_gemvs.restype = ctypes.c_float

RUNS = 15

def sm_clock():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"]
    ).decode().strip().split()[0])

def main():
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()

    eng1 = lm03.Engine(128, packed, emb, norms, rope)
    eng2 = lm03b.Engine2(128, packed, emb, norms, rope)

    lib1, lib2 = lm03._lib, lm03b._lib2
    # warm all paths
    lib1.mk_time_gemvs(2)
    lib2.mk2_time_matvec(2, 0)
    lib2.mk2_time_matvec(2, 1)
    torch.cuda.synchronize()

    sep, nobar, bar = [], [], []
    clk = []
    for _ in range(RUNS):
        sep.append(lib1.mk_time_gemvs(4))
        nobar.append(lib2.mk2_time_matvec(4, 0))
        bar.append(lib2.mk2_time_matvec(4, 1))
        clk.append(sm_clock())
    med = statistics.median
    print(f"SM clock med {med(clk)} MHz")
    print(f"separate 113 GEMVs : {med(sep):.4f} ms  "
          f"(p10 {statistics.quantiles(sep, n=10)[0]:.4f} "
          f"p90 {statistics.quantiles(sep, n=10)[8]:.4f})")
    print(f"mega2 nobar        : {med(nobar):.4f} ms "
          f"(p10 {statistics.quantiles(nobar, n=10)[0]:.4f} "
          f"p90 {statistics.quantiles(nobar, n=10)[8]:.4f})")
    print(f"mega2 bar          : {med(bar):.4f} ms "
          f"(p10 {statistics.quantiles(bar, n=10)[0]:.4f} "
          f"p90 {statistics.quantiles(bar, n=10)[8]:.4f})")
    for name, v in (("nobar", nobar), ("bar", bar)):
        r = med(sep) / med(v)
        print(f"ratio sep/{name} = {r:.3f}  ({'PASS' if r >= 0.9 else 'KILL'}"
              f" vs >=0.9)")
    extra(eng2)

def extra(eng2):
    lib2 = lm03b._lib2
    lib2.mk2_time_barriers.argtypes = [ctypes.c_int, ctypes.c_int]
    lib2.mk2_time_barriers.restype = ctypes.c_float
    t = lib2.mk2_time_barriers(143, 50)
    print(f"143 empty barriers: {t:.4f} ms -> {t*1000/143:.2f} us/barrier")

if __name__ == "__main__":
    main()
