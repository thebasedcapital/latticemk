"""Pack quantized weights into the kernel layouts of kernels/gemv.cu and drive it via ctypes."""

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from lmk.quant import Quantized

FMT = {"f16": 0, "int4": 1, "v4e0": 2, "v4e1": 3, "v4e2": 4, "v2e0": 5, "v2e1": 6}


class Desc(ctypes.Structure):
    _fields_ = [("fmt", ctypes.c_int), ("out", ctypes.c_int), ("in_", ctypes.c_int), ("rht", ctypes.c_int),
                ("codes", ctypes.c_void_p), ("hi", ctypes.c_void_p), ("meta", ctypes.c_void_p),
                ("lut", ctypes.c_void_p), ("x", ctypes.c_void_p), ("y", ctypes.c_void_p), ("signs", ctypes.c_void_p)]


_lib = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "build" / "libgemv.so"))
_lib.gemv_run.argtypes = [ctypes.POINTER(Desc), ctypes.c_int]
_lib.gemv_time.argtypes = [ctypes.POINTER(Desc), ctypes.c_int, ctypes.c_int]
_lib.gemv_time.restype = ctypes.c_float
_lib.read_gbps.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.read_gbps.restype = ctypes.c_float


@dataclass
class Packed:
    fmt: str
    out: int
    n_in: int
    codes: torch.Tensor  # streamed from DRAM every token, like hi and meta
    hi: torch.Tensor | None = None
    meta: torch.Tensor | None = None
    lut: torch.Tensor | None = None  # SMEM-resident, not streamed
    signs: torch.Tensor | None = None

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.codes, self.hi, self.meta) if t is not None)

    def desc(self, x: torch.Tensor, y: torch.Tensor) -> Desc:
        ptr = lambda t: None if t is None else t.data_ptr()
        return Desc(FMT[self.fmt], self.out, self.n_in, int(self.signs is not None), ptr(self.codes), ptr(self.hi),
                    ptr(self.meta), ptr(self.lut), x.data_ptr(), y.data_ptr(), ptr(self.signs))


def _u32(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t >= 1 << 31, t - (1 << 32), t).to(torch.int32)


def _by_rows(fn, codes: torch.Tensor, rows: int = 16384):
    """Apply fn to row chunks and concatenate each returned plane (int64 temporaries of lm_head exceed 1 GB)."""
    parts = [fn(codes[i : i + rows]) for i in range(0, len(codes), rows)]
    return tuple(None if p[0] is None else torch.cat(p).contiguous() for p in zip(*parts))


def pack_f16(w: torch.Tensor) -> Packed:
    return Packed("f16", *w.shape, w.half().contiguous())


def pack_int4(q: Quantized, signs=None) -> Packed:
    out, n_in = q.codes.shape
    sh = torch.arange(4, device=q.codes.device) * 4

    def words(codes):
        c = codes.to(torch.int64).reshape(len(codes), n_in // 8, 4, 2)
        return (_u32((c[..., 0] << sh).sum(-1) | (c[..., 1] << (sh + 16)).sum(-1)),)  # pair j -> bits 4j, 16+4j

    (word,) = _by_rows(words, q.codes)
    meta = torch.stack([q.scale, q.zero], -1).contiguous()
    return Packed("int4", out, n_in, word, meta=meta, signs=signs)


def pack_vq(q: Quantized, cb: torch.Tensor, signs=None) -> Packed:
    d = cb.shape[1]
    k = int(math.log2(cb.shape[0]))
    base = 8 if d == 4 else 4
    e = k - base
    out = q.codes.shape[0]
    n_in = q.codes.shape[1] * d
    per = 32 // d
    t = torch.arange(per, device=q.codes.device)

    def planes(codes):
        idx = codes.to(torch.int64)
        ic = idx.reshape(len(idx), n_in // 32, per)
        if d == 4:
            lo = (idx & 0xFF).to(torch.uint8)
        else:
            nib = (ic & 0xF) << (4 * (t % 8))
            lo = _u32(torch.stack([nib[..., :8].sum(-1), nib[..., 8:].sum(-1)], -1))
        if not e:
            return lo, None
        hb = (((ic >> base) & ((1 << e) - 1)) << (e * t)).sum(-1)
        if e * per == 8:
            return lo, hb.to(torch.uint8)
        return lo, torch.where(hb >= 1 << 15, hb - (1 << 16), hb).to(torch.int16)

    lo, hi = _by_rows(planes, q.codes)
    fmt = f"v{d}e{e}"
    assert fmt in FMT, fmt
    return Packed(fmt, out, n_in, lo, hi, q.scale.contiguous(), cb.half().contiguous(), signs)


def run(descs: list[Desc]) -> None:
    arr = (Desc * len(descs))(*descs)
    rc = _lib.gemv_run(arr, len(descs))
    if rc:
        raise RuntimeError(f"gemv_run failed: {rc}")


def time_ms(descs: list[Desc], iters: int = 50) -> float:
    arr = (Desc * len(descs))(*descs)
    ms = _lib.gemv_time(arr, len(descs), iters)
    if ms < 0:
        raise RuntimeError(f"gemv_time failed: {ms}")
    return ms


def read_gbps(nbytes: int = 1 << 28, iters: int = 20) -> float:
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    return _lib.read_gbps(buf.data_ptr(), nbytes, iters)
