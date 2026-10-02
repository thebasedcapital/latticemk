"""Vector codebooks decoded by table lookup.

Every codebook is a float32 array [2**k, dim]; row i is the decoded value of index i.
The GPU kernel only ever sees this table, so any codebook with the same shape is
drop-in: A_n (simplex lattice) truncations and Lloyd-optimised tables compete on MSE alone.
"""

import ctypes
import itertools
import math
from pathlib import Path

import numpy as np
import torch


def hyperplane_basis(n: int) -> np.ndarray:
    """Orthonormal [n, n+1] basis of {x in R^{n+1} : sum(x) = 0} (Helmert rows)."""
    b = np.zeros((n, n + 1))
    for i in range(1, n + 1):
        b[i - 1, :i] = 1.0
        b[i - 1, i] = -float(i)
        b[i - 1] /= math.sqrt(i * (i + 1))
    return b


def an_codebook(n: int, k: int) -> np.ndarray:
    """The 2**k points of A_n closest to the origin, in n-dim coordinates, unit-RMS scaled.

    A_n = {x in Z^{n+1} : sum x = 0}; its Delaunay cells are regular simplices.
    Ties on the cutting shell are broken lexicographically (deterministic, slightly asymmetric).
    """
    size = 1 << k
    # Ball radius needed: vol(A_n cell) = sqrt(n+1); ball of 2**k cells, plus margin.
    vol_ball_unit = math.pi ** (n / 2) / math.gamma(n / 2 + 1)
    r = (size * math.sqrt(n + 1) / vol_ball_unit) ** (1 / n) * 1.3 + 1
    m = int(math.ceil(r))
    rng = range(-m, m + 1)
    pts = np.array([p for p in itertools.product(rng, repeat=n) if abs(sum(p)) <= m], dtype=np.int64)
    pts = np.concatenate([pts, -pts.sum(1, keepdims=True)], axis=1)  # last coord closes sum=0
    norm2 = (pts * pts).sum(1)
    pts, norm2 = pts[norm2 <= r * r], norm2[norm2 <= r * r]
    order = np.lexsort(tuple(pts[:, ::-1].T) + (norm2,))
    chosen = pts[order[:size]]
    assert len(chosen) == size, (n, k, len(chosen))
    cb = chosen @ hyperplane_basis(n).T  # [size, n]
    return (cb / math.sqrt((cb * cb).mean())).astype(np.float32)


def lloyd_codebook(dim: int, k: int, samples: int = 1 << 22, iters: int = 60, seed: int = 0) -> np.ndarray:
    """k-means codebook for an i.i.d. N(0,1) source, unit-RMS scaled. Unstructured baseline."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(samples, dim, device="cuda", generator=g)
    c = x[torch.randperm(samples, device="cuda", generator=g)[: 1 << k]].clone()
    for _ in range(iters):
        idx = nearest(x, c).long()
        s = torch.zeros_like(c).index_add_(0, idx, x)
        cnt = torch.bincount(idx, minlength=len(c)).clamp_min(1).unsqueeze(1)
        c = s / cnt
    c = c.cpu().numpy()
    return (c / math.sqrt((c * c).mean())).astype(np.float32)


_lib = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "build" / "libvqenc.so"))
_lib.vq_nearest.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def nearest(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Exact nearest-codeword index (int32) for rows of x [N, d] against cb [K, d] (CUDA float32)."""
    x, cb = x.contiguous().float(), cb.contiguous().float()
    out = torch.empty(len(x), dtype=torch.int32, device=x.device)
    torch.cuda.synchronize()
    rc = _lib.vq_nearest(x.data_ptr(), len(x), cb.data_ptr(), cb.shape[0], cb.shape[1], out.data_ptr())
    if rc != 0:
        raise RuntimeError(f"vq_nearest failed: {rc}")
    return out
