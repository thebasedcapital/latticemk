"""Compare megakv's append quantizer to the Python codec on captured KV.

Run through scripts/gpu.sh .venv/bin/python bench/lm09/quant_parity.py.
"""
import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kvcodec.codec_lm09 import make_lm09  # noqa: E402

lib = ctypes.CDLL(str(ROOT / "kernels/megakernel_kv/libmegakv.so"))
lib.mkv_quant_probe.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_int] * 2
lib.mkv_quant_probe.restype = ctypes.c_int
codec = make_lm09("lm09-k8t128r0-v4t128r0")

for kind, bits in (("k", 8), ("v", 4)):
    data = torch.load(ROOT / f"kvcodec/data/{kind}.pt", map_location="cpu",
                      weights_only=True)
    source = data[0, 0, :16].contiguous().cuda()
    del data
    rows = source.shape[0] * source.shape[1]
    codes = torch.empty(rows * 128 * bits // 8, dtype=torch.uint8, device="cuda")
    metadata = torch.empty(rows * 2, dtype=torch.half, device="cuda")
    rc = lib.mkv_quant_probe(source.data_ptr(), codes.data_ptr(),
                            metadata.data_ptr(), rows, bits)
    assert rc == 0, f"CUDA probe launch failed: {rc}"
    torch.cuda.synchronize()
    if bits == 4:
        raw = torch.stack((codes & 15, codes >> 4), -1).reshape(rows, 128)
    else:
        raw = codes.reshape(rows, 128)
    packed = codec.encode(source.float(), kind)
    want = packed.payload["codes"].reshape(rows, 128)
    want_meta = torch.stack((packed.payload["scale"],
                             packed.payload["zero"]), -1)
    got_meta = metadata.reshape(rows, 2)
    cm = (raw != want).sum().item()
    mm = (got_meta != want_meta).sum().item()
    max_meta = (got_meta.float() - want_meta.float()).abs().max().item()
    print(f"{kind}: codes mismatch {cm}/{want.numel()}, metadata mismatch "
          f"{mm}/{want_meta.numel()}, max metadata diff {max_meta}")
    assert cm == 0 and mm == 0, f"{kind} quantizer is not Python-codec identical"
print("RESULT: PASS")
