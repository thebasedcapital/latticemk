"""Build contract of an mt-engine candidate, so gate rebuilds (jitter, tier-3 dumps) and the sequential
reference use the same compiler flags and thread counts as the candidate itself.

Optional file `gate_contract.json` next to the candidate library:
    {"nvcc_flags": ["--fmad=false"], "threads": {"1": 1024, "2": 512, "3": 512, "4": 512, "5": 512},
     "seq_lib": "libmt1.so"}
`seq_lib` is relative to the library's directory: the candidate's own M=1 build, used as the sequential reference.
Without the file, the LM-14 contract applies (default contraction, 1024 threads except M=5, sequential reference
kernels/megakernel_mt/libmt1.so).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = {"nvcc_flags": [], "threads": {"1": 1024, "2": 1024, "3": 1024, "4": 1024, "5": 512},
           "seq_lib": str(ROOT / "kernels/megakernel_mt/libmt1.so")}


def load(library) -> dict:
    folder = Path(library).resolve().parent
    file = folder / "gate_contract.json"
    if not file.exists():
        return dict(DEFAULT, source="default (LM-14)")
    data = json.loads(file.read_text())
    contract = {"nvcc_flags": list(data.get("nvcc_flags", DEFAULT["nvcc_flags"])),
                "threads": {str(k): int(v) for k, v in data.get("threads", DEFAULT["threads"]).items()},
                "seq_lib": str((folder / data["seq_lib"]).resolve()) if "seq_lib" in data else DEFAULT["seq_lib"],
                "source": str(file)}
    if not Path(contract["seq_lib"]).exists():
        raise FileNotFoundError(f"gate_contract.json seq_lib missing: {contract['seq_lib']}")
    return contract


def threads(contract: dict, m: int) -> int:
    return contract["threads"][str(m)]


def calibration_library(library, m: int) -> Path:
    """Select a trusted layer baseline by compiler/thread contract, not candidate path."""
    candidate = load(library)
    fields = ("nvcc_flags", "threads")
    if all(candidate[field] == DEFAULT[field] for field in fields):
        return ROOT / f"kernels/megakernel_mt/libmt{m}.so"
    accepted = ROOT / f"kernels/megakernel_mt2/libmt{m}.so"
    known = load(accepted)
    if all(candidate[field] == known[field] for field in fields):
        return accepted
    raise ValueError("no accepted tier-3 calibration for candidate compiler/thread contract")
