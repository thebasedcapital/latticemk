"""Compare v2 and LM-10 logits, generated tokens, and multi-step state.

This diagnostic detects a sampler disagreement with its own logits and a
cross-step race missed by one-step teacher-forced checks.
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "lm03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm03  # noqa: E402
import lm03b  # noqa: E402

NPRE, NDEC = 8, int(sys.argv[1]) if len(sys.argv) > 1 else 16
CTX_CAP = 256


def main():
    packed, _ = lm03.pack_weights(want_deq=False)
    emb = lm03.load("model.embed_tokens.weight").half().contiguous()
    norms = lm03.norm_table()
    rope = lm03.make_rope().cuda()
    from lmk.model import SNAPSHOT
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    e3 = lm03b.Engine3(CTX_CAP, packed, emb, norms, rope)
    reference = []
    for trial in range(20):
        e3.prefill(ids)
        for s in range(NDEC):
            e3.mega(1)
            observed = int(e3.bufs["tok"][0].item())
            if trial == 0:
                reference.append(observed)
            logits = e3.logits().cpu()
            best = int(logits.argmax())
            if observed != best:
                partial = e3.bufs["argp"].cpu().reshape(36, 2)
                largest = partial[:, 0].topk(5)
                print(f"trial={trial} step={s} sampler {observed} "
                      f"!= logits argmax {best}", flush=True)
                print("top partials:", [(int(i), float(partial[i, 0]),
                                         int(partial[i, 1])) for i in
                                        largest.indices], flush=True)
                print("top logits:", logits.topk(5), flush=True)
                raise AssertionError("sampled token disagrees with logits")
    e3.prefill(ids)
    multi = e3.decode(NDEC)[len(ids):len(ids) + NDEC].tolist()
    if multi != reference:
        raise AssertionError(f"multi-step token history diverged: {multi} != {reference}")
    print(f"PASS: {20 * NDEC} single-step tokens and {NDEC} multi-step tokens")

if __name__ == "__main__":
    main()
