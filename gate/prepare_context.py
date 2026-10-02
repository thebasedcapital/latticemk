"""Generate CPU HF fp32 fake-GPTQ reference for every scale attention warp slice."""
from pathlib import Path
import json
import time
import torch
from transformers import AutoTokenizer
import core


def slice_rows(length):
    """Mirror scale attn2's ceil-divided warp ranges in its two half ranges."""
    halves = ((length + 1) // 2, length // 2)
    return [[max(0, min((size + 31) // 32, size - warp * ((size + 31) // 32)))
             for warp in range(32)] for size in halves]


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    started = time.perf_counter()
    deq = torch.load(core.ROOT / 'bench/lm12/weights_int4_gptq_deq.pt', map_location='cpu')
    net, snapshot = core.reference_model('1.7B', deq)
    del deq
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    text = '\n'.join(f'Item {i}: integer {i*i}, remainder {i%7}. The next item follows.' for i in range(512))
    prompt = tokenizer(text, return_tensors='pt').input_ids[0][:2049]
    assert len(prompt) == 2049
    out = net(input_ids=prompt[None], use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    lines, forced = [], []
    for step in range(4):
        line = out.logits[0, -1].cpu().clone()
        lines.append(line)
        forced.append(int(line.argmax()))
        if step < 3:
            out = net(input_ids=torch.tensor([[forced[-1]]]), past_key_values=past,
                      use_cache=True, logits_to_keep=1)
            past = out.past_key_values
    path = core.HERE / 'cache/scale-context-2049.pt'
    path.parent.mkdir(exist_ok=True)
    torch.save({'name': 'context-2049-varied', 'prompt': prompt, 'steps': 4,
                'reference': torch.stack(lines), 'forced': torch.tensor(forced)}, path)
    coverage = slice_rows(2049)
    assert all(rows > 0 for half in coverage for rows in half)
    metadata = {'length': 2049, 'steps': 4, 'half_ranges': 2, 'warps_per_half': 32,
                'minimum_rows_per_warp': min(rows for half in coverage for rows in half),
                'rows_per_half': coverage, 'wall_s': time.perf_counter()-started,
                'reference': 'CPU HF fp32 fake-GPTQ', 'file': str(path.relative_to(core.ROOT))}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
