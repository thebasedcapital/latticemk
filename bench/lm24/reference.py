"""CPU fp32 fake-GPTQ, one full-history forward per eviction mask."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lmk.model import SNAPSHOT

HERE = Path(__file__).resolve().parent


def transcript(turn_count=12, turn_size=32):
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT)
    start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    words = tokenizer.encode(' apple river stone blue tree', add_special_tokens=False)
    ids, spans = [], []
    for i in range(turn_count):
        role = 'system' if i == 0 else ('user' if i % 2 else 'assistant')
        header = tokenizer.encode(f'<|im_start|>{role}\n', add_special_tokens=False)
        body = tokenizer.encode(f'Turn {i}. Remember the number {i*7}.', add_special_tokens=False)
        room = turn_size - len(header) - 1
        if room < 0:
            raise ValueError('turn too short for header')
        content = (body + words * turn_size)[:room]
        a = len(ids)
        ids.extend(header + content + [end])
        spans.append((a, len(ids)))
        assert ids[a] == start and len(ids)-a == turn_size
    return ids, spans, start, end


def plan(ids, spans, event_turns=(6,10), last_n=2):
    """Events occur before the token following each specified complete turn."""
    event_positions = {spans[i-1][1] for i in event_turns}
    live, events = [], {}
    mask = torch.full((len(ids), len(ids)), torch.finfo(torch.float32).min)
    complete_ends = {b for a,b in spans}
    for q in range(len(ids)):
        if q in event_positions:
            live_turns, begin = [], 0
            for physical, logical in enumerate(live):
                if logical+1 in complete_ends:
                    live_turns.append((begin,physical+1))
                    begin = physical+1
            keep = sorted(set(live_turns[:1] + live_turns[-last_n:]))
            events[q] = keep
            live = [live[p] for a,b in keep for p in range(a,b)]
        live.append(q)
        mask[q, live] = 0
    return events, mask[None,None]


def model():
    torch.set_num_threads(8)
    net = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32,
                                               attn_implementation='eager').eval()
    deq = torch.load(ROOT/'bench/lm11/weights_int4_gptq_deq.pt', map_location='cpu')
    modules = dict(net.named_modules())
    for l in range(28):
        p = f'model.layers.{l}.'
        qkv = deq[f'L{l}.qkv']
        for kind,a,b in [('q',0,2048),('k',2048,3072),('v',3072,4096)]:
            modules[p+f'self_attn.{kind}_proj'].weight.data = qkv[a:b]
        modules[p+'self_attn.o_proj'].weight.data = deq[f'L{l}.o']
        gu = deq[f'L{l}.gu']
        modules[p+'mlp.gate_proj'].weight.data = gu[:3072]
        modules[p+'mlp.up_proj'].weight.data = gu[3072:]
        modules[p+'mlp.down_proj'].weight.data = deq[f'L{l}.down']
    net.lm_head.weight = torch.nn.Parameter(deq['lm_head'])
    return net


@torch.no_grad()
def main():
    ids, spans, start, end = transcript()
    events, mask = plan(ids, spans)
    net = model()
    inputs = torch.tensor([ids])
    positions = torch.arange(len(ids))[None]
    compact = net(input_ids=inputs, attention_mask=mask, position_ids=positions,
                  use_cache=False).logits[0].contiguous()
    causal = torch.full_like(mask, torch.finfo(torch.float32).min)
    causal[0,0].masked_fill_(torch.ones(len(ids),len(ids),dtype=torch.bool).tril(),0)
    # A fresh full-history control uses identical token IDs and logical positions.
    control = net(input_ids=inputs, attention_mask=causal, position_ids=positions,
                  use_cache=False).logits[0].contiguous()
    torch.save({'ids':ids,'spans':spans,'events':events,'compact':compact,
                'control':control,'im_start':start,'im_end':end}, HERE/'reference.pt')
    print(json.dumps({'tokens':len(ids),'events':list(events),
                      'reference':'one fp32 full-history forward per mask',
                      'path':str(HERE/'reference.pt')}))


if __name__ == '__main__':
    main()
