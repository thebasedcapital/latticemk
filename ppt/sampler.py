"""Fixed-horizon PPT, arXiv:2609.38104 equations 30 and 11."""
from dataclasses import dataclass, field
import math
import random

GSM8K = dict(powers=[2.0, 2.3, 2.6, 3.0, 3.5, 4.0], horizon=3072, block=192, rounds=10)


@dataclass
class Record:
    tokens: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    zeta: list = field(default_factory=list)
    terminal: bool = False

    def prefix(self, n):
        return Record(self.tokens[:n], self.logp[:n], self.zeta[:n], False)


def local_ratio(old, new, restart, rung):
    return sum(z[rung] for z in new.zeta[restart:]) - sum(z[rung] for z in old.zeta[restart:])


def swap_ratio(left, right, a, b):
    return (b-a) * (sum(left.logp)-sum(right.logp))


def accept(logratio, rng):
    return math.log(max(rng.random(), 1e-300)) < min(0.0, logratio)


def sweep(records, powers, rng, stats):
    for k in range(len(powers)-1):
        stats['swap_attempts'][k] += 1
        identical = records[k].tokens == records[k+1].tokens
        stats.setdefault('swap_nonidentity_attempts', [0]*(len(powers)-1))[k] += not identical
        if accept(swap_ratio(records[k], records[k+1], powers[k], powers[k+1]), rng):
            records[k], records[k+1] = records[k+1], records[k]
            stats['swap_accepts'][k] += 1
            stats.setdefault('swap_nonidentity_accepts', [0]*(len(powers)-1))[k] += not identical


def ppt(engine, prompt, seed, config=GSM8K, checkpoint=None, state=None, deadline=None):
    import time
    powers, horizon, block, rounds = (config[k] for k in ('powers','horizon','block','rounds'))
    rng = random.Random(seed)
    records = [Record() for _ in powers]
    stats = dict(local_attempts=[0]*len(powers), local_accepts=[0]*len(powers),
                 local_self=[0]*len(powers), swap_attempts=[0]*(len(powers)-1),
                 swap_accepts=[0]*(len(powers)-1))
    start_stage, start_round = 0, -1
    if state:
        records = [Record(**r) for r in state['records']]
        stats = state['stats']
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, list) else x
        rng.setstate(tuples(state['rng']))
        start_stage, start_round = state['stage'], state['round']
    def save(stage, round_):
        if checkpoint:
            from dataclasses import asdict
            checkpoint(dict(records=[asdict(r) for r in records], stats=stats,
                            rng=rng.getstate(), stage=stage, round=round_))
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError('Chunk deadline reached after checkpoint')
    for stage, end in enumerate(range(block, horizon+block, block)):
        if stage < start_stage:
            continue
        end = min(end, horizon)
        if stage != start_stage or start_round == -1:
            jobs = [(k, r, powers[k], end) for k,r in enumerate(records) if not r.terminal]
            if jobs:
                generated = engine.generate(prompt, [(r,a,t) for _,r,a,t in jobs])
                for (k,_,_,_), r in zip(jobs, generated):
                    records[k] = r
            save(stage, 0)
        for round_ in range(start_round if stage == start_stage and start_round >= 0 else 0, rounds):
            jobs = []
            for k, old in enumerate(records):
                restart = rng.randrange(end)
                stats['local_attempts'][k] += 1
                if old.terminal and restart >= len(old.tokens):
                    stats['local_self'][k] += 1
                    stats['local_accepts'][k] += 1
                else:
                    jobs.append((k, restart, old.prefix(restart), powers[k], end))
            if jobs:
                generated = engine.generate(prompt, [(r,a,t) for _,_,r,a,t in jobs])
                for (k,restart,_,_,_), new in zip(jobs, generated):
                    if accept(local_ratio(records[k],new,restart,k),rng):
                        records[k] = new
                        stats['local_accepts'][k] += 1
            sweep(records,powers,rng,stats)
            save(stage, round_+1)
        start_round = -1
        save(stage+1, -1)
    return records[-1], stats


class HFEngine:
    def __init__(self, model_id, powers, seed):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True,
                        dtype=torch.float16, attn_implementation='sdpa').eval().cuda()
        self.powers = powers
        self.generator = torch.Generator(device='cuda').manual_seed(seed)
        self.generated = self.prefill = self.decode_steps = 0
        self.eos = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|im_end|>')}

    def prompt(self, question):
        encoded = self.tokenizer.apply_chat_template([dict(role='user', content=
            'Solve the following math problem step by step. End your response with "####" followed by the numerical answer.\n\n'+question)],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)
        return encoded['input_ids']

    def text(self, record):
        return self.tokenizer.decode(record.tokens, skip_special_tokens=True)

    def generate(self, prompt, jobs):
        torch = self.torch
        # Variable-length restarts can nearly double the padded physical KV
        # horizon. Keep all six Markov replicas, but bound simultaneous caches.
        if len(jobs)>2 and max(end for _,_,end in jobs)>768:
            results=[]
            for start in range(0,len(jobs),2):
                results.extend(self.generate(prompt,jobs[start:start+2]))
            return results
        records = [r.prefix(len(r.tokens)) for r,_,_ in jobs]
        prefixes = [prompt+r.tokens for r in records]
        width = max(map(len,prefixes))
        ids = torch.full((len(jobs),width), self.tokenizer.pad_token_id, device='cuda', dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,p in enumerate(prefixes):
            ids[i,-len(p):] = torch.tensor(p,device='cuda')
            mask[i,-len(p):] = 1
        self.prefill += sum(map(len,prefixes))
        powers = torch.tensor(self.powers,device='cuda',dtype=torch.float32)
        alpha = torch.tensor([a for _,a,_ in jobs],device='cuda')
        active = [True]*len(jobs)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask,
                position_ids=mask.cumsum(-1).sub(1).clamp(min=0),use_cache=True,logits_to_keep=1)
            while any(active):
                logits = out.logits[:,-1,:].float()
                lp = logits.log_softmax(-1)
                z = torch.logsumexp(lp[:,None,:]*powers[None,:,None],dim=-1)
                greedy = alpha == 0
                proposal = (logits * alpha[:,None]).softmax(-1)
                sampled = torch.multinomial(proposal,1,generator=self.generator).squeeze(-1)
                sampled = torch.where(greedy, logits.argmax(-1), sampled)
                chosen = lp.gather(1,sampled[:,None]).squeeze(-1)
                token_cpu = sampled.tolist()
                lp_cpu, z_cpu = chosen.tolist(), z.tolist()
                for i in range(len(jobs)):
                    if active[i]:
                        r = records[i]
                        r.tokens.append(token_cpu[i]); r.logp.append(lp_cpu[i]); r.zeta.append(z_cpu[i])
                        self.generated += 1
                        r.terminal = token_cpu[i] in self.eos
                        active[i] = not r.terminal and len(r.tokens)<jobs[i][2]
                if not any(active):
                    break
                self.decode_steps += len(jobs)
                mask = torch.cat([mask,torch.ones((len(jobs),1),device='cuda',dtype=mask.dtype)],dim=-1)
                out = self.model(input_ids=sampled[:,None], attention_mask=mask,
                    position_ids=(mask.sum(-1)-1)[:,None], past_key_values=out.past_key_values,
                    use_cache=True,logits_to_keep=1)
        return records
