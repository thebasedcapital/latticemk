"""Check cached scores against float64 equations on the SAME fp16 logits.
Teacher-forced versus cached fp16 inference is a separate drift diagnostic,
not an arbitrary acceptance tolerance or a different scoring target.
"""
import json
from pathlib import Path
import time
from sampler import HFEngine, GSM8K, Record

engine=HFEngine('Qwen/Qwen3-1.7B-Base',GSM8K['powers'],42)
prompt=engine.prompt('If two boxes each hold three pencils, how many pencils are there?')
torch=engine.torch; captured=[]
handle=engine.model.register_forward_hook(lambda module,args,out: captured.append(out.logits[:,-1,:].detach().cpu().double()))
started=time.monotonic()
records=engine.generate(prompt,[(Record(),a,24) for a in GSM8K['powers']])
handle.remove()
equation_errors=[]; drift=[]
with torch.inference_mode():
    for k,record in enumerate(records):
        logits=torch.stack([step[k] for step in captured[:len(record.tokens)]])
        lp=logits.log_softmax(-1)
        chosen=lp.gather(1,torch.tensor(record.tokens)[:,None]).squeeze(-1)
        powers=torch.tensor(GSM8K['powers'],dtype=torch.float64)
        z=torch.logsumexp(lp[:,None,:]*powers[None,:,None],-1)
        equation_errors.append(dict(logp=float((chosen-torch.tensor(record.logp)).abs().max()),
                                    zeta=float((z-torch.tensor(record.zeta)).abs().max())))
        ids=torch.tensor([prompt+record.tokens[:-1]],device='cuda')
        logits=engine.model(input_ids=ids,use_cache=False).logits[0,len(prompt)-1:,:].float()
        lp=logits.log_softmax(-1)
        chosen=lp.gather(1,torch.tensor(record.tokens,device='cuda')[:,None]).squeeze(-1)
        powers=torch.tensor(GSM8K['powers'],device='cuda')
        z=torch.logsumexp(lp[:,None,:]*powers[None,:,None],-1)
        drift.append(dict(logp=float((chosen-torch.tensor(record.logp,device='cuda')).abs().max()),
                          zeta=float((z-torch.tensor(record.zeta,device='cuda')).abs().max())))
error=max(max(e.values()) for e in equation_errors)
payload=dict(equation_errors=equation_errors,teacher_forcing_drift=drift,tokens=engine.generated,
    prefill_tokens=engine.prefill,seconds=time.monotonic()-started,vocab_size=engine.model.config.vocab_size,
    precision='fp16 model; float32 cached scores independently evaluated in CPU float64',
    passed=error<1e-5,equation_tolerance=1e-5,
    diagnostic_note='Teacher-forcing drift is expected for fp16 batch/shape changes and is not evidence that cached log probabilities omit vocabulary mass.')
Path(__file__).with_name('engine-proof.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload)); assert payload['passed']
