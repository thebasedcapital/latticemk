from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
SRC = next((Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots').iterdir())
WIKI = next((Path.home()/'.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots').iterdir())
text='\n\n'.join(pq.read_table(WIKI/'wikitext-2-raw-v1/test-00000-of-00001.parquet')['text'].to_pylist())
x=AutoTokenizer.from_pretrained(SRC)(text,return_tensors='pt').input_ids[0][:2048].cuda()
model=AutoModelForCausalLM.from_pretrained(SRC,dtype=torch.float16).cuda().eval()
with torch.inference_mode():
    h=model.model(x.unsqueeze(0)).last_hidden_state[0]
    losses=[]
    for j in range(1024,2047,128):
        e=min(j+128,2047)
        logits=model.lm_head(h[j:e]).float()
        losses.append(torch.nn.functional.cross_entropy(logits,x[j+1:e+1],reduction='sum').item())
    import math
    print('HF first-window PPL', math.exp(sum(losses)/1023),flush=True)
