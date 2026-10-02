"""Generate one GPU-bounded 64-token fp32 fake-quant reference per prompt."""
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'bench/lm11'),str(ROOT)]
from check_correctness_gptq import hf_reference,PROMPTS
from lmk.model import SNAPSHOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('prompt',type=int);a=ap.parse_args()
    ids=AutoTokenizer.from_pretrained(SNAPSHOT)(PROMPTS[a.prompt],return_tensors='pt').input_ids[0]
    deq=torch.load(ROOT/'bench/lm11/weights_int4_gptq_deq.pt',map_location='cpu')
    tokens,logits=hf_reference(deq,ids)
    path=Path(__file__).with_name(f'reference-{a.prompt}.pt')
    torch.save(dict(prompt=ids,tokens=tokens,logits=logits),path)
    print(f'{path}: 64 teacher-forced reference positions',flush=True)
if __name__=='__main__':main()
