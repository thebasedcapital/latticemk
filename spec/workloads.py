"""Fixed public-source prompt suite; no private repository data is read."""
import ast
import collections
import functools
import hashlib
import heapq
import inspect
import json
import platform
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SEED = 25


def tokenizer():
    return AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', local_files_only=True)


def build():
    rng = random.Random(SEED)
    tok = tokenizer()
    wiki_root = Path.home()/'.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots'
    wiki = sorted(wiki_root.glob('*/wikitext-2-raw-v1/test-00000-of-00001.parquet'))[0]
    text = '\n\n'.join(pq.read_table(wiki)['text'].to_pylist())
    wiki_ids = tok.encode(text, add_special_tokens=False)
    modules = [ast, functools, heapq, statistics, collections]
    sources = [inspect.getsource(m) for m in modules]
    rows = []
    for category in ('code', 'rag', 'summarization', 'chat'):
        for i in range(20):
            target = (128, 512, 1024, 1900)[i % 4]
            if category == 'code':
                module = modules[i//4]
                source = sources[i//4]
                ids = tok.encode(source, add_special_tokens=False)
                start = rng.randrange(max(1, len(ids)-target))
                body = tok.decode(ids[start:start+target])
                lead = 'Rewrite the Python source below. Add a docstring and rename the local variable result to output. Preserve all other lines.\n\n```python\n'
                tail = '\n```\nRewritten source:\n```python\n'
                source_path = f'{module.__name__}/__init__.py' if module is collections else f'{module.__name__}.py'
                provenance = f'https://github.com/python/cpython/blob/v{platform.python_version()}/Lib/{source_path}'
                source_hash = hashlib.sha256(source.encode()).hexdigest()
            else:
                start = rng.randrange(len(wiki_ids)-2200)
                body = tok.decode(wiki_ids[start:start+target])
                if category == 'rag':
                    cue = ' '.join(body.split()[:7])
                    lead = 'Passage:\n'
                    tail = f'\n\nQuestion: What does the passage say immediately after "{cue}"? Copy the next two sentences exactly.\nAnswer: {cue}'
                elif category == 'summarization':
                    lead = 'Summarize the following passage in three sentences.\n\nPassage:\n'
                    tail = '\n\nSummary:\n'
                else:
                    lead = 'Continue the following discussion with your own ideas rather than quoting the passage.\n\n'
                    tail = '\n\nI think the most interesting question here is'
                provenance = 'Salesforce/wikitext:wikitext-2-raw-v1:test'
                source_hash = hashlib.sha256(text.encode()).hexdigest()
            # Preserve the task and closing cue while trimming only the source.
            budget = target-len(tok.encode(lead+tail, add_special_tokens=False))
            body_ids = tok.encode(body, add_special_tokens=False)[:max(1, budget)]
            prompt = lead+tok.decode(body_ids)+tail
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            rows.append({'id': f'{category}-{i:02d}', 'category': category,
                         'seed': SEED, 'prompt': prompt, 'prompt_ids': prompt_ids,
                         'prompt_tokens': len(prompt_ids),
                         'generate_tokens': (64, 96, 128, 256)[i % 4],
                         'source': provenance, 'source_sha256': source_hash,
                         'source_offset_tokens': start})
    (HERE/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps({'prompts': len(rows), 'seed': SEED,
                      'prompt_tokens_min': min(r['prompt_tokens'] for r in rows),
                      'prompt_tokens_max': max(r['prompt_tokens'] for r in rows)}))


if __name__ == '__main__':
    build()
