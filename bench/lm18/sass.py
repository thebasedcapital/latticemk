"""Save disassembly and static instruction counts for focused GEMM loops."""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',default=['original']);args=ap.parse_args()
rows=[]
for variant in args.variants:
    for m in (1,4):
        lib=ROOT/f'kernels/megakernel_mt2/libshape_{variant}_m{m}.so'
        text=subprocess.check_output([str(ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'),'--dump-sass',str(lib)],text=True,stderr=subprocess.DEVNULL)
        (HERE/f'sass-{variant}-m{m}.txt').write_text(text)
        for section in text.split('Function : ')[1:]:
            name=section.splitlines()[0].strip()
            if not name.startswith('_Z5shape'):continue
            instructions=[]
            # Turing control layout is an assumed ISA decoding contract:
            # stall bits 105..108, write barrier 110..112, wait mask 116..121.
            controls={int(a,16):int(c,16) for a,c in re.findall(r'/\*([0-9a-f]+)\*/[^\n]+\n\s+/\* (0x[0-9a-f]+) \*/',section)}
            for line in section.splitlines()[1:]:
                if '/*' not in line or '*/' not in line:continue
                left,rest=line.split('*/',1);rest=rest.split(';',1)[0].strip()
                if not rest or rest.startswith('/*'):continue
                try:address=int(left.split('/*')[-1],16)
                except ValueError:continue
                fields=rest.split();op=fields[1] if fields[0].startswith('@') else fields[0]
                instructions.append((address,op,fields))
            counts=Counter(op for _,op,_ in instructions);loops=[]
            for address,op,fields in instructions:
                if op not in ('BRA','BRA.U'):continue
                try:target=int(fields[fields.index(op)+1],16)
                except (ValueError,IndexError):continue
                if target>=address:continue
                body=[x for x in instructions if target<=x[0]<=address]
                loads=sum(x[1].startswith('LDG.E.128') for x in body)
                if not 1<=loads<=3 or not any(x[1].startswith('HFMA2') for x in body):continue
                loops.append(dict(begin=hex(target),end=hex(address),instructions=len(body),int4_chunks=loads,weights=32*loads,instructions_per_weight=len(body)/(32*loads),hfma2=sum(x[1].startswith('HFMA2') for x in body),lds=sum(x[1].startswith('LDS') for x in body),local_ops=sum(x[1].startswith(('LDL','STL')) for x in body)))
                loops[-1]['encoded_stall_cycles']=sum((controls.get(a,0)>>41)&15 for a,_,_ in body)
                loops[-1]['wait_mask_sites']=sum(bool((controls.get(a,0)>>52)&63) for a,_,_ in body)
                loops[-1]['hfma_wait_mask_sites']=sum(op.startswith('HFMA2') and bool((controls.get(a,0)>>52)&63) for a,op,_ in body)
                loops[-1]['lds_write_barrier_sites']=sum(op.startswith('LDS') and ((controls.get(a,0)>>46)&7)!=7 for a,op,_ in body)
            rows.append(dict(variant=variant,m=m,kernel=name,counts=dict(counts),row_loops=loops))
(HERE/'sass.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
