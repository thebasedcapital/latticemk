"""Static row-loop instruction density, not hardware execution counters."""
import json
import subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for lib in ('libmt1.so','libmt4.so','libmt4mma.so','libmt4fp32.so','libmt4initial.so'):
    text=subprocess.check_output([str(ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'),'--dump-sass',str(ROOT/'kernels/megakernel_mt'/lib)],text=True,stderr=subprocess.DEVNULL)
    kernels=[]
    for section in text.split('Function : ')[1:]:
        name=section.splitlines()[0].strip();instructions=[]
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
            if op!='BRA':continue
            try:target=int(fields[fields.index('BRA')+1],16)
            except (ValueError,IndexError):continue
            if target>=address:continue
            body=[x for x in instructions if target<=x[0]<=address]
            loads=sum(x[1].startswith('LDG.E.128') for x in body)
            hfm=sum(x[1].startswith('HFMA2') for x in body)
            if not 1<=loads<=3 or not hfm:continue
            loops.append(dict(begin=hex(target),end=hex(address),instructions=len(body),int4_chunk_loads=loads,weights_per_thread=32*loads,static_thread_instructions_per_weight=len(body)/(32*loads),hfma2=hfm,lds=sum(x[1].startswith('LDS') for x in body),local_ops=sum(x[1].startswith(('LDL','STL')) for x in body)))
        kernels.append(dict(kernel=name,instruction_counts=dict(sorted(counts.items())),hfma2=sum(v for k,v in counts.items() if k.startswith('HFMA2')),hmma=sum(v for k,v in counts.items() if k.startswith('HMMA')),local_ops=sum(v for k,v in counts.items() if k.startswith(('LDL','STL'))),row_loops=loops))
    rows.append(dict(library=lib,kernels=kernels))
Path(__file__).with_name('sass.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{ 'library':r['library'],'kernels':[{k:v for k,v in x.items() if k!='instruction_counts'} for x in r['kernels']]} for r in rows]),flush=True)
