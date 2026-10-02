"""Emit the implemented barrier schedule, including block-wide KV append."""
import importlib.util
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('v2schedule',ROOT/'kernels/megakernel_v2/sched_gen2.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)

def emit(m,ctx,mode,path):
    # Match the runtime's compact cache allocation, not v2's fixed stride.
    v2.MAXPOS=ctx+8
    for b in ('kcache','vcache'):
        v2.BUF_BYTES[b]=v2.NLAY*v2.MAXPOS*v2.KVB
    columns=[v2.build(ctx+(c if mode=='causal' else 0)) for c in range(m)]
    def name(b,c):
        return b if b in ('emb','norms','rope') or (b in ('kcache','vcache') and mode=='causal') else f'{b}.{c}'
    buffers={name(b,c):n for c in range(m) for b,n in v2.BUF_BYTES.items()}
    g=v2.Gen(ctx)
    phases=len(columns[0].flags)
    for phase in range(phases):
        bodies=[{'reads':[],'writes':[]} for _ in range(36)]
        appenders=[{'reads':[],'writes':[]} for _ in range(36)]
        phase_name=columns[0].flags[phase].removesuffix('_done')
        for col,source in enumerate(columns):
            for c,t in enumerate(source.tasks[phase*36:(phase+1)*36]):
                for op in ('reads','writes'):
                    for b,a,z in t[op]:
                        item=(name(b,col),a,z)
                        if m>1 and phase_name.endswith('.attn') and op=='writes' and b in ('kcache','vcache'):
                            appenders[c]['writes'].append(item)
                        else:bodies[c][op].append(item)
                if m>1 and phase_name.endswith('.attn') and appenders[c]['writes']:
                    appenders[c]['reads'].extend((name(b,col),a,z) for b,a,z in t['reads'] if b in ('qkv','rope'))
        if m>1 and phase_name.endswith('.attn'):g.phase(phase_name.replace('.attn','.append'),appenders)
        g.phase(phase_name,bodies)
    ir=v2.ir_json(g)
    ir['buffers']=[{'id':b,'bytes':n} for b,n in buffers.items()]
    ir['note']=f'M={m}, mode={mode}, cache stride={ctx+8}, first input position={ctx-1}. Every phase ends in a 36-CTA barrier. M>1 appends all columns before attention. Inputs are teacher-forced; all position logits are returned. Causal columns share cache and mask at ctx-1+column; batch columns have independent cache and positions.'
    Path(path).write_text(json.dumps(ir,separators=(',',':'))+'\n')

if __name__=='__main__':
    here=Path(__file__).with_name('schedules');here.mkdir(exist_ok=True)
    checks=[]
    for mode in ('causal','batch'):
        for m in (1,2,3,4,5):
            for ctx in (128,2048):
                path=here/f'{mode}_m{m}_ctx{ctx}.json';emit(m,ctx,mode,path)
                print(path)
                result=subprocess.run([str(ROOT/'validator/target/release/schedcheck'),str(path)],capture_output=True,text=True)
                checks.append(dict(path=str(path.relative_to(ROOT)),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
                print(result.stdout.strip())
                if result.returncode:raise SystemExit(result.stderr)
    (ROOT/'bench/lm22/schedules.json').write_text(json.dumps(checks,indent=2)+'\n')
