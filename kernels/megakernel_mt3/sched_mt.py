"""Emit fused append/attention with concurrent batch work and local causal KV."""
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
        phase_name=columns[0].flags[phase].removesuffix('_done')
        attention=phase_name.endswith('.attn')
        layer=int(phase_name.split('.')[0][1:]) if attention else None
        for col,source in enumerate(columns):
            for c,t in enumerate(source.tasks[phase*36:(phase+1)*36]):
                dst=((col*32+c)//(2 if m>1 else 1))%36 if attention and mode=='batch' and c<32 else c
                for op in ('reads','writes'):
                    for b,a,z in t[op]:
                        if attention and op=='reads' and b in ('kcache','vcache') and mode=='causal':
                            # All in-block rows come from local qkv, never the cache.
                            first=layer*v2.MAXPOS*v2.KVB
                            head=a%v2.KVB
                            z=min(z,first+(ctx-2)*v2.KVB+head+256)
                            if z<=a:continue
                        bodies[dst][op].append((name(b,col),a,z))
                if attention and c<32:
                    kvh=c//4
                    # Setup materializes this KV head even in the first half.
                    for offset in (v2.QROWS,v2.QROWS+v2.KVROWS):
                        a=(offset+kvh*128)*2
                        bodies[dst]['reads'].append((name('qkv',col),a,a+256))
        g.phase(phase_name,bodies)
    ir=v2.ir_json(g)
    ir['buffers']=[{'id':b,'bytes':n} for b,n in buffers.items()]
    ir['note']=f'M={m}, mode={mode}, cache stride={ctx+8}, first input position={ctx-1}. Fused append/attention ends in one 36-CTA barrier. Batch maps pairs of (sequence, head, half) jobs across all 36 CTAs. Causal queries share cache loads and materialize all in-block rows locally; no cache reads of in-block rows. Original 32 virtual partitions and output partial layout are unchanged.'
    Path(path).write_text(json.dumps(ir,separators=(',',':'))+'\n')

if __name__=='__main__':
    here=ROOT/'bench/lm23/schedules';here.mkdir(exist_ok=True)
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
    (ROOT/'bench/lm23/schedules.json').write_text(json.dumps(checks,indent=2)+'\n')
