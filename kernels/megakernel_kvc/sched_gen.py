"""IR v1 for unchanged decode phases and actual gather/scatter launch grids."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('v2_schedule',ROOT/'kernels/megakernel_v2/sched_gen2.py')
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)


def decode(logical, physical):
    ir = v2.ir_json(v2.build(physical+1))
    for task in ir['tasks']:
        for r in task['reads']:
            if r['buffer']=='rope':
                r['begin'],r['end'] = logical*1024,(logical+1)*1024
    ir['buffers'] = [dict(b,bytes=max(b['bytes'],(logical+1)*1024))
                     if b['id']=='rope' else b for b in ir['buffers']]
    ir['note'] = ('V2 phase order and barriers unchanged. RoPE uses logical '
                  f'{logical}; cache append uses physical {physical}. '
                  'Logical/physical state loads precede the first phase; '
                  'both counters advance after the final barrier. No compaction '
                  'occurs during a decode launch. Cross-step token and counter '
                  'dependencies are enforced by stream ordering outside IR v1.')
    return ir


def event(physical, ranges):
    rows = [row for a,b in ranges for row in range(a,b)]
    kept = len(rows)
    units = kept*128
    blocks = 32*28*2
    buffers = [dict(id=name,bytes=28*8704*2048) for name in ('kcache','vcache')]
    buffers += [dict(id='scratch',bytes=2*28*kept*2048),
                dict(id='rows',bytes=kept*4),dict(id='kv_len',bytes=4)]
    tasks = []
    def region(buf,begin,end):
        return dict(buffer=buf,begin=begin,end=end)
    for gather in (True,False):
        for which in range(2):
            cache = ('kcache','vcache')[which]
            for layer in range(28):
                for x in range(32):
                    block = x+32*(layer+28*which)
                    reads,writes = [],[]
                    for i in range(x*256,units,32*256):
                        end = min(i+256,units)
                        sb = ((which*28+layer)*units+i)*16
                        se = ((which*28+layer)*units+end)*16
                        if gather:
                            # Each contiguous vector chunk covers at most two
                            # cache rows, whose physical addresses may differ.
                            for j in range(i,end,128):
                                row = j//128
                                cb = (layer*8704+rows[row])*2048
                                reads.append(region(cache,cb,cb+2048))
                                reads.append(region('rows',row*4,row*4+4))
                            writes.append(region('scratch',sb,se))
                        else:
                            reads.append(region('scratch',sb,se))
                            cb = layer*8704*2048+i*16
                            writes.append(region(cache,cb,cb+(end-i)*16))
                    tasks.append(dict(id=f'{"gather" if gather else "scatter"}.{block}',
                                      block=block,order=0 if gather else 1,
                                      reads=reads,writes=writes,
                                      waits=[] if gather else [dict(flag='gather_done',value=blocks)],
                                      sets=[dict(flag='gather_done' if gather else 'scatter_done',add=1)]))
    tasks.append(dict(id='update_kv_len',block=0,order=2,reads=[],
                      writes=[region('kv_len',0,4)],
                      waits=[dict(flag='scatter_done',value=blocks)],sets=[]))
    return dict(version=1,model='qwen3-0.6b',blocks=blocks,flags_per_step=2,
                note=f'Actual grid (32,28,2), 256 threads, physical {physical}, retained {kept}. '
                     'Stream completion orders gather before scatter and state update. '
                     'Rows are immutable during the event; rope_pos is untouched.',
                buffers=buffers,flags=[dict(id='gather_done'),dict(id='scatter_done')],tasks=tasks)


def main():
    target = HERE/'schedules'
    target.mkdir(exist_ok=True)
    cases = {'decode_128':decode(128,128),'decode_2048':decode(2048,2048),
             'decode_compacted_8192_2048':decode(8192,2048),
             'compact_8192_2048':event(8192,[(0,256),(6400,8192)]),
             'compact_192_96':event(192,[(0,32),(128,192)])}
    for name,ir in cases.items():
        path = target/(name+'.json')
        path.write_text(json.dumps(ir,separators=(',',':')))
        print(f'{path}: {len(ir["tasks"])} tasks')


if __name__=='__main__':
    main()
