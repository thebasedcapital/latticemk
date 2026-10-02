"""Static first-group instruction counts and compiler resource records."""
import collections,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;KERNEL=ROOT/'kernels/megakernel_mma'
def main():
    cu=ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'
    text=subprocess.check_output([str(cu),'--dump-sass',str(KERNEL/'libshape_sw1_m4.so')],text=True)
    ins=re.findall(r'/\*([0-9a-f]+)\*/\s+((?:@!?P\d\s+)?[A-Z][^;]*);',text)
    h=[i for i,(_,x) in enumerate(ins) if x.startswith('HMMA')];a=h[0]
    while a>0 and not ins[a][1].startswith('SHFL.IDX'):a-=1
    while a>0 and not ins[a-1][1].startswith('BRA'):a-=1
    body=[x for _,x in ins[a:h[15]+1]]
    counts=collections.Counter(re.sub(r'^@!?P\d\s+','',x).split()[0] for x in body)
    summary=dict(variant='sw1',m=4,range_hex=[ins[a][0],ins[h[15]][0]],method='static disassembly first unrolled 128-K group; excludes group loads and loop control; not retired counters',instructions=dict(counts),instructions_total=len(body),HMMA_total_across_four_width_branches=len(h))
    (HERE/'sass-counts.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    builds=[]
    for p in sorted(KERNEL.glob('build-*.log')):
        entries=re.findall(r"Compiling entry function '([^']+)'.*?([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads.*?Used ([0-9]+) registers.*?([0-9]+) bytes smem",p.read_text(),re.S)
        builds.extend(dict(log=p.name,entry=e,stack_bytes=int(st),spill_stores=int(ss),spill_loads=int(sl),registers=int(r),smem=int(sm)) for e,st,ss,sl,r,sm in entries)
    (HERE/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
if __name__=='__main__':main()
