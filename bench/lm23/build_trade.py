"""Build spill-free fully inlined alternatives at fewer threads, never selected."""
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    folder = ROOT/'gate/build/lm23-thread-trade'
    folder.mkdir(parents=True,exist_ok=True)
    source = (ROOT/'kernels/megakernel_mt3/mega_mt.cu').read_text()
    source = source.replace('#if MT == 5\n#define ATTQUAL __noinline__','#if 0\n#define ATTQUAL __noinline__')
    source = source.replace('#define NORMQUAL __noinline__','#define NORMQUAL')
    path = folder/'inline.cu'
    path.write_text(source)
    cuda = Path.home()/'.local/cuda-13.3'
    records = []
    for m,threads in [(1,512),(5,256)]:
        command = [str(cuda/'bin/nvcc'),'-O3','-arch=sm_75','-std=c++17','-allow-unsupported-compiler',
                   '-L'+str(cuda/'lib'),'-I'+str(ROOT/'kernels/megakernel_mt3'),'-Xcompiler','-fPIC',
                   '-shared','-cudart','static','--fmad=false','-Xptxas=-v',f'-DMT={m}',f'-DTHREADS={threads}',
                   str(path),'-o',str(folder/f'libmt{m}.so')]
        result = subprocess.run(command,capture_output=True,text=True,timeout=120)
        text = result.stdout+result.stderr
        (folder/f'build-m{m}.log').write_text(text)
        print(text,flush=True)
        stats = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',text)
        if result.returncode or not stats or any(int(x) for row in stats for x in row):
            raise RuntimeError(f'thread-trade M{m} build failed spill gate')
        records.append(dict(tag='measured',m=m,threads=threads,stack_spill_bytes=stats,
                            command=command,log=str((folder/f'build-m{m}.log').relative_to(ROOT))))
    (HERE/'trade-builds.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
