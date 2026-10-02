"""Run the scoped LM-23 cumulative gates, each in its own bounded GPU job."""
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    configs = [('mt',2,'batch',2,'megakernel_mt3'), ('mt',2,'causal',2,'megakernel_mt3'),
               ('mt',4,'batch',2,'megakernel_mt3'), ('mt',4,'causal',3,'megakernel_mt3'),
               ('v2',1,'batch',3,None), ('mt',4,'causal',3,'megakernel_mt2')]
    for engine,m,mode,tier,folder in configs:
        args = ['scripts/gpu.sh','timeout','285','.venv/bin/python','gate/run.py',
                '--engine',engine,'--tier',str(tier),'--output',str(HERE/'tiers.jsonl')]
        if folder:
            args += ['--m',str(m),'--mode',mode,'--lib',f'kernels/{folder}/libmt{m}.so']
        result = subprocess.run(args,cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
