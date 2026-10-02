"""Serialize the five LM-23 timing sessions with bounded GPU lock ownership."""
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def main():
    for mode,ctx in [('batch',128),('batch',2048),('causal',128),('causal',2048),('causal',8192)]:
        command = ['scripts/gpu.sh','--timing','timeout','285','.venv/bin/python',
                   'bench/lm23/bench.py','--mode',mode,'--ctx',str(ctx)]
        result = subprocess.run(command,cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
