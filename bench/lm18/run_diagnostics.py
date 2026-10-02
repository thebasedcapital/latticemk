"""Run under one gpu.sh lock; each subprocess is a scoped LM-18 experiment."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--proof',action='store_true');args=ap.parse_args()
steps=[('check_gate.py',[]),('detail_profile.py',[]),('contraction_cost.py',[]),('gemv.py',[]),('bench.py',['--ctx','128','--runs','27'])] if args.proof else [('check_contraction.py',[]),('profile.py',[]),('gemv.py',[]),('stage_profile.py',[]),('shape_probe.py',['--variants','original','preload']),('bench.py',['--ctx','128','--runs','27'])]
for script,params in steps:
    result=subprocess.run([sys.executable,str(HERE/script),*params])
    # The pass benchmark deliberately returns nonzero at the accepted kill line.
    if result.returncode:
        expected_kill=script=='bench.py' and result.returncode==1 and json.loads((HERE/'decision-128.json').read_text())['kill']
        if not expected_kill:raise SystemExit(result.returncode)
