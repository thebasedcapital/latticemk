"""Finish one faithful pilot, with bounded chunks and an explicit GPU-time cap.
This CPU supervisor invokes EVERY GPU subprocess through scripts/gpu.sh.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import os
import signal

ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser(); ap.add_argument('--chunks',type=int,default=12)
ap.add_argument('--gap',type=int,default=60); ap.add_argument('--cap-seconds',type=int,default=3600)
args=ap.parse_args()
logpath=ROOT/'ppt/chunks.jsonl'
# The previous GPU chunk just ended. Leave a fair gap before the first request.
time.sleep(args.gap)
previous=[json.loads(line) for line in logpath.read_text().splitlines()] if logpath.exists() else []
charged=sum(row['wall_seconds'] for row in previous)
offset=len(previous)
for chunk in range(args.chunks):
    hard=min(590,args.cap_seconds-charged)
    if hard<15:
        break
    soft=max(1,int(hard-250))
    t0=time.monotonic(); timeout=False
    process=subprocess.Popen(['scripts/gpu.sh','.venv/bin/python','ppt/run.py',
        '--pilot','--count','1','--seconds',str(soft)],cwd=ROOT,start_new_session=True)
    try:
        code=process.wait(timeout=hard)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        process.wait()
        timeout=True; code=124
    charged+=time.monotonic()-t0
    row=dict(chunk=offset+chunk,wall_seconds=time.monotonic()-t0,returncode=code,
             hard_timeout=timeout,max_chunk_seconds=hard,soft_checkpoint_seconds=soft,charged_total_seconds=charged)
    with logpath.open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    results=ROOT/'ppt/pilot.jsonl'
    rows=[json.loads(l) for l in results.read_text().splitlines()] if results.exists() else []
    if any(r['model']=='1.7B' and r['method']=='ppt' for r in rows):
        break
    if code not in (0,124):
        break
    if chunk+1<args.chunks:
        time.sleep(args.gap)
