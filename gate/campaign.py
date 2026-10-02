"""Rebuild the 41 wave-6 survivors and a deterministic stratified sample of 30 kills."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'gate'
sys.path.insert(0, str(HERE))


def selected():
    latest = {}
    for line in (ROOT / 'mutation/results.jsonl').read_text().splitlines():
        row = json.loads(line)
        latest[row['mutant_id']] = row
    eligible = [r for r in latest.values() if not r.get('equivalent') and r.get('family') != 'control'
                and r.get('build') == 'PASS' and r.get('family') != 'validator_only']
    survivors = [r for r in eligible if r.get('gpu_stage', r['stage']) == 'SURVIVED'
                 and r.get('extra_stage') == 'SURVIVED']
    assert len(survivors) == 41, len(survivors)
    kills = []
    rng = random.Random(17)
    for family in ('attention', 'bounds', 'dequant', 'precision', 'synchronization'):
        group = sorted([r for r in eligible if r['family'] == family and r not in survivors], key=lambda r:r['mutant_id'])
        kills.extend(rng.sample(group, 6))
    return [{**r, 'cohort': 'survivor'} for r in sorted(survivors, key=lambda r:r['mutant_id'])] + [
        {**r, 'cohort': 'previous-kill'} for r in sorted(kills, key=lambda r:r['mutant_id'])]


def prepare():
    spec = importlib.util.spec_from_file_location('gate_campaign_mutator', ROOT / 'mutation/mutate.py')
    mutator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutator)
    mutator.HERE = HERE
    rows = selected()
    path = HERE / 'campaign-manifest.jsonl'
    prior = {r['mutant_id']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    pending = [r for r in rows if r['mutant_id'] not in prior or not (ROOT / prior[r['mutant_id']]['library']).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(mutator.build_one, r):r for r in pending}
        for job in concurrent.futures.as_completed(jobs):
            row = job.result()
            with path.open('a') as out:
                out.write(json.dumps(row)+'\n')
            print(row['mutant_id'], row['build'], flush=True)
    print(json.dumps({'selected': len(rows), 'survivors': 41, 'previous_kills': 30, 'rebuilt': len(pending)}))

def prebuild():
    """Compile probe copies on CPU without holding the GPU lock."""
    import debug
    import suites
    manifest = [json.loads(line) for line in (HERE/'campaign-manifest.jsonl').read_text().splitlines()]
    def build(row):
        engine = 'v2' if row['model'] == '0.6B' else 'scale'
        library = ROOT / row['library']
        debug.build(engine, library)
        suites.jitter_module(engine, library)
        return row['mutant_id']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for job in concurrent.futures.as_completed([pool.submit(build, r) for r in manifest]):
            print(job.result(), 'PROBES_BUILT', flush=True)



def evaluate(index, baseline=True, inside_lock=False, deadline=285):
    rows = [json.loads(line) for line in (HERE / 'campaign-manifest.jsonl').read_text().splitlines()]
    row = sorted({r['mutant_id']:r for r in rows}.values(), key=lambda r:(r['cohort'] != 'survivor',r['mutant_id']))[index]
    engine = 'v2' if row['model'] == '0.6B' else 'scale'
    library = ROOT / row['library']
    outfile = HERE / 'build' / f'result-{index}-{int(baseline)}-{time.time_ns()}.jsonl'
    cmd = ['timeout', str(deadline), str(ROOT / '.venv/bin/python'),
           str(HERE / 'run.py'), '--engine', engine, '--lib', str(library), '--tier', '3', '--output', str(outfile)]
    if not inside_lock:
        cmd.insert(0, str(ROOT / 'scripts/gpu.sh'))
    if baseline:
        base = ROOT / ('kernels/megakernel_v2/libmega2.so' if engine == 'v2' else 'kernels/megakernel_scale/libmega_scale.so')
        cmd += ['--baseline', str(base)]
    stamp = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if outfile.exists():
        result = json.loads(outfile.read_text().splitlines()[-1])
    else:
        result = {'pass': False, 'infrastructure_error': True, 'returncode': proc.returncode,
                  'stderr': proc.stderr[-2000:], 'stdout': proc.stdout[-2000:]}
    saved = {'index': index, 'mutant_id': row['mutant_id'], 'cohort': row['cohort'],
             'family': row['family'], 'operator': row['operator'], 'layer': row['site']['layer'],
             'library': row['library'], 'regression_enabled': baseline,
             'wall_s': time.perf_counter()-stamp, 'gate': result}
    with (HERE / 'campaign-results.jsonl').open('a') as out:
        out.write(json.dumps(saved)+'\n')
    print(json.dumps(saved), flush=True)
    subprocess.run([str(ROOT/'.venv/bin/python'), str(HERE/'summarize.py')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'prebuild', 'evaluate', 'evaluate-all', 'evaluate-batch'))
    parser.add_argument('--index', type=int)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--without-baseline', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'prebuild':
        prebuild()
    elif args.action == 'evaluate-all':
        results = HERE / 'campaign-results.jsonl'
        done = {r['index'] for r in map(json.loads, results.read_text().splitlines())
                if r.get('regression_enabled') == (not args.without_baseline)
                and not r['gate'].get('infrastructure_error') and 'error' not in r['gate']} if results.exists() else set()
        pending = [i for i in range(71) if i not in done]
        manifest = [json.loads(line) for line in (HERE/'campaign-manifest.jsonl').read_text().splitlines()]
        ordered = sorted({r['mutant_id']:r for r in manifest}.values(), key=lambda r:(r['cohort'] != 'survivor',r['mutant_id']))
        batches, small = [], []
        for index in pending:
            if ordered[index]['model'] == '1.7B':
                if small:
                    batches.append(small)
                    small = []
                batches.append([index])
            else:
                small.append(index)
                if len(small) == 2:
                    batches.append(small)
                    small = []
        if small:
            batches.append(small)
        for indices in batches:
            cmd = [str(ROOT/'scripts/gpu.sh'), 'timeout', '285', str(ROOT/'.venv/bin/python'),
                   str(HERE/'campaign.py'), 'evaluate-batch', '--indices', *map(str, indices)]
            if args.without_baseline:
                cmd.append('--without-baseline')
            subprocess.run(cmd, check=True)
    elif args.action == 'evaluate-batch':
        if not args.indices or len(args.indices) > 2:
            parser.error('evaluate-batch requires one or two --indices under the GPU wrapper')
        for index in args.indices:
            evaluate(index, not args.without_baseline, inside_lock=True,
                     deadline=135 if len(args.indices) == 2 else 270)
    elif args.index is None:
        parser.error('evaluate requires --index; one candidate per GPU job')
    else:
        evaluate(args.index, not args.without_baseline)


if __name__ == '__main__':
    main()
