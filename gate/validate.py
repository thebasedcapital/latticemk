"""Three consecutive complete passes for each accepted engine configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'gate'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('v2', 'scale', 'mt'))
    parser.add_argument('--m', type=int, default=1)
    args = parser.parse_args()
    configurations = [(args.engine, args.m)] if args.engine else [('v2',1), ('scale',1), ('mt',1), ('mt',4)]
    output = HERE / 'validation-originals.jsonl'
    for engine, m in configurations:
        for repeat in range(3):
            result_file = HERE / 'build' / f'validation-{engine}-{m}-{repeat}-{time.time_ns()}.jsonl'
            result_file.parent.mkdir(parents=True, exist_ok=True)
            library = ROOT / (f'kernels/megakernel_mt/libmt{m}.so' if engine == 'mt' else
                              'kernels/megakernel_v2/libmega2.so' if engine == 'v2' else
                              'kernels/megakernel_scale/libmega_scale.so')
            command = [str(ROOT/'scripts/gpu.sh'), 'timeout', '285', str(ROOT/'.venv/bin/python'),
                       str(HERE/'run.py'), '--engine', engine, '--tier', '3', '--baseline', str(library),
                       '--m', str(m), '--mode', 'batch', '--output', str(result_file)]
            proc = subprocess.run(command, capture_output=True, text=True)
            if result_file.exists():
                result = json.loads(result_file.read_text().splitlines()[-1])
            else:
                result = {'pass': False, 'infrastructure_error': True, 'returncode': proc.returncode,
                          'stdout': proc.stdout[-2000:], 'stderr':proc.stderr[-2000:]}
            record = {'engine':engine,'m':m,'mode':'batch','repeat':repeat+1,'gate':result}
            with output.open('a') as out:
                out.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if not result['pass']:
                raise SystemExit(1)


if __name__ == '__main__':
    main()
