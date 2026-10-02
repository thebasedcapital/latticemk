"""Run an existing package driver without replacing its published result files.

Usage: python scripts/reproduce_capture.py RESULT [RESULT ...] -- COMMAND ...
Fresh JSONL rows and generated JSONs go under publish/reproduction/<original path>.
The original files are restored on success, failure and ordinary interruption.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    split = sys.argv.index('--')
    files = [Path(name) for name in sys.argv[1:split]]
    if not files or not sys.argv[split + 1:]:
        raise SystemExit('Expected result paths followed by -- COMMAND')
    if any(path.is_absolute() or '..' in path.parts for path in files):
        raise SystemExit('Result paths must be repository-relative')
    before = {path: path.read_bytes() if path.exists() else None for path in files}
    run = None
    try:
        run = subprocess.run(sys.argv[split + 1:], cwd=ROOT)
        return run.returncode
    finally:
        try:
            for path, original in before.items():
                if path.exists():
                    generated = path.read_bytes()
                    if generated != original or (run is not None and run.returncode == 0 and path.suffix != '.jsonl'):
                        if path.suffix == '.jsonl' and original is not None:
                            if not generated.startswith(original):
                                raise RuntimeError(f'{path}: driver replaced rather than appended JSONL')
                            generated = generated[len(original):]
                        destination = ROOT / 'publish/reproduction' / path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if path.suffix == '.jsonl':
                            with destination.open('ab') as handle:
                                handle.write(generated)
                        else:
                            destination.write_bytes(generated)
        finally:
            for path, original in before.items():
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original)


if __name__ == '__main__':
    raise SystemExit(main())
