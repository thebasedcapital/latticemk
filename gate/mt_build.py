"""Reproducible timing-skew copies of the selected mt source, never originals."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def resolve_source(library):
    from debug import resolve_source as selected_source
    source, _ = selected_source('mt', library)
    return source


def build(library, m, kind='jitter'):
    if kind != 'jitter':
        raise ValueError(kind)
    from debug import resolve_source as selected_source
    source, defines = selected_source('mt', library)
    import contract
    built = contract.load(library)
    text = source.read_text()
    # Uniform block barriers remain uniform. Vary arrival times by CTA, warp, site.
    site = 0
    def skew():
        nonlocal site
        site += 1
        return ('{ unsigned gate_skew = ((blockIdx.x * 1664525u + '
                f'(threadIdx.x >> 5) * 1013904223u + {site}u * 747796405u) >> 8) & 255u; '
                '__nanosleep(gate_skew + ((threadIdx.x >> 5) == (blockIdx.x % 32) ? 3072u : 0u)); '
                '__syncthreads(); __nanosleep(gate_skew); }')
    def instrument(content):
        content = content.replace('__syncthreads();', 'GATE_SYNC_SITE')
        while 'GATE_SYNC_SITE' in content:
            content = content.replace('GATE_SYNC_SITE', skew(), 1)
        return content
    text = instrument(text)
    headers = {p.name: instrument(p.read_text()).encode() for p in source.parent.glob('*.cuh')}
    digest = hashlib.sha256(text.encode() + b''.join(headers[k] for k in sorted(headers)) + repr((m, defines, built['nvcc_flags'], built['threads'])).encode()).hexdigest()[:16]
    folder = ROOT / 'gate/build' / f'mt-jitter-m{m}-{digest}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'jitter.so'
    if not output.exists():
        (folder / 'mega_mt.cu').write_text(text)
        for name, content in headers.items():
            (folder / name).write_bytes(content)
        cuda = Path.home() / '.local/cuda-13.3'
        command = [str(cuda / 'bin/nvcc'), '-O3', '-arch=sm_75', '-std=c++17', '-allow-unsupported-compiler', '-L'+str(cuda/'lib'), '-Xcompiler', '-fPIC', '-shared', '-cudart', 'static', *built['nvcc_flags'], f'-DMT={m}', f'-DTHREADS={contract.threads(built, m)}', *['-D'+d for d in defines], '-o', str(output), str(folder/'mega_mt.cu')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (folder/'build.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'mt jitter compile failed: {folder / "build.log"}')
        (folder/'manifest.json').write_text(json.dumps({'source':str(source), 'library':str(Path(library).resolve()), 'm':m, 'barriers':site, 'command':command}, indent=2)+'\n')
    return output
