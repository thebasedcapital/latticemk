"""Candidate-source-only CUDA dump builds. No production source is modified."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'gate'
VERSION = 'local-layer-v4'
ORIGINAL = {
    'v2': (ROOT / 'kernels/megakernel_v2/libmega2.so', ROOT / 'kernels/megakernel_v2/mega2.cu'),
    'scale': (ROOT / 'kernels/megakernel_scale/libmega_scale.so', ROOT / 'kernels/megakernel_scale/mega_scale.cu'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def associate(library, source, defines=None):
    """Record an explicit association, bound to both file hashes."""
    library, source = Path(library).resolve(), Path(source).resolve()
    record = {'library': str(library), 'source': str(source), 'library_sha256': digest(library),
              'source_sha256': digest(source), 'defines': list(defines or ()),
              'explicit_defines': defines is not None}
    folder = HERE / 'cache' / 'sources'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (record['library_sha256'] + '.json')).write_text(json.dumps(record, indent=2))
    return record


def resolve_source(engine, library, source=None):
    library = Path(library).resolve()
    if not library.is_file():
        raise FileNotFoundError(f'candidate library missing: {library}')
    key = digest(library)
    record = HERE / 'cache' / 'sources' / (key + '.json')
    if source is not None:
        return Path(source).resolve(), ()
    if record.exists():
        value = json.loads(record.read_text())
        path = Path(value['source'])
        if digest(path) != value['source_sha256']:
            raise ValueError('associated candidate source changed; re-associate explicitly')
        return path, tuple(value['defines'])
    if engine in ORIGINAL and library == ORIGINAL[engine][0].resolve():
        return ORIGINAL[engine][1], ()
    if engine == 'mt':
        original = ROOT / 'kernels/megakernel_mt'
        if library.parent == original and re.fullmatch(r'libmt[1-5]\.so', library.name):
            return original / 'mega_mt.cu', ()
    adjacent = library.with_suffix('.cu')
    if adjacent.is_file():
        return adjacent, ()
    raise FileNotFoundError(f'no candidate source/debug path for {library}; use debug.associate(library, source, defines)')


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'unsupported candidate debug insertion point: {old[:100]!r}')
    return text.replace(old, new, 1)


def instrument(text, engine):
    """Inject local HF incoming states and dump the candidate's actual staged attention."""
    mt = engine == 'mt'
    col = 'col' if mt else '0'
    x = 'xloc[col]' if mt else 'xloc'
    helper = f'''
__device__ const __half* gate_in = nullptr;
__device__ float* gate_hidden = nullptr;
__device__ float* gate_attention = nullptr;
__shared__ int gate_layer;
__device__ float gate_norm(int layer, int col) {{
  float ss = 0.f;
  for (int i = threadIdx.x; i < HID; i += THREADS) {{
    const __half v = gate_in[((int64_t)col * NLAY + layer) * HID + i];
    {x}[i] = v;
    const float f = __half2float(v);
    ss += f * f;
  }}
  return rsqrtf(block_sum(ss) / HID + RMS_EPS);
}}
__device__ void gate_dump_hidden(const __half* down, int layer, int col) {{
  if (gate_hidden && blockIdx.x == 0)
    for (int i = threadIdx.x; i < HID; i += THREADS)
      gate_hidden[((int64_t)col * NLAY + layer) * HID + i] =
        __half2float(__float2half_rn(__half2float({x}[i]) + __half2float(down[i])));
}}
extern "C" int gate_debug_set(int64_t input, int64_t hidden, int64_t attention) {{
  const __half* a = reinterpret_cast<const __half*>(input);
  float* b = reinterpret_cast<float*>(hidden);
  float* c = reinterpret_cast<float*>(attention);
  cudaError_t e = cudaMemcpyToSymbol(gate_in, &a, sizeof(a));
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(gate_hidden, &b, sizeof(b));
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(gate_attention, &c, sizeof(c));
  return (int)e;
}}
'''
    marker = '// qkv/gu/lm prologue:'
    text = _once(text, marker, helper + '\n' + marker)
    old = 'const float r = xresidual(p, src, embed' + (', col);' if mt else ');')
    new = f'''if (wsel < NLAY && threadIdx.x == 0) gate_layer = wsel;
  __syncthreads();
  const float r = gate_in && wsel < NLAY ? gate_norm(wsel, {col}) :
      xresidual(p, src, embed{', col' if mt else ''});'''
    text = _once(text, old, new)
    # Capture each pro_attnc variant, including mutated clone, before stage_store.
    pattern = r'(__device__ void pro_attnc\w*\([^\n]+\) \{)(.*?)(\n\})'
    count = 0
    def attention(match):
        nonlocal count
        count += 1
        body = match.group(2)
        stamp = f'''if (gate_attention && blockIdx.x == 0)
      for (int j = 0; j < 8; ++j)
        gate_attention[((int64_t){col} * NLAY + gate_layer) * QROWS + i + j] = __half2float(o[j]);
    SX4_WRITE(idx, o);'''
        if body.count('SX4_WRITE(idx, o);') != 1:
            raise ValueError('candidate attention staging cannot be instrumented')
        return match.group(1) + body.replace('SX4_WRITE(idx, o);', stamp) + match.group(3)
    text = re.sub(pattern, attention, text, flags=re.S)
    if not count:
        raise ValueError('candidate has no attention dump insertion point')
    if mt:
        text = _once(text, 'gemv_run(p.w[l*4+3],cta);gbar(p.bar,expect);',
                     'gemv_run(p.w[l*4+3],cta);gbar(p.bar,expect);\n    for(int m=0;m<MT;++m) gate_dump_hidden(column(p,m).dout,l,m);\n    __syncthreads();')
    else:
        marker = '    }\n    // lm_head: residual += dout + final norm -> sx; logits (f32)'
        text = _once(text, marker,
                     '      gate_dump_hidden(p.dout,l,0);\n      __syncthreads();\n' + marker)
    return text


def build(engine, library, m=1, source=None, defines=None):
    record = HERE / 'cache' / 'sources' / (digest(library) + '.json')
    explicit_flags = defines is not None
    if record.exists():
        explicit_flags |= json.loads(record.read_text()).get('explicit_defines', True)
    source, associated = resolve_source(engine, library, source)
    if not source.is_file():
        raise FileNotFoundError(f'candidate source missing: {source}')
    flags = list(associated) + list(defines or ())
    if engine == 'scale' and not explicit_flags and 'FP32_DOT' not in flags:
        flags.append('FP32_DOT')
    if engine == 'mt':
        flags.extend([f'MT={m}', f'THREADS={512 if m == 5 else 1024}'])
    manifest = {'version': VERSION, 'engine': engine, 'm': m, 'source': str(source),
                'implementation_sha256': digest(Path(__file__)),
                'source_sha256': digest(source), 'library': str(Path(library).resolve()),
                'library_sha256': digest(library), 'defines': flags,
                'headers': {p.name: digest(p) for p in source.parent.glob('*.cuh')}}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    folder = HERE / 'build' / f'debug-{engine}-{key}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'debug.so'
    if output.exists() and (folder / 'manifest.json').exists():
        return output, json.loads((folder / 'manifest.json').read_text())
    copy = folder / 'debug.cu'
    copy.write_text(instrument(source.read_text(), engine))
    for header in source.parent.glob('*.cuh'):
        (folder / header.name).write_bytes(header.read_bytes())
    cuda = Path.home() / '.local/cuda-13.3'
    command = [str(cuda / 'bin/nvcc'), '-O3', '-arch=sm_75', '-std=c++17',
               '-allow-unsupported-compiler', '-L' + str(cuda / 'lib'), '-I' + str(folder),
               '-Xcompiler', '-fPIC', '-shared', '-cudart', 'static', *['-D' + f for f in flags],
               '-o', str(output), str(copy)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (folder / 'build.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f'candidate debug compile failed: {folder / "build.log"}')
    manifest['command'] = command
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return output, manifest


def bind(library, inputs, hidden, attention):
    lib = ctypes.CDLL(str(library))
    try:
        fn = lib.gate_debug_set
    except AttributeError as exc:
        raise RuntimeError('candidate library has no debug dump path') from exc
    fn.argtypes = [ctypes.c_int64] * 3
    fn.restype = ctypes.c_int
    status = fn(inputs.data_ptr() if inputs is not None else 0,
                hidden.data_ptr() if hidden is not None else 0,
                attention.data_ptr() if attention is not None else 0)
    if status:
        raise RuntimeError(f'gate_debug_set CUDA error {status}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['v2', 'scale', 'mt'], required=True)
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--define', action='append')
    parser.add_argument('--m', type=int, default=1)
    args = parser.parse_args()
    if args.source:
        associate(args.lib, args.source, args.define)
    path, manifest = build(args.engine, args.lib, args.m)
    print(json.dumps({'debug_library': str(path), **manifest}))
