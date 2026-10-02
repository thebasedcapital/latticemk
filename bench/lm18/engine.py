"""One weight-sharing pass, either causal columns or independent caches."""
import ctypes
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'bench/lm03'), str(ROOT / 'bench/lm03b')]
import lm03
import lm03b

class Engine:
    def __init__(self, m, mode, packed, emb, norms, rope, variant='', cap=256):
        if not 1 <= m <= 5 or mode not in ('causal', 'batch'):
            raise ValueError('M must be 1..5; mode must be causal or batch')
        self.m, self.mode = m, mode
        self.cap = cap
        self.lib = ctypes.CDLL(str(ROOT / f'kernels/megakernel_mt2/libmt{m}{variant}.so'))
        self.lib.mt_init.argtypes = [ctypes.POINTER(ctypes.c_int64)] * 3 + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2
        self.lib.mt_run.restype = ctypes.c_int
        self.lib.mt_time.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.keep = packed, emb, norms, rope
        caches = m if mode == 'batch' else 1
        specs = {'xn': (m*1024, torch.half), 'qkv': (m*4096, torch.half),
                 'oo': (m*1024, torch.half), 'gu': (m*6144, torch.half),
                 'dout': (m*1024, torch.half), 'kc': (caches*28*cap*1024, torch.half),
                 'vc': (caches*28*cap*1024, torch.half), 'part': (m*32*130, torch.float32),
                 'logits': (m*151936, torch.float32), 'argp': (m*72, torch.float32),
                 'pos': (m, torch.int32), 'bar': (1, torch.int32),
                 'xpad': (m*3072, torch.half), 'tok': (m, torch.int32),
                 'tok_hist': (m*cap, torch.int32)}
        self.bufs = {k: torch.zeros(n, dtype=d, device='cuda') for k,(n,d) in specs.items()}
        names = lm03.matrix_names()
        order = ('xn','qkv','oo','gu','dout','kc','vc','part','logits','argp','pos','bar','xpad')
        self.init_args = (lm03._i64([self.bufs[k].data_ptr() for k in order]),
                          lm03._i64([packed[n]['codes'].data_ptr() for n in names]),
                          lm03._i64([packed[n]['meta'].data_ptr() for n in names]),
                          emb.data_ptr(), norms.data_ptr(), rope.data_ptr(),
                          self.bufs['tok'].data_ptr(), self.bufs['tok_hist'].data_ptr(), mode=='batch', cap)
        rc = self.lib.mt_init(*self.init_args)
        if rc: raise RuntimeError(f'mt_init {rc}')

    def run(self, tokens, positions=None):
        if len(tokens) != self.m: raise ValueError('one input token per column required')
        self.bufs['tok'].copy_(torch.as_tensor(tokens, dtype=torch.int32, device='cuda'))
        if positions is not None:
            if self.mode == 'causal': positions = [int(positions)] * self.m
            self.bufs['pos'].copy_(torch.as_tensor(positions, dtype=torch.int32, device='cuda'))
        torch.cuda.synchronize()
        self.activate()
        rc = self.lib.mt_run()
        if rc: raise RuntimeError(f'mt_run CUDA error {rc}')
        return self.bufs['logits'].view(self.m,151936).clone()

    def time(self, ctx, n=1):
        self.activate()
        out = (ctypes.c_float*n)()
        rc = self.lib.mt_time(ctx-1,n,out)
        if rc: raise RuntimeError(f'mt_time CUDA error {rc}')
        return list(out)
    def time_gemv(self, n=1):
        self.activate()
        self.lib.mt_time_gemv.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        out = (ctypes.c_float*n)()
        rc = self.lib.mt_time_gemv(n,out)
        if rc: raise RuntimeError(f'mt_time_gemv CUDA error {rc}')
        return list(out)

    def activate(self):
        rc = self.lib.mt_init(*self.init_args)
        if rc: raise RuntimeError(f'mt_init {rc}')

def shared():
    packed = torch.load(ROOT/'bench/lm11/weights_int4_gptq.pt',map_location='cpu')
    packed = {k:{f:t.cuda() for f,t in v.items()} for k,v in packed.items()}
    return packed, lm03.load('model.embed_tokens.weight').half().contiguous(), lm03.norm_table(), lm03.make_rope().cuda()
