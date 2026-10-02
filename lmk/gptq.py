"""Hessian-aware rounding: GPTQ generalised to d-dim vector codes (block LDLQ, as in QuIP#).

For a column block B quantized jointly, with U = upper Cholesky factor of H^-1, the remaining columns R get
    W_R -= E_B U_BB^-1 U_BR          (E_B = W_B - Q_B);  d = 1 reduces to plain GPTQ.
Group scales are chosen when their 128-column group is reached, from the error-updated weights.
"""

import torch

from lmk.codebooks import nearest
from lmk.quant import GROUP


def h_key(name: str) -> str:
    """Linears that read the same input share a Hessian (q/k/v, gate/up)."""
    return name.replace("k_proj", "q_proj").replace("v_proj", "q_proj").replace("up_proj", "gate_proj")


def calib_hessians(model, windows: torch.Tensor, passes: int = 2) -> dict[str, torch.Tensor]:
    """H = E[x x^T] per distinct Linear input (keyed by h_key) over calibration windows [n, seq]; returned on CPU.

    All Hessians together are ~1.8 GB fp32, so layers are split over `passes` full forward passes.
    """
    n_layers = len(model.model.layers)
    out = {}
    for p in range(passes):
        lo, hi = p * n_layers // passes, (p + 1) * n_layers // passes
        hs, hooks = {}, []

        def acc(name, x):
            x = x.reshape(-1, x.shape[-1]).float()
            hs[name] = hs.get(name, 0) + x.T @ x

        for name, m in model.model.named_modules():
            if (isinstance(m, torch.nn.Linear) and h_key(name) == name
                    and lo <= int(name.split(".")[1]) < hi):
                hooks.append(m.register_forward_hook(lambda _m, inp, _o, n=f"model.{name}": acc(n, inp[0])))
        with torch.no_grad():
            for w in windows:
                h = model.model(w.unsqueeze(0).cuda()).last_hidden_state
                if p == passes - 1:
                    acc("lm_head", h)  # from hidden states: no [seq, vocab] logits
        for hk in hooks:
            hk.remove()
        out.update({n: (h / windows.numel()).cpu() for n, h in hs.items()})
        del hs
        torch.cuda.empty_cache()
    return out


def _prep(h: torch.Tensor, w: torch.Tensor, damp: float):
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp * torch.diag(h).mean())
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(hinv, upper=True)


def _ldlq(w, h, d, damp, start_group, quant_block):
    """Shared driver. quant_block(wb [out, d], col) -> qb; start_group(wg [out, GROUP], g) sets group params."""
    w = w.clone()
    out, n = w.shape
    u = _prep(h, w, damp)
    q = torch.zeros_like(w)
    for g0 in range(0, n, GROUP):
        g1 = g0 + GROUP
        start_group(w[:, g0:g1], g0 // GROUP)
        scaled = torch.zeros(out, GROUP, device=w.device)
        for b0 in range(g0, g1, d):
            b1 = b0 + d
            qb = quant_block(w[:, b0:b1], b0)
            q[:, b0:b1] = qb
            ubb_inv = torch.linalg.inv(u[b0:b1, b0:b1])  # d x d upper-triangular
            s = (w[:, b0:b1] - qb) @ ubb_inv
            scaled[:, b0 - g0 : b1 - g0] = s
            w[:, b1:g1] -= s @ u[b0:b1, b1:g1]
        w[:, g1:] -= scaled @ u[g0:g1, g1:]
    return q


def gptq_int(w: torch.Tensor, h: torch.Tensor, bits: int, damp: float = 0.01, shrinks=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)):
    qmax = (1 << bits) - 1
    st = {}

    def start_group(wg, _g):
        lo, hi = wg.amin(1, keepdim=True), wg.amax(1, keepdim=True)
        best = None
        for s in shrinks:
            scale = ((hi - lo) * s / qmax).clamp_min(1e-10).half().float()
            zp = torch.round(-lo * s / scale).clamp(0, qmax)
            off = (-zp * scale).half().float()  # stored format: w = q * scale + off (fp16 off)
            rec = torch.clamp(torch.round(wg / scale) + zp, 0, qmax) * scale + off
            err = (rec - wg).pow(2).sum(1, keepdim=True)
            if best is None:
                best = [err, scale, zp, off]
            else:
                take = err < best[0]
                best = [torch.where(take, a, b) for a, b in zip((err, scale, zp, off), best)]
        st["scale"], st["zp"], st["off"] = best[1], best[2], best[3]

    def quant_block(wb, _c):
        return torch.clamp(torch.round(wb / st["scale"]) + st["zp"], 0, qmax) * st["scale"] + st["off"]

    return _ldlq(w, h, 1, damp, start_group, quant_block)


def gptq_vq(w: torch.Tensor, h: torch.Tensor, cb: torch.Tensor, damp: float = 0.01, alphas=None):
    d = cb.shape[1]
    alphas = torch.linspace(0.6, 1.5, 19).tolist() if alphas is None else alphas
    st = {}

    def start_group(wg, _g):
        rms = wg.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-10)
        best_err = best_scale = None
        for a in alphas:
            scale = (rms * a).half().float()
            idx = nearest((wg / scale).reshape(-1, d), cb).long()
            err = ((cb[idx].reshape(wg.shape) * scale - wg) ** 2).sum(1, keepdim=True)
            if best_err is None:
                best_err, best_scale = err, scale
            else:
                take = err < best_err
                best_err, best_scale = torch.where(take, err, best_err), torch.where(take, scale, best_scale)
        st["scale"] = best_scale

    def quant_block(wb, _c):
        return cb[nearest(wb / st["scale"], cb).long()] * st["scale"]

    return _ldlq(w, h, d, damp, start_group, quant_block)
