from __future__ import annotations
import torch
def stratified_samples(n_samples=64, near=0.1, far=2.0, device="cpu"):
    t = torch.linspace(near, far, n_samples, device=device)
    mids = 0.5 * (t[:-1] + t[1:])
    lower = torch.cat([t[:1], mids])
    upper = torch.cat([mids, t[-1:]])
    u = torch.rand_like(lower)
    return lower + (upper - lower) * u
