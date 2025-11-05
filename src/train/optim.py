from __future__ import annotations
import torch
def build_optimizer(params, lr=2e-4, wd=1e-4):
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
def build_scheduler(optim, warmup=100, total=1000):
    def lr_lambda(step):
        if step < warmup: return float(step) / float(max(1, warmup))
        return max(0.0, 0.5*(1 + (1 - (step-warmup)/(total-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
