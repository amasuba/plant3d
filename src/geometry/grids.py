from __future__ import annotations
import torch

class DenseGrid:
    def __init__(self, res=(64,64,64), feat_dim=256, device="cpu"):
        D,H,W = res
        self.features = torch.zeros((1, feat_dim, D, H, W), device=device)
    def to(self, device):
        self.features = self.features.to(device); return self
