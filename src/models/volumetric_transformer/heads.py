from __future__ import annotations
import torch, torch.nn as nn

class Heads(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.density = nn.Conv3d(in_dim, 1, 1)
        self.color   = nn.Conv3d(in_dim, 3, 1)
        # Bias density toward near-zero at init: sigmoid(-5) ~ 0.007
        nn.init.constant_(self.density.bias, -2.0)
    def forward(self, x):
        sigma = torch.sigmoid(self.density(x))
        color = torch.sigmoid(self.color(x))
        return sigma, color
