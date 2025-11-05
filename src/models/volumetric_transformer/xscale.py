from __future__ import annotations
import torch, torch.nn as nn
from .blocks import VolBlock

class CrossScalePyramid(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.down = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.block_low = VolBlock(dim)
        self.block_high = VolBlock(dim)
    def forward(self, x):
        low = self.down(x)
        low = self.block_low(low)
        high = self.block_high(x)
        up_low = self.up(low)
        if up_low.shape[-3:] != x.shape[-3:]:
            up_low = torch.nn.functional.interpolate(up_low, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return (up_low + high) * 0.5
