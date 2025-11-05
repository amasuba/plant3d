from __future__ import annotations
import torch, torch.nn as nn
from ..geometry.lift import lift_features_to_grid
from ..models.volumetric_transformer.blocks import VolBlock
from ..models.volumetric_transformer.xscale import CrossScalePyramid
from ..models.volumetric_transformer.heads import Heads
class VolumetricModel(nn.Module):
    def __init__(self, feat_dim=256, vol_dim=128):
        super().__init__()
        self.embed = nn.Conv3d(feat_dim, vol_dim, 1)
        self.block = VolBlock(vol_dim)
        self.xscale = CrossScalePyramid(vol_dim)
        self.heads = Heads(vol_dim)
    def forward(self, f2d):
        v = lift_features_to_grid(f2d)
        v = self.embed(v)
        v = self.block(v)
        v = self.xscale(v)
        sigma, color = self.heads(v)
        return sigma, color, v
