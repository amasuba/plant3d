from __future__ import annotations
import torch, torch.nn as nn
from ..models.refinement.cross_grounding import CrossGrounding
from ..models.volumetric_transformer.heads import Heads
class RefinementModel(nn.Module):
    def __init__(self, c3d=128, c2d=256):
        super().__init__()
        self.xattn = CrossGrounding(c3d, c2d)
        self.heads = Heads(c3d)
    def forward(self, vol_feat, img_feat):
        fused = self.xattn(vol_feat, img_feat)
        sigma, color = self.heads(fused)
        return sigma, color, fused
