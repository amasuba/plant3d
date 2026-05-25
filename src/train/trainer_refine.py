from __future__ import annotations
import torch, torch.nn as nn
from ..models.refinement.cross_grounding import CrossGrounding
from ..models.volumetric_transformer.heads import Heads
from ..features.dino_utils import EMA

class RefinementModel(nn.Module):
    def __init__(self, c3d=128, c2d=256, use_ema=True):
        super().__init__()
        self.xattn = CrossGrounding(c3d, c2d)
        self.heads  = Heads(c3d)
        self.use_ema = use_ema
        self._ema = None
    def init_ema(self, decay=0.99):
        if self.use_ema: self._ema = EMA(self, decay=decay)
    def update_ema(self):
        if self._ema: self._ema.update(self)
    @torch.no_grad()
    def teacher_forward(self, vol_feat, img_feat):
        if self._ema is None: raise RuntimeError("Call init_ema() first.")
        st = {k:v.clone() for k,v in self.state_dict().items()}
        self._ema.copy_to(self)
        out = self.forward(vol_feat, img_feat)
        self.load_state_dict(st)
        return out
    def forward(self, vol_feat, img_feat):
        fused = self.xattn(vol_feat, img_feat)
        sigma, color = self.heads(fused)
        return sigma, color, fused
