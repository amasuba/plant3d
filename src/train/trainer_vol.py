from __future__ import annotations
from typing import List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from ..geometry.lift import lift_features_to_grid
from ..models.volumetric_transformer.blocks import VolBlock
from ..models.volumetric_transformer.xscale import CrossScalePyramid
from ..models.volumetric_transformer.heads import Heads

DEFAULT_GRID_RES: Tuple[int,int,int] = (64,64,64)

class VolumetricModel(nn.Module):
    def __init__(self, feat_dim:int=256, vol_dim:int=128, grid_res:Tuple[int,int,int]=DEFAULT_GRID_RES):
        super().__init__()
        self.grid_res = grid_res
        self.embed  = nn.Conv3d(feat_dim, vol_dim, 1)
        self.block  = VolBlock(vol_dim)
        self.xscale = CrossScalePyramid(vol_dim)
        self.heads  = Heads(vol_dim)
    def forward(self, f2d, depth_map=None, depth_near=0.4, depth_far=2.5):
        v = lift_features_to_grid(f2d, depth_map=depth_map, grid_res=self.grid_res,
                                  near=depth_near, far=depth_far)
        v = self.embed(v); v = self.block(v); v = self.xscale(v)
        sigma, color = self.heads(v)
        return sigma, color, v

def multiview_consistency_loss(model, view_features, depth_maps, depth_near=0.4, depth_far=2.5):
    if len(view_features) < 2:
        return torch.tensor(0.0, device=view_features[0].device)
    avg_feat = torch.stack(view_features).mean(0)
    with torch.no_grad():
        sigma_c, _, _ = model(avg_feat, depth_near=depth_near, depth_far=depth_far)
    loss = torch.tensor(0.0, device=view_features[0].device)
    for f2d, depth in zip(view_features, depth_maps):
        sv, _, _ = model(f2d, depth_map=depth, depth_near=depth_near, depth_far=depth_far)
        loss = loss + F.mse_loss(sv, sigma_c.detach())
    return loss / len(view_features)
