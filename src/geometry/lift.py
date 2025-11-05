from __future__ import annotations
import torch, torch.nn.functional as F

def lift_features_to_grid(feat_map: torch.Tensor, grid_res=(64,64,64)):
    B,C,h,w = feat_map.shape
    D,H,W = grid_res
    f2d = F.interpolate(feat_map, size=(H,W), mode="bilinear", align_corners=False)
    f3d = f2d.unsqueeze(2).repeat(1,1,D,1,1) * torch.linspace(1.0, 0.5, D, device=f2d.device)[None,None,:,None,None]
    return f3d
