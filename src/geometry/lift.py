from __future__ import annotations
from typing import Optional, Tuple
import torch, torch.nn.functional as F

def lift_features_to_grid(feat_map, depth_map=None, grid_res=(32,32,32), near=0.4, far=2.5):
    B,C,h,w = feat_map.shape
    D,H,W   = grid_res
    f2d = F.interpolate(feat_map, size=(H,W), mode="bilinear", align_corners=False)
    if depth_map is None:
        decay = torch.linspace(1.0, 0.5, D, device=feat_map.device)
        return f2d.unsqueeze(2).repeat(1,1,D,1,1) * decay[None,None,:,None,None]
    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(0).expand(B,-1,-1)
    depth_f = F.interpolate(depth_map.unsqueeze(1).float(), size=(H,W),
                            mode="bilinear", align_corners=False)[:,0]
    d_idx_f = ((depth_f.clamp(near,far)-near)/(far-near))*(D-1)
    d_range  = torch.arange(D, dtype=torch.float32, device=feat_map.device)
    sigma_d  = max(1.5, D/10.0)
    weights  = torch.exp(-0.5*((d_range[None,None,None,:]-d_idx_f.unsqueeze(-1))/sigma_d)**2)
    weights  = weights/(weights.sum(-1,keepdim=True)+1e-6)
    return f2d.unsqueeze(2) * weights.permute(0,3,1,2).unsqueeze(1)
