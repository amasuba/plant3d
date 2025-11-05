from __future__ import annotations
import torch, torch.nn.functional as F
def sample_grid(tri_field, pts):
    B,C,D,H,W = tri_field.shape
    grid = pts.view(B,-1,1,1,3)*2-1
    grid = grid[..., [2,1,0]]
    grid = grid.view(B, -1, 1, 1, 3)
    sampled = F.grid_sample(tri_field, grid, align_corners=False)
    return sampled.view(B, C, -1)
def volume_render(sigma, color, alphas):
    B,_,K = sigma.shape
    T = torch.cumprod(torch.cat([torch.ones(B,1, device=sigma.device), (1-alphas)[:,:-1]], dim=1), dim=1)
    weights = (alphas * T)
    rgb = (color * weights.unsqueeze(1)).sum(dim=-1)
    acc = weights.sum(dim=-1, keepdim=True)
    return rgb, acc
