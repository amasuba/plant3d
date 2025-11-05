from __future__ import annotations
import torch, torch.nn as nn
from einops import rearrange
class CrossGrounding(nn.Module):
    def __init__(self, c3d=128, c2d=256, heads=4):
        super().__init__()
        self.q = nn.Linear(c3d, c3d)
        self.k = nn.Linear(c2d, c3d)
        self.v = nn.Linear(c2d, c3d)
        self.proj = nn.Linear(c3d, c3d)
        self.heads = heads
        self.scale = (c3d // heads) ** -0.5
    def forward(self, vol_feat, img_feat):
        B,C,D,H,W = vol_feat.shape
        N = D*H*W
        q = self.q(vol_feat.permute(0,2,3,4,1).reshape(B,N,C))
        k = self.k(img_feat)
        v = self.v(img_feat)
        h = self.heads
        q = rearrange(q, "b n (h c) -> b h n c", h=h)
        k = rearrange(k, "b m (h c) -> b h m c", h=h)
        v = rearrange(v, "b m (h c) -> b h m c", h=h)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n c -> b n (h c)")
        out = self.proj(out)
        out = out.view(B,D,H,W,C).permute(0,4,1,2,3)
        return out
