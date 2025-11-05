from __future__ import annotations
import torch, torch.nn as nn
from einops import rearrange

class Windowed3DAttn(nn.Module):
    def __init__(self, dim, window=(4,4,4), heads=4):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.window = window
    def forward(self, x):
        B,D,H,W,C = x.shape
        wd,wh,ww = self.window
        assert D%wd==0 and H%wh==0 and W%ww==0
        xw = rearrange(x, "b d h w c -> b (d wd) (h wh) (w ww) wd wh ww c", wd=wd, wh=wh, ww=ww)
        qkv = self.qkv(xw).chunk(3, dim=-1)
        q,k,v = qkv
        q = rearrange(q, "b n m l wd wh ww (h ch) -> b n m l h (wd wh ww) ch", h=self.heads)
        k = rearrange(k, "b n m l wd wh ww (h ch) -> b n m l h (wd wh ww) ch", h=self.heads)
        v = rearrange(v, "b n m l wd wh ww (h ch) -> b n m l h (wd wh ww) ch", h=self.heads)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b n m l h s ch -> b n m l s (h ch)")
        out = self.proj(out)
        out = rearrange(out, "b (d) (h) (w) (wd wh ww c) -> b d h w c", d=D//wd, h=H//wh, w=W//ww, wd=wd, wh=wh, ww=ww)
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*hidden), nn.GELU(), nn.Linear(dim*hidden, dim))
    def forward(self, x): return self.net(x)

class VolBlock(nn.Module):
    def __init__(self, dim, window=(4,4,4), heads=4):
        super().__init__()
        self.attn = Windowed3DAttn(dim, window, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        h = self.attn(self.norm1(x)) + x
        h = self.mlp(self.norm2(h)) + h
        return h.permute(0,4,1,2,3)
