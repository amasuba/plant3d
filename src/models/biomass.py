from __future__ import annotations
import torch, torch.nn as nn

class BiomassProxy(nn.Module):
    def __init__(self, voxel_size_m=0.01):
        super().__init__()
        self.voxel_volume = voxel_size_m**3
    def forward(self, sigma):
        return sigma.sum(dim=(1,2,3,4)) * self.voxel_volume

class BiomassHead(nn.Module):
    def __init__(self, feat_dim=128, hidden=64):
        super().__init__()
        self.feat_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim+3, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, 1), nn.Softplus())
    def forward(self, sigma, vol_feat):
        feat = self.feat_pool(vol_feat).flatten(1)
        occ_mean = sigma.mean(dim=(1,2,3,4))
        occ_std  = sigma.std(dim=(1,2,3,4))
        occ_max  = sigma.flatten(1).max(dim=1).values
        x = torch.cat([feat, torch.stack([occ_mean,occ_std,occ_max],dim=1)], dim=1)
        return self.mlp(x).squeeze(1)

def calibrate_biomass_head(head, sigma, vol_feat, labels_g, epochs=200, lr=1e-3):
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    stats = {"losses": []}
    for _ in range(epochs):
        pred = head(sigma, vol_feat)
        loss = (pred - labels_g).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        stats["losses"].append(float(loss.item()))
    head.eval()
    return stats
