from __future__ import annotations
import torch, torch.nn as nn
import timm

class DINOBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224.dino", freeze: bool = True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0, features_only=True)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        feats = self.vit(x)
        return feats[-1]
