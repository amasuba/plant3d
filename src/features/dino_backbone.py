from __future__ import annotations
import torch, torch.nn as nn
import timm

class DINOBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.dino",
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
        )
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Optionally unfreeze the last N transformer blocks (H2 ablation).
        # ViT-B/16 has blocks 0-11.  unfreeze_last_n_blocks=2 unfreezes 10,11.
        # This adds ~14 M trainable params, fits comfortably in 8 GB VRAM.
        if not freeze:
            # Full fine-tune (requires >= 16 GB VRAM with vol transformer)
            for p in self.parameters():
                p.requires_grad = True
        elif unfreeze_last_n_blocks > 0:
            total_blocks = 12  # ViT-B/16
            start = total_blocks - unfreeze_last_n_blocks
            unfreeze_prefixes = tuple(
                f"model.blocks.{i}." for i in range(start, total_blocks)
            )
            for name, param in self.named_parameters():
                if name.startswith(unfreeze_prefixes):
                    param.requires_grad = True
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"DINOBackbone: last {unfreeze_last_n_blocks} blocks unfrozen"
                  f" ({n_train/1e6:.1f}M trainable params)")

    def forward(self, x: torch.Tensor):
        feats = self.vit(x)
        return feats[-1]
