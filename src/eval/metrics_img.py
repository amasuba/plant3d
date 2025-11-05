from __future__ import annotations
import torch, torch.nn.functional as F
def psnr(pred, gt, eps=1e-8):
    mse = F.mse_loss(pred, gt)
    return (-10.0 * torch.log10(mse + eps)).item()
