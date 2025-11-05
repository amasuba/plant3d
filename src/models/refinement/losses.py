from __future__ import annotations
import torch
from ...geometry.priors import smoothness_l2
def photometric_l1(pred, gt): return (pred-gt).abs().mean()
def silhouette_bce(pred_a, gt_m):
    eps=1e-6
    return -(gt_m*torch.log(pred_a+eps) + (1-gt_m)*torch.log(1-pred_a+eps)).mean()
def geometry_smoothness(sigma): return smoothness_l2(sigma)
