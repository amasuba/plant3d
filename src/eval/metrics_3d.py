from __future__ import annotations
import torch
def iou_vox(pred, gt, thresh=0.5):
    p = (pred>=thresh).float()
    g = (gt>=0.5).float()
    inter = (p*g).sum()
    union = (p+g - p*g).sum() + 1e-6
    return (inter/union).item()
def fscore(chamfer_dists, delta=0.01):
    p = (chamfer_dists <= delta).float().mean().item()
    r = p
    return (2*p*r)/(p+r+1e-6)
