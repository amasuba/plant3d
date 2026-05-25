from __future__ import annotations
import torch

def mae(pred, gt): return (pred-gt).abs().mean().item()
def rmse(pred, gt): return (pred-gt).pow(2).mean().sqrt().item()
def mare(pred, gt, eps=1.0): return ((pred-gt).abs()/(gt.abs()+eps)).mean().item()
def biomass_metrics(pred, gt): return {"mae_g":mae(pred,gt),"rmse_g":rmse(pred,gt),"mare":mare(pred,gt)}
