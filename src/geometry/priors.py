from __future__ import annotations
import torch

def smoothness_l2(volume: torch.Tensor):
    dx = (volume[:,:,1:]-volume[:,:,:-1]).pow(2).mean()
    dy = (volume[:,:,:,1:]-volume[:,:,:,:-1]).pow(2).mean()
    dz = (volume[:,:,:,:,1:]-volume[:,:,:,:,:-1]).pow(2).mean()
    return dx+dy+dz
