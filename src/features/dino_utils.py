from __future__ import annotations
import torch

class EMA:
    def __init__(self, model, decay=0.996):
        self.shadow = {k: v.clone().detach() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
