from __future__ import annotations
from typing import Dict, Any
import yaml
from torch.utils.data import DataLoader
from .datasets import MultiViewPlantDataset

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_dataloader(cfg: Dict[str, Any]):
    data_root = cfg['data']['root']
    scene = cfg['data']['scene_list'][0]
    ds = MultiViewPlantDataset(data_root, scene)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
