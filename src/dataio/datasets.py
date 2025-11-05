from __future__ import annotations
import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from .camera import load_intrinsics, load_pose

class MultiViewPlantDataset(Dataset):
    def __init__(self, data_root: str, scene_id: str):
        self.root = os.path.join(data_root, scene_id)
        self.rgb_paths = sorted(glob.glob(os.path.join(self.root, "rgb", "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.root, "mask", "*.png")))
        self.pose_paths = sorted(glob.glob(os.path.join(self.root, "poses", "*.json")))
        self.K = load_intrinsics(os.path.join(self.root, "intrinsics.json"))
        assert len(self.rgb_paths) == len(self.pose_paths), "RGB and pose count mismatch"
    def __len__(self): return len(self.rgb_paths)
    def __getitem__(self, idx):
        rgb = np.array(Image.open(self.rgb_paths[idx]).convert("RGB"))
        mask = None
        if len(self.mask_paths)>idx:
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))/255.0
        E = load_pose(self.pose_paths[idx])
        sample = {
            "image": torch.from_numpy(rgb).permute(2,0,1).float()/255.0,
            "mask": None if mask is None else torch.from_numpy(mask).float(),
            "pose": E, "K": self.K,
            "path": self.rgb_paths[idx],
        }
        return sample
