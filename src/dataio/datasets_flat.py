# src/dataio/datasets_flat.py
from __future__ import annotations
import os, re, glob, json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

# --- simple 3x3 intrinsics structure for Kinect (edit to your calibration) ---
@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    def as_matrix(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float32)

# Regex to parse names like:
# "90_degrees_rgb_plant_1.jpg", "0_degrees_depth_plant_1.npy",
# or optional camera tag: "90_degrees_rgb_cam_red_plant_1.jpg"
PAT = re.compile(
    r'(?P<angle>\d+)_degrees_(?P<mod>rgb|depth)'
    r'(?:_cam_(?P<cam>red|green))?'
    r'_plant_(?P<pid>\d+)'
)

def yaw_matrix_deg(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c,s = np.cos(th), np.sin(th)
    R = np.array([[ c,0, s],
                  [ 0,1, 0],
                  [-s,0, c]], dtype=np.float32)  # y-up yaw
    return R

def world_to_cam(R_wc: np.ndarray, t_wc: np.ndarray):
    # Return [R|t] that maps world->camera (OpenCV convention)
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc
    return R_cw, t_cw

class FlatPlantDataset(Dataset):
    """
    Reads a flat directory and parses angle/modality/plant_id(/camera).
    Builds extrinsics by yawing the rig by 'angle' degrees around Y.
    Camera in rig is offset +/- baseline/2 along X for red/green.
    """
    def __init__(
        self,
        folder: str,
        plant_id: int,
        intr_red: Dict[str,float],
        intr_green: Dict[str,float],
        baseline_m: float = 0.40,             # set your real baseline
        cam_order_fallback: bool = True       # if filenames lack cam tag
    ):
        self.folder = folder
        self.pid = str(plant_id)
        self.K_red = Intrinsics(**intr_red)
        self.K_green = Intrinsics(**intr_green)
        self.baseline = baseline_m
        self.cam_order_fallback = cam_order_fallback

        self.items = self._scan()

    def _scan(self):
        images = sorted(glob.glob(os.path.join(self.folder, "*.*")))
        # pair rgb and depth by stem (angle, plant, optional cam)
        by_key = {}
        for p in images:
            name = os.path.basename(p)
            m = PAT.search(name)
            if not m: 
                continue
            d = m.groupdict()
            if d["pid"] != self.pid:
                continue
            angle = int(d["angle"])
            mod   = d["mod"]
            cam   = d.get("cam")  # may be None
            stem  = (angle, cam)
            entry = by_key.setdefault(stem, {"angle": angle, "cam": cam, "rgb": None, "depth": None})
            if mod == "rgb" and p.lower().endswith((".jpg",".jpeg",".png")):
                entry["rgb"] = p
            elif mod == "depth" and p.lower().endswith(".npy"):
                entry["depth"] = p

        # If cam tag missing, assign alternating files to red/green deterministically per angle
        # (or you can hardcode a mapping file).
        # We split by angle and order.
        by_angle = {}
        for (angle, cam), e in by_key.items():
            by_angle.setdefault(angle, []).append(e)

        items = []
        for angle, lst in by_angle.items():
            lst = sorted(lst, key=lambda x: (x["rgb"] or x["depth"] or ""))  # deterministic
            for i, e in enumerate(lst):
                cam = e["cam"]
                if cam is None and self.cam_order_fallback:
                    cam = "red" if (i % 2 == 0) else "green"
                e["cam"] = cam
                if e["rgb"] is None:
                    continue  # require rgb
                items.append(e)
        return sorted(items, key=lambda x: (x["angle"], x["cam"] or ""))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        rgb = np.array(Image.open(rec["rgb"]).convert("RGB"), dtype=np.uint8)

        # Optional depth (.npy meters)
        depth = None
        if rec["depth"] is not None:
            depth = np.load(rec["depth"]).astype(np.float32)  # HxW (meters)

        # Intrinsics per camera
        if rec["cam"] == "green":
            K = self.K_green
            x_offset = -self.baseline/2.0
        else:
            K = self.K_red
            x_offset = +self.baseline/2.0

        # Rig-to-world at given angle (plant at world origin)
        R_wr = yaw_matrix_deg(rec["angle"])
        t_wr = np.array([0,0,0], dtype=np.float32)

        # Camera-in-rig: translate along X by +/- baseline/2
        R_rc = np.eye(3, dtype=np.float32)
        t_rc = np.array([x_offset, 0, 0], dtype=np.float32)

        # Compose world->camera: T_wc = T_rc ∘ T_wr
        R_wc = R_rc @ R_wr
        t_wc = R_rc @ t_wr + t_rc
        R, t = world_to_cam(R_wc, t_wc)

        sample = {
            "image": torch.from_numpy(rgb).permute(2,0,1).float()/255.0,   # (3,H,W)
            "depth": None if depth is None else torch.from_numpy(depth),   # (H,W)
            "K": torch.from_numpy(K.as_matrix()),                          # (3,3)
            "R": torch.from_numpy(R), "t": torch.from_numpy(t),            # (3,3), (3,)
            "angle": rec["angle"], "cam": rec["cam"], "path": rec["rgb"]
        }
        return sample
