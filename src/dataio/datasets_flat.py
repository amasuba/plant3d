from __future__ import annotations
import os, re, glob, math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from .camera_profiles import detect_profile, get_profile, CameraProfile

_DEPTH_MM_TO_M = 1.0 / 1000.0
PLANT_NEAR_M: float = 0.5
PLANT_FAR_M:  float = 4.5



# Default physical rig geometry (metres).
# Camera is fixed; plant rotates on turntable.
ORBIT_RADIUS_M: float = 1.1     # distance from plant centre to camera
ORBIT_ELEVATION_DEG: float = 10.0

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    def as_matrix(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]],dtype=np.float32)

PAT = re.compile(
    r"(?P<angle>\d+)_degrees_(?P<mod>rgb|depth|filtered)"
    r"(?:_cam_(?P<cam>red|green))?_plant_(?P<pid>\d+)", re.IGNORECASE)

def orbit_extrinsics(angle_deg, radius=ORBIT_RADIUS_M, elevation_deg=ORBIT_ELEVATION_DEG):
    """Camera on a sphere of given radius, looking toward the world origin."""
    th = math.radians(angle_deg); el = math.radians(elevation_deg)
    cx = radius*math.cos(el)*math.sin(th)
    cy = radius*math.sin(el)
    cz = radius*math.cos(el)*math.cos(th)
    cam = np.array([cx,cy,cz],dtype=np.float32)
    fwd = -cam/(np.linalg.norm(cam)+1e-8)
    up0 = np.array([0,1,0],dtype=np.float32)
    right = np.cross(fwd,up0); right /= np.linalg.norm(right)+1e-8
    up = np.cross(right,fwd)
    R_cw = np.stack([right,-up,fwd]).astype(np.float32)
    t_cw = (-R_cw@cam).astype(np.float32)
    return R_cw, t_cw

class FlatPlantDataset(Dataset):
    """
    Reads flat-directory turntable RGB-D captures.

    Camera intrinsics are derived from the camera profile (auto-detected or
    specified via camera_key).  Depth values are converted from mm to metres.
    """
    def __init__(self, folder, plant_id, intr_red, intr_green,
                 baseline_m=0.0, cam_order_fallback=True,
                 radius=ORBIT_RADIUS_M, elevation_deg=ORBIT_ELEVATION_DEG,
                 camera_key=None):
        self.folder=folder; self.pid=str(plant_id)
        # Support both dict (legacy) and CameraProfile intrinsics
        if isinstance(intr_red, dict):
            self.K_red=Intrinsics(**intr_red); self.K_green=Intrinsics(**intr_green)
        else:  # CameraProfile
            self.K_red=Intrinsics(fx=intr_red.fx,fy=intr_red.fy,cx=intr_red.cx,cy=intr_red.cy)
            self.K_green=self.K_red
        self.baseline=baseline_m; self.cam_order_fallback=cam_order_fallback
        self.radius=radius; self.elevation_deg=elevation_deg
        self.camera_key=camera_key
        self.items=self._scan()

    def _scan(self):
        by_key={}
        for p in sorted(glob.glob(os.path.join(self.folder,"*.*"))):
            name=os.path.basename(p); m=PAT.search(name)
            if not m: continue
            d=m.groupdict(); mod=d["mod"].lower()
            cam=(d.get("cam") or "").lower() or None
            if d["pid"]!=self.pid: continue
            angle=int(d["angle"]); stem=(angle,cam)
            e=by_key.setdefault(stem,{"angle":angle,"cam":cam,"rgb":None,"depth":None,"filtered":None})
            if mod=="rgb" and p.lower().endswith((".jpg",".jpeg",".png")): e["rgb"]=p
            elif mod=="depth" and p.lower().endswith(".npy"): e["depth"]=p
            elif mod=="filtered" and p.lower().endswith(".npy"): e["filtered"]=p
        by_angle={}
        for (angle,cam),e in by_key.items(): by_angle.setdefault(angle,[]).append(e)
        items=[]
        for angle,lst in by_angle.items():
            lst=sorted(lst,key=lambda x:x["rgb"] or x["depth"] or x["filtered"] or "")
            for i,e in enumerate(lst):
                if e["cam"] is None and self.cam_order_fallback: e["cam"]="red" if i%2==0 else "green"
                if e["rgb"] is None: continue
                items.append(e)
        return sorted(items,key=lambda x:(x["angle"],x["cam"] or ""))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec=self.items[idx]
        rgb=np.array(Image.open(rec["rgb"]).convert("RGB"),dtype=np.uint8)
        depth=None
        dp=rec.get("filtered") or rec.get("depth")
        if dp and dp.lower().endswith(".npy"):
            raw=np.load(dp).astype(np.float32)
            # Auto-detect camera from depth shape if not specified
            if self.camera_key:
                prof=get_profile(self.camera_key)
            else:
                prof=detect_profile(raw.shape[0],raw.shape[1])
            if prof and prof.depth_unit=="mm":
                depth=raw*_DEPTH_MM_TO_M
            elif prof and prof.depth_unit=="m":
                depth=raw
            else:
                depth=raw*_DEPTH_MM_TO_M  # fallback: assume mm
        K=self.K_green if rec["cam"]=="green" else self.K_red
        R,t=orbit_extrinsics(rec["angle"],self.radius,self.elevation_deg)
        return {
            "image":  torch.from_numpy(rgb).permute(2,0,1).float()/255.0,
            "depth":  None if depth is None else torch.from_numpy(depth),
            "K":      torch.from_numpy(K.as_matrix()),
            "R":      torch.from_numpy(R), "t": torch.from_numpy(t),
            "angle":  rec["angle"], "cam": rec["cam"], "path": rec["rgb"],
        }

# Convenience: derive intrinsic dict from camera profile
def intrinsics_from_profile(profile: CameraProfile) -> dict:
    return {"fx": profile.fx, "fy": profile.fy, "cx": profile.cx, "cy": profile.cy}
