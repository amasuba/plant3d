from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float

@dataclass
class Extrinsics:
    R: np.ndarray
    t: np.ndarray

def load_intrinsics(path: str) -> Intrinsics:
    with open(path, 'r') as f:
        j = json.load(f)
    return Intrinsics(j['fx'], j['fy'], j['cx'], j['cy'])

def load_pose(path: str) -> Extrinsics:
    with open(path, 'r') as f:
        j = json.load(f)
    R = np.array(j['R'], dtype=np.float32).reshape(3,3)
    t = np.array(j['t'], dtype=np.float32).reshape(3)
    return Extrinsics(R, t)
