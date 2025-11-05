from __future__ import annotations
import numpy as np
from PIL import Image
def save_side_by_side(a, b, path):
    a = (np.clip(a*255,0,255)).astype(np.uint8)
    b = (np.clip(b*255,0,255)).astype(np.uint8)
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1]+b.shape[1]
    out = np.zeros((h,w,3), dtype=np.uint8)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1]] = b
    Image.fromarray(out).save(path)
