from __future__ import annotations
import numpy as np
from skimage import measure
import trimesh
def marching_cubes_from_sigma(sigma, thresh=0.5, spacing=(1.0,1.0,1.0)):
    verts, faces, normals, _ = measure.marching_cubes(sigma, level=thresh, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh
