#!/usr/bin/env python
"""Clean and smooth a raw marching-cubes mesh for MeshLab."""
import argparse, os, sys
import trimesh

def postprocess(input_path, output_path, smooth_iters=5, smooth_lambda=0.5,
                min_component_ratio=0.01):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh = trimesh.load(input_path, process=False)
    print(f"Raw : {len(mesh.vertices):,} verts  {len(mesh.faces):,} faces")
    # keep large components
    comps = mesh.split(only_watertight=False)
    thresh = min_component_ratio * len(mesh.faces)
    big = [c for c in comps if len(c.faces) >= thresh]
    if not big: big = [max(comps, key=lambda m: len(m.faces))]
    mesh = trimesh.util.concatenate(big)
    # preserve vertex colours if present
    vc = None
    if hasattr(mesh.visual, "vertex_colors"):
        vc = mesh.visual.vertex_colors.copy()
    trimesh.smoothing.filter_laplacian(mesh, lamb=smooth_lambda, iterations=smooth_iters)
    if vc is not None and len(vc) == len(mesh.vertices):
        mesh.visual.vertex_colors = vc
    mesh.remove_unreferenced_vertices()
    print(f"Clean: {len(mesh.vertices):,} verts  {len(mesh.faces):,} faces")
    mesh.export(output_path)
    print(f"Saved: {output_path}")
    return mesh

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smooth-iters",        type=int,   default=5)
    ap.add_argument("--smooth-lambda",       type=float, default=0.5)
    ap.add_argument("--min-component-ratio", type=float, default=0.01)
    args = ap.parse_args()
    postprocess(args.input, args.output,
                args.smooth_iters, args.smooth_lambda, args.min_component_ratio)
