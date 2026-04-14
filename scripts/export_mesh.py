from __future__ import annotations

import argparse
import os
import sys

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

from src.eval.mesh import marching_cubes_from_sigma


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export a mesh from a saved sigma volume.")
	parser.add_argument("--sigma", type=str, required=True, help="Path to sigma .npy volume.")
	parser.add_argument("--out", type=str, required=True, help="Output mesh path (.ply or .obj).")
	parser.add_argument("--thresh", type=float, default=0.5, help="Marching cubes threshold.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	sigma = np.load(args.sigma)
	if sigma.ndim == 5:
		sigma = sigma[0, 0]
	elif sigma.ndim == 4:
		sigma = sigma[0]
	elif sigma.ndim != 3:
		raise ValueError(f"Expected 3D/4D/5D sigma volume, got shape {sigma.shape}")

	mesh = marching_cubes_from_sigma(sigma, thresh=args.thresh)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	mesh.export(args.out)
	print(f"Exported mesh: {args.out}")


if __name__ == "__main__":
	main()
