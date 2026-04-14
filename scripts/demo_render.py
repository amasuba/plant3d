from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

from src.dataio.datasets_flat import FlatPlantDataset
from src.pipeline import estimate_intrinsics, render_volume_image, save_image, scale_intrinsics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Render a preview from saved sigma/color volumes.")
	parser.add_argument("--data-folder", type=str, default="./dataset/plant_data", help="Flat dataset folder.")
	parser.add_argument("--sigma", type=str, default="./outputs/stage_30_volume/sigma.npy", help="Path to sigma .npy")
	parser.add_argument("--color", type=str, default="./outputs/stage_30_volume/color.npy", help="Path to color .npy")
	parser.add_argument("--out", type=str, default="./outputs/stage_50_render/demo_render.png", help="Output render path")
	parser.add_argument("--plant-id", type=int, default=1, help="Plant ID")
	parser.add_argument("--render-size", type=int, default=256, help="Render side length")
	parser.add_argument("--samples", type=int, default=64, help="Ray samples")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	sigma = torch.from_numpy(np.load(args.sigma)).float()
	color = torch.from_numpy(np.load(args.color)).float()
	if sigma.ndim == 4:
		sigma = sigma.unsqueeze(0)
	if color.ndim == 4:
		color = color.unsqueeze(0)

	rgb_files = sorted(
		[
			p
			for p in os.listdir(args.data_folder)
			if "rgb" in p.lower() and p.lower().endswith((".jpg", ".jpeg", ".png"))
		]
	)
	if not rgb_files:
		raise ValueError(f"No RGB files found in {args.data_folder}")

	sample_path = os.path.join(args.data_folder, rgb_files[0])
	sample = torch.from_numpy(np.array(Image.open(sample_path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
	intr = estimate_intrinsics(sample)
	ds = FlatPlantDataset(args.data_folder, plant_id=args.plant_id, intr_red=intr, intr_green=intr)
	first = ds[0]

	side = (args.render_size, args.render_size)
	K = scale_intrinsics(first["K"].float(), first["image"].shape[1:], side)
	rgb, _ = render_volume_image(
		sigma=sigma,
		color=color,
		K=K,
		R=first["R"].float(),
		t=first["t"].float(),
		output_size=side,
		n_samples=args.samples,
	)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	save_image(rgb, args.out)
	print(f"Saved render preview: {args.out}")


if __name__ == "__main__":
	main()
