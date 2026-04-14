from __future__ import annotations
import argparse
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plant3D end-to-end pipeline on a flat plant dataset.")
    parser.add_argument("--data-folder", type=str, default="./dataset/plant_data", help="Path to flat dataset folder.")
    parser.add_argument("--plant-id", type=int, default=1, help="Plant ID to load from the dataset.")
    parser.add_argument("--output-root", type=str, default="./outputs", help="Root directory for generated stage artifacts.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda).")
    parser.add_argument("--epochs", type=int, default=0, help="Number of volumetric training epochs to run. Use 0 to skip training.")
    parser.add_argument("--baseline", type=float, default=0.40, help="Stereo baseline in meters for flat dataset camera rig.")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered preview image side length.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained DINO weights.")
    parser.add_argument("--no-freeze", dest="freeze_dino", action="store_false", help="Do not freeze DINO backbone weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_size = (args.render_size, args.render_size)
    result = run_pipeline(
        data_folder=args.data_folder,
        plant_id=args.plant_id,
        output_root=args.output_root,
        device_name=args.device,
        epochs=args.epochs,
        baseline_m=args.baseline,
        render_size=render_size,
        pretrained_dino=args.pretrained,
        freeze_dino=args.freeze_dino,
    )
    print("Pipeline finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
