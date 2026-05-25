from __future__ import annotations
import argparse
import os
import sys
from typing import Any, Dict, Optional

import yaml

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

from src.pipeline import run_pipeline


DEFAULTS: Dict[str, Any] = {
    "data_folder": "./dataset/plant_data",
    "plant_id": 1,
    "output_root": "./outputs",
    "device": "cpu",
    "epochs": 0,
    "baseline": 0.40,
    "render_size": 256,
    "refine_epochs": 1,
    "pretrained": True,
    "freeze_dino": True,
    "lambda_consistency": 0.1,
    "use_ema_refine": True,
    "biomass_labels_csv": None,
    "biomass_calibration_epochs": 200,
}


def _load_preset_map() -> Dict[str, Dict[str, Any]]:
    cfg_path = os.path.abspath(os.path.join(here, "..", "src", "config", "presets.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("Invalid presets.yaml: expected top-level 'presets' mapping")
    return presets


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    values: Dict[str, Any] = dict(DEFAULTS)
    presets = _load_preset_map()

    if args.preset is not None:
        if args.preset not in presets:
            raise ValueError(
                f"Unknown preset '{args.preset}'. Available: {', '.join(sorted(presets.keys()))}"
            )
        values.update(presets[args.preset])

    # Explicit CLI values override preset values.
    for key in values.keys():
        if getattr(args, key, None) is not None:
            values[key] = getattr(args, key)

    for key, value in values.items():
        setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plant3D end-to-end pipeline on a flat plant dataset.")
    parser.add_argument("--preset", type=str, default=None, help="Preset from src/config/presets.yaml (e.g., fast, research).")
    parser.add_argument("--list-presets", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--data-folder", type=str, default=None, help="Path to flat dataset folder.")
    parser.add_argument("--plant-id", type=int, default=None, help="Plant ID to load from the dataset.")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for generated stage artifacts.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu or cuda).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of volumetric training epochs to run. Use 0 to skip training.")
    parser.add_argument("--baseline", type=float, default=None, help="Stereo baseline in meters for flat dataset camera rig.")
    parser.add_argument("--render-size", type=int, default=None, help="Rendered preview image side length.")
    parser.add_argument("--refine-epochs", type=int, default=None, help="Number of geometry refinement epochs.")

    parser.add_argument("--lambda-consistency", type=float, default=None, help="Weight for multi-view self-supervised consistency loss (0 to disable).")
    parser.add_argument("--biomass-labels-csv", type=str, default=None, help="CSV with columns [plant_id, biomass_g] for BiomassHead calibration.")
    parser.add_argument("--biomass-calibration-epochs", type=int, default=None, help="Fine-tuning epochs for BiomassHead.")

    pretrained_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--lambda-consistency", type=float, default=None, help="Weight for multi-view self-supervised consistency loss (0 to disable).")
    parser.add_argument("--biomass-labels-csv", type=str, default=None, help="CSV with columns [plant_id, biomass_g] for BiomassHead calibration.")
    parser.add_argument("--biomass-calibration-epochs", type=int, default=None, help="Fine-tuning epochs for BiomassHead.")

    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pretrained DINO weights.")
    parser.add_argument("--lambda-consistency", type=float, default=None, help="Weight for multi-view self-supervised consistency loss (0 to disable).")
    parser.add_argument("--biomass-labels-csv", type=str, default=None, help="CSV with columns [plant_id, biomass_g] for BiomassHead calibration.")
    parser.add_argument("--biomass-calibration-epochs", type=int, default=None, help="Fine-tuning epochs for BiomassHead.")

    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained DINO weights.")
    parser.set_defaults(pretrained=None)

    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze", dest="freeze_dino", action="store_true", help="Freeze DINO backbone weights.")
    freeze_group.add_argument("--no-freeze", dest="freeze_dino", action="store_false", help="Do not freeze DINO backbone weights.")
    parser.set_defaults(freeze_dino=None)

    raw = parser.parse_args()

    if raw.list_presets:
        presets = _load_preset_map()
        print("Available presets:\n")
        for name, vals in sorted(presets.items()):
            flags = ", ".join(f"{k}={v}" for k, v in vals.items())
            print(f"  {name:28s} {flags}")
        raise SystemExit(0)

    return _resolve_args(raw)


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
        refine_epochs=args.refine_epochs,
        pretrained_dino=args.pretrained,
        freeze_dino=args.freeze_dino,
    )
    print("Pipeline finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

