#!/usr/bin/env python
"""Run a batch of ablation presets and collect metrics into a summary table.

Usage
-----
# Run all ablation presets:
    python scripts/run_ablations.py --data-folder ./dataset/plant_data

# Run a specific subset:
    python scripts/run_ablations.py --presets ablation_pretrain_on ablation_pretrain_off

# Override output root:
    python scripts/run_ablations.py --output-root ./outputs/ablations
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

import yaml

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

from src.pipeline import run_pipeline

ABLATION_PREFIX = "ablation_"


def _load_preset_map() -> Dict[str, Dict[str, Any]]:
    cfg_path = os.path.join(here, "..", "src", "config", "presets.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("Invalid presets.yaml")
    return presets


def _ablation_presets(presets: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(k for k in presets if k.startswith(ABLATION_PREFIX))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch ablation runner for Plant3D.")
    ap.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Specific preset names to run. Defaults to all ablation_* presets.",
    )
    ap.add_argument("--data-folder", type=str, default="./dataset/plant_data")
    ap.add_argument("--plant-id", type=int, default=1)
    ap.add_argument("--output-root", type=str, default="./outputs/ablations")
    ap.add_argument("--baseline", type=float, default=0.40)
    ap.add_argument("--list", action="store_true", help="Print ablation presets and exit.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    presets = _load_preset_map()

    if args.list:
        for name in _ablation_presets(presets):
            flags = ", ".join(f"{k}={v}" for k, v in presets[name].items())
            print(f"  {name:28s} {flags}")
        return

    names = args.presets if args.presets else _ablation_presets(presets)
    if not names:
        print("No ablation presets found. Add presets prefixed with 'ablation_' to presets.yaml.")
        return

    summary_rows: List[Dict[str, Any]] = []

    for name in names:
        if name not in presets:
            print(f"[SKIP] Unknown preset: {name}")
            continue

        cfg = presets[name]
        run_output = os.path.join(args.output_root, name)
        print(f"\n{'='*60}")
        print(f"Running preset: {name}")
        print(f"  config: {cfg}")
        print(f"  output: {run_output}")
        print(f"{'='*60}\n")

        t0 = time.time()
        try:
            result = run_pipeline(
                data_folder=args.data_folder,
                plant_id=args.plant_id,
                output_root=run_output,
                device_name=cfg.get("device", "cpu"),
                epochs=cfg.get("epochs", 0),
                baseline_m=args.baseline,
                render_size=(cfg.get("render_size", 256), cfg.get("render_size", 256)),
                refine_epochs=cfg.get("refine_epochs", 0),
                pretrained_dino=cfg.get("pretrained", True),
                freeze_dino=cfg.get("freeze_dino", True),
            )
            elapsed = time.time() - t0
            metrics = result.get("metrics", {})
            row = {
                "preset": name,
                "status": "ok",
                "elapsed_s": round(elapsed, 1),
                "mse_mean": metrics.get("mse_mean", ""),
                "psnr_mean": metrics.get("psnr_mean", ""),
                "occupancy_ratio": metrics.get("occupancy_ratio", ""),
            }
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[FAIL] {name}: {exc}")
            row = {
                "preset": name,
                "status": f"fail: {exc}",
                "elapsed_s": round(elapsed, 1),
                "mse_mean": "",
                "psnr_mean": "",
                "occupancy_ratio": "",
            }
        summary_rows.append(row)

    # Write summary CSV
    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "ablation_summary.csv")
    fieldnames = ["preset", "status", "elapsed_s", "mse_mean", "psnr_mean", "occupancy_ratio"]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"Ablation summary written to {summary_path}")
    print(f"{'='*60}")
    print(f"\n{'preset':28s} {'status':8s} {'elapsed':>8s} {'MSE':>10s} {'PSNR':>10s} {'occ':>8s}")
    print("-" * 76)
    for r in summary_rows:
        print(
            f"{r['preset']:28s} {r['status']:8s} {r['elapsed_s']:>7.1f}s"
            f" {str(r['mse_mean']):>10s} {str(r['psnr_mean']):>10s} {str(r['occupancy_ratio']):>8s}"
        )


if __name__ == "__main__":
    main()
