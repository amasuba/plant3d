# Plant3D: Executable Architecture (April 2026)

This README reflects the architecture that currently runs in code.

The active pipeline combines:
1. DINO feature extraction from multi-view RGB images.
2. Volumetric transformer prediction of density/color fields.
3. Optional refinement with 2D-3D cross-grounding and geometry smoothness prior.
4. Differentiable rendering-based supervision/evaluation.
5. Mesh extraction with marching cubes.

## Current System Flow

```text
Flat RGB dataset
  -> intrinsics estimate + synthetic camera poses
  -> DINO features (stage_20_dino)
  -> lift 2D features to 3D grid
  -> volumetric transformer (sigma, color)
  -> optional volumetric training (stage_40_vol_train)
  -> optional refinement training (stage_60_refine)
  -> render previews (stage_50_render)
  -> eval metrics.csv + report.txt (stage_70_eval)
  -> mesh export .ply (stage_60_refine)
```

Primary runtime entrypoint:
1. `scripts/run_pipeline.py`
2. `src/pipeline.py`

## Install

```bash
git clone https://github.com/amasuba/plant3d.git
cd plant3d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run the full executable path with light defaults:

```bash
python scripts/run_pipeline.py \
  --data-folder ./dataset/plant_data \
  --epochs 0 \
  --refine-epochs 1 \
  --device cpu \
  --render-size 64 \
  --no-pretrained
```

Use built-in experiment presets:

```bash
python scripts/run_pipeline.py --preset fast
python scripts/run_pipeline.py --preset research
```

Useful flags:
1. `--preset`: loads defaults from `src/config/presets.yaml`.
1. `--epochs`: volumetric training epochs (0 skips).
2. `--refine-epochs`: refinement epochs (0 skips).
3. `--no-pretrained`: disables pretrained DINO weights.
4. `--no-freeze`: allows DINO gradients.

Preset values can be overridden by explicit CLI flags.

### Available Presets

| Preset | Purpose | Device | Epochs | Refine | Render | VRAM |
|---|---|---|---|---|---|---|
| `fast` | Quick iteration / CI | cpu | 0 | 1 | 64 | — |
| `research` | Full experiment | cuda | 12 | 4 | 256 | ≈8 GB |
| `research_light` | Lighter GPU run | cuda | 6 | 2 | 128 | ≈4 GB |

List all presets (including ablation):

```bash
python scripts/run_pipeline.py --list-presets
```

### Ablation Studies

Four ablation axes are pre-configured:

| ID | Hypothesis | Presets |
|---|---|---|
| A1 | Pretrained DINO features improve reconstruction | `ablation_pretrain_on` / `ablation_pretrain_off` |
| A2 | Freezing the backbone retains transfer quality | `ablation_freeze_on` / `ablation_freeze_off` |
| A3 | Cross-grounding refinement improves geometry | `ablation_refine_on` / `ablation_refine_off` |
| A4 | More volumetric epochs improve supervision | `ablation_epochs_short` / `ablation_epochs_long` |

Run all ablations with one command:

```bash
python scripts/run_ablations.py --data-folder ./dataset/plant_data
```

Or run a specific pair:

```bash
python scripts/run_ablations.py --presets ablation_pretrain_on ablation_pretrain_off
```

Results are collected into `outputs/ablations/ablation_summary.csv`.

## Input Data Format (Implemented)

The active runner uses `FlatPlantDataset` with flat filenames like:
1. `0_degrees_RGB_plant_1.jpg`
2. `90_degrees_depth_plant_1.npy`
3. Optional camera tag: `..._cam_red_...` or `..._cam_green_...`

Depth can be `.npy` or image files (`.png/.jpg/.jpeg`).

## Output Artifacts

Generated under `outputs/`:
1. `stage_20_dino`: per-view feature files (`view_XX.npz`).
2. `stage_30_volume`: `sigma.npy`, `color.npy`.
3. `stage_40_vol_train`: volumetric checkpoints and loss log (if enabled).
4. `stage_50_render`: rendered preview image and alpha accumulation map.
5. `stage_60_refine`: refinement checkpoints (if enabled) and mesh.
6. `stage_70_eval`: `metrics.csv` and `report.txt`.

## Utility Scripts

1. `scripts/export_mesh.py`: exports `.ply/.obj` from a saved sigma volume.
2. `scripts/demo_render.py`: renders a preview image from saved sigma/color volumes.

## Implemented vs Planned

Implemented in runtime path:
1. DINO features.
2. Volumetric transformer inference/training.
3. Refinement model training.
4. Render-space metrics: MSE and PSNR.
5. Mesh extraction.

Present in repository but not wired into the main `run_pipeline` path yet:
1. Scene-folder calibration dataset path (`MultiViewPlantDataset`).
2. Full 3D metrics pipeline (IoU/Chamfer/F-score end-to-end integration).
3. Notebook-first capture/calibration outputs as required runtime inputs.

## Notes on Diagram Alignment

`sys_diagram.pdf` is currently conceptual. The executable architecture is the flow documented above and implemented in `src/pipeline.py`.

## License

University of Pretoria copyright policy applies.

