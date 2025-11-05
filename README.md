# Dual-Transformer Image→3D for Fruit Plants

Reconstruct botanically faithful **3D plant geometry** from multi-view RGB (±Depth). The system couples a **DINO ViT** backbone, a **Volumetric Transformer** for 3D reasoning, **Geometry-Grounded Refinement**, and a **Differentiable Renderer**. Each stage produces **viewable artifacts** (images, videos, meshes) for quick validation in **Google Colab** or **Kaggle**.

> **Highlights**
>
> * Self-supervised DINO features → robust view-invariant tokens
> * 2D→3D multi-view **lifting** into a (sparse) voxel grid
> * **Volumetric Transformer** with windowed 3D attention + ray-slice attention
> * **Refinement** via cross-attention (2D↔3D) + differential-geometry & botanical priors
> * End-to-end **rendering supervision** (no 3D GT required)

---

## 📁 Repository Structure

```
plant3d/
├─ notebooks/
│  ├─ 00_quickstart_colab.ipynb
│  ├─ 00_quickstart_kaggle.ipynb
│  ├─ 10_capture_and_calibration.ipynb
│  ├─ 20_dino_precompute_features.ipynb
│  ├─ 30_lift_2d_to_3d_volume.ipynb
│  ├─ 40_volumetric_transformer_train.ipynb
│  ├─ 50_renderer_view_synthesis.ipynb
│  ├─ 60_geometry_refinement_train.ipynb
│  └─ 70_evaluation_and_visualization.ipynb
├─ src/
│  ├─ config/
│  │  ├─ base.yaml
│  │  ├─ colab.yaml
│  │  └─ kaggle.yaml
│  ├─ dataio/
│  │  ├─ datasets.py          # MV dataset, masks, depth
│  │  ├─ camera.py            # K/E structs, pose I/O, helpers
│  │  └─ loaders.py           # dataloaders, caching
│  ├─ features/
│  │  ├─ dino_backbone.py     # DINO student/teacher wrappers
│  │  └─ dino_utils.py        # multi-crop, EMA, centering
│  ├─ geometry/
│  │  ├─ lift.py              # 2D→3D projection, voxel agg
│  │  ├─ grids.py             # dense/sparse octree grids
│  │  └─ priors.py            # smoothness/curvature/connectivity
│  ├─ models/
│  │  ├─ volumetric_transformer/
│  │  │  ├─ blocks.py         # windowed 3D attn, ray-slice attn
│  │  │  ├─ xscale.py         # cross-scale fusion pyramid
│  │  │  └─ heads.py          # σ (occupancy), c (color)
│  │  ├─ refinement/
│  │  │  ├─ cross_grounding.py # 2D↔3D cross-attention
│  │  │  └─ losses.py          # geometry-aware losses
│  │  └─ renderer/
│  │     ├─ rays.py           # sampling, stratified along t
│  │     └─ render.py         # α-compositing, color/depth/opacity
│  ├─ train/
│  │  ├─ trainer_vol.py       # train volumetric transformer
│  │  ├─ trainer_refine.py    # train refinement + priors
│  │  └─ optim.py             # schedulers, mixed precision, EMA
│  ├─ eval/
│  │  ├─ metrics_3d.py        # IoU, Chamfer-L2, F-score
│  │  ├─ metrics_img.py       # PSNR, SSIM
│  │  └─ mesh.py              # marching cubes, exports (PLY/OBJ)
│  └─ viz/
│     ├─ gallery.py           # grids, side-by-side, GIF/MP4
│     └─ tensorboard.py
├─ outputs/
│  ├─ stage_10_capture/       # rectified images, K/E, masks
│  ├─ stage_20_dino/          # feature maps (H/8×W/8×d)
│  ├─ stage_30_volume/        # voxel features (npz), previews
│  ├─ stage_40_vol_train/     # ckpts, TB logs, novel views
│  ├─ stage_50_render/        # rendered novel views, depth
│  ├─ stage_60_refine/        # refined volumes, meshes
│  └─ stage_70_eval/          # metrics.csv, plots
├─ scripts/
│  ├─ prepare_colab.sh
│  ├─ prepare_kaggle.sh
│  ├─ export_mesh.py
│  └─ demo_render.py
├─ requirements.txt
├─ pyproject.toml             # or setup.cfg / setup.py
├─ README.md                  # (this file)
└─ LICENSE

```

---

## 🚀 Quickstart

### Option A — Google Colab

1. Open `notebooks/00_quickstart_colab.ipynb`.
2. Run the setup cell (installs `requirements.txt`, optional Drive mount).
3. Set `DATA_ROOT` and `RUN_NAME` in the config cell.
4. Execute stages sequentially or jump to the desired notebook.

### Option B — Kaggle

1. In a Kaggle Notebook:

   ```bash
   !git clone https://github.com/<you>/plant3d.git
   %cd plant3d
   !pip -q install -r requirements.txt
   ```
2. Open `notebooks/00_quickstart_kaggle.ipynb`, set `DATA_ROOT=/kaggle/input/<dataset>`.

---

## 🔧 Installation (local)

```bash
git clone https://github.com/<you>/plant3d.git
cd plant3d
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Edit one of:

* `src/config/base.yaml` – defaults
* `src/config/colab.yaml` – paths & small GPU settings
* `src/config/kaggle.yaml` – Kaggle paths

**Example (`src/config/base.yaml`):**

```yaml
data:
  root: /path/to/DATA_ROOT
  scene_list: [sample_plant_001]
grid:
  type: sparse_octree
  base_res: 64
  max_res: 128
dino:
  pretrained: "facebook/dino-vitb16"
  freeze: true
train:
  epochs: 50
  batch_size: 1
  lr: 2.0e-4
renderer:
  n_samples: 64
loss:
  lambda_sil: 1.0
  lambda_depth: 0.2
  lambda_geom: 0.5
  lambda_dino: 0.1
```

---

## 🧱 Stages & Outputs (what you’ll see)

1. **Capture & Calibration** → rectified images, verified **K/E**, QC mosaics
2. **DINO Features** → dense maps, attention visualizations
3. **2D→3D Lift** → (sparse) voxel feature volumes, slice previews
4. **Volumetric Transformer** → occupancy/color fields, novel-view renders
5. **Renderer** → predicted color/depth/opacity, loss curves
6. **Refinement** → improved thin structures, **meshes (PLY/OBJ)**
7. **Evaluation** → IoU, Chamfer-L2, F-score, PSNR/SSIM, CSV reports

Artifacts are saved under `outputs/stage_*/…` (PNGs, GIF/MP4 turntables, OBJ/PLY meshes, CSV metrics).

---

## 🧪 Minimal Data Schema

```
DATA_ROOT/
└─ sample_plant_001/
   ├─ rgb/*.png
   ├─ mask/*.png
   ├─ depth/*.png           # optional
   ├─ intrinsics.json       # {fx, fy, cx, cy}
   └─ poses/*.json          # {R: 3x3, t: 3x1} per view (world→cam)
```

---

## 🏋️ Training Recipes

* **Small GPU**: 64³ base grid, windowed 3D attention (N1=2), ray-slice (N2=1), AMP on.
* **Medium**: 96³–128³ with sparse octree, gradient checkpointing.
* **Refinement**: freeze σ head, train cross-attention + priors, then fine-tune end-to-end.

---

## 🔌 Extending

* Swap DINO for other ViTs in `src/features/`.
* Replace voxel grid with tri-plane/hybrid (edit `geometry/grids.py`).
* Add depth sensors: enable `loss.lambda_depth` and `data.depth=true`.

---

## 📊 Evaluation

* 3D: **IoU**, **Chamfer-L2**, **F-score**
* Rendering: **PSNR/SSIM**
* Mesh export: `scripts/export_mesh.py` → PLY/OBJ (Meshlab/Blender ready)

---

## 📜 License

Add your chosen license in `LICENSE`.

---

## 🙋 Support

* Open a GitHub Issue for bugs or feature requests.
* Want a generated scaffold (empty modules + starter notebooks) for **PyTorch** with **TensorBoard**/**wandb**? Ask and specify your preference.

