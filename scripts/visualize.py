#!/usr/bin/env python
"""Generate visualizations: DINO feature maps, volume slices, render comparisons,
and ablation metric charts.

Usage
-----
    python scripts/visualize.py --output-root ./outputs/research_gpu
    python scripts/visualize.py --output-root ./outputs/research_gpu --ablation-root ./outputs/ablations
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_npz_feature(path: str) -> np.ndarray:
    """Load a DINO feature .npz and return (C, H, W) array."""
    data = np.load(path)
    feat = data["feature"]  # (1, C, H, W)
    if feat.ndim == 4:
        feat = feat[0]
    return feat


def pca_reduce(features: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce (C, H, W) feature map to (n_components, H, W) via PCA."""
    C, H, W = features.shape
    flat = features.reshape(C, -1).T  # (H*W, C)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Take top components (eigenvalues sorted ascending)
    top = eigvecs[:, -n_components:][:, ::-1]
    projected = centered @ top  # (H*W, n_components)
    projected = projected.reshape(H, W, n_components)
    # Normalize to [0, 1] per channel
    for c in range(n_components):
        mn, mx = projected[:, :, c].min(), projected[:, :, c].max()
        if mx - mn > 1e-8:
            projected[:, :, c] = (projected[:, :, c] - mn) / (mx - mn)
    return projected


# ---------------------------------------------------------------------------
# 1. DINO Feature Map Visualizations
# ---------------------------------------------------------------------------

def plot_dino_features(output_root: str, viz_dir: str) -> List[str]:
    """Visualize DINO features: top-k channel activations + PCA color map."""
    dino_dir = os.path.join(output_root, "stage_20_dino")
    if not os.path.isdir(dino_dir):
        print(f"  [skip] No DINO features at {dino_dir}")
        return []

    npz_files = sorted(glob.glob(os.path.join(dino_dir, "view_*.npz")))
    if not npz_files:
        return []

    saved = []
    n_views = len(npz_files)

    # --- Per-view: top-8 channel activations ---
    for npz_path in npz_files:
        view_name = os.path.splitext(os.path.basename(npz_path))[0]
        feat = load_npz_feature(npz_path)  # (C, H, W)
        C, H, W = feat.shape

        # Channel activation magnitudes
        channel_energy = (feat ** 2).sum(axis=(1, 2))
        top_k = 8
        top_indices = np.argsort(channel_energy)[-top_k:][::-1]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"DINO Feature Channels — {view_name} (shape: {C}×{H}×{W})", fontsize=14)
        for i, idx in enumerate(top_indices):
            ax = axes[i // 4, i % 4]
            im = ax.imshow(feat[idx], cmap="viridis")
            ax.set_title(f"Ch {idx} (E={channel_energy[idx]:.1f})", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"dino_channels_{view_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # --- PCA feature maps (all views side by side) ---
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
    if n_views == 1:
        axes = [axes]
    fig.suptitle("DINO Features — PCA Projection (RGB)", fontsize=14)
    for i, npz_path in enumerate(npz_files):
        view_name = os.path.splitext(os.path.basename(npz_path))[0]
        feat = load_npz_feature(npz_path)
        pca_rgb = pca_reduce(feat, n_components=3)
        axes[i].imshow(pca_rgb)
        axes[i].set_title(view_name, fontsize=11)
        axes[i].axis("off")
    plt.tight_layout()
    path = os.path.join(viz_dir, "dino_pca_all_views.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Feature similarity matrix (cosine) ---
    all_feats = []
    view_names = []
    for npz_path in npz_files:
        feat = load_npz_feature(npz_path)
        all_feats.append(feat.reshape(feat.shape[0], -1).mean(axis=1))
        view_names.append(os.path.splitext(os.path.basename(npz_path))[0])
    all_feats = np.stack(all_feats)  # (N, C)
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8
    all_feats_norm = all_feats / norms
    sim = all_feats_norm @ all_feats_norm.T

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(view_names)))
    ax.set_xticklabels(view_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(view_names)))
    ax.set_yticklabels(view_names, fontsize=9)
    ax.set_title("DINO Feature Cosine Similarity (view means)", fontsize=12)
    for r in range(len(view_names)):
        for c in range(len(view_names)):
            ax.text(c, r, f"{sim[r, c]:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if sim[r, c] > 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(viz_dir, "dino_similarity_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Channel activation distribution ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for npz_path in npz_files:
        view_name = os.path.splitext(os.path.basename(npz_path))[0]
        feat = load_npz_feature(npz_path)
        energy = (feat ** 2).sum(axis=(1, 2))
        energy_sorted = np.sort(energy)[::-1]
        ax.plot(energy_sorted, label=view_name, alpha=0.8)
    ax.set_xlabel("Channel rank (by energy)")
    ax.set_ylabel("Channel energy (L2)")
    ax.set_title("DINO Feature Channel Energy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(viz_dir, "dino_channel_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# 2. Volume Visualizations (sigma + color slices)
# ---------------------------------------------------------------------------

def plot_volume_slices(output_root: str, viz_dir: str) -> List[str]:
    """Visualize sigma density and color volumes as axis-aligned slices."""
    vol_dir = os.path.join(output_root, "stage_30_volume")
    sigma_path = os.path.join(vol_dir, "sigma.npy")
    color_path = os.path.join(vol_dir, "color.npy")

    if not os.path.isfile(sigma_path):
        print(f"  [skip] No sigma volume at {sigma_path}")
        return []

    saved = []
    sigma = np.load(sigma_path)  # (1, 1, D, H, W)
    if sigma.ndim == 5:
        sigma = sigma[0, 0]
    D, H, W = sigma.shape

    # --- Sigma slices along all 3 axes ---
    axis_names = ["Depth (Z)", "Height (Y)", "Width (X)"]
    n_slices = 8
    fig, axes = plt.subplots(3, n_slices, figsize=(2.5 * n_slices, 8))
    fig.suptitle(f"Sigma Density Volume — axis slices (shape: {D}×{H}×{W})", fontsize=14)
    for axis_idx, (axis_name, dim_size) in enumerate(zip(axis_names, [D, H, W])):
        indices = np.linspace(0, dim_size - 1, n_slices, dtype=int)
        for j, idx in enumerate(indices):
            if axis_idx == 0:
                sl = sigma[idx, :, :]
            elif axis_idx == 1:
                sl = sigma[:, idx, :]
            else:
                sl = sigma[:, :, idx]
            im = axes[axis_idx, j].imshow(sl, cmap="inferno", vmin=sigma.min(), vmax=sigma.max())
            axes[axis_idx, j].set_title(f"{axis_name}={idx}", fontsize=8)
            axes[axis_idx, j].axis("off")
        axes[axis_idx, 0].set_ylabel(axis_name, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Density (σ)")
    plt.tight_layout()
    path = os.path.join(viz_dir, "volume_sigma_slices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Sigma MIP (maximum intensity projection) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sigma — Maximum Intensity Projection", fontsize=14)
    projections = [sigma.max(axis=0), sigma.max(axis=1), sigma.max(axis=2)]
    proj_names = ["Front (max over Z)", "Top (max over Y)", "Side (max over X)"]
    for i, (proj, name) in enumerate(zip(projections, proj_names)):
        im = axes[i].imshow(proj, cmap="inferno")
        axes[i].set_title(name, fontsize=11)
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(viz_dir, "volume_sigma_mip.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Sigma histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = sigma.flatten()
    ax.hist(vals, bins=100, color="coral", edgecolor="black", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", label="Mesh threshold (0.5)")
    ax.set_xlabel("Sigma value")
    ax.set_ylabel("Voxel count")
    ax.set_title(f"Sigma Distribution (min={vals.min():.4f}, max={vals.max():.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(viz_dir, "volume_sigma_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Color volume slices (mid-plane) ---
    if os.path.isfile(color_path):
        color = np.load(color_path)  # (1, 3, D, H, W)
        if color.ndim == 5:
            color = color[0]  # (3, D, H, W)
        C_ch, cD, cH, cW = color.shape

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Color Volume — Mid-plane Slices", fontsize=14)
        mid_indices = [cD // 2, cH // 2, cW // 2]
        slice_names = [f"Z={mid_indices[0]}", f"Y={mid_indices[1]}", f"X={mid_indices[2]}"]
        slices_rgb = [
            np.clip(color[:, mid_indices[0], :, :].transpose(1, 2, 0), 0, 1),
            np.clip(color[:, :, mid_indices[1], :].transpose(1, 2, 0), 0, 1),
            np.clip(color[:, :, :, mid_indices[2]].transpose(1, 2, 0), 0, 1),
        ]
        for i, (sl, name) in enumerate(zip(slices_rgb, slice_names)):
            axes[i].imshow(sl)
            axes[i].set_title(f"Color at {name}", fontsize=11)
            axes[i].axis("off")
        plt.tight_layout()
        path = os.path.join(viz_dir, "volume_color_slices.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# 3. Render Visualizations
# ---------------------------------------------------------------------------

def plot_renders(output_root: str, viz_dir: str) -> List[str]:
    """Show rendered previews and accumulation maps."""
    render_dir = os.path.join(output_root, "stage_50_render")
    saved = []

    render_img_path = os.path.join(render_dir, "render_01.png")
    acc_path = os.path.join(render_dir, "render_acc.npy")

    if not os.path.isfile(render_img_path) and not os.path.isfile(acc_path):
        print(f"  [skip] No render artifacts at {render_dir}")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Volume Rendering Output", fontsize=14)

    if os.path.isfile(render_img_path):
        img = np.array(Image.open(render_img_path))
        axes[0].imshow(img)
        axes[0].set_title("Rendered RGB", fontsize=11)
    else:
        axes[0].text(0.5, 0.5, "No render image", ha="center", va="center")
    axes[0].axis("off")

    if os.path.isfile(acc_path):
        acc = np.load(acc_path)
        if acc.ndim > 2:
            acc = acc.squeeze()
        im = axes[1].imshow(acc, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Accumulation (Alpha) Map", fontsize=11)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "No acc map", ha="center", va="center")
    axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(viz_dir, "render_output.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# 4. Ablation Comparisons
# ---------------------------------------------------------------------------

def plot_ablation_metrics(ablation_root: str, viz_dir: str) -> List[str]:
    """Bar charts and grouped comparisons for ablation metrics."""
    csv_path = os.path.join(ablation_root, "ablation_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"  [skip] No ablation_summary.csv at {csv_path}")
        return []

    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["status"] == "ok":
                rows.append(row)

    if not rows:
        return []

    saved = []
    names = [r["preset"] for r in rows]
    short_names = [n.replace("ablation_", "") for n in names]
    mse = [float(r["mse_mean"]) for r in rows]
    psnr = [float(r["psnr_mean"]) for r in rows]
    elapsed = [float(r["elapsed_s"]) for r in rows]

    # --- Overall bar chart ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Ablation Study — Metrics Overview", fontsize=14)

    colors_mse = plt.cm.RdYlGn_r(Normalize(vmin=min(mse), vmax=max(mse))(mse))
    bars1 = ax1.barh(short_names, mse, color=colors_mse, edgecolor="black", alpha=0.85)
    ax1.set_xlabel("MSE (lower is better)")
    ax1.set_title("Mean Squared Error by Preset")
    ax1.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars1, mse):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=9)

    colors_psnr = plt.cm.RdYlGn(Normalize(vmin=min(psnr), vmax=max(psnr))(psnr))
    bars2 = ax2.barh(short_names, psnr, color=colors_psnr, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("PSNR dB (higher is better)")
    ax2.set_title("Peak Signal-to-Noise Ratio by Preset")
    ax2.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars2, psnr):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(viz_dir, "ablation_metrics_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Grouped pair comparisons ---
    pairs = [
        ("A1: Pretrained DINO", "pretrain_on", "pretrain_off"),
        ("A2: Frozen Backbone", "freeze_on", "freeze_off"),
        ("A3: Refinement", "refine_on", "refine_off"),
        ("A4: Training Depth", "epochs_long", "epochs_short"),
    ]
    lookup = {n.replace("ablation_", ""): i for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ablation Paired Comparisons — MSE & PSNR", fontsize=14)
    for idx, (title, key_a, key_b) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        if key_a in lookup and key_b in lookup:
            ia, ib = lookup[key_a], lookup[key_b]
            x = np.arange(2)
            bar_width = 0.35
            bars_m = ax.bar(x - bar_width / 2, [mse[ia], mse[ib]], bar_width,
                            label="MSE", color="#e74c3c", alpha=0.8, edgecolor="black")
            ax2_twin = ax.twinx()
            bars_p = ax2_twin.bar(x + bar_width / 2, [psnr[ia], psnr[ib]], bar_width,
                                  label="PSNR", color="#2ecc71", alpha=0.8, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels([key_a, key_b], fontsize=10)
            ax.set_ylabel("MSE ↓", color="#e74c3c")
            ax2_twin.set_ylabel("PSNR (dB) ↑", color="#2ecc71")
            ax.set_title(title, fontsize=12, fontweight="bold")

            # Annotate values
            for bar, v in zip(bars_m, [mse[ia], mse[ib]]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#e74c3c")
            for bar, v in zip(bars_p, [psnr[ia], psnr[ib]]):
                ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                              f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#2ecc71")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"Missing: {key_a} or {key_b}", ha="center", va="center")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(viz_dir, "ablation_paired_comparisons.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Timing chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_t = plt.cm.Blues(Normalize(vmin=min(elapsed), vmax=max(elapsed))(elapsed))
    bars = ax.barh(short_names, elapsed, color=colors_t, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Ablation Runtime Comparison (GPU)")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars, elapsed):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}s", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(viz_dir, "ablation_runtime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# 5. Ablation Render Gallery
# ---------------------------------------------------------------------------

def plot_ablation_render_gallery(ablation_root: str, viz_dir: str) -> List[str]:
    """Side-by-side rendered RGB from each ablation run."""
    subdirs = sorted([
        d for d in os.listdir(ablation_root)
        if os.path.isdir(os.path.join(ablation_root, d)) and d.startswith("ablation_")
    ])
    if not subdirs:
        return []

    images = []
    labels = []
    for d in subdirs:
        render_path = os.path.join(ablation_root, d, "stage_50_render", "render_01.png")
        if os.path.isfile(render_path):
            images.append(np.array(Image.open(render_path)))
            labels.append(d.replace("ablation_", ""))

    if not images:
        return []

    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle("Ablation Render Gallery", fontsize=14)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i])
            ax.set_title(labels[i], fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    saved = []
    path = os.path.join(viz_dir, "ablation_render_gallery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Plant3D visualizations.")
    ap.add_argument("--output-root", type=str, default="./outputs/research_gpu",
                    help="Pipeline output directory with stage_* folders.")
    ap.add_argument("--ablation-root", type=str, default="./outputs/ablations",
                    help="Ablation outputs directory with ablation_summary.csv.")
    ap.add_argument("--viz-dir", type=str, default=None,
                    help="Where to save visualizations (default: <output-root>/visualizations).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    viz_dir = args.viz_dir or os.path.join(args.output_root, "visualizations")
    ensure_dir(viz_dir)

    print(f"Saving visualizations to: {viz_dir}\n")

    all_saved: List[str] = []

    print("[1/5] DINO Feature Maps...")
    all_saved.extend(plot_dino_features(args.output_root, viz_dir))

    print("[2/5] Volume Slices...")
    all_saved.extend(plot_volume_slices(args.output_root, viz_dir))

    print("[3/5] Render Output...")
    all_saved.extend(plot_renders(args.output_root, viz_dir))

    if args.ablation_root and os.path.isdir(args.ablation_root):
        print("[4/5] Ablation Metrics...")
        all_saved.extend(plot_ablation_metrics(args.ablation_root, viz_dir))

        print("[5/5] Ablation Render Gallery...")
        all_saved.extend(plot_ablation_render_gallery(args.ablation_root, viz_dir))
    else:
        print("[4/5] Ablation Metrics... [skip] no ablation root")
        print("[5/5] Ablation Render Gallery... [skip]")

    print(f"\nDone — {len(all_saved)} images saved:")
    for p in all_saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
