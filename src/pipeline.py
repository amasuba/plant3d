from __future__ import annotations
import csv
import os
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataio.datasets_flat import FlatPlantDataset
from .features.dino_backbone import DINOBackbone
from .geometry.lift import lift_features_to_grid
from .train.trainer_vol import VolumetricModel
from .train.trainer_refine import RefinementModel
from .models.renderer.render import sample_grid, volume_render
from .models.refinement.losses import geometry_smoothness
from .eval.metrics_img import psnr as psnr_metric
from .eval.mesh import marching_cubes_from_sigma

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(rgb: np.ndarray, path: str) -> None:
    rgb = (np.clip(rgb * 255.0, 0, 255)).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def estimate_intrinsics(image: torch.Tensor, fx_scale: float = 0.9) -> Dict[str, float]:
    _, h, w = image.shape
    f = max(h, w) * fx_scale
    return {"fx": float(f), "fy": float(f), "cx": float(w / 2.0), "cy": float(h / 2.0)}


def scale_intrinsics(
    K: torch.Tensor, orig_size: Tuple[int, int], target_size: Tuple[int, int]
) -> torch.Tensor:
    orig_h, orig_w = orig_size
    target_h, target_w = target_size
    scale_x = float(target_w) / float(orig_w)
    scale_y = float(target_h) / float(orig_h)
    K = K.clone().float()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def preprocess_image(
    image: torch.Tensor, size: int = 224, device: Optional[torch.device] = None
) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if device is not None:
        image = image.to(device)
    image = F.interpolate(
        image, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = IMAGENET_MEAN.to(image.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def build_dino_backbone(
    model_name: str = "vit_base_patch16_224.dino",
    pretrained: bool = True,
    freeze: bool = True,
) -> nn.Module:
    try:
        return DINOBackbone(model_name=model_name, pretrained=pretrained, freeze=freeze)
    except Exception as exc:
        print(f"Warning: could not initialize pretrained DINO model: {exc}")
        print("Falling back to an uninitialized DINO backbone. Training may still run.")
        return DINOBackbone(model_name=model_name, pretrained=False, freeze=False)


def build_flat_dataset(
    folder: str, plant_id: int = 1, baseline_m: float = 0.40
) -> FlatPlantDataset:
    sample_paths = sorted(
        [
            p
            for p in os.listdir(folder)
            if "RGB" in p and p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if len(sample_paths) == 0:
        raise ValueError(
            f"No RGB files found in {folder}. Expected names like '0_degrees_RGB_plant_1.jpg'."
        )
    sample_image = Image.open(os.path.join(folder, sample_paths[0]))
    sample_tensor = torch.from_numpy(
        np.array(sample_image).astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    intr = estimate_intrinsics(sample_tensor)
    return FlatPlantDataset(
        folder, plant_id=plant_id, intr_red=intr, intr_green=intr, baseline_m=baseline_m
    )


def compute_camera_rays(
    K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = K.device
    ys = torch.arange(0.5, height + 0.5, device=device)
    xs = torch.arange(0.5, width + 0.5, device=device)
    jj, ii = torch.meshgrid(ys, xs, indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = torch.stack([(ii - cx) / fx, (jj - cy) / fy, torch.ones_like(ii)], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs_world = dirs @ R.T
    origin_world = (-R.T @ t).view(3)
    return origin_world, dirs_world


def render_volume_tensors(
    sigma: torch.Tensor,
    color: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    output_size: Tuple[int, int] = (128, 128),
    n_samples: int = 64,
    near: float = 0.1,
    far: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sigma.device
    height, width = output_size
    origin, dirs = compute_camera_rays(K, R, t, height, width)
    t_vals = torch.linspace(near, far, n_samples, device=device).view(1, 1, -1, 1)
    pts = origin.view(1, 1, 1, 3) + dirs.unsqueeze(-2) * t_vals
    pts_norm = (pts + 1.0) * 0.5
    pts_flat = pts_norm.reshape(1, -1, 3)
    sigma_samples = sample_grid(sigma, pts_flat)
    color_samples = sample_grid(color, pts_flat)
    delta = float((far - near) / n_samples)
    alphas = 1.0 - torch.exp(-sigma_samples * delta)
    rgb, acc = volume_render(sigma_samples, color_samples, alphas)
    rgb = rgb.view(1, 3, height, width).clamp(0.0, 1.0)
    acc = acc.view(1, 1, height, width).clamp(0.0, 1.0)
    return rgb, acc


def render_volume_image(
    sigma: torch.Tensor,
    color: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    output_size: Tuple[int, int] = (128, 128),
    n_samples: int = 64,
    near: float = 0.1,
    far: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_t, acc_t = render_volume_tensors(
        sigma=sigma,
        color=color,
        K=K,
        R=R,
        t=t,
        output_size=output_size,
        n_samples=n_samples,
        near=near,
        far=far,
    )
    rgb = rgb_t[0].permute(1, 2, 0).detach().cpu().numpy()
    acc = acc_t[0, 0].detach().cpu().numpy()
    return rgb, acc


def _extract_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if any(p.requires_grad for p in backbone.parameters()):
        return backbone(x)
    with torch.no_grad():
        return backbone(x)


def average_dino_features(
    dataset: FlatPlantDataset,
    backbone: nn.Module,
    device: torch.device,
    image_size: int = 224,
    save_dir: Optional[str] = None,
) -> torch.Tensor:
    backbone.eval()
    feature_sum = None
    count = 0
    for idx, sample in enumerate(dataset):
        image = sample["image"].to(device)
        x = preprocess_image(image, size=image_size, device=device)
        with torch.no_grad():
            f2d = backbone(x)
        if feature_sum is None:
            feature_sum = f2d.detach().clone()
        else:
            feature_sum += f2d.detach()
        count += 1
        if save_dir is not None:
            ensure_dir(save_dir)
            np.savez_compressed(
                os.path.join(save_dir, f"view_{idx:02d}.npz"), feature=f2d.cpu().numpy()
            )
    if count == 0:
        raise ValueError("Dataset is empty; no DINO features were extracted.")
    return feature_sum / float(count)


def train_volumetric_model(
    dataset: FlatPlantDataset,
    backbone: nn.Module,
    model: nn.Module,
    device: torch.device,
    output_dir: str,
    epochs: int = 2,
    image_size: int = 224,
    render_size: Tuple[int, int] = (128, 128),
    n_samples: int = 32,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 50)
    )
    stats: Dict[str, Any] = {"losses": []}
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx, sample in enumerate(dataset):
            image = sample["image"].to(device)
            K = sample["K"].to(device)
            R = sample["R"].to(device)
            t = sample["t"].to(device)
            target = F.interpolate(
                image.unsqueeze(0),
                size=render_size,
                mode="bilinear",
                align_corners=False,
            )
            x = preprocess_image(image, size=image_size, device=device)
            f2d = _extract_features(backbone, x)
            sigma, color, _ = model(f2d)
            Ks = scale_intrinsics(K, image.shape[1:], render_size)
            rgb_pred, _ = render_volume_tensors(
                sigma, color, Ks, R, t, output_size=render_size, n_samples=n_samples
            )
            loss = F.mse_loss(rgb_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
        epoch_loss /= float(len(dataset))
        stats["losses"].append(epoch_loss)
        checkpoint_path = os.path.join(
            output_dir, f"volumetric_epoch_{epoch+1:02d}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.6f}")
    return stats


def train_refinement_model(
    dataset: FlatPlantDataset,
    backbone: nn.Module,
    vol_model: nn.Module,
    avg_f2d: torch.Tensor,
    device: torch.device,
    output_dir: str,
    epochs: int = 1,
    image_size: int = 224,
    render_size: Tuple[int, int] = (128, 128),
    n_samples: int = 32,
    lambda_smooth: float = 0.05,
) -> Tuple[RefinementModel, Dict[str, Any], torch.Tensor, torch.Tensor]:
    ensure_dir(output_dir)
    with torch.no_grad():
        _, _, base_vol = vol_model(avg_f2d.to(device))
    c3d = int(base_vol.shape[1])
    c2d = int(avg_f2d.shape[1])
    refine_model = RefinementModel(c3d=c3d, c2d=c2d).to(device)
    optimizer = torch.optim.AdamW(refine_model.parameters(), lr=1e-4, weight_decay=1e-4)

    stats: Dict[str, Any] = {"losses": []}
    refine_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sample in dataset:
            image = sample["image"].to(device)
            K = sample["K"].to(device)
            R = sample["R"].to(device)
            t = sample["t"].to(device)

            target = F.interpolate(
                image.unsqueeze(0),
                size=render_size,
                mode="bilinear",
                align_corners=False,
            )
            x = preprocess_image(image, size=image_size, device=device)
            f2d = _extract_features(backbone, x)
            img_tokens = f2d.flatten(2).transpose(1, 2)

            sigma_r, color_r, _ = refine_model(base_vol.detach(), img_tokens)
            Ks = scale_intrinsics(K, image.shape[1:], render_size)
            rgb_pred, _ = render_volume_tensors(
                sigma_r, color_r, Ks, R, t, output_size=render_size, n_samples=n_samples
            )

            loss_recon = F.mse_loss(rgb_pred, target)
            loss_smooth = geometry_smoothness(sigma_r)
            loss = loss_recon + lambda_smooth * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        epoch_loss /= float(len(dataset))
        stats["losses"].append(epoch_loss)
        ckpt = os.path.join(output_dir, f"refine_epoch_{epoch+1:02d}.pth")
        torch.save(refine_model.state_dict(), ckpt)
        print(f"Refinement epoch {epoch+1}/{epochs} loss={epoch_loss:.6f}")

    refine_model.eval()
    with torch.no_grad():
        avg_tokens = avg_f2d.to(device).flatten(2).transpose(1, 2)
        sigma_final, color_final, _ = refine_model(base_vol, avg_tokens)
    return refine_model, stats, sigma_final, color_final


def evaluate_render_metrics(
    dataset: FlatPlantDataset,
    sigma: torch.Tensor,
    color: torch.Tensor,
    device: torch.device,
    output_dir: str,
    render_size: Tuple[int, int] = (256, 256),
    n_samples: int = 64,
) -> Dict[str, float]:
    ensure_dir(output_dir)
    rows = []
    psnr_values = []
    mse_values = []
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            image = sample["image"].to(device)
            K = sample["K"].to(device)
            R = sample["R"].to(device)
            t = sample["t"].to(device)

            target = F.interpolate(
                image.unsqueeze(0),
                size=render_size,
                mode="bilinear",
                align_corners=False,
            )
            Ks = scale_intrinsics(K, image.shape[1:], render_size)
            rgb_pred, _ = render_volume_tensors(
                sigma, color, Ks, R, t, output_size=render_size, n_samples=n_samples
            )
            mse = float(F.mse_loss(rgb_pred, target).item())
            psnr_val = float(psnr_metric(rgb_pred, target))
            mse_values.append(mse)
            psnr_values.append(psnr_val)
            rows.append({"view_idx": idx, "mse": mse, "psnr": psnr_val})

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["view_idx", "mse", "psnr"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "mse_mean": float(np.mean(mse_values)) if mse_values else float("nan"),
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "occupancy_ratio": float((sigma >= 0.5).float().mean().item()),
    }


def export_volume_artifacts(
    sigma: torch.Tensor,
    color: torch.Tensor,
    dataset: FlatPlantDataset,
    device: torch.device,
    output_root: str,
    render_size: Tuple[int, int] = (256, 256),
    n_samples: int = 64,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    stage30 = os.path.join(output_root, "stage_30_volume")
    stage50 = os.path.join(output_root, "stage_50_render")
    stage60 = os.path.join(output_root, "stage_60_refine")
    stage70 = os.path.join(output_root, "stage_70_eval")
    ensure_dir(stage30)
    ensure_dir(stage50)
    ensure_dir(stage60)
    ensure_dir(stage70)

    with torch.no_grad():
        np.save(os.path.join(stage30, "sigma.npy"), sigma.cpu().numpy())
        np.save(os.path.join(stage30, "color.npy"), color.cpu().numpy())

        if len(dataset) > 0:
            first = dataset[0]
            K = first["K"].to(device)
            R = first["R"].to(device)
            t = first["t"].to(device)
            Ks = scale_intrinsics(K, dataset[0]["image"].shape[1:], render_size)
            rgb_pred, acc = render_volume_image(
                sigma, color, Ks, R, t, output_size=render_size, n_samples=n_samples
            )
            save_image(rgb_pred, os.path.join(stage50, "render_01.png"))
            np.save(os.path.join(stage50, "render_acc.npy"), acc)

        sigma_np = sigma[0, 0].cpu().numpy()
        mesh = marching_cubes_from_sigma(sigma_np, thresh=0.5)
        mesh_path = os.path.join(stage60, "plant_volume_mesh.ply")
        mesh.export(mesh_path)
        print(f"Exported mesh to {mesh_path}")

        with open(os.path.join(stage70, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write("Exported volumetric artifacts from plant pipeline.\n")
            fh.write(f"sigma shape: {sigma_np.shape}\n")
            fh.write(f"render image: {os.path.join(stage50, 'render_01.png')}\n")
            fh.write(f"mesh: {mesh_path}\n")
            if metrics is not None:
                fh.write(f"mse_mean: {metrics['mse_mean']:.6f}\n")
                fh.write(f"psnr_mean: {metrics['psnr_mean']:.3f}\n")
                fh.write(f"occupancy_ratio: {metrics['occupancy_ratio']:.6f}\n")


def run_pipeline(
    data_folder: str,
    plant_id: int = 1,
    output_root: str = "./outputs",
    device_name: str = "cpu",
    epochs: int = 0,
    baseline_m: float = 0.40,
    image_size: int = 224,
    render_size: Tuple[int, int] = (256, 256),
    n_render_samples: int = 48,
    refine_epochs: int = 1,
    dino_model_name: str = "vit_base_patch16_224.dino",
    pretrained_dino: bool = True,
    freeze_dino: bool = True,
) -> Dict[str, Any]:
    device = torch.device(
        device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu"
    )
    output_root = os.path.abspath(output_root)
    ensure_dir(output_root)
    stage20 = os.path.join(output_root, "stage_20_dino")
    ensure_dir(stage20)
    print(f"Loading dataset from {data_folder}")
    dataset = build_flat_dataset(data_folder, plant_id=plant_id, baseline_m=baseline_m)
    backbone = build_dino_backbone(
        model_name=dino_model_name, pretrained=pretrained_dino, freeze=freeze_dino
    ).to(device)
    sample_image = dataset[0]["image"]
    feature_sample = preprocess_image(sample_image, size=image_size, device=device)
    with torch.no_grad():
        sample_feat = backbone(feature_sample)
    feat_dim = sample_feat.shape[1]
    model = VolumetricModel(feat_dim=feat_dim, vol_dim=128).to(device)
    print(f"Extracting DINO features for {len(dataset)} views")
    avg_f2d = average_dino_features(
        dataset, backbone, device, image_size=image_size, save_dir=stage20
    )
    if epochs > 0:
        stage40 = os.path.join(output_root, "stage_40_vol_train")
        ensure_dir(stage40)
        stats = train_volumetric_model(
            dataset=dataset,
            backbone=backbone,
            model=model,
            device=device,
            output_dir=stage40,
            epochs=epochs,
            image_size=image_size,
            render_size=(128, 128),
            n_samples=32,
        )
        with open(
            os.path.join(stage40, "training_stats.txt"), "w", encoding="utf-8"
        ) as fh:
            for epoch, loss in enumerate(stats["losses"], start=1):
                fh.write(f"epoch_{epoch:02d}: {loss:.6f}\n")
        print(f"Training complete. Best loss: {min(stats['losses']):.6f}")
    else:
        print("Skipping volumetric training (epochs=0). Using untrained model weights.")

    with torch.no_grad():
        sigma_base, color_base, _ = model(avg_f2d.to(device))

    sigma_final, color_final = sigma_base, color_base
    refine_stats: Optional[Dict[str, Any]] = None
    if refine_epochs > 0:
        stage60_train = os.path.join(output_root, "stage_60_refine")
        refine_model, refine_stats, sigma_ref, color_ref = train_refinement_model(
            dataset=dataset,
            backbone=backbone,
            vol_model=model,
            avg_f2d=avg_f2d,
            device=device,
            output_dir=stage60_train,
            epochs=refine_epochs,
            image_size=image_size,
            render_size=(128, 128),
            n_samples=32,
        )
        sigma_final, color_final = sigma_ref, color_ref
        if refine_stats["losses"]:
            print(f"Refinement complete. Best loss: {min(refine_stats['losses']):.6f}")
    else:
        print("Skipping refinement (refine_epochs=0).")

    stage70 = os.path.join(output_root, "stage_70_eval")
    metrics = evaluate_render_metrics(
        dataset=dataset,
        sigma=sigma_final,
        color=color_final,
        device=device,
        output_dir=stage70,
        render_size=render_size,
        n_samples=n_render_samples,
    )

    export_volume_artifacts(
        sigma=sigma_final,
        color=color_final,
        dataset=dataset,
        device=device,
        output_root=output_root,
        render_size=render_size,
        n_samples=n_render_samples,
        metrics=metrics,
    )
    return {
        "output_root": output_root,
        "dataset_size": len(dataset),
        "dino_feature_dim": feat_dim,
        "epochs": epochs,
        "refine_epochs": refine_epochs,
        "metrics": metrics,
    }
