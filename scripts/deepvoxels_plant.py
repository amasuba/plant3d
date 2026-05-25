#!/usr/bin/env python
"""Prepare plant dataset in DeepVoxels format, train DeepVoxels, and render a
360-degree rotating video of the reconstructed plant.

This script:
1. Converts the flat plant dataset to DeepVoxels directory format
   (rgb/, pose/, intrinsics.txt)
2. Initializes the missing pytorch_prototyping submodule
3. Trains DeepVoxels on the plant views
4. Generates circular test poses and renders a 360 rotation video

Usage
-----
Full pipeline (prepare + train + render):
    python scripts/deepvoxels_plant.py --plant-data ./dataset/plant_data --epochs 200

Prepare data only:
    python scripts/deepvoxels_plant.py --plant-data ./dataset/plant_data --prepare-only

Render from checkpoint:
    python scripts/deepvoxels_plant.py --render-only --checkpoint path/to/model.pth \
        --plant-data ./dataset/plant_data
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

here = os.path.abspath(os.path.dirname(__file__))
PLANT3D_ROOT = os.path.abspath(os.path.join(here, ".."))
DEEPVOXELS_ROOT = os.path.abspath(os.path.join(PLANT3D_ROOT, "..", "deepvoxels"))


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def yaw_rotation_y(deg: float) -> np.ndarray:
    """Y-axis rotation matrix for a given angle in degrees."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def camera_to_world_pose(angle_deg: float, radius: float = 1.5) -> np.ndarray:
    """Build a 4x4 camera-to-world pose matrix for a turntable setup.

    DeepVoxels uses OpenCV convention: Y-down, Z-into-image.
    The camera orbits around Y-axis at a given radius, looking at the origin.
    """
    # Camera position on circle in XZ plane
    th = np.deg2rad(angle_deg)
    cam_x = radius * np.sin(th)
    cam_z = radius * np.cos(th)
    cam_pos = np.array([cam_x, 0.0, cam_z])

    # Look at origin
    forward = -cam_pos / np.linalg.norm(cam_pos)  # camera Z points toward origin
    # World up is -Y in OpenCV (Y points down)
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # Y-axis of camera (points down)

    # Camera-to-world: columns are camera axes in world coords
    R = np.eye(4, dtype=np.float64)
    R[:3, 0] = right
    R[:3, 1] = down
    R[:3, 2] = forward
    R[:3, 3] = cam_pos
    return R


def prepare_deepvoxels_dataset(
    plant_data_dir: str,
    output_dir: str,
    plant_id: int = 1,
    img_size: int = 512,
    radius: float = 1.5,
) -> str:
    """Convert flat plant dataset to DeepVoxels format."""
    rgb_dir = os.path.join(output_dir, "rgb")
    pose_dir = os.path.join(output_dir, "pose")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # Turntable angles present in the dataset
    angles = [0, 90, 180, 270]
    view_idx = 0

    for angle in angles:
        # Try .npy first, fall back to jpg
        npy_path = os.path.join(plant_data_dir, f"{angle}_degrees_RGB_plant_{plant_id}.npy")
        jpg_path = os.path.join(plant_data_dir, f"{angle}_degrees_RGB_plant_{plant_id}.jpg")

        if os.path.isfile(npy_path):
            rgb = np.load(npy_path)
        elif os.path.isfile(jpg_path):
            rgb = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            print(f"  [warn] No RGB for angle {angle}, skipping")
            continue

        # Square crop from centre
        h, w = rgb.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        rgb = rgb[cy - min_dim // 2:cy + min_dim // 2,
                   cx - min_dim // 2:cx + min_dim // 2]

        # Resize to target size
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Save image
        img_path = os.path.join(rgb_dir, f"{view_idx:05d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Save camera-to-world pose
        pose = camera_to_world_pose(angle, radius=radius)
        pose_path = os.path.join(pose_dir, f"{view_idx:05d}.txt")
        with open(pose_path, "w") as f:
            f.write(" ".join(f"{v:.10f}" for v in pose.reshape(-1)) + "\n")

        print(f"  View {view_idx}: angle={angle}° -> {img_path}")
        view_idx += 1

    # Write intrinsics.txt
    # DeepVoxels format: f cx cy / origin / near_plane / scale / height width
    f = img_size * 0.9  # focal length in pixels
    cx = img_size / 2.0
    cy = img_size / 2.0
    origin = np.array([0.0, 0.0, 0.0])
    # scale = total extent of the voxel grid in world units
    # depth_max = scale + near_plane must exceed camera_distance + grid_half
    # Camera at radius from origin, grid centered at origin,
    # so we need scale such that the frustum [near_plane, scale + near_plane]
    # covers distances [0, 2*radius] from camera.
    near_plane = max(0.1, radius - 1.0)  # start frustum before the grid
    scale = radius * 2.0  # grid spans 2*radius to cover the full scene

    intrinsics_path = os.path.join(output_dir, "intrinsics.txt")
    with open(intrinsics_path, "w") as fh:
        fh.write(f"{f} {cx} {cy}\n")
        fh.write(f"{origin[0]} {origin[1]} {origin[2]}\n")
        fh.write(f"{near_plane}\n")
        fh.write(f"{scale}\n")
        fh.write(f"{img_size} {img_size}\n")

    print(f"  Intrinsics written to {intrinsics_path}")
    print(f"  Prepared {view_idx} views in {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# 2. Generate circular test poses for video
# ---------------------------------------------------------------------------

def generate_circular_poses(
    output_dir: str,
    n_frames: int = 120,
    radius: float = 1.5,
    elevation_deg: float = -15.0,
) -> str:
    """Generate a full 360-degree circular camera trajectory for rendering."""
    test_dir = os.path.join(output_dir, "test_poses")
    pose_dir = os.path.join(test_dir, "pose")
    os.makedirs(pose_dir, exist_ok=True)

    for i in range(n_frames):
        angle = (360.0 / n_frames) * i

        # Camera position: orbit in XZ plane with slight elevation
        th = np.deg2rad(angle)
        el = np.deg2rad(elevation_deg)
        cam_x = radius * np.cos(el) * np.sin(th)
        cam_y = radius * np.sin(el)
        cam_z = radius * np.cos(el) * np.cos(th)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, -1.0, 0.0])
        right = np.cross(forward, world_up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm_r
        down = np.cross(forward, right)

        R = np.eye(4, dtype=np.float64)
        R[:3, 0] = right
        R[:3, 1] = down
        R[:3, 2] = forward
        R[:3, 3] = cam_pos

        pose_path = os.path.join(pose_dir, f"{i:06d}.txt")
        with open(pose_path, "w") as f:
            f.write(" ".join(f"{v:.10f}" for v in R.reshape(-1)) + "\n")

    print(f"  Generated {n_frames} circular poses in {pose_dir}")
    return test_dir


# ---------------------------------------------------------------------------
# 3. Ensure pytorch_prototyping submodule
# ---------------------------------------------------------------------------

def ensure_pytorch_prototyping():
    """Clone pytorch_prototyping if the submodule dir is empty."""
    proto_dir = os.path.join(DEEPVOXELS_ROOT, "pytorch_prototyping")
    marker = os.path.join(proto_dir, "pytorch_prototyping.py")

    if os.path.isfile(marker):
        print("  pytorch_prototyping already present")
        return

    print("  Cloning pytorch_prototyping submodule...")
    # Try git submodule update first
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=DEEPVOXELS_ROOT,
            check=True,
            capture_output=True,
        )
        if os.path.isfile(marker):
            print("  Submodule initialized via git")
            return
    except Exception:
        pass

    # Fallback: clone manually
    if os.path.isdir(proto_dir):
        shutil.rmtree(proto_dir)
    subprocess.run(
        ["git", "clone", "https://github.com/vsitzmann/pytorch_prototyping.git", proto_dir],
        check=True,
        capture_output=True,
    )
    print("  Cloned pytorch_prototyping successfully")


# ---------------------------------------------------------------------------
# 4. Compatibility patches for modern PyTorch
# ---------------------------------------------------------------------------

def patch_deepvoxels_for_modern_torch():
    """Apply minimal patches to make DeepVoxels run on PyTorch 2.x."""
    patches = {}

    # Patch custom_layers.py: torch.Tensor types
    custom_layers_path = os.path.join(DEEPVOXELS_ROOT, "custom_layers.py")
    if os.path.isfile(custom_layers_path):
        with open(custom_layers_path, "r", encoding="utf-8") as f:
            code = f.read()
        original = code
        # torch.cuda.FloatTensor -> torch.FloatTensor or device handling
        code = code.replace("Variable(", "(")  # Variable is deprecated
        if code != original:
            with open(custom_layers_path, "w", encoding="utf-8") as f:
                f.write(code)
            patches["custom_layers.py"] = True

    # Patch projection.py: same Variable removal
    proj_path = os.path.join(DEEPVOXELS_ROOT, "projection.py")
    if os.path.isfile(proj_path):
        with open(proj_path, "r", encoding="utf-8") as f:
            code = f.read()
        original = code
        code = code.replace("Variable(", "(")
        if code != original:
            with open(proj_path, "w", encoding="utf-8") as f:
                f.write(code)
            patches["projection.py"] = True

    # Patch run_deepvoxels.py: num_workers issue on Windows
    run_path = os.path.join(DEEPVOXELS_ROOT, "run_deepvoxels.py")
    if os.path.isfile(run_path):
        with open(run_path, "r", encoding="utf-8") as f:
            code = f.read()
        original = code
        code = code.replace("num_workers=8", "num_workers=0")
        code = code.replace("num_workers=4", "num_workers=0")
        if code != original:
            with open(run_path, "w", encoding="utf-8") as f:
                f.write(code)
            patches["run_deepvoxels.py"] = True

    # Patch dataio.py: Variable removal
    dataio_path = os.path.join(DEEPVOXELS_ROOT, "dataio.py")
    if os.path.isfile(dataio_path):
        with open(dataio_path, "r", encoding="utf-8") as f:
            code = f.read()
        original = code
        code = code.replace("Variable(", "(")
        if code != original:
            with open(dataio_path, "w", encoding="utf-8") as f:
                f.write(code)
            patches["dataio.py"] = True

    if patches:
        print(f"  Patched files for modern PyTorch: {', '.join(patches.keys())}")
    else:
        print("  No patches needed")


# ---------------------------------------------------------------------------
# 5. Train DeepVoxels
# ---------------------------------------------------------------------------

def train_deepvoxels(
    data_root: str,
    logging_root: str,
    max_epoch: int = 200,
    img_sidelength: int = 512,
    grid_dim: int = 32,
    lr: float = 0.0004,
) -> str:
    """Launch DeepVoxels training."""
    os.makedirs(logging_root, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(DEEPVOXELS_ROOT, "run_deepvoxels.py"),
        "--train_test", "train",
        "--data_root", data_root,
        "--logging_root", logging_root,
        "--max_epoch", str(max_epoch),
        "--img_sidelength", str(img_sidelength),
        "--grid_dim", str(grid_dim),
        "--lr", str(lr),
        "--l1_weight", "200",
        "--num_trgt", "1",
        "--sampling_pattern", "all",
    ]

    print(f"\n  Training command:\n    {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = DEEPVOXELS_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, cwd=DEEPVOXELS_ROOT, env=env)
    if proc.returncode != 0:
        print(f"  [warn] Training exited with code {proc.returncode}")

    # Find the latest checkpoint
    checkpoints = []
    for root, dirs, files in os.walk(logging_root):
        for f in files:
            if f.endswith(".pth"):
                checkpoints.append(os.path.join(root, f))

    if checkpoints:
        latest = max(checkpoints, key=os.path.getmtime)
        print(f"  Latest checkpoint: {latest}")
        return latest
    else:
        print("  [warn] No checkpoint found after training")
        return ""


# ---------------------------------------------------------------------------
# 6. Test / render from checkpoint
# ---------------------------------------------------------------------------

def render_test_trajectory(
    data_root: str,
    test_pose_dir: str,
    logging_root: str,
    checkpoint: str,
    img_sidelength: int = 512,
    grid_dim: int = 32,
) -> str:
    """Render images for a test trajectory using the trained model."""
    # Copy test poses to data_root as the test expects them there
    test_data_root = os.path.join(os.path.dirname(data_root), "deepvoxels_test_data")
    os.makedirs(test_data_root, exist_ok=True)

    # Copy intrinsics
    shutil.copy2(
        os.path.join(data_root, "intrinsics.txt"),
        os.path.join(test_data_root, "intrinsics.txt"),
    )

    # Copy/link pose dir
    dst_pose = os.path.join(test_data_root, "pose")
    if os.path.isdir(dst_pose):
        shutil.rmtree(dst_pose)
    shutil.copytree(os.path.join(test_pose_dir, "pose"), dst_pose)

    cmd = [
        sys.executable,
        os.path.join(DEEPVOXELS_ROOT, "run_deepvoxels.py"),
        "--train_test", "test",
        "--data_root", test_data_root,
        "--logging_root", logging_root,
        "--checkpoint", checkpoint,
        "--img_sidelength", str(img_sidelength),
        "--grid_dim", str(grid_dim),
    ]

    print(f"\n  Render command:\n    {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = DEEPVOXELS_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, cwd=DEEPVOXELS_ROOT, env=env)
    if proc.returncode != 0:
        print(f"  [warn] Render exited with code {proc.returncode}")

    # Find the rendered images directory
    test_traj_dir = os.path.join(logging_root, "test_traj")
    if os.path.isdir(test_traj_dir):
        subdirs = sorted(
            [os.path.join(test_traj_dir, d) for d in os.listdir(test_traj_dir)
             if os.path.isdir(os.path.join(test_traj_dir, d))],
            key=os.path.getmtime
        )
        if subdirs:
            return subdirs[-1]
    return test_traj_dir


# ---------------------------------------------------------------------------
# 7. Assemble rotation video from rendered frames
# ---------------------------------------------------------------------------

def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: int = 30,
) -> str:
    """Compile rendered PNG frames into an MP4 video."""
    frame_paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
         if f.lower().endswith(".png") and not f.startswith("depth")]
    )
    # Filter out the depth subfolder frames
    frame_paths = [p for p in frame_paths if os.path.isfile(p)]

    if not frame_paths:
        print(f"  [warn] No frames found in {frames_dir}")
        return ""

    # Read first frame to get dimensions
    first = cv2.imread(frame_paths[0])
    if first is None:
        print(f"  [warn] Cannot read {frame_paths[0]}")
        return ""

    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        # DeepVoxels outputs 16-bit; normalize to 8-bit
        if frame.dtype == np.uint16:
            frame = (frame.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"  Video saved: {output_path} ({len(frame_paths)} frames, {fps} fps, {w}x{h})")
    return output_path


# ---------------------------------------------------------------------------
# Fallback standalone renderer (no DeepVoxels dependency)
# ---------------------------------------------------------------------------

def render_plant3d_rotation_video(
    output_root: str,
    video_path: str,
    n_frames: int = 120,
    fps: int = 30,
    render_size: Tuple[int, int] = (256, 256),
) -> str:
    """Render a rotation video using the Plant3D pipeline's own volume renderer
    as a fallback if DeepVoxels training fails."""
    sys.path.insert(0, PLANT3D_ROOT)
    import torch
    from src.pipeline import (
        render_volume_image, scale_intrinsics, camera_to_world_pose as _unused,
    )

    sigma_path = os.path.join(output_root, "stage_30_volume", "sigma.npy")
    color_path = os.path.join(output_root, "stage_30_volume", "color.npy")

    if not os.path.isfile(sigma_path):
        print(f"  [fallback] No volume at {sigma_path}")
        return ""

    sigma = torch.from_numpy(np.load(sigma_path)).float()
    color = torch.from_numpy(np.load(color_path)).float()

    # Simple intrinsic
    f = max(render_size) * 0.9
    K = torch.tensor([
        [f, 0, render_size[1] / 2.0],
        [0, f, render_size[0] / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32)

    frames = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        c, s = math.cos(angle), math.sin(angle)
        R = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ], dtype=torch.float32)
        t = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)

        rgb, _ = render_volume_image(
            sigma, color, K, R, t,
            output_size=render_size,
            n_samples=64,
        )
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        frames.append(rgb_uint8)

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Fallback video saved: {video_path} ({n_frames} frames)")
    return video_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DeepVoxels plant reconstruction + 360° video")
    ap.add_argument("--plant-data", type=str, default="./dataset/plant_data",
                    help="Path to flat plant data directory")
    ap.add_argument("--plant-id", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="./outputs/deepvoxels_plant",
                    help="Working directory for DeepVoxels data and outputs")
    ap.add_argument("--epochs", type=int, default=200,
                    help="DeepVoxels training epochs")
    ap.add_argument("--img-size", type=int, default=512,
                    help="Image sidelength for DeepVoxels")
    ap.add_argument("--grid-dim", type=int, default=32,
                    help="Voxel grid dimension")
    ap.add_argument("--n-frames", type=int, default=120,
                    help="Number of frames in rotation video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--radius", type=float, default=1.5,
                    help="Camera orbit radius")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Only prepare data, skip training and rendering")
    ap.add_argument("--render-only", action="store_true",
                    help="Only render from an existing checkpoint")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Path to DeepVoxels checkpoint for render-only mode")
    ap.add_argument("--use-plant3d-fallback", action="store_true",
                    help="Use Plant3D volume renderer instead of DeepVoxels")
    ap.add_argument("--plant3d-output", type=str, default="./outputs/research_gpu",
                    help="Plant3D output root (for fallback renderer)")
    return ap.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(args.output_dir, "data")
    log_dir = os.path.join(args.output_dir, "logs")
    video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Plant3D fallback path ---
    if args.use_plant3d_fallback:
        print("\n=== Using Plant3D volume renderer for rotation video ===")
        video_path = os.path.join(video_dir, "plant_rotation_plant3d.mp4")
        render_plant3d_rotation_video(
            args.plant3d_output, video_path,
            n_frames=args.n_frames, fps=args.fps,
        )
        return

    # --- Step 1: Prepare data ---
    print("\n=== Step 1: Preparing DeepVoxels dataset ===")
    prepare_deepvoxels_dataset(
        plant_data_dir=args.plant_data,
        output_dir=data_dir,
        plant_id=args.plant_id,
        img_size=args.img_size,
        radius=args.radius,
    )

    if args.prepare_only:
        print("\nData prepared. Exiting (--prepare-only).")
        return

    # --- Step 2: Ensure submodule ---
    print("\n=== Step 2: Ensuring pytorch_prototyping ===")
    ensure_pytorch_prototyping()

    # --- Step 3: Patch for modern PyTorch ---
    print("\n=== Step 3: Patching DeepVoxels for modern PyTorch ===")
    patch_deepvoxels_for_modern_torch()

    # --- Step 4: Generate test poses ---
    print("\n=== Step 4: Generating 360° test poses ===")
    test_dir = generate_circular_poses(
        data_dir,
        n_frames=args.n_frames,
        radius=args.radius,
    )

    # --- Step 5: Train or use existing checkpoint ---
    checkpoint = args.checkpoint
    if not args.render_only:
        print("\n=== Step 5: Training DeepVoxels ===")
        checkpoint = train_deepvoxels(
            data_root=data_dir,
            logging_root=log_dir,
            max_epoch=args.epochs,
            img_sidelength=args.img_size,
            grid_dim=args.grid_dim,
        )

    if not checkpoint or not os.path.isfile(checkpoint):
        print("\n  No checkpoint available. Falling back to Plant3D renderer...")
        video_path = os.path.join(video_dir, "plant_rotation_plant3d.mp4")
        render_plant3d_rotation_video(
            args.plant3d_output, video_path,
            n_frames=args.n_frames, fps=args.fps,
        )
        return

    # --- Step 6: Render rotation ---
    print("\n=== Step 6: Rendering rotation trajectory ===")
    frames_dir = render_test_trajectory(
        data_root=data_dir,
        test_pose_dir=test_dir,
        logging_root=log_dir,
        checkpoint=checkpoint,
        img_sidelength=args.img_size,
        grid_dim=args.grid_dim,
    )

    # --- Step 7: Compile video ---
    print("\n=== Step 7: Assembling rotation video ===")
    video_path = os.path.join(video_dir, "plant_rotation_deepvoxels.mp4")
    frames_to_video(frames_dir, video_path, fps=args.fps)

    print(f"\n=== Done! Video: {video_path} ===")


if __name__ == "__main__":
    main()
