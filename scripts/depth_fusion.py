#!/usr/bin/env python
"""
Depth fusion: build a 3D coloured mesh and point cloud from turntable RGB-D views.

Supported cameras (auto-detected from depth image shape):
  kinect_v1         Kinect v1 / Xbox 360  640x480 depth
  kinect_v2         Kinect v2 / Xbox One  512x424 depth   <-- default for this dataset
  azure_nfov        Azure Kinect NFOV Unbinned  640x576
  azure_nfov_binned Azure Kinect NFOV 2x2       320x288
  azure_wfov        Azure Kinect WFOV Unbinned  1024x1024
  azure_wfov_binned Azure Kinect WFOV 2x2       512x512
  realsense_d415    Intel RealSense D415  512x424

Usage:
    python scripts/depth_fusion.py --data-folder dataset/plant_data --output-dir outputs/my_plant

    # Override camera (if auto-detect is wrong):
    python scripts/depth_fusion.py --camera azure_nfov --data-folder dataset/plant_data

    # List all supported cameras:
    python scripts/depth_fusion.py --list-cameras
"""
from __future__ import annotations
import argparse, math, os, struct, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh
from PIL import Image
from scipy.spatial import KDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataio.camera_profiles import (
    detect_profile, get_profile, list_profiles, CameraProfile, PROFILES
)

# Physical camera position: distance from turntable centre (metres).
# Measured from depth data: non-zero min ~ 0.5 m for Kinect v2.
# Adjust if your rig is different.
CAM_X = 0.0    # lateral offset (0 = centred)
CAM_Y = 0.3    # vertical offset above turntable plane
CAM_Z = 1.1    # horizontal distance from plant centre


def backproject(depth_mm: np.ndarray, angle_deg: float,
                profile: CameraProfile,
                near_mm: float, far_mm: float) -> np.ndarray:
    """
    Back-project a depth image into world space and rotate to 0-degree frame.

    The plant sits at the world origin on a turntable.  The camera is fixed
    at position [CAM_X, CAM_Y, CAM_Z].  At angle_deg the plant has rotated,
    so we rotate the resulting points by -angle_deg to align everything.
    """
    H, W = depth_mm.shape
    ys, xs = np.mgrid[0:H, 0:W]
    valid = (depth_mm > near_mm) & (depth_mm < far_mm)
    d = depth_mm[valid].astype(np.float32)
    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    d_m  = d / 1000.0    # mm -> metres

    # Camera-frame: X right, Y down, Z forward (standard depth camera convention)
    px = (xs_v - profile.cx) / profile.fx * d_m
    py = (ys_v - profile.cy) / profile.fy * d_m
    pz = d_m

    # World frame: X right, Y up, Z toward camera
    pw = np.stack([px,
                   CAM_Y - py,        # Y_world = cam_height - Y_cam (Y up)
                   CAM_Z - pz], axis=-1)    # Z_world = cam_z - depth

    # Rotate by -angle_deg around Y to align to 0-degree reference frame
    th = math.radians(-angle_deg)
    c, s = math.cos(th), math.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return (Ry @ pw.T).T


def load_depth(folder: str, angle: int, plant_id: int) -> np.ndarray | None:
    """Load filtered depth (preferred) or raw depth NPY for a turntable angle."""
    for tag in ("filtered", "depth"):
        p = os.path.join(folder, f"{angle}_degrees_{tag}_plant_{plant_id}.npy")
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def save_ply_colored(path: str, pts: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "wb") as f:
        hdr = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode()
        f.write(hdr)
        for pt, cl in zip(pts, colors):
            f.write(struct.pack("<fffBBB",
                float(pt[0]), float(pt[1]), float(pt[2]),
                int(cl[0]),   int(cl[1]),   int(cl[2])))


def run(folder: str, plant_id: int, grid: int,
        profile: CameraProfile,
        near_mm: float, far_mm: float,
        smooth_sigma: float, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCamera: {profile.name}")
    print(f"  Depth: {profile.depth_w}x{profile.depth_h}  "
          f"fx={profile.fx:.1f} fy={profile.fy:.1f} "
          f"cx={profile.cx:.1f} cy={profile.cy:.1f}")
    print(f"  Depth range: {near_mm:.0f} - {far_mm:.0f} mm\n")

    # Discover turntable angles from filenames
    angles = sorted({
        int(f.split("_")[0])
        for f in os.listdir(folder)
        if f.endswith(".npy") and f"plant_{plant_id}" in f
    })
    if not angles:
        raise ValueError(f"No depth NPY files found in {folder}")
    print(f"Turntable angles found: {angles}\n")

    occupancy  = np.zeros((grid, grid, grid), dtype=np.float32)
    color_acc  = np.zeros((3, grid, grid, grid), dtype=np.float32)
    color_cnt  = np.zeros((grid, grid, grid), dtype=np.float32)
    all_pts:  list[np.ndarray] = []
    all_rgb:  list[np.ndarray] = []

    for angle in angles:
        d_mm = load_depth(folder, angle, plant_id)
        if d_mm is None:
            print(f"  {angle:3d}deg: no depth file — skipping")
            continue

        rgb_path = os.path.join(
            folder, f"{angle}_degrees_RGB_plant_{plant_id}.jpg"
        )
        if not os.path.exists(rgb_path):
            rgb_path = rgb_path.replace("_RGB_", "_rgb_")
        if not os.path.exists(rgb_path):
            print(f"  {angle:3d}deg: no RGB file — skipping colour")
            continue

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        pts = backproject(d_mm, angle, profile, near_mm, far_mm)

        if len(pts) == 0:
            print(f"  {angle:3d}deg: no valid depth pixels (check --near-mm / --far-mm)")
            continue

        # Map world coords [-1,1]^3 -> voxel indices [0, grid-1]
        idx = ((pts + 1.0) * 0.5 * (grid - 1)).astype(int)
        in_vol = np.all((idx >= 0) & (idx < grid), axis=1)
        idx_v = idx[in_vol]
        pts_v = pts[in_vol]

        # Accumulate occupancy
        np.add.at(occupancy, (idx_v[:,2], idx_v[:,1], idx_v[:,0]), 1)

        # Sample RGB for each valid depth pixel
        H_d, W_d = d_mm.shape
        H_r, W_r = rgb.shape[:2]
        valid_mask = (d_mm > near_mm) & (d_mm < far_mm)
        ys_d, xs_d = np.mgrid[0:H_d, 0:W_d]
        ys_vd = ys_d[valid_mask][in_vol]
        xs_vd = xs_d[valid_mask][in_vol]
        ys_r = np.clip((ys_vd * H_r / H_d).astype(int), 0, H_r - 1)
        xs_r = np.clip((xs_vd * W_r / W_d).astype(int), 0, W_r - 1)
        clr_f = rgb[ys_r, xs_r].astype(np.float32) / 255.0

        np.add.at(color_acc[0], (idx_v[:,2], idx_v[:,1], idx_v[:,0]), clr_f[:,0])
        np.add.at(color_acc[1], (idx_v[:,2], idx_v[:,1], idx_v[:,0]), clr_f[:,1])
        np.add.at(color_acc[2], (idx_v[:,2], idx_v[:,1], idx_v[:,0]), clr_f[:,2])
        np.add.at(color_cnt,    (idx_v[:,2], idx_v[:,1], idx_v[:,0]), 1)

        all_pts.append(pts_v)
        all_rgb.append((clr_f * 255).astype(np.uint8))
        print(f"  {angle:3d}deg: {len(pts_v):,} pts in volume")

    if not all_pts:
        raise RuntimeError(
            "No valid depth pixels found.\n"
            f"  Camera: {profile.name}\n"
            f"  Check --near-mm / --far-mm match your rig distance.\n"
            f"  Current range: {near_mm:.0f} - {far_mm:.0f} mm"
        )

    # Smooth occupancy
    occ_norm   = occupancy / (occupancy.max() + 1e-8)
    occ_smooth = gaussian_filter(occ_norm, sigma=smooth_sigma)
    occ_smooth = occ_smooth / (occ_smooth.max() + 1e-8)

    # Average colour
    nz = color_cnt > 0
    color_acc[:, nz] /= color_cnt[nz]
    color_acc[:, ~nz] = 0.5   # grey for unobserved voxels

    # Save volumes for render_video.py
    np.save(os.path.join(output_dir, "sigma.npy"),
            occ_smooth[np.newaxis, np.newaxis, ...].astype(np.float32))
    np.save(os.path.join(output_dir, "color.npy"),
            color_acc[np.newaxis, ...].astype(np.float32))
    print(f"\nSaved sigma.npy + color.npy  ({grid}^3 grid)")

    # Coloured point cloud
    all_pts_np = np.concatenate(all_pts)
    all_rgb_np = np.concatenate(all_rgb)
    pcd_path = os.path.join(output_dir, "plant_pointcloud.ply")
    save_ply_colored(pcd_path, all_pts_np, all_rgb_np)
    print(f"Saved point cloud: {pcd_path}  ({len(all_pts_np):,} pts)")

    # Marching cubes mesh
    nz_vals = occ_smooth[occ_smooth > 0.01]
    thresh = float(np.percentile(nz_vals, 30)) if len(nz_vals) > 100 else 0.2
    verts, faces, normals, _ = measure.marching_cubes(occ_smooth, level=thresh)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)

    # Colour vertices from nearest point-cloud neighbours
    pts_vox = (all_pts_np + 1.0) * 0.5 * (grid - 1)
    _, nn_idx = KDTree(pts_vox).query(verts, k=5)
    vc = all_rgb_np[nn_idx].mean(axis=1).astype(np.uint8)
    mesh.visual.vertex_colors = np.concatenate(
        [vc, np.full((len(verts), 1), 255, dtype=np.uint8)], axis=1
    )

    mesh_path = os.path.join(output_dir, "plant_mesh_colored.ply")
    mesh.export(mesh_path)
    print(f"Saved mesh: {mesh_path}  "
          f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")

    # Write a camera log so you always know which profile was used
    with open(os.path.join(output_dir, "camera_used.txt"), "w") as f:
        f.write(f"Camera : {profile.name}\n")
        f.write(f"fx={profile.fx}  fy={profile.fy}  "
                f"cx={profile.cx}  cy={profile.cy}\n")
        f.write(f"depth range : {near_mm}-{far_mm} mm\n")
        f.write(f"grid        : {grid}^3\n")

    print(f"\nDone. All outputs in: {output_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a 3D mesh from turntable RGB-D images.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Camera is auto-detected from depth image shape.\n"
            "Use --list-cameras to see all supported profiles.\n"
            "Use --camera to override auto-detection.\n"
            "\nExamples:\n"
            "  python scripts/depth_fusion.py --data-folder dataset/plant_data\n"
            "  python scripts/depth_fusion.py --camera azure_nfov --data-folder dataset/plant_data\n"
            "  python scripts/depth_fusion.py --list-cameras"
        ),
    )
    ap.add_argument("--data-folder",   default="dataset/plant_data")
    ap.add_argument("--plant-id",      type=int,   default=1)
    ap.add_argument("--grid",          type=int,   default=96,
                    help="Voxel grid resolution (default 96 -> 96^3, ~2 cm/voxel)")
    ap.add_argument("--camera",        default=None,
                    help="Camera profile key, e.g. kinect_v2, azure_nfov.  "
                         "Auto-detected if omitted.")
    ap.add_argument("--near-mm",       type=float, default=None,
                    help="Min valid depth in mm  (default: camera profile value)")
    ap.add_argument("--far-mm",        type=float, default=None,
                    help="Max valid depth in mm  (default: camera profile value)")
    ap.add_argument("--smooth",        type=float, default=2.0,
                    help="Gaussian smoothing sigma for the occupancy grid")
    ap.add_argument("--output-dir",    default="outputs/depth_fusion")
    ap.add_argument("--list-cameras",  action="store_true",
                    help="Print all supported camera profiles and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_cameras:
        list_profiles()
        sys.exit(0)

    # ── Resolve camera profile ──────────────────────────────────────────────
    if args.camera:
        profile = get_profile(args.camera)
        print(f"Using specified camera profile: {profile.name}")
    else:
        # Load one depth file to detect shape
        import glob
        depth_files = sorted(glob.glob(
            os.path.join(args.data_folder, f"*_depth_plant_{args.plant_id}.npy")
        ))
        if not depth_files:
            depth_files = sorted(glob.glob(
                os.path.join(args.data_folder, f"*_filtered_plant_{args.plant_id}.npy")
            ))
        if not depth_files:
            sys.exit(f"No depth NPY files found in {args.data_folder}")

        probe = np.load(depth_files[0])
        profile = detect_profile(probe.shape[0], probe.shape[1])
        if profile is None:
            print(f"WARNING: Unknown depth shape {probe.shape} — "
                  f"defaulting to kinect_v2.  Use --camera to specify.")
            profile = get_profile("kinect_v2")
        else:
            print(f"Auto-detected camera: {profile.name}  "
                  f"(depth {probe.shape[1]}x{probe.shape[0]})")

    near_mm = args.near_mm if args.near_mm is not None else profile.near_mm
    far_mm  = args.far_mm  if args.far_mm  is not None else profile.far_mm

    run(
        folder=args.data_folder,
        plant_id=args.plant_id,
        grid=args.grid,
        profile=profile,
        near_mm=near_mm,
        far_mm=far_mm,
        smooth_sigma=args.smooth,
        output_dir=args.output_dir,
    )