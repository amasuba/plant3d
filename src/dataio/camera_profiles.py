"""
Known camera profiles for the Plant3D depth fusion pipeline.

Intrinsics are typical/nominal values.  For maximum accuracy, run the
manufacturer calibration tool and replace the values below with your
specific unit calibration (they vary slightly per camera).

How to get exact values:
  Kinect v1:    OpenNI2 CameraSettings::GetZoomValue() or rgbd_launch
  Kinect v2:    Kinect SDK -> CoordinateMapper -> GetDepthCameraIntrinsics()
                or libfreenect2 -> Registration::getParameters()
  Azure Kinect: k4a_calibration_get_from_raw() or k4abt_tracker_create()
                or 'k4a-tools': k4aviewer -> Calibration tab
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraProfile:
    name: str
    depth_h: int           # depth image height (rows)
    depth_w: int           # depth image width  (cols)
    fx: float              # depth camera focal length X (pixels)
    fy: float              # depth camera focal length Y (pixels)
    cx: float              # depth camera principal point X
    cy: float              # depth camera principal point Y
    depth_unit: str        # "mm" or "m"
    near_mm: float         # minimum reliable depth (mm)
    far_mm: float          # maximum reliable depth (mm)
    description: str = ""


# ── All supported profiles ────────────────────────────────────────────────────
PROFILES: dict[str, CameraProfile] = {

    "kinect_v1": CameraProfile(
        name="Kinect v1",
        depth_h=480, depth_w=640,
        fx=575.8, fy=575.8, cx=319.5, cy=239.5,
        depth_unit="mm",
        near_mm=400, far_mm=4000,
        description="Xbox 360 Kinect / Kinect for Windows v1.  640x480 depth.",
    ),

    "kinect_v2": CameraProfile(
        name="Kinect v2",
        depth_h=424, depth_w=512,
        fx=365.8, fy=365.8, cx=256.3, cy=207.1,
        depth_unit="mm",
        near_mm=500, far_mm=4500,
        description="Xbox One Kinect / Kinect for Windows v2.  512x424 depth + 1920x1080 colour.",
    ),

    "azure_nfov": CameraProfile(
        name="Azure Kinect NFOV Unbinned",
        depth_h=576, depth_w=640,
        fx=504.2, fy=504.2, cx=320.2, cy=288.6,
        depth_unit="mm",
        near_mm=250, far_mm=3860,
        description="Azure Kinect DK in NFOV Unbinned mode.  640x576 depth.",
    ),

    "azure_nfov_binned": CameraProfile(
        name="Azure Kinect NFOV 2x2 Binned",
        depth_h=288, depth_w=320,
        fx=252.1, fy=252.1, cx=160.1, cy=144.3,
        depth_unit="mm",
        near_mm=250, far_mm=5460,
        description="Azure Kinect DK in NFOV 2x2 Binned mode.  320x288 depth.",
    ),

    "azure_wfov": CameraProfile(
        name="Azure Kinect WFOV Unbinned",
        depth_h=1024, depth_w=1024,
        fx=898.9, fy=898.9, cx=512.6, cy=512.5,
        depth_unit="mm",
        near_mm=250, far_mm=2880,
        description="Azure Kinect DK in WFOV Unbinned mode.  1024x1024 depth.",
    ),

    "azure_wfov_binned": CameraProfile(
        name="Azure Kinect WFOV 2x2 Binned",
        depth_h=512, depth_w=512,
        fx=449.4, fy=449.4, cx=256.3, cy=256.2,
        depth_unit="mm",
        near_mm=250, far_mm=2880,
        description="Azure Kinect DK in WFOV 2x2 Binned mode.  512x512 depth.",
    ),

    # ── Fallback: RealSense D415 at 512x424 (previous camera) ─────────────
    "realsense_d415": CameraProfile(
        name="Intel RealSense D415",
        depth_h=424, depth_w=512,
        fx=380.0, fy=380.0, cx=256.0, cy=212.0,
        depth_unit="mm",
        near_mm=350, far_mm=4000,
        description="Intel RealSense D415 depth stream at 512x424.",
    ),
}

# ── Shape -> profile lookup ────────────────────────────────────────────────────
_SHAPE_MAP: dict[tuple[int, int], str] = {
    (p.depth_h, p.depth_w): key for key, p in PROFILES.items()
}
# Prefer Kinect v2 over RealSense for the shared 424x512 shape
_SHAPE_MAP[(424, 512)] = "kinect_v2"


def detect_profile(depth_h: int, depth_w: int) -> Optional[CameraProfile]:
    """Auto-detect camera profile from depth image dimensions."""
    return PROFILES.get(_SHAPE_MAP.get((depth_h, depth_w), ""))


def get_profile(name: str) -> CameraProfile:
    """Look up a profile by key (e.g. 'kinect_v2', 'azure_nfov')."""
    if name not in PROFILES:
        raise ValueError(
            f"Unknown camera profile: {name!r}.  "
            f"Available: {list(PROFILES.keys())}"
        )
    return PROFILES[name]


def list_profiles() -> None:
    print("Available camera profiles:")
    for key, p in PROFILES.items():
        print(f"  {key:22s}  {p.depth_w}x{p.depth_h} depth  {p.description}")
