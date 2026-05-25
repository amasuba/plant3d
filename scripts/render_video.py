#!/usr/bin/env python
"""Render a 360-degree orbit MP4 from sigma.npy + color.npy."""
import argparse, math, os, sys
import numpy as np, torch
import imageio.v3 as iio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import render_volume_tensors

def orbit_frame(sigma, color, az, elev, radius, size, n_samples, device):
    th = math.radians(az); el = math.radians(elev)
    cx = radius*math.cos(el)*math.sin(th)
    cy = radius*math.sin(el)
    cz = radius*math.cos(el)*math.cos(th)
    cam = np.array([cx,cy,cz], dtype=np.float32)
    fwd = -cam/(np.linalg.norm(cam)+1e-8)
    up0 = np.array([0,1,0], dtype=np.float32)
    right = np.cross(fwd,up0); right /= np.linalg.norm(right)+1e-8
    up = np.cross(right,fwd)
    R = np.stack([right,-up,fwd]).astype(np.float32)
    t = (-R@cam).astype(np.float32)
    focal = size/(2*math.tan(math.radians(30)))
    K = torch.tensor([[focal,0,size/2],[0,focal,size/2],[0,0,1]],
                     dtype=torch.float32, device=device)
    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device)
    with torch.no_grad():
        rgb,_,_ = render_volume_tensors(sigma,color,K,R_t,t_t,(size,size),n_samples,0.4,2.2)
    return (rgb[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")

def make_video(sigma_path, color_path, output, n_frames, size, n_samples,
               fps, device_name, multi_elevation):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    sigma = torch.from_numpy(np.load(sigma_path)).to(device)
    color = torch.from_numpy(np.load(color_path)).to(device)
    if sigma.dim()==3: sigma = sigma.unsqueeze(0).unsqueeze(0)
    if color.dim()==4: color = color if color.shape[0]==1 else color.unsqueeze(0)
    passes = [("High",35,1.4),("Mid",15,1.1),("Low",3,0.95)] if multi_elevation \
             else [("Orbit",15,1.1)]
    frames = []
    for label, elev, radius in passes:
        print(f"  Rendering {n_frames} frames [{label} view] ...")
        for i in range(n_frames):
            frames.append(orbit_frame(sigma,color,360*i/n_frames,elev,radius,size,n_samples,device))
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    iio.imwrite(output, np.stack(frames), fps=fps, codec="libx264",
                quality=9, macro_block_size=None)
    print(f"Video: {output}  ({len(frames)} frames @ {fps}fps)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma",     required=True)
    ap.add_argument("--color",     required=True)
    ap.add_argument("--output",    default="outputs/orbit.mp4")
    ap.add_argument("--frames",    type=int,   default=120)
    ap.add_argument("--size",      type=int,   default=512)
    ap.add_argument("--samples",   type=int,   default=128)
    ap.add_argument("--fps",       type=int,   default=30)
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--multi-elevation", action="store_true")
    args = ap.parse_args()
    make_video(args.sigma, args.color, args.output, args.frames,
               args.size, args.samples, args.fps, args.device, args.multi_elevation)
