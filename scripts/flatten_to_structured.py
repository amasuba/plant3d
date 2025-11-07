# scripts/flatten_to_structured.py
import argparse, os, re, shutil, json, numpy as np, pathlib

PAT = re.compile(
    r'(?P<angle>\d+)_degrees_(?P<mod>rgb|depth)'
    r'(?:_cam_(?P<cam>red|green))?'
    r'_plant_(?P<pid>\d+)'
)

def yaw_R(deg):
    th=np.deg2rad(deg); c,s=np.cos(th),np.sin(th)
    return [[ c,0, s],[0,1,0],[-s,0, c]]

def world_to_cam(R_wc, t_wc):
    R_cw = np.array(R_wc).T
    t_cw = (-R_cw @ np.array(t_wc)).tolist()
    return R_cw.tolist(), t_cw

def main(args):
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # intrinsics template
    intr_red = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    intr_green = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    # build index
    by = {}
    for p in src.iterdir():
        m = PAT.search(p.name)
        if not m: continue
        d = m.groupdict()
        if args.plant and d["pid"] != str(args.plant): continue
        angle = int(d["angle"]); mod = d["mod"]; cam = d.get("cam")
        key = (d["pid"], angle, cam, p.suffix.lower())
        by.setdefault((d["pid"], angle, cam), {}).setdefault(mod, p)

    for (pid, angle, cam), mods in sorted(by.items()):
        if cam is None:
            # assign by angle deterministically (red first per angle)
            cam = "red" if len([k for k in by if k[0]==pid and k[1]==angle and (k[2] in (None,"red"))])%2==1 else "green"

        sess = f"session_{angle:03d}_deg"
        base = dst / f"plant_{pid}" / sess / f"cam_{cam}"
        (base/"rgb").mkdir(parents=True, exist_ok=True)
        (base/"depth").mkdir(parents=True, exist_ok=True)
        (base/"poses").mkdir(parents=True, exist_ok=True)

        # copy files
        if "rgb" in mods:
            shutil.copy2(mods["rgb"], base/"rgb"/mods["rgb"].name)
        if "depth" in mods:
            shutil.copy2(mods["depth"], base/"depth"/(mods["depth"].stem + ".npy"))

        # world->rig rotation
        R_wr = yaw_R(angle); t_wr=[0,0,0]
        # cam in rig tx +/- baseline/2
        xoff = (+args.baseline/2) if cam=="red" else (-args.baseline/2)
        R_rc = np.eye(3).tolist(); t_rc=[xoff,0,0]
        # compose world->camera: T_rc ∘ T_wr
        R_wc = (np.array(R_rc) @ np.array(R_wr)).tolist()
        t_wc = (np.array(R_rc) @ np.array(t_wr) + np.array(t_rc)).tolist()
        R,t = world_to_cam(R_wc, t_wc)

        pose = {"R": R, "t": t}
        with open(base/"poses"/f"{angle:03d}_{cam}.json","w") as f: json.dump(pose,f,indent=2)

        # intrinsics once per plant
        plant_root = dst / f"plant_{pid}"
        with open(plant_root / "intrinsics_cam_red.json","w") as f: json.dump(intr_red,f,indent=2)
        with open(plant_root / "intrinsics_cam_green.json","w") as f: json.dump(intr_green,f,indent=2)

    print("Done. Output at:", dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flat folder with images + npy depths")
    ap.add_argument("--dst", required=True, help="structured DATA_ROOT output")
    ap.add_argument("--plant", type=int, default=None)
    ap.add_argument("--baseline", type=float, default=0.40)
    ap.add_argument("--fx", type=float, default=580.0)
    ap.add_argument("--fy", type=float, default=580.0)
    ap.add_argument("--cx", type=float, default=640/2)
    ap.add_argument("--cy", type=float, default=480/2)
    args = ap.parse_args(); main(args)
