from __future__ import annotations
import csv, os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from .dataio.datasets_flat import FlatPlantDataset, PLANT_NEAR_M, PLANT_FAR_M
from .features.dino_backbone import DINOBackbone
from .geometry.lift import lift_features_to_grid
from .train.trainer_vol import VolumetricModel, multiview_consistency_loss, DEFAULT_GRID_RES
from .train.trainer_refine import RefinementModel
from .models.renderer.render import sample_grid, volume_render
from .models.refinement.losses import geometry_smoothness
from .models.biomass import BiomassProxy, BiomassHead, calibrate_biomass_head
from .eval.metrics_img import psnr as psnr_metric
from .eval.metrics_3d import reconstruction_metrics
from .eval.metrics_biomass import biomass_metrics
from .eval.mesh import marching_cubes_from_sigma

IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
RENDER_NEAR = 0.5
RENDER_FAR  = 2.5

def depth_init_sigma(dataset, grid_res, device, depth_near=0.4, depth_far=2.5, dilation=1):
    """
    Build initial sigma field by back-projecting depth maps into the voxel grid.
    This directly encodes the sensor geometry rather than inferring it from RGB.
    """
    D, H, W = grid_res
    occupancy = torch.zeros(D, H, W, device=device)
    count = 0
    for sample in dataset:
        dep = sample.get("depth")
        if dep is None: continue
        dep = dep.to(device)  # (Hd, Wd) in metres
        K = sample["K"].to(device)
        R = sample["R"].to(device)  # R_cw
        t = sample["t"].to(device)  # t_cw
        Hd, Wd = dep.shape
        # Build pixel grid
        ys = torch.arange(Hd, device=device).float()
        xs = torch.arange(Wd, device=device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # Normalise to camera frame (assume K is for original image, rescale)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        # Rescale K for depth map resolution
        sx = Wd / float(sample["image"].shape[2])
        sy = Hd / float(sample["image"].shape[1])
        fx_d, fy_d = fx*sx, fy*sy
        cx_d, cy_d = cx*sx, cy*sy
        # Back-project: direction in camera frame
        dx = (xx - cx_d) / fx_d
        dy = (yy - cy_d) / fy_d
        dirs_cam = torch.stack([dx, dy, torch.ones_like(dx)], dim=-1)  # (Hd, Wd, 3)
        # Scale by depth
        valid = (dep > depth_near) & (dep < depth_far)
        d_vals = dep.unsqueeze(-1)  # (Hd, Wd, 1)
        pts_cam = dirs_cam * d_vals  # (Hd, Wd, 3) in camera frame
        # Transform to world: p_world = R_cw^T (p_cam - t_cw)
        pts_flat = pts_cam.reshape(-1, 3)  # (N, 3)
        pts_world = (pts_flat - t.unsqueeze(0)) @ R  # (N, 3) world coords
        # Map to voxel indices [-1,1]^3 -> [0, grid_size-1]
        pts_norm = (pts_world + 1.0) * 0.5  # [0, 1]^3
        ix = (pts_norm[:, 0] * (W-1)).long()
        iy = (pts_norm[:, 1] * (H-1)).long()
        iz = (pts_norm[:, 2] * (D-1)).long()
        valid_flat = valid.reshape(-1)
        mask = valid_flat & (ix>=0) & (ix<W) & (iy>=0) & (iy<H) & (iz>=0) & (iz<D)
        ix, iy, iz = ix[mask], iy[mask], iz[mask]
        # Scatter into occupancy grid
        flat_idx = iz * H * W + iy * W + ix
        occupancy.reshape(-1).scatter_add_(0, flat_idx,
            torch.ones(flat_idx.shape[0], device=device))
        count += 1
    if count == 0: return None
    # Normalise and clamp
    occ_norm = (occupancy / occupancy.max().clamp(min=1)).clamp(0, 1)
    # Dilate if requested
    if dilation > 0:
        import torch.nn.functional as F_
        occ_norm = F_.max_pool3d(
            occ_norm.unsqueeze(0).unsqueeze(0),
            kernel_size=2*dilation+1, stride=1, padding=dilation
        )[0, 0]
    # Shape to match model sigma output: (1, 1, D, H, W)
    return occ_norm.unsqueeze(0).unsqueeze(0)


def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_image(rgb, path): Image.fromarray((np.clip(rgb*255,0,255)).astype(np.uint8)).save(path)

def estimate_intrinsics(image, fx_scale=0.9):
    _,h,w=image.shape; f=max(h,w)*fx_scale
    return {"fx":float(f),"fy":float(f),"cx":float(w/2),"cy":float(h/2)}

def scale_intrinsics(K, orig, tgt):
    K=K.clone().float()
    K[0,0]*=tgt[1]/orig[1]; K[0,2]*=tgt[1]/orig[1]
    K[1,1]*=tgt[0]/orig[0]; K[1,2]*=tgt[0]/orig[0]
    return K

def preprocess_image(image, size=224, device=None):
    if image.ndim==3: image=image.unsqueeze(0)
    if device: image=image.to(device)
    image=F.interpolate(image,size=(size,size),mode="bilinear",align_corners=False)
    m=IMAGENET_MEAN.to(image.device).view(1,3,1,1)
    s=IMAGENET_STD.to(image.device).view(1,3,1,1)
    return (image-m)/s

def build_dino_backbone(model_name="vit_base_patch16_224.dino",pretrained=True,freeze=True,unfreeze_last_n=0):
    try: return DINOBackbone(model_name=model_name,pretrained=pretrained,freeze=freeze,unfreeze_last_n_blocks=unfreeze_last_n)
    except Exception as e:
        print(f"DINO pretrained failed ({e}), using random init.")
        return DINOBackbone(model_name=model_name,pretrained=False,freeze=False)

def build_flat_dataset(folder, plant_id=1, baseline_m=0.40):
    rgb_files=sorted(p for p in os.listdir(folder)
                     if "RGB" in p.upper() and p.lower().endswith((".jpg",".jpeg",".png")))
    if not rgb_files: raise ValueError(f"No RGB files in {folder}.")
    img=Image.open(os.path.join(folder,rgb_files[0]))
    t=torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1)
    intr=estimate_intrinsics(t)
    return FlatPlantDataset(folder,plant_id=plant_id,intr_red=intr,intr_green=intr,baseline_m=baseline_m)

def compute_camera_rays(K,R,t,height,width):
    dev=K.device
    ys=torch.arange(0.5,height+0.5,device=dev); xs=torch.arange(0.5,width+0.5,device=dev)
    jj,ii=torch.meshgrid(ys,xs,indexing="ij")
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    dirs=torch.stack([(ii-cx)/fx,(jj-cy)/fy,torch.ones_like(ii)],dim=-1)
    dirs=dirs/torch.norm(dirs,dim=-1,keepdim=True)
    return (-R.T@t).view(3), dirs@R.T

def render_volume_tensors(sigma,color,K,R,t,output_size=(128,128),n_samples=64,
                          near=RENDER_NEAR,far=RENDER_FAR):
    dev=sigma.device; H,W=output_size
    origin,dirs=compute_camera_rays(K,R,t,H,W)
    t_vals=torch.linspace(near,far,n_samples,device=dev).view(1,1,-1,1)
    pts=origin.view(1,1,1,3)+dirs.unsqueeze(-2)*t_vals
    pts_flat=((pts+1.0)*0.5).reshape(1,-1,3)
    sigma_s=sample_grid(sigma,pts_flat).view(1,1,H*W,n_samples)
    color_s=sample_grid(color,pts_flat).view(1,3,H*W,n_samples)
    delta=(far-near)/n_samples
    alphas=1.0-torch.exp(-sigma_s*delta)
    trans=torch.cumprod(torch.cat([torch.ones(1,1,H*W,1,device=dev),(1.0-alphas+1e-10)],dim=-1),dim=-1)[...,:-1]
    weights=alphas*trans
    rgb=(color_s*weights).sum(-1).view(1,3,H,W).clamp(0,1)
    acc=weights.sum(-1).view(1,1,H,W).clamp(0,1)
    t_mid=torch.linspace(near,far,n_samples,device=dev).view(1,1,-1)
    depth=(weights.squeeze(1)*t_mid).sum(-1).view(1,1,H,W)
    return rgb,acc,depth

def render_volume_image(sigma,color,K,R,t,output_size=(256,256),n_samples=96):
    r,a,d=render_volume_tensors(sigma,color,K,R,t,output_size,n_samples)
    return r[0].permute(1,2,0).detach().cpu().numpy(),a[0,0].detach().cpu().numpy(),d[0,0].detach().cpu().numpy()

def _extract_features(backbone,x):
    if any(p.requires_grad for p in backbone.parameters()): return backbone(x)
    with torch.no_grad(): return backbone(x)

def average_dino_features(dataset,backbone,device,image_size=224,save_dir=None):
    backbone.eval(); feat_sum=None; view_feats=[]; depth_maps=[]; count=0
    for idx,sample in enumerate(dataset):
        image=sample["image"].to(device)
        depth=sample.get("depth")
        if depth is not None: depth=depth.to(device)
        f2d=_extract_features(backbone,preprocess_image(image,size=image_size,device=device))
        view_feats.append(f2d.detach().clone()); depth_maps.append(depth)
        feat_sum=f2d.detach().clone() if feat_sum is None else feat_sum+f2d.detach()
        count+=1
        if save_dir:
            ensure_dir(save_dir)
            np.savez_compressed(os.path.join(save_dir,f"view_{idx:02d}.npz"),feature=f2d.cpu().numpy())
    if count==0: raise ValueError("Dataset empty.")
    return feat_sum/float(count), view_feats, depth_maps

def train_volumetric_model(dataset,backbone,model,device,output_dir,epochs=12,
                           image_size=224,render_size=(128,128),n_samples=48,
                           lambda_consistency=0.1,lambda_depth=0.5,
                           depth_near=PLANT_NEAR_M,depth_far=PLANT_FAR_M,
                           depth_sigma=None):
    ensure_dir(output_dir)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs*len(dataset)))
    stats={"losses":[],"depth_losses":[],"consistency_losses":[]}
    model.train()
    for epoch in range(epochs):
        ep_r=ep_d=ep_c=0.0; all_f2d=[]; all_dep=[]
        for sample in dataset:
            image=sample["image"].to(device)
            K=sample["K"].to(device); R=sample["R"].to(device); t=sample["t"].to(device)
            dgt=sample.get("depth")
            if dgt is not None: dgt=dgt.to(device)
            target=F.interpolate(image.unsqueeze(0),size=render_size,mode="bilinear",align_corners=False)
            f2d=_extract_features(backbone,preprocess_image(image,size=image_size,device=device))
            all_f2d.append(f2d.detach()); all_dep.append(dgt)
            sigma,color,_=model(f2d,depth_map=dgt,depth_near=depth_near,depth_far=depth_far)
            Ks=scale_intrinsics(K,image.shape[1:],render_size)
            rgb_p,_,dep_p=render_volume_tensors(sigma,color,Ks,R,t,render_size,n_samples)
            loss_recon=F.mse_loss(rgb_p,target)
            # Sparsity prior: encourage empty space (avoid fog collapse)
            loss_sparse = 0.001 * sigma.mean()
            # Entropy regulariser via sigma variance (crisp = high variance)
            loss_entropy = torch.tensor(0.0, device=sigma.device)
            loss = loss_recon + loss_sparse + loss_entropy
            if dgt is not None and lambda_depth>0:
                dgt_r=F.interpolate(dgt.unsqueeze(0).unsqueeze(0).float(),size=render_size,mode="bilinear",align_corners=False)[0,0]
                valid=(dgt_r>RENDER_NEAR)&(dgt_r<RENDER_FAR)
                if valid.any():
                    ld=(dep_p[0,0][valid]-dgt_r[valid]).abs().mean()
                    loss=loss+lambda_depth*ld; ep_d+=float(ld.item())
            # Structural loss: sigma should match depth occupancy where available
            if depth_sigma is not None:
                sigma_tgt = F.interpolate(depth_sigma, size=sigma.shape[-3:],
                                         mode="trilinear", align_corners=False)
                loss = loss + 0.2 * F.mse_loss(sigma, sigma_tgt.detach())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); ep_r+=float(loss.item())
        if len(all_f2d)>=2 and lambda_consistency>0:
            c=multiview_consistency_loss(model,all_f2d,all_dep,depth_near,depth_far)
            (lambda_consistency*c).backward(); opt.step(); opt.zero_grad(); ep_c=float(c.item())
        n=float(len(dataset))
        stats["losses"].append(ep_r/n); stats["depth_losses"].append(ep_d/n)
        stats["consistency_losses"].append(ep_c)
        torch.save(model.state_dict(),os.path.join(output_dir,f"vol_epoch_{epoch+1:02d}.pth"))
        print(f"  Epoch {epoch+1:02d}/{epochs}  recon={ep_r/n:.5f}  depth={ep_d/n:.5f}  consist={ep_c:.5f}")
    return stats

def train_refinement_model(dataset,backbone,vol_model,avg_f2d,device,output_dir,epochs=4,
                           image_size=224,render_size=(128,128),n_samples=48,
                           lambda_smooth=0.05,use_ema=True,
                           depth_near=PLANT_NEAR_M,depth_far=PLANT_FAR_M):
    ensure_dir(output_dir)
    with torch.no_grad():
        _,_,base_vol=vol_model(avg_f2d.to(device),depth_near=depth_near,depth_far=depth_far)
    c3d=int(base_vol.shape[1]); c2d=int(avg_f2d.shape[1])
    rm=RefinementModel(c3d=c3d,c2d=c2d,use_ema=use_ema).to(device)
    if use_ema: rm.init_ema(decay=0.99)
    opt=torch.optim.AdamW(rm.parameters(),lr=1e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs*len(dataset)))
    stats={"losses":[]}; rm.train()
    for epoch in range(epochs):
        ep=0.0
        for sample in dataset:
            image=sample["image"].to(device)
            K=sample["K"].to(device); R=sample["R"].to(device); t=sample["t"].to(device)
            target=F.interpolate(image.unsqueeze(0),size=render_size,mode="bilinear",align_corners=False)
            f2d=_extract_features(backbone,preprocess_image(image,size=image_size,device=device))
            sigma_r,color_r,_=rm(base_vol.detach(),f2d.flatten(2).transpose(1,2))
            Ks=scale_intrinsics(K,image.shape[1:],render_size)
            rgb_p,_,_=render_volume_tensors(sigma_r,color_r,Ks,R,t,render_size,n_samples)
            loss=F.mse_loss(rgb_p,target)+lambda_smooth*geometry_smoothness(sigma_r)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(),1.0)
            opt.step(); sch.step(); rm.update_ema(); ep+=float(loss.item())
        ep/=float(len(dataset)); stats["losses"].append(ep)
        torch.save(rm.state_dict(),os.path.join(output_dir,f"refine_epoch_{epoch+1:02d}.pth"))
        print(f"  Refine {epoch+1:02d}/{epochs}  loss={ep:.5f}")
    rm.eval()
    with torch.no_grad():
        sf,cf,_=rm(base_vol,avg_f2d.to(device).flatten(2).transpose(1,2))
    return rm,stats,sf,cf

def evaluate_render_metrics(dataset,sigma,color,device,output_dir,render_size=(256,256),n_samples=96):
    ensure_dir(output_dir); rows=[]; psnr_v=[]; mse_v=[]
    with torch.no_grad():
        for idx,sample in enumerate(dataset):
            image=sample["image"].to(device)
            K=sample["K"].to(device); R=sample["R"].to(device); t=sample["t"].to(device)
            target=F.interpolate(image.unsqueeze(0),size=render_size,mode="bilinear",align_corners=False)
            Ks=scale_intrinsics(K,image.shape[1:],render_size)
            rgb_p,_,_=render_volume_tensors(sigma,color,Ks,R,t,render_size,n_samples)
            mse=float(F.mse_loss(rgb_p,target).item()); pv=float(psnr_metric(rgb_p,target))
            mse_v.append(mse); psnr_v.append(pv); rows.append({"view_idx":idx,"mse":mse,"psnr":pv})
            save_image(rgb_p[0].permute(1,2,0).cpu().numpy(),
                       os.path.join(output_dir,f"render_view{idx:02d}.png"))
            save_image(image.permute(1,2,0).cpu().numpy(),
                       os.path.join(output_dir,f"gt_view{idx:02d}.png"))
    with open(os.path.join(output_dir,"metrics.csv"),"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=["view_idx","mse","psnr"]); w.writeheader(); w.writerows(rows)
    return {"mse_mean":float(np.mean(mse_v)) if mse_v else float("nan"),
            "psnr_mean":float(np.mean(psnr_v)) if psnr_v else float("nan"),
            **reconstruction_metrics(sigma)}

def run_biomass_stage(sigma,vol_feat,device,output_dir,biomass_labels_csv=None,
                      feat_dim=128,calibration_epochs=200):
    ensure_dir(output_dir); result={}
    with torch.no_grad(): pv=float(BiomassProxy(voxel_size_m=0.01).to(device)(sigma).item())
    result["proxy_volume_m3"]=pv; print(f"  Biomass proxy: {pv:.6f} m3")
    if biomass_labels_csv and os.path.isfile(biomass_labels_csv):
        labels_g=[]
        with open(biomass_labels_csv,newline="",encoding="utf-8") as fh:
            for row in csv.DictReader(fh): labels_g.append(float(row["biomass_g"]))
        if labels_g:
            lt=torch.tensor(labels_g,dtype=torch.float32,device=device); n=len(labels_g)
            sb=sigma.expand(n,-1,-1,-1,-1); fb=vol_feat.expand(n,-1,-1,-1,-1)
            head=BiomassHead(feat_dim=feat_dim).to(device)
            cs=calibrate_biomass_head(head,sb,fb,lt,epochs=calibration_epochs)
            with torch.no_grad(): pg=head(sb,fb)
            bm=biomass_metrics(pg,lt); result.update(bm)
            result["calibration_final_loss"]=cs["losses"][-1]
            torch.save(head.state_dict(),os.path.join(output_dir,"biomass_head.pth"))
            print(f"  MAE={bm['mae_g']:.2f}g RMSE={bm['rmse_g']:.2f}g MARE={bm['mare']:.4f}")
    with open(os.path.join(output_dir,"biomass_report.txt"),"w",encoding="utf-8") as fh:
        fh.write("Stage 80 Biomass\n")
        for k,v in result.items(): fh.write(f"{k}: {v}\n")
    return result

def export_volume_artifacts(sigma,color,dataset,device,output_root,
                            render_size=(512,512),n_samples=128,metrics=None,biomass_result=None):
    for tag in ["stage_30_volume","stage_50_render","stage_60_refine","stage_70_eval"]:
        ensure_dir(os.path.join(output_root,tag))
    s30=os.path.join(output_root,"stage_30_volume")
    s50=os.path.join(output_root,"stage_50_render")
    s60=os.path.join(output_root,"stage_60_refine")
    s70=os.path.join(output_root,"stage_70_eval")
    with torch.no_grad():
        np.save(os.path.join(s30,"sigma.npy"),sigma.cpu().numpy())
        np.save(os.path.join(s30,"color.npy"),color.cpu().numpy())
        if len(dataset)>0:
            samp=dataset[0]; K=samp["K"].to(device); R=samp["R"].to(device); t=samp["t"].to(device)
            Ks=scale_intrinsics(K,samp["image"].shape[1:],render_size)
            rgb_np,_,_=render_volume_image(sigma,color,Ks,R,t,render_size,n_samples)
            save_image(rgb_np,os.path.join(s50,"render_hero.png"))
        sn=sigma[0,0].cpu().numpy(); s_min,s_max=float(sn.min()),float(sn.max())
        thresh=0.5 if s_min<0.5<s_max else 0.5*(s_min+s_max)
        mesh=marching_cubes_from_sigma(sn,thresh=thresh)
        mp=os.path.join(s60,"plant_mesh.ply"); mesh.export(mp)
        print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces -> {mp}")
        with open(os.path.join(s70,"report.txt"),"w",encoding="utf-8") as fh:
            fh.write("=== Plant3D Report ===\n")
            fh.write(f"sigma shape: {sn.shape}\n")
            fh.write(f"sigma range: [{s_min:.4f},{s_max:.4f}]\n")
            fh.write(f"mesh thresh: {thresh:.4f}\n")
            fh.write(f"mesh verts: {len(mesh.vertices)}\n")
            fh.write(f"mesh faces: {len(mesh.faces)}\n")
            if metrics:
                fh.write("\n--- Image metrics ---\n")
                for k,v in metrics.items(): fh.write(f"{k}: {v:.6f}\n")
            if biomass_result:
                fh.write("\n--- Biomass ---\n")
                for k,v in biomass_result.items(): fh.write(f"{k}: {v}\n")

def run_pipeline(data_folder,plant_id=1,output_root="./outputs",device_name="cuda",
                 epochs=12,baseline_m=0.40,image_size=224,render_size=(256,256),
                 n_render_samples=96,refine_epochs=4,
                 dino_model_name="vit_base_patch16_224.dino",pretrained_dino=True,freeze_dino=True,
                 lambda_consistency=0.1,lambda_depth=0.5,use_ema_refine=True,
                 biomass_labels_csv=None,biomass_calibration_epochs=200,
                 grid_res=DEFAULT_GRID_RES,depth_near=PLANT_NEAR_M,depth_far=PLANT_FAR_M):
    device=torch.device(device_name if (torch.cuda.is_available() or device_name=="cpu") else "cpu")
    if device.type=="cuda": torch.backends.cudnn.benchmark=True
    output_root=os.path.abspath(output_root); ensure_dir(output_root)
    print(f"\n{'='*60}")
    print(f"Plant3D  plant_id={plant_id}  device={device}  grid={grid_res}")
    print(f"depth=[{depth_near},{depth_far}]m  epochs={epochs}  refine={refine_epochs}")
    print(f"{'='*60}\n")

    stage20=os.path.join(output_root,"stage_20_dino"); ensure_dir(stage20)
    print("[Stage 20] DINO features ...")
    dataset=build_flat_dataset(data_folder,plant_id=plant_id,baseline_m=baseline_m)
    backbone=build_dino_backbone(dino_model_name,pretrained=pretrained_dino,freeze=freeze_dino).to(device)
    with torch.no_grad():
        probe=preprocess_image(dataset[0]["image"],size=image_size,device=device)
        feat_dim=backbone(probe).shape[1]
    avg_f2d,view_feats,depth_maps=average_dino_features(
        dataset,backbone,device,image_size=image_size,save_dir=stage20)
    print(f"  {len(dataset)} views  feat_dim={feat_dim}\n")

    model=VolumetricModel(feat_dim=feat_dim,vol_dim=128,grid_res=grid_res).to(device)

    # Depth-based sigma initialisation: inject real geometry before photometric training
    print("[Init] Back-projecting depth maps into voxel grid ...")
    depth_sigma = depth_init_sigma(dataset, grid_res, device, depth_near, depth_far, dilation=1)
    if depth_sigma is not None:
        smax = float(depth_sigma.max().item())
        print(f"  Depth occupancy init: max={smax:.4f}  shape={depth_sigma.shape}")
        # Pre-set density head bias to match depth occupancy
        # sigma_raw = logit(depth_sigma.clamp(0.01, 0.99))  -> stored in the conv bias via a forward pass
        # Simpler: directly scale the depth_sigma to match the model output range during eval
        # We will inject it as a soft constraint in training (not directly overwrite weights)
    print()

    if epochs>0:
        print(f"[Stage 40] Volumetric training ({epochs} epochs) ...")
        stage40=os.path.join(output_root,"stage_40_vol_train")
        stats=train_volumetric_model(dataset,backbone,model,device,stage40,epochs,
                                     image_size,(128,128),32,lambda_consistency,lambda_depth,depth_near,depth_far,
                                     depth_sigma=depth_sigma)
        with open(os.path.join(stage40,"training_stats.txt"),"w") as fh:
            for i,(r,d,c) in enumerate(zip(stats["losses"],stats["depth_losses"],
                                            stats["consistency_losses"]),1):
                fh.write(f"epoch_{i:02d}: recon={r:.6f} depth={d:.6f} consist={c:.6f}\n")
        print(f"  Best recon: {min(stats['losses']):.6f}\n")

    adep=depth_maps[0] if depth_maps else None
    with torch.no_grad():
        sigma_base,color_base,vf_base=model(avg_f2d.to(device),depth_map=adep,depth_near=depth_near,depth_far=depth_far)
    sf,cf,vf=sigma_base,color_base,vf_base

    if refine_epochs>0:
        print(f"[Stage 60] Refinement ({refine_epochs} epochs) ...")
        stage60=os.path.join(output_root,"stage_60_refine")
        rm,rs,sigma_r,color_r=train_refinement_model(dataset,backbone,model,avg_f2d,device,
                                                      stage60,refine_epochs,image_size,(128,128),32,
                                                      0.05,use_ema_refine,depth_near,depth_far)
        sf,cf=sigma_r,color_r
        with torch.no_grad():
            _,_,vf=rm(vf_base,avg_f2d.to(device).flatten(2).transpose(1,2))
        print(f"  Best refine: {min(rs['losses']):.6f}\n")

    print("[Stage 70] Evaluation ...")
    stage70=os.path.join(output_root,"stage_70_eval")
    metrics=evaluate_render_metrics(dataset,sf,cf,device,stage70,render_size,n_render_samples)
    print(f"  MSE={metrics['mse_mean']:.5f}  PSNR={metrics['psnr_mean']:.2f}dB  occ={metrics['occupancy_ratio']:.4f}\n")

    print("[Stage 80] Biomass ...")
    br=run_biomass_stage(sf,vf,device,os.path.join(output_root,"stage_80_biomass"),
                         biomass_labels_csv,int(vf.shape[1]),biomass_calibration_epochs)

    print("\n[Export] ...")
    export_volume_artifacts(sf,cf,dataset,device,output_root,render_size,n_render_samples,metrics,br)
    print(f"\n{'='*60}\nDone -> {output_root}\n{'='*60}\n")
    return {"output_root":output_root,"dataset_size":len(dataset),"dino_feature_dim":feat_dim,
            "grid_res":grid_res,"epochs":epochs,"refine_epochs":refine_epochs,"metrics":metrics,"biomass":br}
