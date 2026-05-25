from __future__ import annotations
import torch

def iou_vox(pred, gt, thresh=0.5):
    p=(pred>=thresh).float(); g=(gt>=0.5).float()
    inter=(p*g).sum(); union=(p+g-p*g).sum()+1e-6
    return (inter/union).item()

def chamfer_distance(pred_pts, gt_pts):
    if pred_pts.shape[0]==0 or gt_pts.shape[0]==0: return float("inf")
    diff=pred_pts.unsqueeze(1)-gt_pts.unsqueeze(0); dist2=(diff**2).sum(-1)
    return (dist2.min(1).values.mean()+dist2.min(0).values.mean()).item()

def fscore(pred_pts, gt_pts, delta=0.02):
    if pred_pts.shape[0]==0 or gt_pts.shape[0]==0: return 0.0
    diff=pred_pts.unsqueeze(1)-gt_pts.unsqueeze(0); dist=(diff**2).sum(-1).sqrt()
    pr=(dist.min(1).values<=delta).float().mean().item()
    rc=(dist.min(0).values<=delta).float().mean().item()
    return (2*pr*rc)/(pr+rc+1e-6)

def sigma_to_pointcloud(sigma, thresh=0.5, max_pts=4096):
    vol=sigma[0,0] if sigma.dim()==5 else sigma.squeeze()
    D,H,W=vol.shape; occ=(vol>=thresh).nonzero(as_tuple=False).float()
    if occ.shape[0]==0: return occ
    occ=occ/torch.tensor([D,H,W],dtype=torch.float32,device=sigma.device).clamp(min=1)
    if occ.shape[0]>max_pts:
        occ=occ[torch.randperm(occ.shape[0],device=sigma.device)[:max_pts]]
    return occ

def reconstruction_metrics(pred_sigma, gt_sigma=None, iou_thresh=0.5, fscore_delta=0.02, max_pts=4096):
    # Use adaptive threshold if nothing exceeds the hard threshold
    s_min, s_max = float(pred_sigma.min()), float(pred_sigma.max())
    adaptive = iou_thresh if s_min < iou_thresh < s_max else 0.5 * (s_min + s_max)
    metrics={"occupancy_ratio":float((pred_sigma>=adaptive).float().mean().item()),
             "sigma_min":s_min, "sigma_max":s_max, "adaptive_thresh":adaptive}
    if gt_sigma is None: return metrics
    metrics["iou"]=iou_vox(pred_sigma,gt_sigma,thresh=iou_thresh)
    pp=sigma_to_pointcloud(pred_sigma,thresh=iou_thresh,max_pts=max_pts)
    gp=sigma_to_pointcloud(gt_sigma,thresh=0.5,max_pts=max_pts)
    metrics["chamfer"]=chamfer_distance(pp,gp)
    metrics["fscore"]=fscore(pp,gp,delta=fscore_delta)
    return metrics
