from __future__ import annotations
import fnmatch
import re
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
import torch
import torch.nn as nn

@dataclass
class MaskInfo:
    name:str
    sparsity_target: float
    numel: int
    zeros: int

def matches_any(name:str, patterns: List[str]):
    return any(fnmatch.fnmatch(name, p) for p in patterns)

def extract_layer_index(param_name:str):
    m = re.search(r"transformer\.h\.(\d+)\.", param_name)
    if not m:
        return None

    return int(m.group(1))

def sparsity_by_layer(layer_idx: int, schedule: Dict[str, Any]):
    stype = schedule.get("type", "uniform")
    if stype == "uniform":
        return float(schedule.get("sparsity", 0.5))
    
    if stype == "piecewise_by_layer":
        rules = schedule.get("rules", [])
        for r in rules:
            lo, hi = r["layer_range"]
            if lo <= layer_idx <= hi:
                return float(r["sparsity"])
        
        return float(schedule.get("default", 0.0))
    
    raise ValueError(f"Unknown schedule type: {stype}")

def make_mask_by_sparsity(weight: torch.Tensor, sparsity: float):
    if sparsity <= 0.0:
        return torch.ones_like(weight, dtype=torch.bool)
    if sparsity >= 1.0:
        return torch.zeros_like(weight, dtype=torch.bool)
    
    w = weight.detach()
    abs_w = w.float().abs().view(-1)
    k = int(abs_w.numel() * sparsity)

    if k <= 0:
        return torch.ones_like(weight, dtype=torch.bool)
    if k >= abs_w.numel():
        return torch.zeros_like(weight, dtype=torch.bool)

    thresh = torch.kthvalue(abs_w, k).values.item()
    mask = (w.float().abs() > thresh)

    return mask.to(dtype=torch.bool)

class MagnitudePruning:
    def compute_masks(self, model: nn.Module, prune_cfg: Dict[str, Any]):
        include = prune_cfg.get("include", ["*.weight"])
        exclude = prune_cfg.get("exclude", [])
        schedule = prune_cfg.get("schedule", {"type": "uniform", "sparsity": prune_cfg.get("sparsity", 0.5)})

        masks = {}
        infos = []

        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue

            if exclude and matches_any(name, exclude):
                continue
        
            if include and not matches_any(name, include):
                continue
                
            layer_idx = extract_layer_index(name)
            if layer_idx is None:
                s = float(schedule.get("default", schedule.get("sparsity", 0.0)))
            else:
                s = sparsity_by_layer(layer_idx, schedule)
            
            mask = make_mask_by_sparsity(param.data, s)
            masks[name] = mask
            zeros = int((~mask).sum().item())
            infos.append(MaskInfo(name, s, mask.numel(), zeros))

        return masks, infos
    
    def apply_masks(self, model, masks):
        name_to_param = dict(model.named_parameters())
        for name, mask in masks.items():
            if name not in name_to_param:
                raise KeyError(f"Mask name {name} not found")
            p = name_to_param[name]
            p.data.mul_(mask.to(dtype=p.data.dtype, device=p.data.device))

    def summarize(self, infos):
        total = sum(info.numel for info in infos)
        zeros = sum(info.zeros for info in infos)
        overall = (zeros / total) if total > 0 else 0.0

        per_layer = {}
        for i in infos:
            layer = extract_layer_index(i.name)
            key = str(layer) if layer is not None else "none"
            if key not in per_layer:
                per_layer[key] = {"numel": 0, "zeros": 0}
            per_layer[key]["numel"] += i.numel
            per_layer[key]["zeros"] += i.zeros
        
        sparsity_by_layer = {
            k: (v["zeros"] / v["numel"] if v["numel"] else 0.0) for k,v in per_layer.items()
        }

        return {
            "overall_sparsity": overall,
            "num_params_masked": len(infos),
            "sparsity_by_layer": sparsity_by_layer
        }