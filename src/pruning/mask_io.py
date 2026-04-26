from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch

MaskDict = Dict[str, torch.Tensor]

def save_masks(masks: MaskDict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_masks = {
        name: mask.detach().to(device="cpu", dtype=torch.bool) 
        for name, mask in masks.items()
    }

    torch.save(cpu_masks, path)


def load_masks(path:str | Path, device = None):
    path = Path(path)

    if not path.exists():
        return FileNotFoundError(f"Mask file not found : {path}")
    
    masks = torch.load(path, map_location="cpu")

    loaded_masks = {}
    for name, mask in masks.items():
        mask = mask.to(dtype=torch.bool)
        if device is not None:
            mask = mask.to(device)

        loaded_masks[name] = mask
    
    return loaded_masks