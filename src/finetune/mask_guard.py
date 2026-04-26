from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

MaskDict = Dict[str, torch.Tensor]

class MaskedFineTuneGuard:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.name_to_param = dict(model.named_parameters())

        # self.validate_masks()

    
    @torch.no_grad()
    def apply_masks(self):
        print(self.masks)
        for name, mask in self.masks.items():
            param = self.name_to_param[name]
            mask = mask.to(device=param.device, dtype=param.dtype)
            param.data.mul_(mask)

    
    @torch.no_grad()
    def zero_pruned_gradients(self):
        for name, mask in self.masks.items():
            param = self.name_to_param[name]
            if param.grad is None:
                continue

            mask = mask.to(device=param.grad.device, dtype=param.grad.dtype)
            param.grad.mul_(mask)

    
    @torch.no_grad()
    def count_pruned_nonzero_weights(self):
        total_nonzero = 0
        for name, mask in self.masks.items():
            param = self.name_to_param[name]
            mask = mask.to(device=param.device, dtype=torch.bool)

            pruned_values = param.data[~mask]
            total_nonzero ++ int((pruned_values != 0).sum().item())

        return total_nonzero
    

