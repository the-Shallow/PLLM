import torch
import torch.nn as nn
import re
from dataclasses import dataclass
from transformers.pytorch_utils import Conv1D

@dataclass
class MaskInfo:
    name: str
    sparsity_target: float
    numel: int
    zeros: int

def extract_layer_index(param_name:str):
    m = re.search(r"transformer\.h\.(\d+)\.", param_name)
    if not m:
        return None

    return int(m.group(1))

## Recursively find layers that match the layer name specified and return a dictionary filled with the modules that match
def find_layers(module, layers=(nn.Linear, Conv1D), name=''):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

## loop through the layers in a model and return the global sparsity % of the model
def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

## loop through a singular layer and return it's local sparsity %
def check_layer_sparsity(module):
    pass

def return_given_alpha(alpha, sort_res, W_norm, temp_norm, pre_sum):
    threshold_cumsum = alpha * pre_sum
    sort_mask = temp_norm <= threshold_cumsum.reshape((-1,1))
    threshold = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1,keepdims=True)-1)
    W_mask = (W_norm <= threshold)
    curr_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, curr_sparsity

class WrappedGPT:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.in_features = layer.weight.shape[1]  # input dimension
        self.out_features = layer.weight.shape[0] # output dimension

        # scaler_row accumulates input norms per row (matches input features)
        self.scaler_row = torch.zeros(self.layer.in_features, device=self.layer.weight.device)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
        inp: [batch, in_features] for Linear/Conv1D
        out: [batch, out_features] (not used here)
        """
        if len(inp.shape) == 3:
            # Flatten sequence dimension for transformers
            inp = inp.reshape(-1, inp.shape[-1])  # [batch*seq_len, in_features]

        batch_size = inp.shape[0]

        # Update scaler_row using running average
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        # Compute row-wise norms (per input feature)
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=0) ** 2 / self.nsamples