import torch
import torch.nn as nn

from ..lib.util import *


def magnitude_prune(model, ratio, prune_n=0, prune_m=0):
    """ 
    Loop through the layers in a model and prune the neurons based on their magnitudes. 
    For unstructured pruning, the ratio is applied and the resulting weight tensor is pruned based on the specified ratio
    For structured pruning, the parameters N and M are used to apply N:M pruning to the weight tensor
    
    Inputs:
        model - the pytorch LLM to prune
        ratio - the sparsity ratio
        prune_n, prune_m - The sparsity type N:M
    Outputs:
        None
    """

    layers = model.model.layers
    # Loop through the layers in the specified model
    for i in range(len(layers)):
        layer = layer[i]
        subset = find_layers(layer)
        # Loop through the subsets of the layer
        for name in subset:
            W = subset[name].weight.data
            W_norm = torch.abs(W)

            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                # Iterate over the columns of W_norm
                for j in range(W_norm.shape[1]):
                    if j % prune_m == 0:
                        # Create a temporary M-wide slice of W_norm
                        temp = W_norm[:,j:(j+prune_m)].float()
                        # Find the indices of the N lowest magnitudes in the temporary column slice
                        lowk_i = torch.topk(temp, prune_n, dim=1,largest=False)[1]
                        # Update the W_mask tensor with the indices of the N lowest magnitudes
                        W_mask.scatter_(1, j+lowk_i, True)
            else:
                # Calculate the threshold to maintain the sparsity ratio provided
                threshold = torch.sort(W_norm.flatten().cuda())[0][int(W.numel()*ratio)].cpu()
                W_mask = (W_norm<=threshold)

            # Zero out the neurons below the threshold
            W[W_mask] = 0