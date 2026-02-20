import torch
import torch.nn as nn

# Magnitude Pruning testing

prune_n, prune_m = 2,4
# prune_n, prune_m = 0,0
ratio = 0.5
W = ((torch.rand(10, 10)*100)-50).int()
W = W[torch.randperm(W.size()[0])]
W = W[:,torch.randperm(W.size()[1])]
print(W)
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
    threshold = torch.sort(W_norm.flatten())[0][int(W.numel()*ratio)]
    W_mask = (W_norm<=threshold)

W[W_mask] = 0
print(W)