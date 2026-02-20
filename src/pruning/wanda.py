import torch
import torch.nn as nn

from ..lib.util import *

def wanda_prune(model, ratio=0.5, prune_n=0, prune_m=0, nsamples=128):

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps,outs,attention_mask,position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev),position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_,inp,out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            W_norm = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_mask = (torch.zeros_like(W_norm)==1)

            if prune_n != 0:
                for j in range(W_norm.shape[1]):
                    temp = W_norm[:,j:j+prune_m].float()
                    lowk_i = torch.topk(temp, prune_n, 1,False)[1]
                    W_mask.scatter_(1,j+lowk_i,True)
            else:
                sort_res = torch.sort(W_norm, dim=-1, stable=True)

                temp_norm = torch.cumsum(sort_res[0], dim=1)
                pre_sum = W_norm.sum(dim=1)

                alpha = 0.4
                alpha_hist = [0., 0.8]

                W_mask, curr_sparsity = return_given_alpha(alpha, sort_res, W_norm, temp_norm, pre_sum)
                while (torch.abs(curr_sparsity - ratio) > 0.001) and (alpha_hist[1]-alpha_hist[0]>0.001):
                    if curr_sparsity > ratio:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha

                    alpha = alpha_new
                    W_mask,curr_sparsity = return_given_alpha(alpha, sort_res, W_norm, temp_norm, pre_sum)
                
            subset[name].weight.data[W_mask] = 0

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),attention_mask=attention_mask, position_ids=position_ids)[0]
        
        outs, inps = inps, outs
