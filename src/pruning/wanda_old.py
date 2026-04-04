from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, List

from tqdm import tqdm

from ..lib.util import *
from ..lib.data import *

import gc

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("prep calibration")
    seqlen = getattr(model.config, "max_position_embeddings",
             getattr(model.config, "n_positions",
             getattr(model.config, "max_seq_len", 512)))
    model_type = model.config.model_type
    if model_type in ("llama", "mistral"):
        layers = model.model.layers
    elif model_type == "opt":
        layers = model.model.decoder.layers
    elif model_type in ("gpt2", "gptj"):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # dev = model.hf_device_map["model.embed_tokens"]
    hf_device_map = getattr(model, "hf_device_map", {})
    if "model.embed_tokens" in hf_device_map:
        device = hf_device_map["model.embed_tokens"]

    # dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((len(dataloader), seqlen, model.config.hidden_size), dtype=dtype, device=device)
    # inps.requires_grad = False
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # class Catcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module
    #     def forward(self, inp, *args, **kwargs):
    #         inps[cache['i']] = inp
    #         cache['i'] += 1
    #         cache['attention_mask'] = kwargs.get('attention_mask', None)
    #         cache['position_ids'] = kwargs.get('position_ids', None)
    #         cache['position_embeddings'] = kwargs.get('position_embeddings', None)
    #         raise ValueError
    # layers[0] = Catcher(layers[0])
    # for batch in dataloader:
    #     try:
    #         # GPT2 Fix
    #         #model(batch[0].to(device))
    #         model(input_ids=batch[0].to(device))
    #     except ValueError:
    #         pass 
    # layers[0] = layers[0].module

    # outs = torch.zeros_like(inps)
    dtype = next(iter(model.parameters())).dtype
    cache = {'inps': [], 'attention_mask': None, "position_ids": None, "position_embeddings": None}
    print("catch")
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            # Store on CPU immediately — don't hold 128 samples on GPU
            cache['inps'].append(inp.detach().cpu())
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            cache['position_ids'] = kwargs.get('position_ids', None)
            cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    torch.cuda.empty_cache()
    gc.collect()
    for idx, batch in enumerate(dataloader):
        print(f"  Batch {idx}...")
        try:
            model(input_ids=batch[0].to(device))
        except ValueError:
            print(f"  Batch {idx} caught OK, cache size: {len(cache['inps'])}")
        except Exception as e:
            print(f"  Batch {idx} unexpected exception: {type(e).__name__}: {e}")
            raise e
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"  Batch {idx} loop iteration complete")
    layers[0] = layers[0].module
    inps = cache['inps']
    outs = [None] * len(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings

class WandaPruning:
    def compute_masks(self,model: nn.Module, prune_cfg: Dict[str, Any], tokenizer, device=torch.device("cuda:0")):
        print("wanda pruning")
        ratio = prune_cfg.get("ratio", 0.5)
        prune_n = prune_cfg.get("prune_n", 0)
        prune_m = prune_cfg.get("prune_m", 0)
        nsamples = prune_cfg.get("nsamples", 64)
        seed = prune_cfg.get("seed", 0)
        
        if(prune_n!=0):
            print(f"N:M Sparsity: {prune_n}:{prune_m}")

        seqlen = getattr(model.config, "max_position_embeddings",
                 getattr(model.config, "n_positions",
                 getattr(model.config, "max_seq_len", 512)))
        model_type = model.config.model_type
        if model_type in ("llama", "mistral"):
            layers = model.model.layers
        elif model_type == "opt":
            layers = model.model.decoder.layers
        elif model_type in ("gpt2", "gptj"):
            layers = model.transformer.h
            seqlen = model.config.n_positions
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print("loading calibdation data")
        dataloader, _ = get_loaders("wikitext2",nsamples=nsamples,seed=seed,seqlen=seqlen,tokenizer=tokenizer)
        print("dataset loading complete")
        with torch.no_grad():
            inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)


        masks: Dict[str, torch.Tensor] = {}
        infos: List[MaskInfo] = []
        print(len(layers))
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            # Need to change how hf_device_map is defined per
            hf_device_map = getattr(model, "hf_device_map", {})
            if f"model.layers.{i}" in hf_device_map:
                dev = hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids, position_embeddings = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                    position_embeddings.to(dev),
                )

            # Register hooks to collect activation statistics
            wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [
                subset[name].register_forward_hook(add_batch(name))
                for name in wrapped_layers
            ]

            # for j in range(nsamples):
            #     with torch.no_grad():
            #         outs[j] = layer(
            #             inps[j].unsqueeze(0),
            #             attention_mask=attention_mask,
            #             position_ids=position_ids,
            #             position_embeddings=position_embeddings
            #         )[0]
            print("loading")
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]

            for h in handles:
                h.remove()
            print(subset)
            for name in subset:
                param_name = f"transformer.h.{i}.{name}.weight"
                W_norm = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape(1, -1))
                W_mask = (torch.zeros_like(W_norm)==1)

                if prune_n != 0:
                    for j in range(0, W_norm.shape[1], prune_m):
                        temp = W_norm[:, j:j+prune_m].float()
                        n_to_prune = min(prune_n, temp.shape[1]) #handle last partial block <m wide
                        lowk_i = torch.topk(temp, n_to_prune, dim=1, largest=False)[1]
                        W_mask.scatter_(1, j + lowk_i, True)
                    
                    #calculate sparsity_achieved for N:M pruning
                    num_zeros = (~W_mask).sum().item()
                    num_total = W_mask.numel()
                    sparsity_achieved = num_zeros / num_total
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

                    sparsity_achieved = curr_sparsity.item()
                    
                subset[name].weight.data[W_mask] = 0
                masks[param_name] = ~W_mask
                zeros = int(W_mask.sum().item())
                infos.append(MaskInfo(param_name, sparsity_achieved, W_mask.numel(), zeros))

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0),attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

            inps, outs = outs, inps

        return masks, infos

    def apply_masks(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        model_type = model.config.model_type
        if model_type in ("llama", "mistral"):
            layers = model.model.layers
        elif model_type == "opt":
            layers = model.model.decoder.layers
        elif model_type in ("gpt2", "gptj"):
            layers = model.transformer.h
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            for name in subset:
                param_name = f"model.layers.{i}.{name}.weight"
                if param_name not in masks:
                    continue
                mask = masks[param_name]
                W = subset[name].weight.data
                # Zero out pruned weights (mask=False means pruned)
                W.mul_(mask.to(dtype=W.dtype, device=W.device))

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
            "num_tensors_masked": len(infos),
            "num_params_masked": zeros,
            "total_params" : total,
            "sparsity_by_layer": sparsity_by_layer
        }