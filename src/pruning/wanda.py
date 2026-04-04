from __future__ import annotations
import gc
import torch
import torch.nn as nn
from typing import Dict, Any, List

from tqdm import tqdm

from ..lib.util import *
from ..lib.data import *

def _compute_position_embeddings(model, inp_dev, pos_ids_dev):
    """
    Compute (cos, sin) position embeddings directly, bypassing the need to
    locate and correctly call the rotary_emb module.
    Uses the same logic as LlamaRotaryEmbedding.forward() internally.
    """
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = getattr(config, "rope_theta", 10000.0)
    dim = head_dim  # rotary operates on head_dim, not hidden_size

    device = inp_dev.device
    dtype = inp_dev.dtype

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    # pos_ids_dev: [1, seqlen]
    freqs = torch.outer(pos_ids_dev[0].float(), inv_freq)  # [seqlen, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                # [seqlen, dim]
    cos = emb.cos()[None, :, :].to(dtype=dtype)            # [1, seqlen, dim]
    sin = emb.sin()[None, :, :].to(dtype=dtype)            # [1, seqlen, dim]
    return cos, sin

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model_type = model.config.model_type
    if model_type in ("llama", "mistral"):
        layers = model.model.layers
    elif model_type == "opt":
        layers = model.model.decoder.layers
    elif model_type in ("gpt2", "gptj"):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    hf_device_map = getattr(model, "hf_device_map", {})
    if "model.embed_tokens" in hf_device_map:
        device = hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    # Use a list — avoids pre-allocating a large CPU/GPU tensor
    cache = {'inps': [], 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            cache['inps'].append(inp.detach().squeeze(0).cpu().to(dtype=dtype))  # store as [seqlen, hidden]
            if cache['attention_mask'] is None:
                am = kwargs.get('attention_mask', None)
                cache['attention_mask'] = am.cpu() if am is not None else None
            if cache['position_ids'] is None:
                pi = kwargs.get('position_ids', None)
                cache['position_ids'] = pi.cpu() if pi is not None else None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(input_ids=batch[0].to(device))
        except ValueError:
            pass
        torch.cuda.empty_cache()
    layers[0] = layers[0].module

    inps = cache['inps']           # list of (1, seqlen, hidden) CPU tensors
    outs = [None] * len(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    gc.collect()
    torch.cuda.empty_cache()

    return inps, outs, attention_mask, position_ids


def _to_dev(t, dev):
    """Move a tensor or tuple of tensors to device; pass through None."""
    if t is None:
        return None
    if isinstance(t, tuple):
        return tuple(x.to(dev) if isinstance(x, torch.Tensor) else x for x in t)
    return t.to(dev)

class WandaPruning:
    def compute_masks(self, model: nn.Module, prune_cfg: Dict[str, Any], tokenizer, device=torch.device("cuda:0")):

        ratio = prune_cfg.get("ratio", 0.5)
        prune_n = prune_cfg.get("prune_n", 0)
        prune_m = prune_cfg.get("prune_m", 0)
        nsamples = prune_cfg.get("nsamples", 128)
        seed = prune_cfg.get("seed", 0)

        if prune_n != 0:
            print(f"N:M Sparsity: {prune_n}:{prune_m}")

        seqlen = min(
            getattr(model.config, "max_position_embeddings",
            getattr(model.config, "n_positions",
            getattr(model.config, "max_seq_len", 512))),
            2048
        )

        model_type = model.config.model_type
        if model_type in ("llama", "mistral"):
            layers = model.model.layers
        elif model_type == "opt":
            layers = model.model.decoder.layers
        elif model_type in ("gpt2", "gptj"):
            layers = model.transformer.h
            seqlen = min(model.config.n_positions, 2048)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print("loading calibration data")
        dataloader, _ = get_loaders("wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
        print("dataset loading complete")

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

        nsamples = len(inps)

        masks: Dict[str, torch.Tensor] = {}
        infos: List[MaskInfo] = []

        for i, layer in enumerate(layers):
            subset = find_layers(layer)

            hf_device_map = getattr(model, "hf_device_map", {})
            if f"model.layers.{i}" in hf_device_map:
                dev = hf_device_map[f"model.layers.{i}"]
            else:
                dev = device

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

            # Forward pass to collect activation stats
            for j in range(nsamples):
                with torch.no_grad():
                    inp_dev = inps[j].to(dev)
                    if inp_dev.dim() == 2:          # [seqlen, hidden] → [1, seqlen, hidden]
                        inp_dev = inp_dev.unsqueeze(0)
                    seqlen = inp_dev.shape[1]
                    pos_ids_dev = torch.arange(seqlen, device=dev).unsqueeze(0)
                    position_embeddings = _compute_position_embeddings(model, inp_dev, pos_ids_dev)
                    out = layer(
                        inp_dev,
                        attention_mask=None,
                        position_ids=pos_ids_dev,
                        position_embeddings=position_embeddings,
                    )[0]
                    outs[j] = out.squeeze(0).cpu()  # store as [seqlen, hidden]
                torch.cuda.empty_cache()

            for h in handles:
                h.remove()

            for name in subset:
                param_name = f"model.layers.{i}.{name}.weight"
                W_norm = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape(1, -1))
                W_mask = (torch.zeros_like(W_norm) == 1)

                if prune_n != 0:
                    for j in range(0, W_norm.shape[1], prune_m):
                        temp = W_norm[:, j:j+prune_m].float()
                        n_to_prune = min(prune_n, temp.shape[1])
                        lowk_i = torch.topk(temp, n_to_prune, dim=1, largest=False)[1]
                        W_mask.scatter_(1, j + lowk_i, True)

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
                    while (torch.abs(curr_sparsity - ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] > 0.001):
                        if curr_sparsity > ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, curr_sparsity = return_given_alpha(alpha, sort_res, W_norm, temp_norm, pre_sum)

                    sparsity_achieved = curr_sparsity.item()

                subset[name].weight.data[W_mask] = 0
                masks[param_name] = ~W_mask
                zeros = int(W_mask.sum().item())
                infos.append(MaskInfo(param_name, sparsity_achieved, W_mask.numel(), zeros))

            # Second forward pass with pruned weights to propagate activations to next layer
            for j in range(nsamples):
                with torch.no_grad():
                    inp_dev = inps[j].to(dev)
                    if inp_dev.dim() == 2:          # [seqlen, hidden] → [1, seqlen, hidden]
                        inp_dev = inp_dev.unsqueeze(0)
                    seqlen = inp_dev.shape[1]
                    pos_ids_dev = torch.arange(seqlen, device=dev).unsqueeze(0)
                    position_embeddings = _compute_position_embeddings(model, inp_dev, pos_ids_dev)
                    out = layer(
                        inp_dev,
                        attention_mask=None,
                        position_ids=pos_ids_dev,
                        position_embeddings=position_embeddings,
                    )[0]
                    outs[j] = out.squeeze(0).cpu()  # store as [seqlen, hidden]
                torch.cuda.empty_cache()

            # Swap: outs become inputs for the next layer
            inps, outs = outs, [None] * nsamples

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
            k: (v["zeros"] / v["numel"] if v["numel"] else 0.0) for k, v in per_layer.items()
        }

        return {
            "overall_sparsity": overall,
            "num_tensors_masked": len(infos),
            "num_params_masked": zeros,
            "total_params": total,
            "sparsity_by_layer": sparsity_by_layer
        }