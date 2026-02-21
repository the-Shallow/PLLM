from typing import Any, Dict
import json
import os
import time
import torch

from src.models.load_model import load_model
from src.pruning.registry import get_pruner
from src.eval.generate import generate

def run_experiment(cfg):
    run_id = cfg["experiment"]["name"] + "_" + str(int(time.time()))

    out_dir = os.path.join("outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)


    model, tokenizer = load_model(cfg["model"])

    pr_cfg = cfg.get("prune",{})
    metrics = None
    if pr_cfg.get("enabled", False):
        pruner = get_pruner(pr_cfg.get("method", "magnitude"))
        masks, infos = pruner.compute_masks(model, pr_cfg)
        pruner.apply_masks(model, masks)
        summary = pruner.summarize(infos)

        torch.save({k:v.cpu() for k, v in masks.items()}, os.path.join(out_dir,"masks.pt"))
        with open(os.path.join(out_dir,"prune_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        metrics = {
            "run_id" : run_id,
            "prune_summary": summary["overall_sparsity"],
            "num_params_masked": summary["num_params_masked"]
        }
        with open(os.path.join(out_dir,"metrics.json"), "w") as f:
            json.dump(metrics,f, indent=2)


    if cfg["eval"]["mode"] == "prompt_test":
        outputs = []
        for prompt in cfg["eval"]["prompts"]:
            text = generate(model, tokenizer, prompt)
            outputs.append({"prompt": prompt, "output": text})
        
        with open(os.path.join(out_dir,"prompt_summary.json"), "w") as f:
            json.dump(outputs, f, indent=2)
    
    return metrics if metrics else {"run_id": run_id}