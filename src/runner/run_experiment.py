from typing import Any, Dict
import json
import os
import time
import torch

from src.models.load_model import load_model
from src.pruning.registry import get_pruner
from src.eval.datasets import load_prompts
from src.eval.generate import generate_with_scores, generate_n_samples
from src.eval.metrics import score_output, aggregate_metrics
from src.runner.report import print_rich_table, save_charts
from src.runner.logging import logger

def get_path(profile_cfg, key, default):
    return profile_cfg.get("paths", {}).get(key, default)

def run_experiment(cfg, profile_cfg):
    run_id = cfg["experiment"]["name"] + "_" + str(int(time.time()))

    # out_dir = os.path.join("outputs", run_id)
    base_output_dir = get_path(profile_cfg,"output_dir","outputs")
    out_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {out_dir}")

    logger.info(f"Loading model and tokenizer")
    model, tokenizer, device = load_model(cfg["model"], profile_cfg)


    # ------------------------------------------------------------------
    # Pruning (optional)
    # ------------------------------------------------------------------
    pr_cfg = cfg.get("prune", {})
    prune_metrics = None
    if pr_cfg.get("enabled", False):
        logger.info(f"Pruning enabled. Method: {pr_cfg.get('method', 'magnitude')}")
        pruner = get_pruner(pr_cfg.get("method", "magnitude"))
        masks, infos = pruner.compute_masks(model, pr_cfg, tokenizer, device)
        pruner.apply_masks(model, masks)
        summary = pruner.summarize(infos)

        # torch.save({k: v.cpu() for k, v in masks.items()}, os.path.join(out_dir, "masks.pt"))
        with open(os.path.join(out_dir, "prune_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        prune_metrics = {
            "prune_summary": summary["overall_sparsity"],
            "num_params_masked": summary["num_params_masked"],
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("eval", {})
    eval_metrics: Dict[str, Any] = {}

    if eval_cfg.get("mode") == "prompt_test":
        logger.info(f"Evaluation mode: Prompt test")
        num_samples = eval_cfg.get("num_samples", 1)
        lns_threshold = eval_cfg.get("lns_threshold", -2.0)
        # entropy_threshold = eval_cfg.get("entropy_threshold", 1.5)  # entropy disabled, using lns_threshold

        records = load_prompts(eval_cfg)
        outputs = []

        for record in records:
            if num_samples > 1:
                samples = generate_n_samples(model, tokenizer, record.prompt, n=num_samples)
                avg_logprob = sum(s.avg_logprob for s in samples) / num_samples
                # avg_entropy = sum(s.avg_entropy for s in samples) / num_samples  # entropy disabled
                output_text = samples[0].text
                sample_list = [
                    {
                        "text": s.text,
                        "lns_score": round(s.avg_logprob, 4),
                        # "entropy": round(s.avg_entropy, 4),  # entropy disabled
                    }
                    for s in samples
                ]
            else:
                result = generate_with_scores(model, tokenizer, record.prompt)
                avg_logprob = result.avg_logprob
                # avg_entropy = result.avg_entropy  # entropy disabled
                output_text = result.text
                sample_list = None

            record_dict = {
                "bucket": record.bucket,
                "answer": record.answer,
                "expected_behavior": record.expected_behavior,
            }
            scores = score_output(record_dict, output_text, avg_logprob, lns_threshold)

            entry: Dict[str, Any] = {
                "id": record.id,
                "bucket": record.bucket,
                "prompt": record.prompt,
                "output": output_text,
                **scores,
            }
            if record.answer is not None:
                entry["answer"] = record.answer
            if record.expected_behavior is not None:
                entry["expected_behavior"] = record.expected_behavior
            if sample_list is not None:
                entry["samples"] = sample_list

            outputs.append(entry)

        with open(os.path.join(out_dir, "prompt_summary.json"), "w") as f:
            json.dump(outputs, f, indent=2)

        eval_metrics = aggregate_metrics(outputs)

        # Print terminal table + save charts
        print_rich_table(outputs, eval_metrics)
        save_charts(outputs, eval_metrics, out_dir)

    # ------------------------------------------------------------------
    # Final metrics.json — always written, merges prune + eval
    # ------------------------------------------------------------------
    metrics: Dict[str, Any] = {"run_id": run_id}
    if prune_metrics:
        metrics.update(prune_metrics)
    if eval_metrics:
        metrics.update(eval_metrics)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
