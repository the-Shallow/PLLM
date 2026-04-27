from __future__ import annotations

import os
import json
from typing import Any, Dict, List
# from venv import logger

from ..runner.report import print_rich_table, save_charts
from src.eval.judge import judge_outputs_file
from .metrics import compute_auprc, compute_auroc, compute_prr, aggregate_metrics
# from src.eval.report import save_plots  # assuming you already have this


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_path(profile_cfg, key, default):
    return profile_cfg.get("paths", {}).get(key, default)


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_postprocess(input_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    judged_path = os.path.join(outdir, "judged_outputs.json")
    metrics_path = os.path.join(outdir, "metrics.json")

    print("🔍 Step 1: Judging with LLM...")
    judge_outputs_file(
        input_path=input_path,
        output_path=judged_path,
        model="gpt-5.4-mini",
    )

    print("📊 Step 2: Computing metrics...")
    judged_outputs = load_json(judged_path)
    for entry in judged_outputs:
            lns_score = entry.get("lns_score")
            is_correct = entry.get("is_correct")

            # if lns_score is None or is_correct is None:
                # logger.warning(f"Entry {entry['id']} is missing lns_score or is_correct. Skipping.")
                # continue

            # is_certain = lns_score > lns_threshold
            # entry["is_certain"] = is_certain
            is_certain = entry.get("is_certain")

            if is_correct and is_certain:
                entry["bucket_label"] = "correct_certain"
            elif is_correct and not is_certain:
                entry["bucket_label"] = "correct_uncertain"
            elif not is_correct and is_certain:
                entry["bucket_label"] = "incorrect_certain"
            else:
                entry["bucket_label"] = "incorrect_uncertain"
            
            # logger.debug(f"Entry {entry['id']} - lns_score: {lns_score}, is_correct: {is_correct}, is_certain: {is_certain}, bucket_label: {entry['bucket_label']}")

    metrics = aggregate_metrics(judged_outputs)
    save_json(metrics_path, metrics)

    # base_output_dir = get_path(profile_cfg,"output_dir","outputs")
    # out_dir = os.path.join(base_output_dir, run_id)
    # os.makedirs(out_dir, exist_ok=True)

    print("📈 Step 3: Generating plots...")
    # save_plots(judged_outputs, metrics, outdir)
    print_rich_table(judged_outputs, metrics)
    save_charts(judged_outputs, metrics, outdir)

    print("✅ Done!")
    print(f"📁 Outputs saved to: {outdir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    run_postprocess(args.input, args.outdir)