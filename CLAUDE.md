# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

PLLM is a research framework for pruning and evaluating LLMs. It loads a HuggingFace model, optionally applies weight pruning (zeroing out weights), then runs an evaluation pass. Results are saved to `outputs/<run_id>/`.

## Running Experiments

```bash
# Run from repo root
python -m src.main --experiment configs/experiments/gpt2_mag_layerwise_mmlu.yaml
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the standalone pruning logic test:
```bash
python src/pruning/test.py
```

## Experiment Config Structure

Configs live in `configs/experiments/*.yaml`. Each has three top-level keys:

```yaml
experiment:
  name: <run-name>

model:
  name: <huggingface-model-id>   # e.g. "gpt2", "microsoft/phi-3-mini"

prune:
  enabled: true/false
  method: magnitude               # only "magnitude" is wired into registry
  include: ["transformer.h.*.attn.*.weight", ...]   # fnmatch glob patterns
  exclude: ["wte", "wpe", "ln_", ...]
  schedule:
    type: uniform                 # flat sparsity across all matched layers
    sparsity: 0.5
    # OR
    type: piecewise_by_layer      # different sparsity per layer index range
    rules:
      - {layer_range: [0, 3], sparsity: 0.9}
    default: 0.0

eval:
  mode: prompt_test
  prompts:
    - "Q: What is ..."
```

## Architecture

**Execution flow** (`src/main.py` → `src/runner/run_experiment.py`):
1. Load model + tokenizer from HuggingFace (`src/models/load_model.py`) — device is hardcoded to `"mps"` (Apple Silicon)
2. If `prune.enabled`, get a pruner from the registry, compute masks, apply masks (zero out weights), save `masks.pt` + `prune_summary.json`
3. Run eval (currently only `prompt_test` mode: generate text per prompt, save `prompt_summary.json`)

**Pruning registry** (`src/pruning/registry.py`): maps method name strings → pruner classes. Currently only `"magnitude"` is registered. To add a new method, implement a class with `compute_masks(model, prune_cfg) → (masks, infos)`, `apply_masks(model, masks)`, and `summarize(infos)`, then add it to `PRUNER_REGISTRY`.

**MagnitudePruning** (`src/pruning/magnitude.py`): the active implementation uses `make_mask_by_sparsity` to threshold weights by absolute value. Layer index is extracted via regex matching `transformer.h.(\d+).` (GPT-2 naming). Note: the file contains a merge conflict artifact — the old function-based implementation (`magnitude_prune`) precedes the `=======` marker on line 49; the class-based implementation below that line is what's actually imported.

**Wanda pruning** (`src/pruning/wanda.py`): contains the Wanda algorithm (activation-weighted magnitude) as a standalone function. It is not yet wired into the registry.

**Utilities** (`src/lib/util.py`): `find_layers`, `check_sparsity`, `check_layer_sparsity` are stubs. `WrappedGPT` (used by Wanda) and `return_given_alpha` are implemented.

**Stubs**: `src/eval/datasets.py`, `src/eval/metrics.py`, `src/eval/calibration.py`, `src/runner/logging.py`, `src/runner/io.py` are all currently empty (placeholder files).

## Output Files

Each run creates `outputs/<experiment-name>_<unix-timestamp>/`:
- `masks.pt` — saved weight masks (dict of bool tensors, CPU)
- `prune_summary.json` — overall sparsity, per-layer sparsity breakdown
- `metrics.json` — run_id, overall sparsity, number of params masked
- `prompt_summary.json` — list of `{prompt, output}` for prompt_test mode
