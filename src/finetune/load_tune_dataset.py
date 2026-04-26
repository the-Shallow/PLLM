from __future__ import annotations
from typing import Dict, Any, Optional
from datasets import Dataset, load_dataset

def format_alpaca_prompt(example:Dict[str, Any]):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def load_alpaca_subset(max_rows = 1000, split = "train", seed = None):
    dataset = load_dataset("tatsu-lab/alpaca", split=split)
    if max_rows is not None:
        max_rows = min(max_rows, len(dataset))
        dataset = dataset.select(range(max_rows))

    
    dataset = dataset.map(
        lambda example: {"text": format_alpaca_prompt(example)},
        remove_columns=[],
    )

    return dataset


def save_alpaca_subset(output_path, max_rows=1000, split= "train", seed = None):
    dataset = load_alpaca_subset(
        max_rows=max_rows,
        split=split,
        seed=seed
    )

    dataset.to_json(output_path, orient="records", lines=True)


def load_wikitext_subset(
    max_rows = 1000,
    split = "train",
    seed = None,
    source = "EleutherAI/wikitext_document_level",
    subset = "wikitext-103-v1"
):
    dataset = load_dataset(source, subset, split=split)
    text_column = "page"

    if text_column not in dataset.column_names:
        if "text" in dataset.column_names:
            text_column = "text"

    dataset = dataset.filter(
        lambda example: example.get(text_column) is not None
        and len(str(example.get(text_column)).strip()) > 0
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    if max_rows is not None:
        max_rows = min(max_rows, len(dataset))
        dataset = dataset.select(range(max_rows))

    dataset = dataset.map(
        lambda example: {"text": str(example[text_column]).strip()},
        remove_columns=[]
    )

    return dataset

def load_finetune_dataset(
    dataset_name:str,
    max_rows = 1000,
    split = "train",
    seed = None,
    **kwargs
):
    dataset_name = dataset_name.lower().strip()

    return load_wikitext_subset(
        max_rows=max_rows,
        split=split,
        seed=seed,
        source=kwargs.get("source", "EleutherAI/wikitext_document_level"),
        subset=kwargs.get("subset", "wikitext-103-v1"),
    )