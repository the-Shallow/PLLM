from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptRecord:
    prompt: str
    id: str = ""
    # Bucket determines how correctness is evaluated.
    # Supported: "factual", "unanswerable", "false_premise", "ambiguous"
    bucket: str = "factual"
    # For factual prompts: the expected answer string.
    answer: Optional[str] = None
    # For unanswerable / false_premise / ambiguous prompts:
    # "refuse", "reject_premise", "acknowledge_ambiguity"
    expected_behavior: Optional[str] = None


def load_prompts(eval_cfg: Dict[str, Any]) -> List[PromptRecord]:
    """
    Load prompts from the eval section of an experiment config.

    Supports two formats:

    Plain string (legacy):
        prompts:
          - "Q: What is 2+2?\nA:"

    Structured dict (full):
        prompts:
          - id: fact_001
            bucket: factual
            prompt: "Q: What is the capital of France?\nA:"
            answer: "Paris"
          - id: unans_001
            bucket: unanswerable
            prompt: "Q: What did I eat for breakfast?\nA:"
            expected_behavior: "refuse"
    """
    raw = eval_cfg.get("prompts", [])
    records: List[PromptRecord] = []

    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append(PromptRecord(
                prompt=item,
                id=f"prompt_{i:03d}",
                bucket="factual",
            ))
        else:
            records.append(PromptRecord(
                id=item.get("id", f"prompt_{i:03d}"),
                bucket=item.get("bucket", "factual"),
                prompt=item["prompt"],
                answer=item.get("answer"),
                expected_behavior=item.get("expected_behavior"),
            ))

    return records
