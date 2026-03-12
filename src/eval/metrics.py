from __future__ import annotations
import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Rule-based phrase lists for each bucket type
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = [
    "i don't know", "i do not know", "i cannot determine", "i can't determine",
    "not enough information", "i have no idea", "i'm not sure", "i am not sure",
    "cannot know", "no way to know", "unable to determine", "impossible to know",
    "i wasn't told", "you haven't told me", "no information provided",
    "i cannot answer", "i don't have access", "i have no way",
]

_PREMISE_REJECTION_PHRASES = [
    "that's incorrect", "that is incorrect", "that's not correct", "that is not correct",
    "that's not true", "that is not true", "the premise is wrong", "the premise is false",
    "that's a false", "actually,", "in fact,", "contrary to", "is actually",
    "is not berlin", "is not true", "this is incorrect", "this is not accurate",
    "that assumption is", "that claim is",
]

_AMBIGUITY_PHRASES = [
    "depends", "could refer to", "which one do you mean", "could mean",
    "more context", "please clarify", "ambiguous", "it's unclear", "it is unclear",
    "more information", "what do you mean", "can you specify",
]


# ---------------------------------------------------------------------------
# Per-bucket correctness scorers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip()


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = _normalize(text)
    return any(p in t for p in phrases)


def _extract_response(output: str) -> str:
    """
    Strip the echoed prompt from the output and return only the generated response.

    The model output includes the full prompt (e.g. "Q: What is...\\nA:") followed
    by the generated continuation. We split on the last "A:" to isolate the response
    part — otherwise short answers like "6" or "12" would match numbers that appear
    anywhere in the question or context, not just in the actual answer.
    """
    # Split on the last occurrence of "A:" (the answer marker in our prompt format)
    parts = output.rsplit("A:", 1)
    return parts[-1] if len(parts) > 1 else output


def score_factual(output: str, answer: str) -> bool:
    """
    Correct if the expected answer appears as a whole word in the generated response.

    Two fixes over plain substring matching:
    1. Only search in the response part (after "A:"), not the full echoed prompt+output.
       Prevents answers like "Tokyo" matching in the question text itself.
    2. Use word-boundary matching (\\b) so short answers like "6" don't match inside
       longer numbers like "16" or "60".
    """
    response = _extract_response(output)
    # \b matches word boundaries — "6" won't match inside "16" or "60"
    pattern = r"\b" + re.escape(_normalize(answer)) + r"\b"
    return bool(re.search(pattern, _normalize(response)))


def score_unanswerable(output: str) -> bool:
    """Correct if the model refuses or expresses genuine uncertainty."""
    return _contains_any(output, _REFUSAL_PHRASES)


def score_false_premise(output: str) -> bool:
    """Correct if the model rejects the false premise rather than accepting it."""
    return _contains_any(output, _PREMISE_REJECTION_PHRASES)


def score_ambiguous(output: str) -> bool:
    """Correct if the model acknowledges ambiguity or asks for clarification."""
    return _contains_any(output, _AMBIGUITY_PHRASES)


_BUCKET_SCORERS = {
    "factual":       lambda output, record: score_factual(output, record.get("answer") or ""),
    "unanswerable":  lambda output, record: score_unanswerable(output),
    "false_premise": lambda output, record: score_false_premise(output),
    "ambiguous":     lambda output, record: score_ambiguous(output),
}


# ---------------------------------------------------------------------------
# Per-prompt scoring
# ---------------------------------------------------------------------------

def score_output(
    record: Dict[str, Any],
    output_text: str,
    avg_logprob: float,
    # ---------------------------------------------------------------------------
    # ENTROPY parameter (commented out — entropy disabled, using LNS for certainty)
    # avg_entropy: float,
    # entropy_threshold: float = 1.5,
    #   When entropy was active, is_certain = avg_entropy < entropy_threshold
    #   i.e. low entropy (peaked distribution) meant the model was certain.
    #   Swapped to LNS because LNS is more directly interpretable:
    #   it measures how probable the chosen tokens actually were.
    # ---------------------------------------------------------------------------
    lns_threshold: float = -2.0,
) -> Dict[str, Any]:
    """
    Score a single generated output against its prompt bucket.

    Returns a dict with:
        is_correct            — whether the output passes the bucket-specific check
        is_hallucination      — True when the model is wrong (is_correct=False)
        is_certain            — True when avg_logprob > lns_threshold (less negative = more confident)
        bucket_label          — one of: correct_certain / correct_uncertain /
                                         incorrect_certain / incorrect_uncertain / unscored
        lns_score             — avg_logprob (LNS confidence signal)
    """
    bucket = record.get("bucket", "factual")
    scorer = _BUCKET_SCORERS.get(bucket)

    if scorer is None or (bucket == "factual" and not record.get("answer")):
        # No scorer available or factual prompt has no reference answer
        is_correct = None
        is_hallucination = None
    else:
        is_correct = scorer(output_text, record)
        is_hallucination = not is_correct

    # is_certain: avg_logprob closer to 0 = model was confident in its token choices.
    # lns_threshold=-2.0 means log-prob per token above -2.0 is considered "certain".
    is_certain = avg_logprob > lns_threshold

    if is_correct is None:
        bucket_label = "unscored"
    elif is_correct and is_certain:
        bucket_label = "correct_certain"
    elif is_correct and not is_certain:
        bucket_label = "correct_uncertain"
    elif not is_correct and is_certain:
        # Most dangerous failure mode: model was wrong but confident — overconfident hallucination
        bucket_label = "incorrect_certain"
    else:
        bucket_label = "incorrect_uncertain"

    return {
        "is_correct": is_correct,
        "is_hallucination": is_hallucination,
        "is_certain": is_certain,
        "bucket_label": bucket_label,
        "lns_score": round(avg_logprob, 4),
        # "entropy": round(avg_entropy, 4),  # entropy disabled
    }


# ---------------------------------------------------------------------------
# Aggregate metrics across all scored prompts
# ---------------------------------------------------------------------------

def aggregate_metrics(scored_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-prompt scores into summary metrics.

    Skips entries with bucket_label="unscored" for accuracy/hallucination counts
    but still includes them in entropy/lns averages.
    """
    all_n = len(scored_outputs)
    scored = [s for s in scored_outputs if s.get("bucket_label") != "unscored"]
    n = len(scored)

    bucket_counts = {
        "correct_certain":    0,
        "correct_uncertain":  0,
        "incorrect_certain":  0,
        "incorrect_uncertain": 0,
    }
    for s in scored:
        label = s.get("bucket_label", "")
        if label in bucket_counts:
            bucket_counts[label] += 1

    num_correct = bucket_counts["correct_certain"] + bucket_counts["correct_uncertain"]
    num_hallucinated = sum(1 for s in scored if s.get("is_hallucination"))
    # Overconfident hallucination: model was wrong AND certain (most dangerous failure mode)
    overconfident = bucket_counts["incorrect_certain"]

    lns_values = [s["lns_score"] for s in scored_outputs if "lns_score" in s]
    # ent_values = [s["entropy"] for s in scored_outputs if "entropy" in s]  # entropy disabled

    return {
        "num_prompts":                      all_n,
        "num_scored":                       n,
        "accuracy":                         round(num_correct / n, 4) if n else None,
        "hallucination_rate":               round(num_hallucinated / n, 4) if n else None,
        "overconfident_hallucination_rate": round(overconfident / n, 4) if n else None,
        "avg_lns_score":                    round(sum(lns_values) / len(lns_values), 4) if lns_values else None,
        # "avg_entropy": round(sum(ent_values) / len(ent_values), 4) if ent_values else None,  # entropy disabled
        **bucket_counts,
    }
