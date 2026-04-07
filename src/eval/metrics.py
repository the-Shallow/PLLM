from __future__ import annotations
import re
from typing import Any, Dict, List
from .judge import judge_record_with_llm

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
    Return only the first answer from the generated output.

    Two things can pollute the output:
    1. Echoed prompt — the model repeats "Q: ... A:" before answering.
       Split on the last "A:" to skip past it.
       NOTE: with generate.py decoding only generated tokens (prompt_len slice),
       the prompt is already stripped, so this is a safety fallback.
    2. Overgeneration — larger models (e.g. Llama 3B) continue generating extra
       Q&A pairs after the actual answer, e.g.:
           " H2O\nQ: How do you spell water?\nA: W-A-T-E-R\n..."
       Truncate at the first "\\nQ:" to keep only the real first answer.
    """
    # Step 1: strip echoed prompt if present (safety fallback)
    # parts = output.rsplit("A:", 1)
    # response = parts[-1] if len(parts) > 1 else output

    # Step 2: cut off overgeneration — anything after the first new question is noise.
    # Use \n+ to handle models that insert one or two blank lines before the next Q.
    response = re.split(r'\n+Q:', output)[0].strip()

    return response


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
    # lns_threshold: float = -2.0,
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


        if not is_correct:
            judged = judge_record_with_llm(record, output_text)
            is_correct = judged.get("is_correct", is_correct)
            is_hallucination = judged.get("is_hallucination", is_hallucination)
            print(f"LLM judge override: is_correct={is_correct}, is_hallucination={is_hallucination}, reason={judged.get('judge_reason')}, confidence={judged.get('judge_confidence')}")

    # is_certain: avg_logprob closer to 0 = model was confident in its token choices.
    # lns_threshold=-2.0 means log-prob per token above -2.0 is considered "certain".
    # is_certain = avg_logprob > lns_threshold

    # if is_correct is None:
    #     bucket_label = "unscored"
    # elif is_correct and is_certain:
    #     bucket_label = "correct_certain"
    # elif is_correct and not is_certain:
    #     bucket_label = "correct_uncertain"
    # elif not is_correct and is_certain:
    #     # Most dangerous failure mode: model was wrong but confident — overconfident hallucination
    #     bucket_label = "incorrect_certain"
    # else:
    #     bucket_label = "incorrect_uncertain"

    return {
        "is_correct": is_correct,
        "is_hallucination": is_hallucination,
        # "is_certain": is_certain,
        # "bucket_label": bucket_label,
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

    y_true, y_score = extract_binary_labels_and_scores(scored_outputs)
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)
    prr = compute_prr(y_true, y_score)
    print(f"AUROC: {auroc}, AUPRC: {auprc}, PRR: {prr}")
    return {
        "num_prompts":                      all_n,
        "num_scored":                       n,
        "accuracy":                         round(num_correct / n, 4) if n else None,
        "hallucination_rate":               round(num_hallucinated / n, 4) if n else None,
        "overconfident_hallucination_rate": round(overconfident / n, 4) if n else None,
        "avg_lns_score":                    round(sum(lns_values) / len(lns_values), 4) if lns_values else None,
        # "avg_entropy": round(sum(ent_values) / len(ent_values), 4) if ent_values else None,  # entropy disabled
        **bucket_counts,
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "prr": round(prr, 4) if prr is not None else None,
    }


def extract_binary_labels_and_scores(outputs):
    y_true = []
    y_score = []

    for entry in outputs:
        is_correct = entry.get("is_correct")
        lns_score = entry.get("lns_score")

        if is_correct is None or lns_score is None:
            continue
        
        y_true.append(0 if is_correct else 1)
        y_score.append(-float(lns_score))

    return y_true, y_score


def binary_clf_curve(y_true, y_score):
    pairs = sorted(zip(y_score,y_true), key=lambda x : x[0], reverse=True)

    fps, tps, thresholds = [], [], []

    tp,fp,prev_score = 0.0,0.0,None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            fps.append(fp)
            tps.append(tp)
            thresholds.append(prev_score)
        
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        
        prev_score = score
    
    if prev_score is not None:
        fps.append(fp)
        tps.append(tp)
        thresholds.append(prev_score)

    return fps, tps, thresholds


def compute_auroc(y_true, y_score):
    if not y_true or len(set(y_true)) < 2:
        return None
    
    fps,tps,_ = binary_clf_curve(y_true,y_score)


    pos = sum(y_true)
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        return None
    
    fpr = [0.0] + [fp/neg for fp in fps]
    tpr = [0.0] + [tp/pos for tp in tps]

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
    return auc

def compute_auprc(y_true,y_score):
    if not y_true or len(set(y_true)) < 2:
        return None

    pairs = sorted(zip(y_score,y_true), key=lambda x: x[0], reverse=True)

    total_pos = sum(y_true)
    if total_pos == 0:
        return None
    
    tp, fp = 0.0, 0.0

    precisions = [1.0]
    recalls = [0.0]

    for _, label in pairs:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        
        precision = tp / (tp + fp)
        recall = tp / total_pos

        precisions.append(precision)
        recalls.append(recall)
    
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2.0
    
    return auc


def precision_fraction(sorted_correct, keep_fraction):
    n = len(sorted_correct)
    if n == 0:
        return 0.0
    
    k = max(1, int(round(n * keep_fraction)))
    kept = sorted_correct[:k]
    return sum(kept) / len(kept)

def area_under_precision_rejection(sorted_correct, num_points = 101):
    rejection_rates, precisions = [], []

    for i in range(num_points):
        reject_rate = i / (num_points - 1)
        keep_fraction = 1.0 - reject_rate
        precision = precision_fraction(sorted_correct, keep_fraction)

        rejection_rates.append(reject_rate)
        precisions.append(precision)

    auc = 0.0
    for i in range(1,len(rejection_rates)):
        auc += (
            (rejection_rates[i] - rejection_rates[i-1]) *
            (precisions[i] + precisions[i-1]) / 2.0
        )
    return auc

def compute_prr(y_true, y_score):
    correctness = [1 - y for y in y_true]
    pairs_unc = sorted(zip(y_score, correctness), key=lambda x : x[0])
    sorted_correct_unc = [c for _,c in pairs_unc]

    sorted_correct_oracle = sorted(correctness, reverse=True)
    base_accuracy = sum(correctness) / len(correctness)
    auc_rand = base_accuracy

    auc_unc = area_under_precision_rejection(sorted_correct_unc)
    auc_oracle = area_under_precision_rejection(sorted_correct_oracle)

    denom = auc_oracle - auc_rand
    if denom <= 0:
        return None
    
    return (auc_unc - auc_rand) / denom