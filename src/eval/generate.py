# =============================================================================
# Original implementation (preserved)
# =============================================================================
# import torch
# from src.models.load_model import load_model
# from src.pruning.registry import get_pruner
#
# def generate(model, tokenizer, prompt, do_sample=True, temperature=0.8, repetition_penalty=1.15, max_length=100):
#     device = next(model.parameters()).device
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature, repetition_penalty=repetition_penalty)
#     return tokenizer.decode(out[0], skip_special_tokens=True)
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class GenerationResult:
    text: str
    token_logprobs: List[float]
    avg_logprob: float   # LNS score: mean log P(chosen token | context) over all generated tokens.
                         # Closer to 0 → model was confident in its choices.
                         # More negative → model was uncertain / low probability tokens were chosen.
    # ---------------------------------------------------------------------------
    # ENTROPY (commented out — preserved for future use)
    # avg_entropy measures the Shannon entropy of the full vocabulary distribution
    # at each generation step, then averages it across all steps.
    #   Low entropy  → probability mass is concentrated on a few tokens (model is peaked/certain)
    #   High entropy → probability mass is spread across many tokens (model is flat/uncertain)
    # Like LNS, this is an UNCERTAINTY signal — it does NOT directly detect hallucinations.
    # Hallucination detection is handled separately via bucket-based phrase matching in metrics.py.
    # Uncomment avg_entropy here and in generate_with_scores() to re-enable it.
    # avg_entropy: float
    # ---------------------------------------------------------------------------


def generate_with_scores(
    model,
    tokenizer,
    prompt: str,
    do_sample: bool = True,
    temperature: float = 0.8,
    repetition_penalty: float = 1.15,
    max_new_tokens: int = 100,
) -> GenerationResult:
    """
    Generate text for a single prompt and return the LNS confidence signal.

    LNS (avg_logprob): average log-probability of the tokens the model chose.
        Closer to 0 → high confidence. More negative → low confidence.
        Used in metrics.py to decide whether the model was "certain" (avg_logprob > lns_threshold).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            output_scores=True,          # gives us per-step logit distributions
            return_dict_in_generate=True,
        )

    # out.scores: tuple of (1, vocab_size) logit tensors, one per generated token.
    # out.sequences: (1, prompt_len + gen_len)
    generated_ids = out.sequences[0, prompt_len:]
    n_steps = min(len(out.scores), len(generated_ids))
    # print(f"Out scores : {out.scores} and n_steps : {n_steps}")
    token_logprobs: List[float] = []

    for step in range(n_steps):
        logits = out.scores[step][0]           # (vocab_size,)
        log_probs = F.log_softmax(logits, dim=-1)

        # Record log-probability of the token that was actually chosen
        token_id = generated_ids[step].item()
        token_logprobs.append(log_probs[token_id].item())

        # ---------------------------------------------------------------------------
        # ENTROPY computation (commented out — see GenerationResult above)
        # probs = log_probs.exp()
        # entropy = -(probs * log_probs).nan_to_num(nan=0.0).sum().item()
        #   nan_to_num is needed because 0 * -inf = nan in floating point
        # entropies.append(entropy)
        # ---------------------------------------------------------------------------

    avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
    # avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0  # entropy disabled

    # Decode only the generated tokens (prompt_len: strips the echoed prompt).
    # out.sequences[0] is the full sequence (prompt + generated), so slicing
    # from prompt_len gives us only what the model actually generated.
    text = tokenizer.decode(out.sequences[0, prompt_len:], skip_special_tokens=True)

    return GenerationResult(
        text=text,
        token_logprobs=token_logprobs,
        avg_logprob=avg_logprob,
        # avg_entropy=avg_entropy,  # entropy disabled
    )


def generate_n_samples(
    model,
    tokenizer,
    prompt: str,
    n: int = 5,
    **kwargs,
) -> List[GenerationResult]:
    """
    Generate n independent samples for the same prompt.
    Useful for measuring response-level consistency (self-consistency).
    """
    return [generate_with_scores(model, tokenizer, prompt, **kwargs) for _ in range(n)]
