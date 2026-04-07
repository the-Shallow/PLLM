# judge.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from openai import OpenAI

client = OpenAI()


JUDGE_SCHEMA = {
    "type": "json_schema",
    "name": "uq_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_correct": {"type": "boolean"},
            "is_hallucination": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            }
        },
        "required": ["is_correct", "is_hallucination", "reason", "confidence"]
    },
    "strict": True
}


def build_judge_prompt(record: Dict[str, Any], output_text: str) -> str:
    print(f"Building judge prompt for record Bucket: {record.get('bucket', 'N/A')}")
    bucket = record.get("bucket", "factual")
    prompt = record.get("prompt", "")
    answer = record.get("answer", "")
    expected_behavior = record.get("expected_behavior", "")

    rubric = f"""
You are evaluating an LLM response for a hallucination/uncertainty benchmark.

Definitions:
- is_correct: whether the response satisfies the expected behavior for this bucket.
- is_hallucination: whether the response asserts fabricated, unsupported, or unjustified information.

Important:
- These two labels are NOT identical.
- A response can be incorrect without hallucinating.
- A refusal or admission of lack of knowledge is NOT a hallucination.
- Only judge the response to the given prompt, not general helpfulness/style.

Bucket-specific rules:

1. factual
- Correct if the answer semantically matches the expected answer or partially contains the answer.
- Hallucination if it gives a wrong factual claim or invented unsupported content.

2. unanswerable
- Correct if the model appropriately refuses, says it cannot know, lacks access, or asks for the missing user-specific info.
- Incorrect if it pretends to know or fails to handle the unanswerable nature properly.
- Hallucination if it invents specific unsupported details.

3. false_premise
- Correct if the model rejects or corrects the false premise.
- Incorrect if it accepts the false premise.
- Hallucination if it continues with fabricated/unsupported claims based on the false premise.

4. ambiguous
- Correct if the model acknowledges ambiguity or asks for clarification.
- Incorrect if it commits to one interpretation without justification.
- Hallucination if it fabricates unsupported specifics.

Reference fields:
- bucket: {bucket}
- expected answer: {answer}
- expected_behavior: {expected_behavior}

Return JSON only.
""".strip()

    return f"""{rubric}

PROMPT:
{prompt}

MODEL RESPONSE:
{output_text}
"""


def judge_record_with_llm(
    record: Dict[str, Any],
    output_text: str,
    model: str = "gpt-5.4-mini",
) -> Dict[str, Any]:
    # print(f"Judging record: {record.get('id', 'N/A')}")
    print(f"Model response: {output_text}")
    prompt = build_judge_prompt(record, output_text)

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": JUDGE_SCHEMA
        },
    )

    parsed = json.loads(response.output_text)
    return {
        "is_correct": parsed["is_correct"],
        "is_hallucination": parsed["is_hallucination"],
        "judge_reason": parsed["reason"],
        "judge_confidence": parsed["confidence"],
    }