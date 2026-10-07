import argparse
import json
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "gpt-5.6-terra"
MAX_OUTPUT_TOKENS = 16

client = OpenAI()


def build_prompt(example):
    evidence_text = "\n\n".join(
        passage["text"]
        for passage in example["retrieved"][:10]
    )

    return f"""Verify the factual accuracy of the claim using only the provided evidence.

SUPPORTS:
The evidence confirms all statements and details in the claim.

REFUTES:
The evidence contradicts or disproves the claim.

NOT ENOUGH INFO:
The evidence does not provide enough information to support or refute the claim.
If only part of the claim is supported and any required detail cannot be confirmed,
choose NOT ENOUGH INFO.

Base your decision solely on the provided evidence.
Do not infer additional details.

Claim:
{example["claim"]}

Evidence:
{evidence_text}

Return exactly one label:
SUPPORTS
REFUTES
NOT ENOUGH INFO"""


def parse_verdict(raw):
    text = raw.strip().upper()

    if text == "SUPPORTS":
        return "SUPPORTS"

    if text == "REFUTES":
        return "REFUTES"

    if text == "NOT ENOUGH INFO":
        return "NOT ENOUGH INFO"

    if "NOT ENOUGH INFO" in text:
        return "NOT ENOUGH INFO"

    if "REFUTES" in text:
        return "REFUTES"

    if "SUPPORTS" in text:
        return "SUPPORTS"

    return "INVALID"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    results = []

    for i, example in enumerate(examples, start=1):
        prompt = build_prompt(example)

        response = client.responses.create(
            model=MODEL_NAME,
            reasoning={
                "effort": "none"
            },
            input=prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )

        raw_output = response.output_text.strip()
        prediction = parse_verdict(raw_output)

        result = {
            "claim_id": example["claim_id"],
            "example_id": example["example_id"],
            "claim_in_example": example["claim_in_example"],
            "claim": example["claim"],
            "prediction": prediction
        }

        results.append(result)

        print(
            f"[{i}/{len(examples)}] "
            f"{example['claim_id']} -> {prediction}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Claims verified:", len(results))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
