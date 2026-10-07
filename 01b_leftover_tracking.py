import argparse
import json
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-terra"

client = OpenAI()


PROMPT = """
You are checking whether all content from an original response is represented
by a list of extracted factual claims.

Your job is NOT to fact-check anything.

Find the content in the original response that is NOT represented by any of
the extracted claims.

Rules:
- A part is represented if its meaning is already conveyed by one or more of
  the extracted claims, even if the wording is different.
- Do not include content that is already represented by a claim.
- If a sentence contains both represented and leftover content, return only
  the leftover part.
- Keep the wording as close to the original response as possible.
- Do not add or infer new information.
- Do not decide whether the leftover content is true, false, or verifiable.
- If there is no leftover content, return an empty list.

Return only valid JSON in this format:
{{"leftover_content": ["leftover 1", "leftover 2"]}}

If there is no leftover content, return:
{{"leftover_content": []}}

Original response:
{response_text}

Extracted claims:
{claims}
"""


parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


# Read the decomposed claims
with open(args.input, "r", encoding="utf-8") as f:
    data = json.load(f)


results = []

for i, example in enumerate(data["examples"], start=1):

    print("=" * 80)
    print(f"{i}/{len(data['examples'])} - {example['id']}")
    print("=" * 80)

    claims_text = "\n".join(
        f"- {claim}"
        for claim in example["claims"]
    )

    prompt = PROMPT.format(
        response_text=example["response"],
        claims=claims_text
    )

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": "none"}
    )

    raw_output = response.output_text.strip()

    # Remove a code block if the model adds one
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_output = "\n".join(lines).strip()

    output = json.loads(raw_output)
    leftover_content = output["leftover_content"]

    print("Leftover:")
    print(leftover_content)
    print()

    result = example.copy()
    result["leftover_content"] = leftover_content

    results.append(result)


output_data = {
    "examples": results
}


output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        output_data,
        f,
        indent=2,
        ensure_ascii=False
    )


print("=" * 80)
print("Saved to:", output_path)
print("=" * 80)
