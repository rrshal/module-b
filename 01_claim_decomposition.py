import argparse
import json
import random
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-terra"

client = OpenAI()


# FactLens prompt with our atomicity addition
PROMPT = '''
We aim to fact-check a textual claim. To make the fact-checking task simpler, we break down a claim into simpler, atomic sub-claims to fact-check as needed. Note that atomic sub-claims refer to unit claims within the original claim, that refer to a single concept that can be independently verified without having to refer to the original claim. Verification of the sub-claims should not require aggregation of facts or multi-hop reasoning over concepts. However, the sub-claim should have all the contextual information preserved from the original claim. 

To enforce atomicity, if a statement contains one subject with multiple independently verifiable objects, properties, roles, or assertions, split them into separate sub-claims.

Your task is to break down a claim into atomic sub-claims for fact checking only if needed. If the original claim itself is a unit claim, do not break it down.

For example:
{demonstrations}

Note how each sub claim contains atomic information to fact check and is brief, yet is contextualized with all the information needed from the original claim.

Now find the sub claims from the following claim.
claim: {claim}
sub_claims: <your output in form of a list> 
'''


# Exact demonstrations from FactLens
DEMONSTRATIONS = [
    '''Claim: Mathias Herrmann acted in eleven movies from 1987 to 2020
Sub-claims: [“Mathias Herrmann acted in eleven movies from 1987 to 2020”]''',

    '''Claim: L-arabinose is more common than D-arabinose with most research has been done on L-arabinose operon which is required for the breakdown of the five-carbon sugar L-arabinose in Escherichia coli.
Sub-claims: [“L-arabinose is more common than D-arabinose”, “Most research has been done on L-arabinose operon”,“L-arabinose operon is required for the breakdown of L-arabinose in Escherichia coli”, “L-arabinose is a five-carbon sugar”]''',

    '''Claim: Forest of the Dead was directed by three directors and was written by Steven Moffat.
Sub-claims:  [“Forest of the Dead was directed by three directors”, “Forest of the Dead was written by Steven Moffat”]''',

    '''Claim: Matthew Busche whose full name is Matthew Craig Busche rode for the RadioShack-Nissan from 2012 to 2016.
Sub-claims:  [“Matthew Busche's full name is Matthew Craig Busche”, “Matthew Busche rode for the RadioShack-Nissan from 2012 to 2016””]'''
]


# We changed FactLens's system message so the output matches our parser
SYSTEM_PROMPT = (
    'Return only valid JSON in this format: '
    '{"sub_claims": ["claim 1", "claim 2"]}'
)


parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--id-field", default="example_id")
parser.add_argument("--response-field", default="generated_response")

args = parser.parse_args()


# Read the input file
with open(args.input, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    input_examples = data["examples"]
else:
    input_examples = data


results = []

for i, example in enumerate(input_examples, start=1):

    example_id = example[args.id_field]
    response_text = example[args.response_field]

    print("=" * 80)
    print(f"{i}/{len(input_examples)} - {example_id}")
    print("=" * 80)

    # FactLens randomly uses 3 of its 4 examples
    chosen_examples = random.sample(DEMONSTRATIONS, 3)
    random.shuffle(chosen_examples)

    demonstrations = "\n\n".join(chosen_examples)

    prompt = PROMPT.format(
        demonstrations=demonstrations,
        claim=response_text
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
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
    claims = output["sub_claims"]

    print("Claims:")

    for claim in claims:
        print("-", claim)

    print()

    results.append({
        "id": example_id,
        "response": response_text,
        "claims": claims,
        "claim_count": len(claims),
        "source_record": example
    })


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


total_claims = 0

for example in results:
    total_claims += example["claim_count"]


print("=" * 80)
print("Total claims:", total_claims)
print("Saved to:", output_path)
print("=" * 80)
