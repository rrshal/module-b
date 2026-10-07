import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--claims", required=True)
parser.add_argument("--evidence", required=True)
parser.add_argument("--verification", required=True)
parser.add_argument("--scores", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--source-dataset", required=True)

args = parser.parse_args()


with open(args.claims, "r", encoding="utf-8") as f:
    claims_data = json.load(f)

with open(args.evidence, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

with open(args.verification, "r", encoding="utf-8") as f:
    verification_data = json.load(f)

with open(args.scores, "r", encoding="utf-8") as f:
    scores_data = json.load(f)


evidence_by_claim = {}

for item in evidence_data:
    evidence_by_claim[item["claim_id"]] = item


verdict_by_claim = {}

for item in verification_data:
    verdict_by_claim[item["claim_id"]] = item["prediction"]


scores_by_example = {}

for item in scores_data:
    scores_by_example[item["example_id"]] = item


final_results = []


for example in claims_data["examples"]:

    example_id = example["id"]
    source = example["source_record"]

    final_claims = []

    for i, claim in enumerate(example["claims"], start=1):

        claim_id = f"{example_id}_claim_{i}"

        evidence = evidence_by_claim[claim_id]

        passages = []

        for passage in evidence["retrieved"]:
            passages.append({
                "passage_id": passage["passage_id"],
                "title": passage["title"],
                "text": passage["text"],
                "retrieval_score": passage["score"]
            })

        final_claims.append({
            "claim_id": claim_id,
            "claim": claim,
            "retrieved_passages": passages,
            "verdict": verdict_by_claim[claim_id]
        })


    score = scores_by_example[example_id]

    final_results.append({
        "id": example_id,
        "source_dataset": args.source_dataset,
        "question": source["question"],
        "response": source["generated_response"],
        "gold_best_answer": source["best_answer"],
        "gold_good_answers": source["correct_answers"],

        "claims": final_claims,

        "leftover_content": example["leftover_content"],

        "claim_count": example["claim_count"],

        "verdict_counts": {
            "supports": score["supports"],
            "refutes": score["refutes"],
            "not_enough_info": score["not_enough_info"]
        },

        "factuality_score": score["factuality_score"]
    })


output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {"examples": final_results},
        f,
        indent=2,
        ensure_ascii=False
    )


print("Examples:", len(final_results))
print("Saved to:", output_path)
