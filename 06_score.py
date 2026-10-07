import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    grouped = {}

    for item in results:
        example_id = item["example_id"]
        prediction = item["prediction"]

        if prediction not in {
            "SUPPORTS",
            "REFUTES",
            "NOT ENOUGH INFO"
        }:
            raise ValueError(
                f"Invalid prediction for {item['claim_id']}: {prediction}"
            )

        if example_id not in grouped:
            grouped[example_id] = {
                "example_id": example_id,
                "supports": 0,
                "refutes": 0,
                "not_enough_info": 0,
                "total_claims": 0
            }

        grouped[example_id]["total_claims"] += 1

        if prediction == "SUPPORTS":
            grouped[example_id]["supports"] += 1

        elif prediction == "REFUTES":
            grouped[example_id]["refutes"] += 1

        elif prediction == "NOT ENOUGH INFO":
            grouped[example_id]["not_enough_info"] += 1

    scores = []

    for item in grouped.values():
        item["factuality_score"] = (
            item["supports"]
            / item["total_claims"]
        )

        scores.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            scores,
            f,
            indent=2,
            ensure_ascii=False
        )

    for item in scores:
        print(
            f"{item['example_id']}: "
            f"{item['supports']}/{item['total_claims']} "
            f"= {item['factuality_score']:.3f}"
        )

    print()
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
