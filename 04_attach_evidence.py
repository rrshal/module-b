import argparse
import csv
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("Claims loaded:", len(results))

    wanted_ids = set()

    for claim in results:
        for passage in claim["retrieved"]:
            wanted_ids.add(
                str(passage["passage_id"])
            )

    print("Unique passage IDs needed:", len(wanted_ids))
    print("Scanning corpus...")

    passages = {}

    with gzip.open(
        args.corpus,
        "rt",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:
            passage_id = row["id"]

            if passage_id in wanted_ids:
                passages[passage_id] = {
                    "title": row["title"],
                    "text": row["text"]
                }

                if len(passages) == len(wanted_ids):
                    break

    print("Passages found:", len(passages))

    missing = wanted_ids - set(passages.keys())

    if missing:
        raise ValueError(
            f"{len(missing)} passage IDs were not found in the corpus."
        )

    for claim in results:
        for passage in claim["retrieved"]:
            passage_id = str(
                passage["passage_id"]
            )

            passage["title"] = (
                passages[passage_id]["title"]
            )

            passage["text"] = (
                passages[passage_id]["text"]
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

    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
