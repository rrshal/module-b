import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# use the official Contriever repository
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "contriever_official"))

from src.contriever import load_retriever


MODEL_NAME = "facebook/contriever"
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings-output", required=True)
    parser.add_argument("--metadata-output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for example in data["examples"]:
        for claim_number, claim in enumerate(example["claims"], start=1):
            records.append({
                "claim_id": f"{example['id']}_claim_{claim_number}",
                "example_id": example["id"],
                "claim_in_example": claim_number,
                "claim": claim
            })

    claims = [record["claim"] for record in records]

    print("Claims loaded:", len(claims))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # official Contriever model and tokenizer
    model, tokenizer, _ = load_retriever(MODEL_NAME)

    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(claims), BATCH_SIZE):
            batch = claims[start:start + BATCH_SIZE]

            # current Transformers equivalent of Contriever's
            # original batch_encode_plus call
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            # official Contriever forward method does the pooling
            embeddings = model(**encoded)

            embeddings = embeddings.cpu().float().numpy()
            all_embeddings.append(embeddings)

            done = min(start + BATCH_SIZE, len(claims))
            print(f"Encoded {done}/{len(claims)}")

    all_embeddings = np.vstack(all_embeddings)

    metadata = []

    for row_index, record in enumerate(records):
        metadata.append({
            "row_index": row_index,
            "claim_id": record["claim_id"],
            "example_id": record["example_id"],
            "claim_in_example": record["claim_in_example"],
            "claim": record["claim"]
        })

    embeddings_path = Path(args.embeddings_output)
    metadata_path = Path(args.metadata_output)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(
        embeddings_path,
        all_embeddings.astype(np.float32)
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Embedding shape:", all_embeddings.shape)
    print("Saved embeddings:", embeddings_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
