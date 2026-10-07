import argparse
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np


# use the official Contriever repository
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "contriever_official"))

from src.index import Indexer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=16)

    args = parser.parse_args()

    queries = np.load(
        args.embeddings
    ).astype("float32")

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert queries.shape[0] == len(metadata)
    assert queries.shape[1] == 768

    print("Query matrix shape:", queries.shape)
    print("Number of claims:", len(metadata))
    print("Top-k:", args.top_k)

    # Store candidates from all shards
    all_candidates = [
        []
        for _ in range(len(metadata))
    ]

    for shard in range(args.num_shards):
        print(
            f"Searching shard {shard:02d} "
            f"({shard + 1}/{args.num_shards})..."
        )

        index_path = (
            Path(args.index_dir)
            / f"index_{shard:02d}.faiss"
        )

        ids_path = (
            Path(args.index_dir)
            / f"ids_{shard:02d}.pkl"
        )

        faiss_index = faiss.read_index(
            str(index_path)
        )

        with open(ids_path, "rb") as f:
            passage_ids = pickle.load(f)

        # Use Contriever's official Indexer
        index = Indexer(768)

        index.index = faiss_index
        index.index_id_to_db_id = passage_ids

        shard_results = index.search_knn(
            queries,
            args.top_k
        )

        for query_idx, result in enumerate(shard_results):
            ids, scores = result

            for passage_id, score in zip(ids, scores):
                all_candidates[query_idx].append({
                    "passage_id": str(passage_id),
                    "score": float(score)
                })

        del index
        del faiss_index
        del passage_ids
        del shard_results

    results = []

    for query_idx, claim_info in enumerate(metadata):
        candidates = all_candidates[query_idx]

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        top_passages = candidates[:args.top_k]

        results.append({
            "claim_id": claim_info["claim_id"],
            "example_id": claim_info["example_id"],
            "claim_in_example": claim_info["claim_in_example"],
            "claim": claim_info["claim"],
            "retrieved": top_passages
        })

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
    print("=" * 70)
    print("RETRIEVAL COMPLETE")
    print("=" * 70)
    print("Claims processed:", len(results))
    print("Top-k per claim:", args.top_k)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
