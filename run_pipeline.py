import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def run(command):
    print()
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        check=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    
    parser.add_argument(
        "--source-dataset",
        required=True
    )

    location_file = BASE_DIR / "data_location.txt"

    if location_file.exists():
        data_dir = Path(location_file.read_text().strip())
    else:
        data_dir = BASE_DIR / "data" / "wikipedia"

    parser.add_argument(
        "--index-dir",
        default=str(data_dir / "faiss_indexes")
    )

    parser.add_argument(
        "--corpus",
        default=str(data_dir / "psgs_w100.tsv.gz")
    )

    parser.add_argument(
        "--id-field",
        default="id"
    )

    parser.add_argument(
        "--response-field",
        default="response"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    claims_file = output_dir / "01_decomposed_claims.json"
    leftover_file = output_dir / "01b_claims_with_leftover.json"
    embeddings_file = output_dir / "02_embeddings.npy"
    metadata_file = output_dir / "02_metadata.json"
    retrieval_file = output_dir / "03_retrieval.json"
    evidence_file = output_dir / "04_retrieval_with_text.json"
    verification_file = output_dir / "05_verification.json"
    scores_file = output_dir / "06_scores.json"

    run([
        "python",
        str(BASE_DIR / "01_claim_decomposition.py"),
        "--input", args.input,
        "--output", str(claims_file),
        "--id-field", args.id_field,
        "--response-field", args.response_field
    ])

    run([
        "python",
        str(BASE_DIR / "01b_leftover_tracking.py"),
        "--input", str(claims_file),
        "--output", str(leftover_file)
    ])

    run([
        "python",
        str(BASE_DIR / "02_claim_embedding.py"),
        "--input", str(claims_file),
        "--embeddings-output", str(embeddings_file),
        "--metadata-output", str(metadata_file)
    ])

    run([
        "python",
        str(BASE_DIR / "03_retrieval.py"),
        "--embeddings", str(embeddings_file),
        "--metadata", str(metadata_file),
        "--index-dir", args.index_dir,
        "--output", str(retrieval_file)
    ])

    run([
        "python",
        str(BASE_DIR / "04_attach_evidence.py"),
        "--input", str(retrieval_file),
        "--corpus", args.corpus,
        "--output", str(evidence_file)
    ])

    run([
        "python",
        str(BASE_DIR / "05_verification.py"),
        "--input", str(evidence_file),
        "--output", str(verification_file)
    ])

    run([
        "python",
        str(BASE_DIR / "06_score.py"),
        "--input", str(verification_file),
        "--output", str(scores_file)
    ])

    final_file = output_dir / "07_final_output.json"

    run([
        "python",
        str(BASE_DIR / "07_final_output.py"),
        "--claims", str(leftover_file),
        "--evidence", str(evidence_file),
        "--verification", str(verification_file),
        "--scores", str(scores_file),
        "--output", str(final_file),
        "--source-dataset", args.source_dataset
    ])

    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("Final JSON:", final_file)


if __name__ == "__main__":
    main()
