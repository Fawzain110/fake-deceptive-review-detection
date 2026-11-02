"""Attach product IDs to the supplied labeled reviews using exact source matches."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("../training_reviews.csv"))
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("../extracted_dataset/public_reviews_dataset_cleaned.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/labelled_reviews.csv"))
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, dtype={"reviewer_id": "string"})
    source = pd.read_csv(
        args.reviews,
        usecols=["reviewer_id", "asin", "review_text"],
        dtype={"reviewer_id": "string", "asin": "string"},
        low_memory=False,
    )
    labels["review_text"] = labels["review_text"].fillna("").astype(str).str.strip()
    source["review_text"] = source["review_text"].fillna("").astype(str).str.strip()
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["reviewer_id", "review_text", "label"])
    source = source.dropna(subset=["reviewer_id", "asin", "review_text"])
    source = source.drop_duplicates(["reviewer_id", "review_text"])

    merged = labels.merge(
        source,
        on=["reviewer_id", "review_text"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise SystemExit(f"Could not map {missing} labeled rows to a product ID")
    merged = merged.drop(columns="_merge")
    merged["label"] = merged["label"].astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged):,} labeled reviews to {args.output}")
    print(f"Reviewers: {merged.reviewer_id.nunique():,} | Products: {merged.asin.nunique():,}")
    print("Labels:")
    print(merged.label.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
