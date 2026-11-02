"""Train text-only and hybrid deceptive-review baselines with leakage-aware features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


GRAPH_COLUMNS = [
    "reviewer_prior_review_count",
    "reviewer_prior_product_degree",
    "product_prior_reviewer_degree",
]


def build_graph_features(history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Calculate bipartite reviewer/product degrees from history only.

    ``history`` must contain training rows only. Test rows never contribute edges.
    The reviewer/product edge for a training row is subtracted (leave-one-out).
    """
    graph = nx.Graph()
    for reviewer, product in history[["reviewer_id", "asin"]].drop_duplicates().itertuples(index=False):
        graph.add_edge(f"r:{reviewer}", f"p:{product}")
    reviewer_counts = history.groupby("reviewer_id").size().to_dict()
    reviewer_products = history.groupby("reviewer_id")["asin"].agg(lambda values: set(values)).to_dict()
    product_reviewers = history.groupby("asin")["reviewer_id"].agg(lambda values: set(values)).to_dict()
    pair_counts = history.groupby(["reviewer_id", "asin"]).size().to_dict()

    rows = rows.copy()
    known_train = rows.index.isin(history.index)
    features = []
    for (reviewer, product), is_train in zip(rows[["reviewer_id", "asin"]].itertuples(index=False, name=None), known_train):
        reviewer_seen = reviewer in reviewer_products and product in reviewer_products[reviewer]
        product_seen = product in product_reviewers and reviewer in product_reviewers[product]
        remove_edge = int(bool(is_train and reviewer_seen and product_seen))
        pair_count = int(pair_counts.get((reviewer, product), 0))
        prior_count = max(0, int(reviewer_counts.get(reviewer, 0)) - remove_edge)
        prior_products = len(reviewer_products.get(reviewer, set()))
        if is_train and pair_count == 1 and reviewer_seen:
            prior_products = max(0, prior_products - 1)
        prior_product_reviewers = len(product_reviewers.get(product, set()))
        if is_train and pair_count == 1 and product_seen:
            prior_product_reviewers = max(0, prior_product_reviewers - 1)
        # Read the graph degrees as a sanity-checked source of the node degrees.
        if reviewer in reviewer_products:
            assert graph.degree(f"r:{reviewer}") == len(reviewer_products[reviewer])
        features.append((prior_count, prior_products, prior_product_reviewers))
    return pd.DataFrame(features, columns=GRAPH_COLUMNS, index=rows.index)


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, backend: str = "tfidf"):
    train_graph = build_graph_features(train, train)
    test_graph = build_graph_features(train, test)
    scaler = StandardScaler()
    xg_train = scaler.fit_transform(train_graph)
    xg_test = scaler.transform(test_graph)

    if backend == "tfidf":
        vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english", min_df=2)
        xt_train = vectorizer.fit_transform(train.review_text.fillna(""))
        xt_test = vectorizer.transform(test.review_text.fillna(""))
        from scipy.sparse import hstack
        x_train = hstack([xt_train, xg_train], format="csr")
        x_test = hstack([xt_test, xg_test], format="csr")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(x_train, train.label.astype(int))
        return model, vectorizer, scaler, x_test, x_train, test_graph, train_graph

    if backend == "bert":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        xt_train = encoder.encode(train.review_text.fillna("").tolist(), show_progress_bar=True, normalize_embeddings=True)
        xt_test = encoder.encode(test.review_text.fillna("").tolist(), show_progress_bar=True, normalize_embeddings=True)
        x_train = np.hstack([xt_train, xg_train])
        x_test = np.hstack([xt_test, xg_test])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(x_train, train.label.astype(int))
        return model, encoder, scaler, x_test, x_train, test_graph, train_graph
    raise ValueError("backend must be 'tfidf' or 'bert'")


def metrics(model, x, y) -> dict[str, float]:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/labelled_reviews.csv"))
    parser.add_argument("--backend", choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    df = pd.read_csv(args.data, dtype={"reviewer_id": "string", "asin": "string"})
    df = df.dropna(subset=["reviewer_id", "asin", "review_text", "label"]).reset_index(drop=True)
    df["review_text"] = df.review_text.astype(str).str.strip()
    df = df[df.review_text.ne("")].reset_index(drop=True)
    df["label"] = df.label.astype(int)

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=df.label)
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    model, encoder, scaler, x_test, x_train, test_graph, train_graph = fit_and_score(train, test, args.backend)
    review_split_metrics = metrics(model, x_test, test.label)

    # A separate cold-start estimate uses reviewers absent from training and text only.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    cold_train_idx, cold_test_idx = next(splitter.split(df, df.label, groups=df.reviewer_id))
    cold_train, cold_test = df.iloc[cold_train_idx], df.iloc[cold_test_idx]
    if args.backend == "tfidf":
        cold_vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english", min_df=2)
        cx_train = cold_vectorizer.fit_transform(cold_train.review_text)
        cx_test = cold_vectorizer.transform(cold_test.review_text)
        cold_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        cold_model.fit(cx_train, cold_train.label)
        cold_metrics = metrics(cold_model, cx_test, cold_test.label)
    else:
        from sentence_transformers import SentenceTransformer
        cold_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        cx_train = cold_encoder.encode(cold_train.review_text.tolist(), show_progress_bar=True, normalize_embeddings=True)
        cx_test = cold_encoder.encode(cold_test.review_text.tolist(), show_progress_bar=True, normalize_embeddings=True)
        cold_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        cold_model.fit(cx_train, cold_train.label)
        cold_metrics = metrics(cold_model, cx_test, cold_test.label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "backend": args.backend,
        "model": model,
        "text_encoder": encoder,
        "graph_scaler": scaler,
        "graph_columns": GRAPH_COLUMNS,
        "feature_defaults": train_graph.median().to_dict(),
        "label_meaning": {0: "reviewer-labeled honest", 1: "reviewer-labeled fake"},
        "cold_start_model": cold_model,
    }
    if args.backend == "tfidf":
        model_bundle["text_vectorizer"] = encoder
        model_bundle["cold_start_vectorizer"] = cold_vectorizer
    else:
        model_bundle["cold_start_encoder"] = cold_encoder
    joblib.dump(model_bundle, args.output_dir / "review_detector.joblib")
    report = {
        "dataset_rows": int(len(df)),
        "reviewers": int(df.reviewer_id.nunique()),
        "products": int(df.asin.nunique()),
        "class_counts": {str(int(k)): int(v) for k, v in df.label.value_counts().sort_index().items()},
        "backend": args.backend,
        "review_random_split_hybrid": review_split_metrics,
        "reviewer_disjoint_cold_start_text_only": cold_metrics,
        "review_random_split_sizes": {"train": int(len(train)), "test": int(len(test))},
        "reviewer_disjoint_split_sizes": {"train": int(len(cold_train)), "test": int(len(cold_test))},
        "caveat": "Labels are supplied reviewer-classification labels, not independently adjudicated fraud ground truth.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
