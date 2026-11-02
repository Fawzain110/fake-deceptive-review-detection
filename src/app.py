"""Streamlit demonstration UI for the saved fake-review detector."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Review Signal Lab", page_icon="🔎", layout="centered")
st.title("Fake Review Signal Lab")
st.caption("A classroom decision-support demo that combines review language with reviewer/product history.")

model_path = Path(__file__).resolve().parents[1] / "artifacts" / "review_detector.joblib"
if not model_path.exists():
    st.warning("Train the model first. See README.md for the setup and training commands.")
    st.stop()

bundle = joblib.load(model_path)
with st.form("review_input"):
    review = st.text_area("Review text", height=150, placeholder="Paste a product review…")
    c1, c2, c3 = st.columns(3)
    prior_reviews = c1.number_input("Prior reviewer reviews", 0, 10000, 0)
    prior_products = c2.number_input("Prior distinct products", 0, 10000, 0)
    product_reviewers = c3.number_input("Prior reviewers on product", 0, 100000, 0)
    submitted = st.form_submit_button("Assess review", type="primary")

if submitted:
    if not review.strip():
        st.error("Enter review text to continue.")
        st.stop()
    graph_frame = pd.DataFrame([{
        "reviewer_prior_review_count": prior_reviews,
        "reviewer_prior_product_degree": prior_products,
        "product_prior_reviewer_degree": product_reviewers,
    }], columns=bundle["graph_columns"])
    graph_scaled = bundle["graph_scaler"].transform(graph_frame)
    if bundle["backend"] == "tfidf":
        text_vector = bundle["text_vectorizer"].transform([review.strip()])
        from scipy.sparse import hstack
        x = hstack([text_vector, graph_scaled], format="csr")
    else:
        text_vector = bundle["text_encoder"].encode([review.strip()], normalize_embeddings=True)
        x = np.hstack([text_vector, graph_scaled])
    probability = float(bundle["model"].predict_proba(x)[0, 1])
    if probability >= 0.5:
        st.metric("Model signal", "Reviewer-labeled fake", f"{probability:.1%} estimated probability")
    else:
        st.metric("Model signal", "Reviewer-labeled honest", f"{1-probability:.1%} estimated probability")
    st.progress(probability)
    st.caption("This score is a model estimate, not proof of deception. The supplied labels are reviewer-classification labels; human review is required.")

with st.expander("How to interpret this demo"):
    st.write("The text model uses TF–IDF by default, or optional Sentence-BERT embeddings when trained with `--backend bert`. Graph inputs are prior counts in the reviewer–product network. If history is unavailable, leave the values at zero; the model then has no behavioral evidence for that case.")
