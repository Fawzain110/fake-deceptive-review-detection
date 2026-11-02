# Fake and Deceptive Review Detection

University minor project exploring how review language and reviewer–product network history can help flag suspicious product reviews.

## Project overview

The project builds a reviewer–product bipartite graph and combines three prior-history signals with review text:

- prior reviews by the reviewer
- distinct products reviewed by the reviewer
- prior reviewers connected to the product

The default text encoder is TF–IDF with word unigrams and bigrams. An optional Sentence-BERT backend is available with `--backend bert`. A class-weighted logistic regression classifier produces a probability estimate. The Streamlit demo accepts a review and optional history values.

This is a coursework prototype for review triage. A model score is not proof of deception and must not be used as the sole basis for moderation.

## Results from the supplied run

The supplied labeled file contains 80,281 reviews, 36,764 reviewers, and 3,031 matched products. Labels are imbalanced: 22,614 label-0 examples and 57,667 label-1 examples. The classification labels come from reviewer-classification fields in the supplied data; they are weak labels, not independently verified outcomes.

| Evaluation setting | Accuracy | Fake precision | Fake recall | Fake F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Random review split, hybrid text + graph | 0.7889 | 0.9123 | 0.7812 | 0.8417 | 0.7625 |
| Reviewer-disjoint cold start, text only | 0.7235 | 0.8597 | 0.7408 | 0.7958 | 0.6838 |

The first split tests reviews where reviewer history can be available. The reviewer-disjoint split tests unseen reviewers and therefore uses text only. Report both; they measure different settings. These values measure agreement with the supplied labels, not validated real-world fraud detection.

## Repository contents

- `src/prepare_dataset.py` joins the labeled review file to product IDs in the supplied full corpus.
- `src/train_model.py` trains the baseline and hybrid models, evaluates both split types, and writes `artifacts/metrics.json` plus `artifacts/review_detector.joblib`.
- `src/app.py` launches the interactive demo.
- `requirements.txt` lists the default Python dependencies.
- `VIVA_PREP.md` gives beginner-friendly explanations and likely presentation questions.

The formal report and viva slides are being prepared separately for university submission. Raw review data, processed review rows, virtual environments, model binaries, and those submission documents are intentionally excluded from this code repository. The data should be obtained from the university-provided source and used according to its terms.

## Run locally on Windows

Place `training_reviews.csv` and `extracted_dataset/public_reviews_dataset_cleaned.csv` in the folder one level above this project folder. Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\prepare_dataset.py
python src\train_model.py
streamlit run src\app.py
```

The app opens at the local address Streamlit prints, normally `http://localhost:8501`. Stop it with **Ctrl+C** in PowerShell. If reviewer history is unknown, leave its values at zero.

For Sentence-BERT embeddings, install `sentence-transformers` in the environment, then run:

```powershell
python -m pip install sentence-transformers
python src\train_model.py --backend bert
```

The BERT backend downloads `sentence-transformers/all-MiniLM-L6-v2` on its first run and needs additional disk space and compute.

## Evaluation design

The random review split is stratified and uses training-only graph history. Each training row's own reviewer–product edge is excluded from its graph counts. The second split groups by reviewer so its test set contains reviewers absent from training. The cold-start model uses only text. Metrics include accuracy, fake-class precision, fake-class recall, fake-class F1, and macro-F1.

## Limitations

- The supplied labels are reviewer-level classifications and are not independently adjudicated review-level fraud labels.
- The graph signals are simple degree/activity counts; prolific activity alone is not evidence of deception.
- The evaluation is not chronological, and review-level random splits are not a replacement for a time-based holdout.
- The optional BERT route is available but the reported scores above are for the TF–IDF run.

