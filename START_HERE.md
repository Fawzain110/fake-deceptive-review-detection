# Start Here

This repository contains the project source code, run instructions, evaluation results, and viva notes. The formal report and presentation are separate university-submission files. Raw data and the trained model are not included in the public repository.

## Before you submit

1. Fill in your name, roll number, college/department, and guide on the separate report and first presentation slide.
2. Ask your guide whether your college has a required cover page, certificate, declaration, or report format. If it does, use that format.
3. Run the project once and copy the measured values from `artifacts/metrics.json` into the report's Results table. Never submit invented performance scores.
4. Start the demo and try one review before your viva.

## Run the project on Windows

Obtain the university-provided data and place `training_reviews.csv` and `extracted_dataset/public_reviews_dataset_cleaned.csv` in the folder one level above this project folder. Open PowerShell in this project folder. In File Explorer, click the address bar, type `powershell`, and press Enter. Copy and run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\prepare_dataset.py
python src\train_model.py
```

To launch the demo:

```powershell
streamlit run src\app.py
```

Your browser will open the demo. Paste a review and click **Assess review**. If you do not know the reviewer's history, leave the history values at zero.

## What to explain in the viva

- The text model looks at word patterns in each review.
- The graph represents reviewers and products as connected nodes. It adds simple history counts.
- A high score is only a warning for human review; it is not proof that someone wrote a fake review.
- The project reports known-reviewer and unseen-reviewer results separately because those are different situations.
- The supplied labels are reviewer classifications, not independently verified outcomes. State this limitation clearly.

If any command shows an error, copy the complete error message and ask for help before trying random fixes.
