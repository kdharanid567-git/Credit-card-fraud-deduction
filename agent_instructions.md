# Agent Instructions

## Project Overview
This is a **Credit Card Fraud Detection** Machine Learning web application.

## Dataset
- Path: `data/dataset.csv`
- Source: Credit Card Fraud 2026
- Rows: ~20 000
- Target column: `is_fraud` (0 = Legitimate, 1 = Fraud)
- Task type: Binary Classification

## Key Files
| File | Purpose |
|------|---------|
| `train_model.py` | Loads data, preprocesses, trains & compares models, saves best pipeline |
| `app.py` | Streamlit UI — loads `models/model.pkl`, accepts user inputs, returns predictions |
| `models/model.pkl` | Joblib-serialised dict containing the pipeline + metadata |
| `data/dataset.csv` | Raw CSV dataset |

## How the pipeline works
1. `train_model.py` reads `data/dataset.csv`.
2. Boolean string columns are converted to 0/1.
3. `transaction_id` is dropped (identifier, not a feature).
4. A `ColumnTransformer` applies `StandardScaler` to numeric columns and
   `OneHotEncoder` to categorical columns.
5. Three classifiers are compared via 5-fold stratified cross-validation
   (metric: F1-macro).
6. The best pipeline is retrained on the full training split and evaluated on
   the 20 % held-out test set.
7. The pipeline + metadata dict is saved with `joblib.dump`.

## Adding a new model
Add the model to the `models` dict in `train_model.py` under `# Model candidates`.
It will automatically be included in the comparison.

## Deployment notes
- Commit `models/model.pkl` to the repo for zero-config Streamlit Cloud deployment.
- All paths use `pathlib.Path(__file__).parent` — no absolute paths.
- No API keys or external services required.
