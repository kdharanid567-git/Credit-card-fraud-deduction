"""
train_model.py
--------------
Trains ML models on the Credit Card Fraud 2026 dataset and saves the best
pipeline to models/model.pkl.

Run:
    python train_model.py
"""

import os
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")
print(f"[INFO] Target distribution:\n{df['is_fraud'].value_counts()}")

# ── Feature Engineering ────────────────────────────────────────────────────────
DROP_COLS  = ["transaction_id"]
TARGET_COL = "is_fraud"

# Convert boolean-like string columns to int
bool_str_cols = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
for col in bool_str_cols:
    if df[col].dtype == object:
        df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col]).astype(int)
    else:
        df[col] = df[col].astype(int)

# Drop identifier columns
df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# ── Column types ───────────────────────────────────────────────────────────────
CAT_COLS = X.select_dtypes(include=["object", "category"]).columns.tolist()
NUM_COLS = X.select_dtypes(include=["number"]).columns.tolist()

print(f"[INFO] Categorical features : {CAT_COLS}")
print(f"[INFO] Numerical features   : {NUM_COLS}")

# ── Preprocessor ──────────────────────────────────────────────────────────────
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ]
)

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Model candidates ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

best_name   = None
best_score  = -1
best_pipeline = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[INFO] === Model Comparison (5-Fold CV F1-macro) ===")
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    mean_score = scores.mean()
    print(f"  {name:25s}  CV F1-macro = {mean_score:.4f}  ± {scores.std():.4f}")
    if mean_score > best_score:
        best_score  = mean_score
        best_name   = name
        best_pipeline = pipe

print(f"\n[INFO] Best model: {best_name}  (CV F1-macro = {best_score:.4f})")

# ── Final training on full train set ─────────────────────────────────────────
print("[INFO] Fitting best model on full training set …")
best_pipeline.fit(X_train, y_train)

# ── Evaluation on held-out test set ──────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n[INFO] === Test Set Evaluation ===")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1-macro : {f1_score(y_test, y_pred, average='macro'):.4f}")

# ── Persist ───────────────────────────────────────────────────────────────────
meta = {
    "pipeline":    best_pipeline,
    "model_name":  best_name,
    "features":    list(X.columns),
    "cat_cols":    CAT_COLS,
    "num_cols":    NUM_COLS,
    "bool_cols":   bool_str_cols,
    "target":      TARGET_COL,
    "task":        "classification",
    "classes":     ["Legitimate", "Fraud"],
    "cat_values":  {col: sorted(X[col].dropna().unique().tolist()) for col in CAT_COLS},
}

joblib.dump(meta, MODEL_PATH)
print(f"\n[INFO] Model + metadata saved to {MODEL_PATH}")
print("[DONE] Training complete.")
