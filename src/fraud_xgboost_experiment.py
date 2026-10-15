from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/claims/claims_core.csv"
OUT = ROOT / "outputs/model/xgboost_experiment.json"

df = pd.read_csv(DATA)
df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
df = df.dropna(subset=["fraud_label"])
df = df[df["fraud_label"].isin([0, 1])].copy()
df["fraud_label"] = df["fraud_label"].astype(int)

y = df["fraud_label"]
X = df.drop(
    columns=["fraud_label", "claim_id", "policy_number"],
    errors="ignore",
)
X = X.loc[:, X.nunique(dropna=False) > 1]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    c for c in X.columns if c not in numeric_cols
]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Compensate for class imbalance using training data only.
negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())
scale_pos_weight = negative / positive

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )),
])

print("Training XGBoost...")
model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]
thresholds = [i / 100 for i in range(5, 81, 5)]

best_threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val, (val_probs >= t).astype(int), zero_division=0
    ),
)

val_pred = (val_probs >= best_threshold).astype(int)
test_probs = model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= best_threshold).astype(int)

result = {
    "model": "XGBoost",
    "xgboost_version": __import__("xgboost").__version__,
    "dataset_rows": len(df),
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),
    "scale_pos_weight": scale_pos_weight,
    "threshold_selected_on_validation": best_threshold,
    "validation_f1": float(
        f1_score(y_val, val_pred, zero_division=0)
    ),
    "validation_precision": float(
        precision_score(y_val, val_pred, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_val, val_pred, zero_division=0)
    ),
    "exploratory_test_f1": float(
        f1_score(y_test, test_pred, zero_division=0)
    ),
    "exploratory_test_precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "exploratory_test_recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "test_confusion_matrix_labels_0_1": confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).tolist(),
    "f1_target": 0.65,
    "f1_target_met": bool(
        f1_score(y_test, test_pred, zero_division=0) >= 0.65
    ),
    "evaluation_note": (
        "Exploratory test result; this test split has been examined "
        "in previous experiments and is not an untouched final benchmark."
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))

print("\nXGBoost results:")
print(json.dumps(result, indent=2))
print(f"\nSaved to: {OUT}")
