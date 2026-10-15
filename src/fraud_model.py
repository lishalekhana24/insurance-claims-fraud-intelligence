from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/claims/claims_core.csv"
OUTPUT_DIR = ROOT / "outputs/model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
df = df.dropna(subset=["fraud_label"])
df = df[df["fraud_label"].isin([0, 1])].copy()
df["fraud_label"] = df["fraud_label"].astype(int)

y = df["fraud_label"]
X = df.drop(columns=["fraud_label", "claim_id", "policy_number"], errors="ignore")
X = X.loc[:, X.nunique(dropna=False) > 1]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

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

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

thresholds = [i / 100 for i in range(5, 56, 5)]
candidates = []
fitted = {}

for name, classifier in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    probs = pipe.predict_proba(X_val)[:, 1]

    for threshold in thresholds:
        pred_val = (probs >= threshold).astype(int)
        candidates.append({
            "model": name,
            "threshold": threshold,
            "validation_f1": float(
                f1_score(y_val, pred_val, zero_division=0)
            ),
            "validation_precision": float(
                precision_score(y_val, pred_val, zero_division=0)
            ),
            "validation_recall": float(
                recall_score(y_val, pred_val, zero_division=0)
            ),
        })

best = max(candidates, key=lambda row: row["validation_f1"])
chosen = fitted[best["model"]]
test_probs = chosen.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= best["threshold"]).astype(int)

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
test_f1 = f1_score(y_test, test_pred, zero_division=0)

results = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "selected_model": best["model"],
    "selected_threshold": best["threshold"],
    "best_validation_metrics": best,
    "test_fraud_precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "test_fraud_recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "test_fraud_f1": float(test_f1),
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "test_confusion_matrix_labels_0_1": cm.tolist(),
    "f1_target": 0.65,
    "f1_target_pass": bool(test_f1 >= 0.65),
    "validation_candidates": candidates,
}

with (OUTPUT_DIR / "evaluation.json").open("w") as f:
    json.dump(results, f, indent=2)

print("Top validation candidates:")
for row in sorted(
    candidates, key=lambda r: r["validation_f1"], reverse=True
)[:10]:
    print(
        f"{row['model']:12s} threshold={row['threshold']:.2f} "
        f"F1={row['validation_f1']:.4f} "
        f"precision={row['validation_precision']:.4f} "
        f"recall={row['validation_recall']:.4f}"
    )

print(f"\nSelected model: {best['model']}")
print(f"Selected threshold: {best['threshold']:.2f}")
print("\nHeld-out test report:")
print(classification_report(
    y_test, test_pred, labels=[0, 1],
    target_names=["non_fraud", "fraud"], zero_division=0
))
print("Test confusion matrix (rows=actual, columns=predicted; labels 0,1):")
print(cm)
print(f"Test fraud F1-score: {test_f1:.4f}")
print(f"F1 >= 0.65: {'PASS' if test_f1 >= 0.65 else 'NOT MET'}")
print(f"Metrics saved to: {OUTPUT_DIR / 'evaluation.json'}")
