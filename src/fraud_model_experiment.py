from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/claims/claims_core.csv"
OUT = ROOT / "outputs/model/class_weight_experiment.json"

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

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

thresholds = [i / 100 for i in range(5, 56, 5)]
weights = ["balanced", "balanced_subsample", None]
results = []

for weight in weights:
    model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=400,
            class_weight=weight,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    best_threshold = max(
        thresholds,
        key=lambda t: f1_score(
            y_val, (val_probs >= t).astype(int), zero_division=0
        ),
    )
    val_pred = (val_probs >= best_threshold).astype(int)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_threshold).astype(int)

    row = {
        "class_weight": weight,
        "threshold_selected_on_validation": best_threshold,
        "validation_f1": float(f1_score(y_val, val_pred, zero_division=0)),
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
    }
    results.append(row)
    print(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\nExperiment results saved to: {OUT}")
print("F1 target: 0.65; evaluate results honestly.")
