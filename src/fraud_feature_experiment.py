from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/claims/claims_core.csv"
OUT = ROOT / "outputs/model/feature_experiment.json"

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

ordinal_orders = {
    "vehicle_price": [
        "less than 20000", "20000 to 29000", "30000 to 39000",
        "40000 to 59000", "60000 to 69000", "more than 69000",
    ],
    "vehicle_age": [
        "new", "2 years", "3 years", "4 years", "5 years",
        "6 years", "7 years", "more than 7",
    ],
    "number_of_supplements": [
        "none", "1 to 2", "3 to 5", "more than 5",
    ],
    "address_change_claim": [
        "no change", "under 6 months", "1 year", "2 to 3 years",
        "4 to 8 years",
    ],
    "number_of_cars": [
        "1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8",
    ],
    "days_policy_accident": [
        "none", "1 to 7", "8 to 15", "15 to 30", "more than 30",
    ],
    "days_policy_claim": [
        "none", "8 to 15", "15 to 30", "more than 30",
    ],
    "past_number_of_claims": [
        "none", "1", "2 to 4", "more than 4",
    ],
    "age_of_policyholder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30",
        "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
}

ordinal_cols = [c for c in ordinal_orders if c in X.columns]
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    c for c in X.columns
    if c not in numeric_cols and c not in ordinal_cols
]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

def make_preprocessor(engineered):
    transformers = [
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ]

    if engineered and ordinal_cols:
        transformers.append((
            "ordinal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(
                    categories=[ordinal_orders[c] for c in ordinal_cols],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            ordinal_cols,
        ))

    onehot_cols = categorical_cols + (
        [] if engineered else ordinal_cols
    )
    if onehot_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            onehot_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")

results = []
thresholds = [i / 100 for i in range(5, 56, 5)]

for name, engineered in [
    ("baseline_onehot", False),
    ("ordinal_engineered", True),
]:
    model = Pipeline([
        ("preprocessor", make_preprocessor(engineered)),
        ("model", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    threshold = max(
        thresholds,
        key=lambda t: f1_score(
            y_val, (val_probs >= t).astype(int), zero_division=0
        ),
    )

    val_pred = (val_probs >= threshold).astype(int)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)

    row = {
        "model": name,
        "ordinal_columns": ordinal_cols if engineered else [],
        "threshold_selected_on_validation": threshold,
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
print(f"\nSaved experiment to: {OUT}")
print("F1 target: 0.65; target met:", any(
    r["exploratory_test_f1"] >= 0.65 for r in results
))
