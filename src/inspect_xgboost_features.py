from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/raw/claims/claims_core.csv")

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
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

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
        scale_pos_weight=negative / positive,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )),
])

print("Training XGBoost on training data...")
model.fit(X_train, y_train)

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_xgb = model.named_steps["model"]
feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": fitted_xgb.feature_importances_,
}).sort_values("importance", ascending=False)

out = ROOT / "outputs/model/xgboost_feature_importance.csv"
out.parent.mkdir(parents=True, exist_ok=True)
importance.to_csv(out, index=False)

print("\nTop 20 encoded features:")
print(importance.head(20).to_string(index=False))
print(f"\nFull feature-importance report saved to: {out}")
