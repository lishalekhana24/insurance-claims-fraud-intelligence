from pathlib import Path
import pandas as pd

path = (
    Path(__file__).resolve().parents[1]
    / "data/raw/claims/claims_core.csv"
)
df = pd.read_csv(path)

print("Dataset shape:", df.shape)
print("\nFraud label counts:")
print(df["fraud_label"].value_counts(dropna=False))

print("\nFraud rate by claim year:")
print(
    df.groupby("claim_year")["fraud_label"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "fraud_rate"})
    .round(4)
)

print("\nFraud rate by categorical feature:")
categorical = [
    "fault", "policy_type", "base_policy", "vehicle_category",
    "vehicle_make", "accident_area", "police_report_filed",
    "witness_present", "agent_type", "customer_sex", "marital_status",
]
for col in categorical:
    if col in df.columns:
        summary = (
            df.groupby(col, dropna=False)["fraud_label"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "fraud_rate"})
            .sort_values("fraud_rate", ascending=False)
        )
        print(f"\n--- {col} ---")
        print(summary.round(4).to_string())

print("\nNumeric feature correlation with fraud label:")
numeric = df.select_dtypes(include="number")
corr = numeric.corr(numeric_only=True)["fraud_label"].drop(
    labels=["fraud_label"], errors="ignore"
)
print(corr.sort_values(key=abs, ascending=False).round(4).to_string())

print("\nPotential leakage/identifier candidates:")
for col in df.columns:
    if any(token in col.lower() for token in ["fraud", "label", "outcome", "decision"]):
        print(col, "unique values:", df[col].nunique(dropna=False))
