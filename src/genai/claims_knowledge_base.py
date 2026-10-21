"""Prepare insurance claims records for the GenAI RAG knowledge base."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "claims" / "claims_core.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "genai"
OUTPUT_PATH = OUTPUT_DIR / "claims_knowledge_base.jsonl"

# Fields excluded from the LLM retrieval text.
# Keep claim IDs only as source references, not as embedding content.
EXCLUDED_FROM_TEXT = {"fraud_label", "policy_number", "customer_sex"}


def main():
    df = pd.read_csv(DATA_PATH, dtype={"claim_id": str})

    required = {"claim_id", "fraud_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            claim_id = str(row["claim_id"])

            # Build descriptive text from available structured fields.
            fields = []
            for col in df.columns:
                if col in EXCLUDED_FROM_TEXT or col == "claim_id":
                    continue
                value = row[col]
                if pd.isna(value):
                    continue
                fields.append(f"{col.replace('_', ' ')}: {value}")

            record = {
                "id": claim_id,
                "text": "Insurance claim record. " + "; ".join(fields),
                "metadata": {
                    "claim_id": claim_id,
                    "claim_year": int(row["claim_year"]),
                    "fraud_label": int(row["fraud_label"]),
                    "source": "claims_core.csv",
                    "text_type": "structured_claim_record",
                },
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Knowledge-base records created: {count}")
    print(f"Output file: {OUTPUT_PATH}")
    print("Note: these are structured records, not original claim narratives.")


if __name__ == "__main__":
    main()