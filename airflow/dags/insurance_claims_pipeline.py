
from datetime import datetime
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(
    os.environ.get(
        "INSURANCE_PROJECT_ROOT",
        str(Path.home() / "insurance-claims-fraud-intelligence"),
    )
)

JAR = PROJECT_ROOT / (
    "target/scala-2.12/"
    "insurance-claims-fraud-intelligence_2.12-0.1.0-SNAPSHOT.jar"
)

HDFS_ROOT = (
    "hdfs://localhost:9000/user/harsha/"
    "insurance-claims-fraud-intelligence"
)

CLAIMS_CSV = PROJECT_ROOT / "data/raw/claims/claims_core.csv"
EVALUATION_JSON = PROJECT_ROOT / "outputs/model/evaluation.json"

with DAG(
    dag_id="insurance_claims_pipeline",
    description="Validate claims and model, then build Silver and Gold",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "insurance-claims-team",
        "depends_on_past": False,
        "retries": 1,
    },
    tags=["insurance", "spark", "fraud"],
) as dag:

    validate_input_files = BashOperator(
        task_id="validate_input_files",
        bash_command="""
        set -euo pipefail
        test -s "$CLAIMS_CSV"
        test -s "$EVALUATION_JSON"
        test -f "$JAR"
        command -v python
        command -v spark-submit
        command -v hdfs
        echo "Input files and required commands are available."
        """,
        append_env=True,
        env={
            "CLAIMS_CSV": str(CLAIMS_CSV),
            "EVALUATION_JSON": str(EVALUATION_JSON),
            "JAR": str(JAR),
        },
    )

    validate_claims_data = BashOperator(
        task_id="validate_claims_data",
        bash_command="""
        set -euo pipefail
        python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["CLAIMS_CSV"])
required = {"claim_id", "policy_number", "fraud_label"}

with path.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    columns = set(reader.fieldnames or [])
    missing = required - columns
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    count = 0
    seen_ids = set()
    for row in reader:
        count += 1
        for field in required:
            if not (row.get(field) or "").strip():
                raise SystemExit(
                    f"Empty {field} at CSV data row {count}"
                )

        claim_id = row["claim_id"].strip()
        if claim_id in seen_ids:
            raise SystemExit(f"Duplicate claim_id: {claim_id}")
        seen_ids.add(claim_id)

        if row["fraud_label"].strip() not in {"0", "1"}:
            raise SystemExit(
                f"Unexpected fraud_label at CSV data row {count}"
            )

    if count != 15420:
        raise SystemExit(f"Expected 15420 rows, found {count}")

print(f"Claims validation passed: {count} rows; unique claim IDs.")
PY
        """,
        append_env=True,
        env={"CLAIMS_CSV": str(CLAIMS_CSV)},
    )

    validate_model_evaluation = BashOperator(
        task_id="validate_model_evaluation",
        bash_command="""
        set -euo pipefail
        python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["EVALUATION_JSON"])
with path.open(encoding="utf-8") as f:
    result = json.load(f)

required = {
    "dataset_rows",
    "selected_model",
    "test_fraud_f1",
    "f1_target",
    "f1_target_pass",
}
missing = required - result.keys()
if missing:
    raise SystemExit(f"Missing evaluation fields: {sorted(missing)}")

if result["dataset_rows"] != 15420:
    raise SystemExit("Evaluation dataset row count is unexpected")

if not isinstance(result["test_fraud_f1"], (int, float)):
    raise SystemExit("test_fraud_f1 must be numeric")

if result["test_fraud_f1"] < result["f1_target"]:
    raise SystemExit("Reported test F1 is below the configured target")

if result["f1_target_pass"] is not True:
    raise SystemExit("Evaluation does not report the F1 target as passed")

print("Model evaluation validation passed.")
print("Model:", result["selected_model"])
print("Test fraud F1:", result["test_fraud_f1"])
print("F1 target:", result["f1_target"])
PY
        """,
        append_env=True,
        env={"EVALUATION_JSON": str(EVALUATION_JSON)},
    )

    validate_prerequisites = BashOperator(
        task_id="validate_prerequisites",
        bash_command="""
        set -euo pipefail
        hdfs dfs -ls /
        echo "HDFS is accessible."
        """,
        append_env=True,
    )

    build_silver_claims = BashOperator(
        task_id="build_silver_claims",
        bash_command="""
        set -euo pipefail
        spark-submit --class SilverClaims --master 'local[*]' "$JAR"
        """,
        append_env=True,
        env={"JAR": str(JAR)},
    )

    build_gold_fraud_summary = BashOperator(
        task_id="build_gold_fraud_summary",
        bash_command="""
        set -euo pipefail
        spark-submit --class GoldFraudSummary --master 'local[*]' "$JAR"
        """,
        append_env=True,
        env={"JAR": str(JAR)},
    )

    verify_hdfs_outputs = BashOperator(
        task_id="verify_hdfs_outputs",
        bash_command="""
        set -euo pipefail
        hdfs dfs -test -d "${HDFS_ROOT}/silver/claims"
        hdfs dfs -test -d "${HDFS_ROOT}/gold/fraud_summary"
        hdfs dfs -test -e "${HDFS_ROOT}/silver/claims/_SUCCESS"
        hdfs dfs -test -e "${HDFS_ROOT}/gold/fraud_summary/_SUCCESS"
        hdfs dfs -ls "${HDFS_ROOT}/silver/claims"
        hdfs dfs -ls "${HDFS_ROOT}/gold/fraud_summary"
        echo "Silver and Gold output checks passed."
        """,
        append_env=True,
        env={"HDFS_ROOT": HDFS_ROOT},
    )

    (
        validate_input_files
        >> validate_claims_data
        >> validate_model_evaluation
        >> validate_prerequisites
        >> build_silver_claims
        >> build_gold_fraud_summary
        >> verify_hdfs_outputs
    )
