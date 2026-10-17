# Airflow Pipeline Execution Evidence

## Run details

- Project: Insurance Claims Fraud Intelligence
- DAG ID: `insurance_claims_pipeline`
- Execution date: 17 October 2026
- Execution method: Airflow CLI DAG test
- Command:
  `airflow dags test insurance_claims_pipeline 2026-10-17`
- Result: SUCCESS

## Tasks executed

1. `validate_input_files` — verified that the claims CSV, model evaluation JSON, compiled JAR, Python, Spark Submit, and HDFS commands were available.
2. `validate_claims_data` — validated required fields, non-empty values, unique claim IDs, binary fraud labels, and the expected row count.
3. `validate_model_evaluation` — checked the evaluation JSON, dataset size, fraud F1-score, and configured target.
4. `validate_prerequisites` — confirmed HDFS was accessible.
5. `build_silver_claims` — executed the Spark Silver processing job.
6. `build_gold_fraud_summary` — executed the Spark Gold summary job.
7. `verify_hdfs_outputs` — verified the Silver and Gold output directories and their `_SUCCESS` markers.

## Verified execution results

- Input claims: 15,420
- Silver records: 15,420
- Records removed by the Silver job: 0
- Fraud-labelled records: 923
- Non-fraud records: 14,497
- Gold summary: yearly fraud statistics for 1994, 1995, and 1996
- Silver output:
  `hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/silver/claims`
- Gold output:
  `hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/gold/fraud_summary`

The terminal output showed both `_SUCCESS` files and ended with:
`Dag run in success state`

## Model evaluation

The project's `outputs/model/evaluation.json` reports:

- Selected model: RandomForest
- Test fraud precision: 0.64
- Test fraud recall: 0.67
- Test fraud F1-score: 0.6548148148148148
- F1 target: 0.65
- Reported target status: PASS

The model metrics above come from the evaluation JSON. The Airflow task validated this file; the DAG run did not retrain the model.

## Evidence limitations

This report summarizes the terminal output captured for the run. It is not a raw Airflow log file or a screenshot. Retain the original terminal output or capture screenshots separately if required by the mentor.

This run verifies the listed validation tasks, Silver/Gold Spark batch jobs, and HDFS output checks. It does not by itself prove completion of every requirement in the overall project specification.
