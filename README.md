
### 1. What I have completed

**1. Dataset preparation**
- Inspected the insurance claims dataset and related policy, vehicle, agent, city, weather, and earthquake datasets.
- The main claims dataset contains 15,420 records.
- Identified data limitations: the historical claims are from 1994–1996, while the weather dataset covers 2020–2025. So, we should not directly join them without a valid location and date relationship.

**2. Environment setup**
- Set up Java, Scala, Apache Spark, Hadoop/HDFS, and Kafka using Docker.
- Created the project structure and loaded the required CSV files into HDFS.

**3. Spark batch processing**
- Implemented `ClaimPolicyValidation.scala` to validate and join claims with policy data.
- Successfully processed 15,420 claims and 15,420 policy records.
- Validation showed zero duplicate claim IDs, zero unmatched claims, and zero null values in the checked key fields.
- Implemented `SilverClaims.scala` to filter invalid records and remove duplicate claim IDs.
- Implemented `GoldFraudSummary.scala` to generate a preliminary fraud summary by claim year and fraud label.

**4. Kafka and real-time processing**
- Created the `claims_intake` Kafka topic with three partitions.
- Created `claims_producer.py` to publish claim events from the CSV dataset to Kafka.
- Implemented Spark Structured Streaming to read Kafka events and write Parquet output to HDFS.
- Successfully tested a separate streaming test topic with one event, `CLM_STREAM_TEST_001`. Spark processed the event, saved it to HDFS, and I verified that it could be read back.

Note: the one-event test is verified; the complete real-time pipeline still needs further testing.

### 2. What needs to be continued

Please continue from the existing code rather than rebuilding everything.

**Priority 1 — Complete the streaming pipeline**
- Verify the main `claims_intake` topic works end-to-end with the Spark streaming application.
- Check that incoming records are actually saved to HDFS.
- Validate checkpoint recovery and restart behavior.
- Avoid rerunning the full 15,420-record producer unnecessarily.

**Priority 2 — Improve the Bronze, Silver, and Gold layers**
- Organize raw, validated, cleaned, and aggregated data properly.
- Add data-quality checks and record-count reconciliation between stages.
- Ensure Bronze preserves the original ingested events and Silver contains cleaned records.
- Expand Gold with useful business metrics, such as fraud counts and rates by year and other valid dimensions.

**Priority 3 — Catastrophe matching**
- Continue weather and earthquake matching using Spark.
- Use only valid location and date relationships.
- Do not create artificial matches between historical claims and unrelated weather records.

**Priority 4 — Fraud detection model**
- Train and evaluate a Decision Tree and Random Forest model.
- Use proper train/test separation and evaluate precision, recall, F1-score, and the confusion matrix.
- Check for data leakage and class imbalance.
- The project specification targets an F1-score of at least 0.65, but we must report the actual measured result.

**Priority 5 — Remaining capstone components**
- Airflow orchestration.
- Hive analytical tables and reporting.
- HBase claim-status history, where supported by the available environment.
- AWS S3, Glue, and Athena integration.
- Databricks Delta Lake and auditability.
- GenAI/RAG with PII redaction.
- Power BI dashboard and business KPIs.
- Reliability testing, documentation, and final integration.

### 3. Important points

- Please review the project specification before deciding the next implementation steps.
- Do not assume every component is completed just because the environment is installed.
- Preserve existing source code and validate each new change.
- Document actual results; do not invent model scores or business improvements.
- Coordinate before changing shared GitHub files so we do not overwrite each other's work.

My work so far mainly covers **initial data preparation, Spark batch processing, Kafka setup, and a verified one-event streaming test**. The remaining work is to turn these pieces into a complete, tested end-to-end capstone.
