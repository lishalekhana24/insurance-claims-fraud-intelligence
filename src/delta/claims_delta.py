
from pathlib import Path
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable

# Project paths
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/raw/claims/claims_core.csv"
TABLE_PATH = ROOT / "outputs/delta/claims_table_demo"
EVIDENCE_PATH = ROOT / "docs/evidence/delta_lake"

# Start Spark with Delta Lake and the local filesystem
builder = (
    SparkSession.builder
    .appName("InsuranceClaimsDelta")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

try:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Claims CSV not found: {CSV_PATH}")

    # Read the original claims dataset
    claims = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(CSV_PATH.as_uri())
    )

    if "claim_id" not in claims.columns:
        raise ValueError("The claims CSV does not contain claim_id")

    # Ensure claim IDs can be used as merge keys
    claims = claims.withColumn("claim_id", claims["claim_id"].cast("string"))

    if claims.filter(
        claims["claim_id"].isNull() | (claims["claim_id"] == "")
    ).limit(1).count():
        raise ValueError("Null or empty claim_id found; fix data before MERGE")

    if claims.groupBy("claim_id").count().filter(
        "count > 1"
    ).limit(1).count():
        raise ValueError("Duplicate claim_id values found; MERGE needs unique keys")

    # Add a demo-only column; original CSV remains unchanged
    claims = claims.withColumn("delta_demo_status", lit("ORIGINAL"))

    # Reset only the dedicated demo table, never the source CSV
    if TABLE_PATH.exists():
        shutil.rmtree(TABLE_PATH)

    # 1. Create initial Delta table (version 0)
    claims.write.format("delta").mode("overwrite").save(str(TABLE_PATH))

    initial_count = spark.read.format("delta").load(str(TABLE_PATH)).count()
    print("Initial Delta table created.")
    print("Version 0 row count:", initial_count)

    # Choose one real claim to update
    first_claim = claims.orderBy("claim_id").limit(1)
    claim_id = first_claim.select("claim_id").first()["claim_id"]

    # Update the demo status of that claim
    update_claim = first_claim.withColumn(
        "delta_demo_status", lit("UPDATED_BY_MERGE")
    )

    # Create one clearly labelled synthetic test claim
    test_claim = (
        first_claim
        .withColumn("claim_id", lit("DELTA_DEMO_INSERT_001"))
        .withColumn("delta_demo_status", lit("SYNTHETIC_TEST"))
    )

    # Avoid accidental collision with a real claim ID
    if claims.filter(
        claims["claim_id"] == "DELTA_DEMO_INSERT_001"
    ).limit(1).count():
        raise ValueError("Demo claim ID already exists in source data")

    source = update_claim.unionByName(test_claim)

    # 2. MERGE: update existing claim and insert synthetic test claim
    delta_table = DeltaTable.forPath(spark, str(TABLE_PATH))

    (
        delta_table.alias("target")
        .merge(source.alias("source"), "target.claim_id = source.claim_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    print("\nMERGE completed.")
    spark.read.format("delta").load(str(TABLE_PATH)).filter(
        "claim_id = 'DELTA_DEMO_INSERT_001'"
    ).select("claim_id", "delta_demo_status").show(truncate=False)

    spark.read.format("delta").load(str(TABLE_PATH)).filter(
    f"claim_id = '{claim_id}'"
   ).select("claim_id", "delta_demo_status").show(truncate=False)
    # 3. Time travel: read version 0 before the MERGE
    old_data = (
        spark.read.format("delta")
        .option("versionAsOf", 0)
        .load(str(TABLE_PATH))
    )

    current_data = spark.read.format("delta").load(str(TABLE_PATH))

    old_count = old_data.count()
    current_count = current_data.count()

    print("Rows in version 0:", old_count)
    print("Rows in current version:", current_count)

    if old_count != initial_count or current_count != initial_count + 1:
        raise RuntimeError("Unexpected row counts; verify the MERGE results")

    # 4. Save audit history as project evidence
    EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)

    history = DeltaTable.forPath(spark, str(TABLE_PATH)).history()

    history.select(
        "version", "timestamp", "operation", "operationMetrics"
    ).show(truncate=False)

    history.select(
        "version", "timestamp", "operation", "operationMetrics"
    ).write.mode("overwrite").json(
        str(EVIDENCE_PATH / "history")
    )

    print("\nAll checks passed.")
    print("Delta table:", TABLE_PATH)
    print("Audit history:", EVIDENCE_PATH / "history")

finally:
    spark.stop()

