from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("LoadInsuranceBucketedTable")
    .master("local[2]")
    .enableHiveSupport()
    .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse")
    .getOrCreate()
)

spark.sql("USE insurance_analytics")

data = spark.read.parquet(
    "file:///home/nopassword/insurance-claims-fraud-intelligence/data/processed/hive_star_schema/fact_claims.parquet"
)

data.write \
    .mode("overwrite") \
    .format("parquet") \
    .bucketBy(4, "policy_number") \
    .saveAsTable("fact_claims_bucketed")

print("Rows loaded:", spark.table("fact_claims_bucketed").count())
print("Bucketed table created successfully.")

spark.stop()
