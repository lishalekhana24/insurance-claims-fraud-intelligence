import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ClaimPolicyValidation {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Insurance Claim Policy Validation")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val claimsPath =
      "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/raw/claims/claims_core.csv"

    val policyPath =
      "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/raw/policy/dim_policy.csv"

    // Read claims
    val claims = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(claimsPath)

    // Read policies
    val policies = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(policyPath)

    println("======================================")
    println("INSURANCE CLAIM VALIDATION")
    println("======================================")

    println("Total claims   : " + claims.count())
    println("Total policies : " + policies.count())

    // Basic quality checks
    val nullClaimIds =
      claims.filter(col("claim_id").isNull).count()

    val nullPolicyNumbers =
      claims.filter(col("policy_number").isNull).count()

    val nullFraudLabels =
      claims.filter(col("fraud_label").isNull).count()

    val duplicateClaimIds =
      claims.count() - claims.select("claim_id").distinct().count()

    println("Null claim IDs       : " + nullClaimIds)
    println("Null policy numbers  : " + nullPolicyNumbers)
    println("Null fraud labels    : " + nullFraudLabels)
    println("Duplicate claim IDs  : " + duplicateClaimIds)

    // Validate claim -> policy relationship
    val unmatchedClaims = claims
      .join(
        policies.select("policy_number"),
        Seq("policy_number"),
        "left_anti"
      )
      .count()

    println("Unmatched claims     : " + unmatchedClaims)

    // Select only policy columns that are not already present in claims.
    // This avoids duplicate/ambiguous columns after the join.
    val policyForJoin = policies.select(
      col("policy_number"),
      col("policy_type").alias("policy_dim_type"),
      col("base_policy").alias("policy_dim_base"),
      col("deductible").alias("policy_dim_deductible")
    )

    // Valid claim-policy join
    val claimPolicy = claims
      .join(policyForJoin, Seq("policy_number"), "left")

    println("Rows after join      : " + claimPolicy.count())

    println("======================================")
    println("FRAUD DISTRIBUTION")
    println("======================================")

    claimPolicy
      .groupBy("fraud_label")
      .count()
      .orderBy("fraud_label")
      .show()

    println("======================================")
    println("SAMPLE VALIDATED DATA")
    println("======================================")

    claimPolicy
      .select(
        "claim_id",
        "policy_number",
        "policy_dim_type",
        "policy_dim_base",
        "policy_dim_deductible",
        "fraud_label"
      )
      .show(10, false)

    // Write validated result to local processed area
    val outputPath =
  "/home/nopassword/insurance-claims-fraud-intelligence/data/processed/claim_policy_validated"

    println("======================================")
println("STARTING PARQUET WRITE")
println("======================================")

claimPolicy.write
  .mode("overwrite")
  .parquet(outputPath)

println("======================================")
println("PARQUET WRITE COMPLETED")
println("Output: " + outputPath)
println("======================================")
    println("Output: " + outputPath)
    println("======================================")

    spark.stop()
  }
}
