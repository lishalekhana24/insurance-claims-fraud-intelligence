
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ClaimPolicyValidation {

  def main(args: Array[String]): Unit = {

    if (args.length < 3) {
      println("Usage: ClaimPolicyValidation <claims.csv> <policies.csv> <output_path>")
      System.exit(1)
    }

    val claimsPath = args(0)
    val policyPath = args(1)
    val outputPath = args(2)

    val spark = SparkSession.builder()
      .appName("Insurance Claim Policy Validation")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {

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

      // Check missing values
      val nullClaimIds =
        claims.filter(col("claim_id").isNull).count()

      val nullPolicyNumbers =
        claims.filter(col("policy_number").isNull).count()

      val nullFraudLabels =
        claims.filter(col("fraud_label").isNull).count()

      // Check duplicate claim IDs
      val duplicateClaimIds =
        claims.count() - claims.select("claim_id").distinct().count()

      println("Null claim IDs       : " + nullClaimIds)
      println("Null policy numbers  : " + nullPolicyNumbers)
      println("Null fraud labels    : " + nullFraudLabels)
      println("Duplicate claim IDs  : " + duplicateClaimIds)

      // Find claims without a matching policy
      val unmatchedClaims = claims
        .join(
          policies.select("policy_number"),
          Seq("policy_number"),
          "left_anti"
        )
        .count()

      println("Unmatched claims     : " + unmatchedClaims)

      // Select policy columns for the join
      val policyForJoin = policies.select(
        col("policy_number"),
        col("policy_type").alias("policy_dim_type"),
        col("base_policy").alias("policy_dim_base"),
        col("deductible").alias("policy_dim_deductible")
      )

      // Join claims with policies
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

      // Save the joined data as Parquet
      claimPolicy.write
        .mode("overwrite")
        .parquet(outputPath)

      println("======================================")
      println("PARQUET WRITE COMPLETED")
      println("Output: " + outputPath)
      println("======================================")

    } finally {
      spark.stop()
    }
  }
}

