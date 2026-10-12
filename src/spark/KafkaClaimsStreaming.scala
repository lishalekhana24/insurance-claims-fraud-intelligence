
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object KafkaClaimsStreaming {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Kafka Claims Streaming to HDFS")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val schema = new StructType()
      .add("claim_id", StringType)
      .add("policy_number", StringType)
      .add("event_type", StringType)
      .add("claim_amount", DoubleType)
      .add("event_time", StringType)
      .add("fraud_label", StringType)
      .add("claim_year", StringType)

    val kafkaStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:19092")
      .option("subscribe", "claims_intake")
      .option("startingOffsets", "latest")
      .load()

    val parsedClaims = kafkaStream
      .select(from_json(
        col("value").cast("string"), schema
      ).alias("claim"))
      .select("claim.*")
      .filter(col("claim_id").isNotNull)
      .filter(col("policy_number").isNotNull)

    val outputPath =
      "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/bronze/streaming_claims"

    val checkpointPath =
      "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/checkpoints/kafka_claims_stream"

    val query = parsedClaims.writeStream
      .format("parquet")
      .outputMode("append")
      .option("path", outputPath)
      .option("checkpointLocation", checkpointPath)
      .start()

    println("Streaming claims from Kafka to HDFS...")
    println("Output: " + outputPath)

    query.awaitTermination()
  }
}

