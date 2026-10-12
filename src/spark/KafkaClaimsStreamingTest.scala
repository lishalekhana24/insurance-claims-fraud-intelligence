
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object KafkaClaimsStreamingTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Kafka Claims Streaming Test")
      .master("local[1]")
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

    val stream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:19092")
      .option("subscribe", "claims_stream_test")
      .option("startingOffsets", "earliest")
      .load()

    val claims = stream
      .select(from_json(col("value").cast("string"), schema).alias("claim"))
      .select("claim.*")
      .filter(col("claim_id").isNotNull)

    val query = claims.writeStream
      .format("parquet")
      .outputMode("append")
      .option(
        "path",
        "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/bronze/streaming_test"
      )
      .option(
        "checkpointLocation",
        "hdfs://localhost:9000/user/nopassword/insurance-claims-fraud-intelligence/checkpoints/streaming_test"
      )
      .start()

    println("Streaming test is running. Waiting for the test event...")
    query.awaitTermination()
  }
}

