import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object KafkaClaimsStreaming {
  def main(args: Array[String]): Unit = {

    val bootstrapServers =
      args.lift(0).getOrElse("localhost:9092")

    val topic =
      args.lift(1).getOrElse("claims_intake")

    val outputPath =
      args.lift(2).getOrElse(
        "hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/bronze/streaming_claims"
      )

    val checkpointPath =
      args.lift(3).getOrElse(
        "hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/checkpoints/kafka_claims_stream"
      )

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
      .add("is_test", BooleanType)

    val kafkaStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("subscribe", topic)
      .option("startingOffsets", "latest")
      .load()

    val parsedClaims = kafkaStream
      .select(
        from_json(col("value").cast("string"), schema).alias("claim")
      )
      .select("claim.*")
      .filter(col("claim_id").isNotNull)
      .filter(col("policy_number").isNotNull)

    val query = parsedClaims.writeStream
      .format("parquet")
      .outputMode("append")
      .option("path", outputPath)
      .option("checkpointLocation", checkpointPath)
      .start()

    println(s"Kafka broker: $bootstrapServers")
    println(s"Topic: $topic")
    println(s"Output: $outputPath")
    println(s"Checkpoint: $checkpointPath")
    println("Streaming claims from Kafka to HDFS...")

    query.awaitTermination()
  }
}
