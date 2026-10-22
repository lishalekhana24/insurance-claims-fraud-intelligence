import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.Instant

import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Put}
import org.apache.hadoop.hbase.util.Bytes

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object KafkaClaimsStreaming {

  private val HBaseTable = "insurance:claim_status"

  private def stringValue(row: Row, column: String): String = {
    val index = row.fieldIndex(column)
    if (row.isNullAt(index)) ""
    else Option(row.getAs[String](column)).getOrElse("")
  }

  private def stableEventId(row: Row): String = {
    val columns = Seq(
      "claim_id",
      "policy_number",
      "event_type",
      "claim_amount",
      "event_time",
      "fraud_label"
    )

    val input = columns.map { column =>
      stringValue(row, column)
    }.mkString("|")

    val digest = MessageDigest.getInstance("SHA-256")
      .digest(input.getBytes(StandardCharsets.UTF_8))

    digest.take(12).map(b => f"${b & 0xff}%02x").mkString
  }

  private def writeBatchToHBase(batchDF: DataFrame): Unit = {
    batchDF.foreachPartition { (rows: Iterator[Row]) =>
      if (rows.hasNext) {
        val conf = HBaseConfiguration.create()
        val connection = ConnectionFactory.createConnection(conf)
        val table = connection.getTable(TableName.valueOf(HBaseTable))

        try {
          rows.foreach { row =>
            val claimId = stringValue(row, "claim_id")
            val eventTime = stringValue(row, "event_time")
            val eventId = stableEventId(row)

            val rowKey = s"$claimId#$eventTime#$eventId"
            val put = new Put(Bytes.toBytes(rowKey))

            def add(family: String, column: String, value: String): Unit = {
              if (value.nonEmpty) {
                put.addColumn(
                  Bytes.toBytes(family),
                  Bytes.toBytes(column),
                  Bytes.toBytes(value)
                )
              }
            }

            add("claim", "claim_id", claimId)
            add("claim", "policy_number", stringValue(row, "policy_number"))
            add("claim", "event_type", stringValue(row, "event_type"))
            add("claim", "event_time", eventTime)
            add("claim", "claim_year", stringValue(row, "claim_year"))
            add("fraud", "fraud_label", stringValue(row, "fraud_label"))
            add("audit", "source_event_id", eventId)
            add("audit", "source_topic", "claims_intake")
            add("audit", "ingested_at", Instant.now().toString)

            val amountIndex = row.fieldIndex("claim_amount")
            if (!row.isNullAt(amountIndex)) {
              add(
                "claim",
                "claim_amount",
                row.getAs[Double]("claim_amount").toString
              )
            }

            val testIndex = row.fieldIndex("is_test")
            if (!row.isNullAt(testIndex)) {
              add(
                "claim",
                "is_test",
                row.getAs[Boolean]("is_test").toString
              )
            }

            table.put(put)
          }
        } finally {
          try {
            table.close()
          } finally {
            connection.close()
          }
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val bootstrapServers = args.lift(0).getOrElse("localhost:9092")
    val topic = args.lift(1).getOrElse("claims_intake")

    val outputPath = args.lift(2).getOrElse(
      "hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/bronze/streaming_claims"
    )

    val checkpointBase = args.lift(3).getOrElse(
      "hdfs://localhost:9000/user/harsha/insurance-claims-fraud-intelligence/checkpoints"
    )

    val spark = SparkSession.builder()
      .appName("Kafka Claims Streaming with HBase")
      .master("local[2]")
      .config("spark.sql.shuffle.partitions", "2")
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
      .select(from_json(col("value").cast("string"), schema).alias("claim"))
      .select("claim.*")
      .filter(col("claim_id").isNotNull)
      .filter(col("policy_number").isNotNull)

    // Existing Parquet sink.
    val parquetQuery = parsedClaims.writeStream
      .format("parquet")
      .outputMode("append")
      .option("path", outputPath)
      .option("checkpointLocation", s"$checkpointBase/kafka_claims_parquet")
      .start()

    // New HBase claim-status timeline sink.
    val hbaseQuery = parsedClaims.writeStream
      .queryName("claims-to-hbase")
      .outputMode("append")
      .option("checkpointLocation", s"$checkpointBase/kafka_claims_hbase")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        println(s"Writing batch $batchId to HBase")
        writeBatchToHBase(batchDF)
      }
      .start()

    println(s"Kafka broker: $bootstrapServers")
    println(s"Kafka topic: $topic")
    println(s"Parquet output: $outputPath")
    println(s"HBase table: $HBaseTable")
    println("Streaming started with local[2].")

    try {
      spark.streams.awaitAnyTermination()
    } finally {
      spark.stop()
    }
  }
}
