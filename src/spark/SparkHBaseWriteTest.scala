import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Get, Put}
import org.apache.hadoop.hbase.util.Bytes

object SparkHBaseWriteTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Spark HBase Write Test")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val tableName = TableName.valueOf("insurance:claim_status")
    val claimId = "SPARK_HBASE_TEST_001"
    val eventTime = java.time.Instant.now().toString
    val rowKey = s"$claimId#$eventTime"

    try {
      import spark.implicits._

      val data: DataFrame = Seq((
        claimId,
        eventTime,
        "SPARK_HBASE_TEST",
        "TEST_SUCCESS",
        "SPARK_EVENT_001"
      )).toDF(
        "claim_id",
        "event_time",
        "event_type",
        "assessment_status",
        "source_event_id"
      )

      data.foreachPartition { (rows: Iterator[Row]) =>
        val conf = HBaseConfiguration.create()
        val connection = ConnectionFactory.createConnection(conf)

        try {
          val table = connection.getTable(TableName.valueOf(tableNameString))

          try {
            rows.foreach { row =>
              val id = row.getAs[String]("claim_id")
              val time = row.getAs[String]("event_time")
              val key = s"$id#$time"
              val put = new Put(Bytes.toBytes(key))

              def add(family: String, column: String, value: String): Unit =
                put.addColumn(
                  Bytes.toBytes(family),
                  Bytes.toBytes(column),
                  Bytes.toBytes(value)
                )

              add("claim", "claim_id", id)
              add("claim", "event_time", time)
              add("claim", "event_type", row.getAs[String]("event_type"))
              add("fraud", "assessment_status",
                row.getAs[String]("assessment_status"))
              add("audit", "source_event_id",
                row.getAs[String]("source_event_id"))

              table.put(put)
              println(s"SPARK_WRITE_OK rowKey=$key")
            }
          } finally {
            table.close()
          }
        } finally {
          connection.close()
        }
      }

      val conf = HBaseConfiguration.create()
      val connection = ConnectionFactory.createConnection(conf)

      try {
        val table = connection.getTable(tableName)

        try {
          val result = table.get(new Get(Bytes.toBytes(rowKey)))

          def read(family: String, column: String): String =
            Bytes.toString(
              result.getValue(Bytes.toBytes(family), Bytes.toBytes(column))
            )

          val storedId = read("claim", "claim_id")
          val storedStatus = read("fraud", "assessment_status")
          val storedEventId = read("audit", "source_event_id")

          require(storedId == claimId, s"Claim ID mismatch: $storedId")
          require(storedStatus == "TEST_SUCCESS",
            s"Status mismatch: $storedStatus")
          require(storedEventId == "SPARK_EVENT_001",
            s"Event ID mismatch: $storedEventId")

          println(s"HBASE_READ_OK claim_id=$storedId")
          println(s"HBASE_READ_OK assessment_status=$storedStatus")
          println(s"HBASE_READ_OK source_event_id=$storedEventId")
          println("SPARK_HBASE_WRITE_TEST_PASSED")
        } finally {
          table.close()
        }
      } finally {
        connection.close()
      }
    } finally {
      spark.stop()
    }
  }
}
