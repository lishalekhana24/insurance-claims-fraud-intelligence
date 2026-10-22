import java.time.Instant

import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Get, Put}
import org.apache.hadoop.hbase.util.Bytes

object HBaseWriteSmokeTest {
  def main(args: Array[String]): Unit = {
    val tableName = TableName.valueOf("insurance:claim_status")
    val claimId = "HBASE_SMOKE_001"
    val eventTime = Instant.now().toString
    val rowKey = s"$claimId#$eventTime"

    val conf = HBaseConfiguration.create()
    val connection = ConnectionFactory.createConnection(conf)

    try {
      val table = connection.getTable(tableName)

      try {
        val put = new Put(Bytes.toBytes(rowKey))

        def add(family: String, column: String, value: String): Unit =
          put.addColumn(
            Bytes.toBytes(family),
            Bytes.toBytes(column),
            Bytes.toBytes(value)
          )

        add("claim", "claim_id", claimId)
        add("claim", "event_type", "HBASE_SMOKE_TEST")
        add("claim", "event_time", eventTime)
        add("fraud", "assessment_status", "TEST_SUCCESS")
        add("audit", "source_event_id", "SMOKE_EVENT_001")

        table.put(put)
        println(s"WRITE_OK rowKey=$rowKey")

        val result = table.get(new Get(Bytes.toBytes(rowKey)))

        val storedClaimId = Bytes.toString(
          result.getValue(Bytes.toBytes("claim"), Bytes.toBytes("claim_id"))
        )
        val storedEventType = Bytes.toString(
          result.getValue(Bytes.toBytes("claim"), Bytes.toBytes("event_type"))
        )
        val storedStatus = Bytes.toString(
          result.getValue(Bytes.toBytes("fraud"), Bytes.toBytes("assessment_status"))
        )

        require(storedClaimId == claimId, "Claim ID verification failed")
        require(storedEventType == "HBASE_SMOKE_TEST", "Event type verification failed")
        require(storedStatus == "TEST_SUCCESS", "Status verification failed")

        println(s"READ_OK claim_id=$storedClaimId")
        println(s"READ_OK event_type=$storedEventType")
        println(s"READ_OK assessment_status=$storedStatus")
        println("HBASE_SMOKE_TEST_PASSED")
      } finally {
        table.close()
      }
    } finally {
      connection.close()
    }
  }
}
