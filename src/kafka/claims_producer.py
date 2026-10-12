
import csv
import json
import time
from datetime import datetime, timezone

from kafka import KafkaProducer

CSV_FILE = "data/raw/claims/claims_core.csv"
KAFKA_TOPIC = "claims_intake"
KAFKA_SERVER = "localhost:19092"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all",
)

try:
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            event = {
                "claim_id": row["claim_id"],
                "policy_number": row["policy_number"],
                "event_type": "CLAIM_RECEIVED",
                "fraud_label": row["fraud_label"],
                "claim_year": row["claim_year"],
                "event_time": datetime.now(timezone.utc).isoformat(),
            }

            producer.send(KAFKA_TOPIC, value=event)
            print(f"Sent claim: {event['claim_id']}")

            # Slow down the replay so we can observe the events.
            time.sleep(0.2)

    producer.flush()
    print("All claim events sent successfully.")

finally:
    producer.close()

