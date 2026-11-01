"""Camera results -> MQTT adapter -> validated, idempotent SQLite load.

The in-process mock has no socket, broker, protocol negotiation, or cloud service.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Protocol

import paho.mqtt.client as mqtt

from camera_quality.generator import SIZE
from camera_quality.messages import not_run_detector, validate_message

TOPIC = "synthetic/camera-01/inspection/v2"


class Publisher(Protocol):
    def publish(self, topic: str, payload: str) -> None: ...


def load_message(database: Path, payload: str | bytes) -> bool:
    """Return True for new image, False for exact replay; reject conflicting IDs."""
    message = validate_message(payload)
    canonical = json.dumps(message.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=10) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS inspections_v2 (
            image_id TEXT PRIMARY KEY, captured_at TEXT NOT NULL,
            source TEXT NOT NULL CHECK(source = 'synthetic'), decision TEXT NOT NULL,
            opencv_decision TEXT NOT NULL, cnn_decision TEXT NOT NULL,
            cnn_probability REAL NOT NULL, yolo_status TEXT NOT NULL,
            yolo_decision TEXT NOT NULL, yolo_boxes_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)""")
        cursor = connection.execute("""INSERT INTO inspections_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(image_id) DO NOTHING""", (
            message.image_id, message.captured_at, message.source, message.decision,
            message.opencv.decision, message.cnn.decision, message.cnn.probability,
            message.yolo.status, message.yolo.decision,
            json.dumps([box.model_dump() for box in message.yolo.boxes], sort_keys=True, allow_nan=False), digest, canonical))
        if cursor.rowcount == 1:
            return True
        existing = connection.execute("SELECT payload_sha256 FROM inspections_v2 WHERE image_id=?", (message.image_id,)).fetchone()
        if existing[0] != digest:
            raise ValueError("Conflicting payload for an existing image_id")
        return False


def make_message(prediction: dict, index: int) -> str:
    # Simulated, fixed clock: timestamps are not actual acquisition times.
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
    detector = prediction.get("yolo", not_run_detector())
    payload = dict(schema_version=2, image_id=prediction["image_id"],
                   camera_id="synthetic-camera-01", source="synthetic",
                   captured_at=instant.isoformat(), width=SIZE, height=SIZE,
                   opencv=prediction["opencv"], cnn=prediction["cnn"], yolo=detector,
                   decision="fail" if "fail" in (prediction["opencv"]["decision"], prediction["cnn"]["decision"], detector["decision"]) else "pass")
    encoded = json.dumps(payload, allow_nan=False)
    validate_message(encoded)
    return encoded


def publish_camera(predictions: list[dict], publisher: Publisher) -> int:
    for index, prediction in enumerate(predictions):
        publisher.publish(TOPIC, make_message(prediction, index))
    return len(predictions)


class MockMQTT:
    """Synchronous test double; exercises serialization/consumer/storage only."""
    def __init__(self, database: Path):
        self.database, self.inserted, self.duplicates = database, 0, 0

    def publish(self, topic: str, payload: str) -> None:
        if topic != TOPIC:
            raise ValueError("Unexpected topic")
        if load_message(self.database, payload):
            self.inserted += 1
        else:
            self.duplicates += 1


class PahoPublisher:
    """Optional real adapter; run only against a separately installed local broker."""
    def __init__(self, host="localhost", port=1883):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="synthetic-camera-publisher")
        connected = Event()
        def on_connect(client, userdata, flags, reason_code, properties):
            if not reason_code.is_failure:
                connected.set()
        self.client.on_connect = on_connect
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()
        if not connected.wait(5):
            self.close()
            raise TimeoutError("MQTT connection not acknowledged")

    def publish(self, topic: str, payload: str) -> None:
        info = self.client.publish(topic, payload, qos=1, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish error: {info.rc}")
        info.wait_for_publish(timeout=5)
        if not info.is_published():
            raise TimeoutError("MQTT publish not acknowledged")

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


def consume(database: Path, host="localhost", port=1883):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="synthetic-camera-consumer", manual_ack=True)
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            raise ConnectionError(str(reason_code))
        client.subscribe(TOPIC, qos=1)
    def on_message(client, userdata, message):
        try:
            load_message(database, message.payload)
        except ValueError as error:
            logging.error("Discarded invalid/conflicting message: %s", error)
        except sqlite3.Error:
            # Leave unacknowledged on storage failure; operator must reconnect/replay.
            logging.exception("SQLite failed; message was not acknowledged")
            client.disconnect()
            return
        client.ack(message.mid, message.qos)
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(host, port, keepalive=30)
    try:
        client.loop_forever()
    finally:
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["mock", "publish", "consume"])
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/local/predictions.json"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/local/inspection.sqlite"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    if args.mode == "consume":
        consume(args.database, args.host, args.port)
        return
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.mode == "mock":
        publisher = MockMQTT(args.database)
        publish_camera(predictions, publisher)
        publish_camera(predictions, publisher)
        print(json.dumps(dict(inserted=publisher.inserted, duplicates=publisher.duplicates)))
    else:
        publisher = PahoPublisher(args.host, args.port)
        try:
            print(f"Published {publish_camera(predictions, publisher)} synthetic results")
        finally:
            publisher.close()


if __name__ == "__main__":
    main()
