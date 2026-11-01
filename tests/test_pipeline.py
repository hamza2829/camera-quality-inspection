import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from camera_quality.messages import validate_message
from camera_quality.pipeline import MockMQTT, PahoPublisher, TOPIC, consume, load_message, make_message, publish_camera


@pytest.fixture
def prediction():
    return dict(image_id="sim-" + "a"*24, opencv=dict(decision="pass", regions=[]),
                cnn=dict(decision="pass", probability=.2, threshold=.5))


def test_message_roundtrip_and_simulated_clock(prediction):
    result = validate_message(make_message(prediction, 3))
    assert result.source == "synthetic" and result.captured_at == "2026-01-01T00:00:03+00:00"
    assert result.decision == "pass"


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("source", "real"), ("image_id", "../../escape"),
    ("width", "96"), ("width", True), ("height", 0),
    ("captured_at", "2026-01-01T00:00:00"), ("captured_at", "not-a-date"),
    ("decision", "fail"), ("unexpected", "secret"),
])
def test_message_rejects_invalid_fields(prediction, field, value):
    payload = json.loads(make_message(prediction, 0))
    payload[field] = value
    with pytest.raises((ValidationError, ValueError)):
        validate_message(json.dumps(payload))


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), float("inf")])
def test_message_rejects_invalid_probabilities(prediction, probability):
    payload = json.loads(make_message(prediction, 0))
    payload["cnn"]["probability"] = probability
    with pytest.raises(ValueError):
        validate_message(json.dumps(payload))


def test_message_rejects_inconsistent_cnn_and_classical_decisions(prediction):
    payload = json.loads(make_message(prediction, 0))
    payload["cnn"]["probability"] = .8
    with pytest.raises(ValueError):
        validate_message(json.dumps(payload))
    payload = json.loads(make_message(prediction, 0))
    payload["opencv"]["decision"] = "fail"
    with pytest.raises(ValueError):
        validate_message(json.dumps(payload))


def test_message_rejects_out_of_bounds_regions(prediction):
    prediction["opencv"] = dict(decision="fail", regions=[dict(x=90, y=40, width=20, height=10, area_px=40., center_x=92., center_y=44.)])
    with pytest.raises(ValueError):
        make_message(prediction, 0)


@pytest.mark.parametrize("payload", ["{broken", "x"*64_001], ids=["malformed", "oversize"])
def test_bad_payload_does_not_create_database(tmp_path, payload):
    database = tmp_path / "results.sqlite"
    with pytest.raises(ValueError):
        load_message(database, payload)
    assert not database.exists()


def test_replay_is_idempotent_across_consumer_restart(tmp_path, prediction):
    database = tmp_path / "results.sqlite"
    first = MockMQTT(database)
    publish_camera([prediction], first)
    second = MockMQTT(database)
    publish_camera([prediction], second)
    assert first.inserted == second.duplicates == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1


def test_concurrent_replay_inserts_one_row(tmp_path, prediction):
    database = tmp_path / "results.sqlite"
    payload = make_message(prediction, 0)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: load_message(database, payload), range(8)))
    assert results.count(True) == 1 and results.count(False) == 7


def test_conflicting_id_is_rejected_without_overwriting(tmp_path, prediction):
    database = tmp_path / "results.sqlite"
    payload = make_message(prediction, 0)
    assert load_message(database, payload)
    conflict = copy.deepcopy(prediction)
    conflict["cnn"]["probability"] = .3
    with pytest.raises(ValueError, match="Conflicting"):
        load_message(database, make_message(conflict, 0))
    assert not load_message(database, payload)


def test_paho_publisher_uses_qos_one_and_waits_for_ack(prediction):
    client = MagicMock()
    client.connect.side_effect = lambda *a, **k: client.on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
    client.publish.return_value.rc = 0
    client.publish.return_value.is_published.return_value = True
    with patch("camera_quality.pipeline.mqtt.Client", return_value=client):
        publisher = PahoPublisher()
        publish_camera([prediction], publisher)
        publisher.close()
    assert client.publish.call_args.kwargs == dict(qos=1, retain=False)
    client.publish.return_value.wait_for_publish.assert_called_once_with(timeout=5)
    client.disconnect.assert_called_once()


def test_paho_publisher_reports_failed_publish(prediction):
    client = MagicMock()
    client.connect.side_effect = lambda *a, **k: client.on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
    client.publish.return_value.rc = 4
    with patch("camera_quality.pipeline.mqtt.Client", return_value=client):
        publisher = PahoPublisher()
        with pytest.raises(RuntimeError, match="publish error"):
            publish_camera([prediction], publisher)
        publisher.close()


def test_consumer_validates_then_commits_and_acknowledges(tmp_path, prediction):
    client = MagicMock()
    good = SimpleNamespace(payload=make_message(prediction, 0).encode(), mid=1, qos=1)
    bad = SimpleNamespace(payload=b'{"source":"real"}', mid=2, qos=1)
    def deliver():
        client.on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
        client.on_message(client, None, good)
        client.on_message(client, None, good)
        client.on_message(client, None, bad)
    client.loop_forever.side_effect = deliver
    database = tmp_path / "results.sqlite"
    def confirm_committed_before_ack(*args):
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
    client.ack.side_effect = confirm_committed_before_ack
    with patch("camera_quality.pipeline.mqtt.Client", return_value=client):
        consume(database)
    client.subscribe.assert_called_once_with(TOPIC, qos=1)
    assert client.ack.call_count == 3
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0] == 1
