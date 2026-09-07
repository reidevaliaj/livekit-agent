import json

from diagnostics import CallDiagnostics


def test_each_call_has_own_file_and_drops_sensitive_fields(tmp_path):
    first = CallDiagnostics(tmp_path, "incoming", "tenant-1", "room-private")
    second = CallDiagnostics(tmp_path, "incoming", "tenant-1", "room-private")
    first.emit(
        "tool_completed",
        tool_name="book_meeting",
        duration_ms=12.2,
        outcome="booked",
        transcript="private speech",
        token="secret",
    )
    second.emit("session_ended", reason="completed")
    first.close()
    second.close()
    assert first.path != second.path
    records = [json.loads(line) for line in first.path.read_text().splitlines()]
    assert records[0]["tool_name"] == "book_meeting"
    assert "private" not in first.path.read_text()
    assert "secret" not in first.path.read_text()


def test_numeric_metrics_keep_zero_and_drop_payloads(tmp_path):
    log = CallDiagnostics(tmp_path, "incoming", "tenant-1", "room")
    log.emit(
        "per_turn_metrics", metrics={"e2e_latency": 0, "text": "private", "ttft": 0.25}
    )
    log.close()
    record = json.loads(log.path.read_text().splitlines()[0])
    assert record["metrics"] == {"e2e_latency": 0, "ttft": 0.25}


def test_missing_realtime_timing_sentinel_is_not_reported_as_negative_latency(tmp_path):
    log = CallDiagnostics(tmp_path, "incoming", "tenant", "room")
    log.emit("per_turn_metrics", metrics={"ttft": -1, "duration": 0, "output_tokens": 20})
    log.close()
    record = json.loads(log.path.read_text().splitlines()[0])
    assert record["metrics"] == {"duration": 0, "output_tokens": 20}
