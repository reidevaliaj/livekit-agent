"""Bounded per-call diagnostics without caller text, tool arguments or credentials."""

import contextlib
import hashlib
import json
import math
import queue
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

FIELDS = {
    "tool_name",
    "duration_ms",
    "outcome",
    "reason",
    "model",
    "voice",
    "language",
    "config_version",
    "sdk_version",
    "turn_detection",
    "eagerness",
    "reasoning_effort",
    "actor",
    "old_state",
    "new_state",
    "user_stop_to_agent_speaking_ms",
    "metrics",
    "error_type",
    "recoverable",
    "stage",
    "measurement",
    "transcript_accepted",
}
METRICS = {
    "e2e_latency",
    "llm_node_ttft",
    "tts_node_ttfb",
    "playback_latency",
    "ttft",
    "duration",
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cached_tokens",
}


class CallDiagnostics:
    def __init__(self, directory: Path, direction: str, tenant_id: str, room: str):
        self.call_id = str(uuid.uuid4())
        self.base = {
            "call_id": self.call_id,
            "direction": direction,
            "tenant_id": tenant_id,
            "room_id": hashlib.sha256(room.encode()).hexdigest()[:16],
        }
        self.started = time.monotonic()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.call_id}.jsonl"
        self._queue = queue.Queue(maxsize=256)
        self._closed = False
        self._worker = threading.Thread(
            target=self._write, daemon=True, name="call-diagnostics"
        )
        self._worker.start()

    def emit(self, event: str, **fields):
        if self._closed:
            return
        record = {
            **self.base,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "elapsed_ms": round((time.monotonic() - self.started) * 1000, 2),
            "event": event,
        }
        for key, value in fields.items():
            if key not in FIELDS:
                continue
            if key == "metrics" and isinstance(value, dict):
                record[key] = {
                    k: v
                    for k, v in value.items()
                    if k in METRICS
                    and isinstance(v, (int, float))
                    and math.isfinite(v)
                    and v >= 0
                }
            elif value is None or isinstance(value, (bool, int)):
                record[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                record[key] = round(value, 3)
            elif isinstance(value, str):
                record[key] = value[:120]
        # Diagnostics must never block voice processing.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(json.dumps(record, ensure_ascii=False))

    def _write(self):
        try:
            with self.path.open("x", encoding="utf-8", buffering=1) as handle:
                self.path.chmod(0o600)
                while True:
                    line = self._queue.get()
                    if line is None:
                        return
                    handle.write(line + "\n")
        except OSError:
            return

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=0.5)
            self._worker.join(timeout=0.5)
        except queue.Full:
            pass
