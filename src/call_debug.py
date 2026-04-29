from __future__ import annotations

import atexit
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG_LOG_PATH = REPO_ROOT / "runtime" / "active_call_debug.log"


class CallDebugLogger:
    def __init__(
        self,
        path: Path | None = None,
        *,
        reset_on_init: bool = True,
        cleanup_on_close: bool = True,
    ) -> None:
        self.path = Path(path or DEFAULT_DEBUG_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_close = cleanup_on_close
        if reset_on_init:
            self.path.write_text("", encoding="utf-8")

        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._closed = False
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="call-debug-writer",
            daemon=True,
        )
        self._writer.start()
        atexit.register(self.close, False)

    def log(self, category: str, event: str, **fields: Any) -> None:
        if self._closed:
            return

        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        parts = [timestamp, category.upper(), event]
        for key, value in fields.items():
            parts.append(f"{key}={self._serialize(value)}")
        self._queue.put(" | ".join(parts))

    def close(self, cleanup: bool | None = None) -> None:
        if self._closed:
            return

        self._closed = True
        self._queue.put(None)
        self._writer.join(timeout=1.5)

        should_cleanup = self._cleanup_on_close if cleanup is None else cleanup
        if should_cleanup:
            self.path.write_text("", encoding="utf-8")

    def _writer_loop(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                handle.write(item + "\n")

    def _serialize(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, str):
            return json.dumps(self._shorten(value), ensure_ascii=False)
        if isinstance(value, (int, float, bool)):
            return json.dumps(value)

        try:
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            elif hasattr(value, "dict"):
                value = value.dict()
            elif hasattr(value, "__dict__"):
                value = {
                    key: val
                    for key, val in vars(value).items()
                    if not key.startswith("_")
                }
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(repr(value), ensure_ascii=False)

    def _shorten(self, value: str, limit: int = 400) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."
