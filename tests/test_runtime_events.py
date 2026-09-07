from types import SimpleNamespace

import pytest

from voice_runtime import CallRuntime


@pytest.mark.asyncio
async def test_full_runtime_binds_real_sdk_session_and_receives_realtime_metrics(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-only")
    events = []
    monkeypatch.setattr(
        "voice_runtime.CallDiagnostics",
        lambda *args: SimpleNamespace(
            emit=lambda event, **fields: events.append((event, fields))
        ),
    )
    runtime = CallRuntime(
        SimpleNamespace(room=SimpleNamespace(name="room")),
        object(),
        {
            "tenant": {"id": "tenant"},
            "config": {"business_name": "Example", "timezone": "UTC"},
        },
        "incoming",
        {},
        SimpleNamespace(identity="sip-test"),
    )
    runtime.session.emit(
        "metrics_collected",
        SimpleNamespace(
            metrics=SimpleNamespace(
                ttft=0.23,
                duration=0.7,
                input_tokens=20,
                output_tokens=12,
                total_tokens=32,
            )
        ),
    )
    assert events[-1] == (
        "per_turn_metrics",
        {
            "metrics": {
                "ttft": 0.23,
                "duration": 0.7,
                "input_tokens": 20,
                "output_tokens": 12,
                "total_tokens": 32,
            }
        },
    )
