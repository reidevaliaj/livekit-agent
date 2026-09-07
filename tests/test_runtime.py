from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from runtime_config import VoiceOptions
from voice_runtime import CallRuntime, build_session, disconnect_sip, fetch_config


@pytest.mark.asyncio
async def test_realtime_sdk_constructs_with_ga_audio_and_low_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    session = build_session(VoiceOptions())
    assert session.llm.model == "gpt-realtime-2.1-mini"
    assert session.llm.capabilities.audio_output is True
    assert session.llm._opts.reasoning.effort == "low"
    assert session.llm._opts.turn_detection.interrupt_response is True


@pytest.mark.asyncio
async def test_actual_sdk_start_serializes_ga_session_and_tools_to_local_websocket(
    monkeypatch,
):
    """Run the real SDK pipeline against a local fake provider, with no billed calls."""
    import asyncio
    import json

    from aiohttp import ClientSession, web

    import voice_runtime
    from receptionist import Receptionist

    received = []
    tools_ready = asyncio.Event()

    async def websocket(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            if message.type == web.WSMsgType.TEXT:
                event = json.loads(message.data)
                received.append(event)
                if event["type"] == "session.update" and event["session"].get("tools"):
                    tools_ready.set()
        return socket

    app = web.Application()
    app.router.add_get("/v1/realtime", websocket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    real_model = voice_runtime.openai.realtime.RealtimeModel
    session = None
    try:
        async with ClientSession() as client:

            def local_model(**kwargs):
                return real_model(
                    **kwargs,
                    api_key="local-test",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    http_session=client,
                )

            monkeypatch.setattr(
                voice_runtime.openai.realtime, "RealtimeModel", local_model
            )
            session = build_session(VoiceOptions(language="it"))
            snapshot = {
                "tenant": {"id": "tenant"},
                "config": {
                    "business_name": "Example",
                    "timezone": "UTC",
                    "enabled_tools": {
                        "calendar_lookup": True,
                        "meeting_creation": True,
                    },
                },
            }
            agent = Receptionist(
                snapshot,
                "incoming",
                "test-room",
                SimpleNamespace(post=AsyncMock()),
                SimpleNamespace(emit=lambda *_args, **_kwargs: None),
                AsyncMock(),
            )
            await session.start(agent=agent, record=False)
            await asyncio.wait_for(tools_ready.wait(), timeout=4)
            updates = [
                event["session"]
                for event in received
                if event["type"] == "session.update"
            ]
            assert any(
                update.get("reasoning") == {"effort": "low"} for update in updates
            )
            assert any(
                update.get("audio", {})
                .get("input", {})
                .get("turn_detection", {})
                .get("eagerness")
                == "high"
                for update in updates
            )
            functions = [tool for update in updates for tool in update.get("tools", [])]
            assert {tool["name"] for tool in functions} == {
                "check_meeting_slot",
                "book_meeting",
                "call_end",
            }
            booking = next(tool for tool in functions if tool["name"] == "book_meeting")
            assert "start_iso" in booking["parameters"]["required"]
            await session.aclose()
            session = None
    finally:
        if session:
            await session.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_bad_configuration_cannot_start_default_business():
    participant = SimpleNamespace(identity="sip-test", attributes={}, metadata="")
    ctx = SimpleNamespace(
        room=SimpleNamespace(name="room"),
        wait_for_participant=AsyncMock(return_value=participant),
    )
    backend = SimpleNamespace(post=AsyncMock(return_value={"ok": False}))
    with pytest.raises(ValueError):
        await fetch_config(ctx, backend, "incoming", {})


@pytest.mark.asyncio
async def test_teardown_targets_only_linked_sip_participant():
    from livekit import rtc

    kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    one = SimpleNamespace(identity="sip-one", kind=kind)
    two = SimpleNamespace(identity="sip-two", kind=kind)
    room = SimpleNamespace(
        name="room",
        remote_participants={"one": one, "two": two},
        disconnect=AsyncMock(),
    )
    removed = AsyncMock()
    shutdown = []
    ctx = SimpleNamespace(
        room=room,
        api=SimpleNamespace(room=SimpleNamespace(remove_participant=removed)),
        shutdown=lambda **kw: shutdown.append(kw),
    )
    await disconnect_sip(ctx, "sip-one")
    removed.assert_awaited_once()
    assert removed.await_args.args[0].identity == "sip-one"
    # The SDK disconnects the agent room after closing the session and its
    # transcript/event streams. Only the linked SIP caller is removed here.
    room.disconnect.assert_not_awaited()
    assert shutdown


@pytest.mark.asyncio
async def test_finish_removes_sip_even_when_farewell_and_provider_fail(monkeypatch):
    runtime = CallRuntime.__new__(CallRuntime)
    runtime._ending = False
    runtime.direction = "outgoing"
    runtime.options = VoiceOptions()
    runtime.participant_identity = "sip-matched"
    runtime.ctx = object()
    runtime.speak = AsyncMock(side_effect=TimeoutError)
    runtime.backend = SimpleNamespace(post=AsyncMock(side_effect=TimeoutError))
    runtime.log = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime.outgoing_payload = lambda: {"outgoing_call_id": "call-matched"}
    disconnected = AsyncMock()
    monkeypatch.setattr("voice_runtime.disconnect_sip", disconnected)
    await runtime.finish()
    await runtime.finish()
    disconnected.assert_awaited_once_with(runtime.ctx, "sip-matched")
    assert runtime.backend.post.await_args.args[1]["outgoing_call_id"] == "call-matched"


@pytest.mark.asyncio
async def test_livekit_first_prepares_voice_before_dial(tmp_path, monkeypatch):
    order = []

    async def start(**_kwargs):
        order.append("session_start")

    async def dial():
        order.append("dial")
        return True

    session = SimpleNamespace(start=start)
    runtime = CallRuntime.__new__(CallRuntime)
    runtime.ctx = SimpleNamespace(room=SimpleNamespace(name="room"))
    runtime.snapshot = {
        "tenant": {"id": "tenant"},
        "config": {"business_name": "Example", "timezone": "UTC"},
        "outgoing": {"opening_phrase": "Hello"},
    }
    runtime.direction = "outgoing"
    runtime.dispatch = {"mode": "livekit_first"}
    runtime.options = VoiceOptions()
    runtime.participant_identity = "sip-target"
    runtime.backend = object()
    runtime.log = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime.session = session
    runtime.dial = dial
    runtime.speak = AsyncMock()
    await runtime.start()
    assert order == ["session_start", "dial"]
    runtime.speak.assert_awaited_once_with("Hello", interruptible=True)


@pytest.mark.asyncio
async def test_final_transcript_retries_identical_event_and_keeps_diagnostic_id():
    runtime = CallRuntime.__new__(CallRuntime)
    runtime._finalized = False
    runtime._tasks = set()
    runtime.direction = "incoming"
    runtime.snapshot = {"tenant": {"id": "tenant"}}
    runtime.ctx = SimpleNamespace(room=SimpleNamespace(name="room"))
    runtime.participant_identity = "sip-id"
    runtime.session = SimpleNamespace(history=SimpleNamespace(messages=lambda: []))
    runtime.session.llm = SimpleNamespace(aclose=AsyncMock())
    runtime.log = SimpleNamespace(
        call_id="diag-uuid", emit=lambda *_args, **_kwargs: None, close=lambda: None
    )
    runtime.backend = SimpleNamespace(
        post=AsyncMock(side_effect=[TimeoutError, {"ok": True, "accepted": True}]),
        aclose=AsyncMock(),
    )
    await runtime.finalize()
    assert runtime.backend.post.await_count == 2
    payloads = [call.args[1] for call in runtime.backend.post.await_args_list]
    assert payloads[0] == payloads[1]
    assert payloads[0]["agent_call_id"] == "diag-uuid"
    runtime.backend.aclose.assert_awaited_once()
