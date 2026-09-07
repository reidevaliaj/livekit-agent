"""Offline behavior and failure-path tests; never connect to an AI provider."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from backend_client import BackendClient, validate_session_config
from receptionist import Receptionist, booking_key, instructions
from runtime_config import REALTIME_MODEL, voice_options


@pytest.fixture
def config():
    return {
        "ok": True,
        "tenant": {"id": "tenant-1", "slug": "example"},
        "config": {
            "business_name": "Example",
            "assistant_language": "it",
            "timezone": "Europe/Rome",
            "enabled_tools": {"calendar_lookup": True, "meeting_creation": True},
            "extra_settings": {"openai_realtime_model": "obsolete-model"},
        },
        "outgoing": {"openai_realtime_voice": "cedar"},
        "call": {"id": "call-1"},
    }


def test_config_failure_never_uses_another_tenant():
    for response in ({"ok": False}, {"ok": True, "config": {}}):
        with pytest.raises(ValueError):
            validate_session_config(response)


def test_only_requested_realtime_model_and_outgoing_voice(config):
    options = voice_options(config, "outgoing")
    assert options.model == REALTIME_MODEL == "gpt-realtime-2.1-mini"
    assert options.voice == "cedar"
    assert options.language == "it"


def test_known_legacy_booking_instruction_is_removed(config):
    legacy = "- Only say a meeting is booked or confirmed when check_meeting_slot returns that the slot is available."
    config["config"]["prompt_appendix"] = (
        legacy + "\nBusiness custom rule: collect the caller name."
    )
    prompt = instructions(config, "incoming")
    assert legacy not in prompt
    assert "Business custom rule: collect the caller name." in prompt
    assert "book_meeting returns booked=true with an event_id" in prompt


@pytest.mark.asyncio
async def test_backend_auth_and_connection_reuse():
    seen = []

    async def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    async with BackendClient(
        "http://127.0.0.1:8000", "unit-token", transport=httpx.MockTransport(handle)
    ) as client:
        identity = id(client.http)
        await client.post("/agent/session-config", {})
        await client.post("/agent/session-config", {})
        assert id(client.http) == identity
    assert len(seen) == 2
    assert all(r.headers["X-Agent-Token"] == "unit-token" for r in seen)
    assert all("X-Internal-API-Key" not in r.headers for r in seen)


@pytest.mark.asyncio
async def test_backend_whole_operation_has_deadline():
    async def slow(_request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={"ok": True})

    async with BackendClient(
        "http://127.0.0.1:8000", "unit-token", transport=httpx.MockTransport(slow)
    ) as client:
        with pytest.raises(TimeoutError):
            await client.post("/agent/session-config", {}, budget=0.01)


def make_agent(config, result=None):
    backend = SimpleNamespace(post=AsyncMock(return_value=result or {"ok": True}))
    diagnostics = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    finish = AsyncMock()
    agent = Receptionist(config, "incoming", "room-1", backend, diagnostics, finish)
    return agent, backend, finish


@pytest.mark.asyncio
async def test_availability_does_not_confirm_booking(config):
    agent, backend, _ = make_agent(config, {"ok": True, "status": "free"})
    result = await agent.check_meeting_slot("2026-10-01T10:00:00+02:00", 30)
    assert result["status"] == "available"
    assert result["booked"] is False
    assert backend.post.await_args.args[1]["room_name"] == "room-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "unknown", "unavailable", "busy"])
async def test_uncertain_booking_never_confirms(config, status):
    agent, _, _ = make_agent(config, {"ok": True, "status": status})
    result = await agent.book_meeting("2026-10-01T10:00:00+02:00", 30)
    assert result["booked"] is False


@pytest.mark.asyncio
async def test_confirmed_booking_needs_event_and_retries_same_key(config):
    agent, backend, _ = make_agent(
        config, {"ok": True, "status": "booked", "event_id": "event-1"}
    )
    first = await agent.book_meeting("2026-10-01T10:00:00+02:00", 30)
    second = await agent.book_meeting("2026-10-01T08:00:00Z", 30)
    assert first["booked"] and second["booked"]
    payloads = [call.args[1] for call in backend.post.await_args_list]
    assert payloads[0]["idempotency_key"] == payloads[1]["idempotency_key"]
    assert (
        booking_key("tenant-1", "room-2", "2026-10-01T08:00:00Z", 30)
        != payloads[0]["idempotency_key"]
    )


@pytest.mark.asyncio
async def test_disabled_calendar_rejected_even_if_function_is_called(config):
    config["config"]["enabled_tools"] = {
        "calendar_lookup": False,
        "meeting_creation": False,
    }
    agent, backend, _ = make_agent(config)
    result = await agent.book_meeting("2026-10-01T10:00:00+02:00", 30)
    assert result["booked"] is False
    backend.post.assert_not_awaited()
    assert all(
        tool.info.name not in {"book_meeting", "check_meeting_slot"}
        for tool in agent.tools
    )


@pytest.mark.asyncio
async def test_end_still_hangs_up_after_backend_failure_and_only_once(config):
    agent, backend, finish = make_agent(config)
    backend.post.side_effect = TimeoutError
    result = await agent.call_end(call_type="general", notes="private customer detail")
    await agent.call_end(call_type="general")
    assert result["details_saved"] is False
    finish.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_errors_do_not_leak_exception_secrets(config):
    agent, backend, _ = make_agent(config)
    backend.post.side_effect = RuntimeError("secret-provider-token")
    result = await agent.check_meeting_slot("2026-10-01T10:00:00+02:00")
    assert result["status"] == "unknown"
    assert "secret-provider-token" not in json.dumps(result)
