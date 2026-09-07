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


def test_holiday_simulation_has_consent_and_test_only_rules_without_meeting_context(
    config,
):
    config["config"]["extra_settings"]["booking_mode"] = "simulation"
    config["config"].update(
        owner_name="Old business owner",
        owner_email="old-business@example.com",
        meeting_duration_minutes=45,
    )
    prompt = instructions(config, "incoming")
    assert "SIMULATED HOLIDAY RESERVATIONS" in prompt
    assert "caller agrees" in prompt
    assert "test reservation" in prompt
    assert "check-in and check-out dates" in prompt
    assert "never claim" in prompt
    assert "calendar, CRM, payment, email or message" in " ".join(prompt.split())
    assert "Use only this business's configured facts" in prompt
    assert "If a tool is unavailable" in prompt
    assert "Use call_end" in prompt
    assert "book_meeting returns" not in prompt
    assert "Default meeting duration" not in prompt
    assert "Old business owner" not in prompt
    assert "old-business@example.com" not in prompt


def test_verbal_reservation_has_no_demo_language_or_calendar_context(config):
    config["config"]["extra_settings"]["booking_mode"] = "verbal_reservation"
    config["config"].update(owner_name="Legacy owner", owner_email="old@example.com")
    agent, _, _ = make_agent(config)
    prompt = instructions(config, "incoming")
    assert "VERBAL HOLIDAY RESERVATIONS" in prompt
    assert "caller agrees" in prompt
    assert "WAIT" in prompt
    assert "no tool is needed" in prompt
    assert "calendar, CRM, payment, email or message" in " ".join(prompt.split())
    assert "book_meeting returns" not in prompt
    assert "Default meeting duration" not in prompt
    assert "old@example.com" not in prompt
    model_context = (
        prompt + " " + " ".join(tool.info.description for tool in agent.tools)
    ).lower()
    for forbidden in (
        "demo",
        "simulation",
        "fictional",
        "test reservation",
        "test stay",
        "di prova",
    ):
        assert forbidden not in model_context


@pytest.mark.asyncio
async def test_verbal_reservation_cannot_invoke_external_booking(config):
    config["config"]["extra_settings"]["booking_mode"] = "verbal_reservation"
    agent, backend, _ = make_agent(config)
    assert {tool.info.name for tool in agent.tools} == {"call_end"}
    assert "does NOT confirm or save a reservation" in agent.tools[0].info.description
    assert (await agent.check_meeting_slot("2026-10-01T10:00:00+02:00"))[
        "booked"
    ] is False
    assert (await agent.book_meeting("2026-10-01T10:00:00+02:00"))["booked"] is False
    backend.post.assert_not_awaited()


@pytest.mark.parametrize(
    ("direction", "mode"),
    [
        ("incoming", None),
        ("incoming", "SIMULATION"),
        ("outgoing", "simulation"),
        ("outgoing", "verbal_reservation"),
    ],
)
def test_real_booking_rules_remain_default_and_outgoing_never_uses_demo(
    config, direction, mode
):
    config["config"]["extra_settings"]["booking_mode"] = mode
    prompt = instructions(config, direction)
    assert "book_meeting returns booked=true with an event_id" in prompt
    assert (
        "Before booking, obtain the caller's agreement to an exact date, time and timezone"
        in prompt
    )
    assert "SIMULATED HOLIDAY RESERVATIONS" not in prompt


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
async def test_simulation_blocks_calendar_even_with_stale_enabled_flags(config):
    config["config"]["extra_settings"]["booking_mode"] = "simulation"
    agent, backend, _ = make_agent(config)
    assert {tool.info.name for tool in agent.tools} == {"call_end"}
    assert "does NOT confirm or save a reservation" in agent.tools[0].info.description
    availability = await agent.check_meeting_slot("2026-10-01T10:00:00+02:00")
    booking = await agent.book_meeting("2026-10-01T10:00:00+02:00")
    assert availability == {"status": "unavailable", "booked": False}
    assert booking == {"status": "unavailable", "booked": False}
    backend.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_simulated_reservation_details_use_existing_internal_call_record(config):
    config["config"]["extra_settings"]["booking_mode"] = "simulation"
    agent, backend, finish = make_agent(config)
    notes = (
        "Test reservation: Casa Demo, 10-15 October 2026, two guests; caller agreed."
    )
    result = await agent.call_end(
        call_type="general", topic="Test reservation", notes=notes
    )
    backend.post.assert_awaited_once()
    assert backend.post.await_args.args[0] == "/events/call-end"
    assert backend.post.await_args.args[1]["notes"] == notes
    assert result["details_saved"] is True
    finish.assert_awaited_once()


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
