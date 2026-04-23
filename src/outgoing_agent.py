import asyncio
import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    room_io,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from call_debug import CallDebugLogger

logger = logging.getLogger("outgoing-agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "https://voice.code-studio.eu")
INTERNAL_API_KEY = (os.getenv("INTERNAL_API_KEY") or "").strip()
DEFAULT_BUSINESS_TIMEZONE = os.getenv("BUSINESS_TIMEZONE", "Europe/Budapest")
DEFAULT_LLM_MODEL = (os.getenv("LLM_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
DEFAULT_TTS_VOICE = (
    os.getenv("DEFAULT_TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    or "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
).strip()
DEFAULT_TTS_SPEED = float((os.getenv("DEFAULT_TTS_SPEED", "1.0") or "1.0").strip() or "1.0")
INTERRUPTION_MODE = (os.getenv("INTERRUPTION_MODE", "adaptive") or "adaptive").strip().lower()
INTERRUPTION_MIN_DURATION = float((os.getenv("INTERRUPTION_MIN_DURATION", "0.5") or "0.5").strip() or "0.5")
FALSE_INTERRUPTION_TIMEOUT_RAW = (os.getenv("FALSE_INTERRUPTION_TIMEOUT", "2.0") or "2.0").strip()
RESUME_FALSE_INTERRUPTION = os.getenv("RESUME_FALSE_INTERRUPTION", "true").strip().lower() == "true"
AGENT_NUM_IDLE_PROCESSES = int(os.getenv("AGENT_NUM_IDLE_PROCESSES", "1").strip() or "1")
AGENT_LOAD_THRESHOLD = float(os.getenv("AGENT_LOAD_THRESHOLD", "0.95").strip() or "0.95")
OUTGOING_AGENT_NAME = (os.getenv("OUTGOING_AGENT_NAME", "outgoing-agent") or "outgoing-agent").strip()
OUTGOING_AGENT_DEBUG_LOG_PATH = (
    os.getenv("OUTGOING_AGENT_DEBUG_LOG_PATH")
    or str(Path(__file__).resolve().parent.parent / "runtime" / "outgoing_call_debug.log")
)

PLATFORM_RULES = """
Rules:
- Speak in a warm, business-professional tone.
- Keep responses short and phone-friendly.
- If the callee says something unclear, ask for clarification kindly.
- Never invent pricing, timelines, or promises that are not in the tenant configuration.
- Ask only the next useful question.
""".strip()

OUTGOING_PLATFORM_RULES = """
Rules:
- This is an outbound call initiated by the business.
- Open with the configured opening phrase once the callee answers.
- Keep replies warm, concise, and conversational.
- If the callee says they are busy, not interested, or wants to stop, be polite and call finish_call.
- If the callee asks what the business does, answer only using the tenant's configured services and notes.
- Never invent offers, availability, or commitments not present in the tenant context.
- If the conversation is clearly complete, call finish_call politely.
""".strip()

LANGUAGE_LABELS = {
    "en": "English",
    "it": "Italian",
    "de": "German",
}

FAREWELL_BY_LANGUAGE = {
    "en": "Thank you for your time. Goodbye.",
    "it": "Grazie per il suo tempo. Arrivederci.",
    "de": "Vielen Dank fuer Ihre Zeit. Auf Wiederhoeren.",
}

SUPPORTED_STT_LANGUAGES = {"en", "it", "de", "multi"}
SUPPORTED_INTERRUPTION_MODES = {"adaptive", "vad"}


def _normalize_tts_speed(value: Any) -> float:
    try:
        speed = float(value if value not in (None, "") else DEFAULT_TTS_SPEED)
    except (TypeError, ValueError):
        speed = DEFAULT_TTS_SPEED
    return min(1.5, max(0.6, speed))


def _normalize_interruption_mode(value: Any) -> str:
    candidate = str(value or INTERRUPTION_MODE or "adaptive").strip().lower()
    return candidate if candidate in SUPPORTED_INTERRUPTION_MODES else "adaptive"


def _normalize_interruption_min_duration(value: Any) -> float:
    try:
        duration = float(value if value not in (None, "") else INTERRUPTION_MIN_DURATION)
    except (TypeError, ValueError):
        duration = INTERRUPTION_MIN_DURATION
    return min(3.0, max(0.05, duration))


def _normalize_false_interruption_timeout(value: Any) -> float | None:
    if value is None:
        value = FALSE_INTERRUPTION_TIMEOUT_RAW
    text = str(value).strip()
    if not text:
        return 2.0
    if text.lower() in {"none", "off", "disabled"}:
        return None
    try:
        timeout = float(text)
    except (TypeError, ValueError):
        timeout = 2.0
    return min(10.0, max(0.1, timeout))


def _normalize_stt_language(value: Any, assistant_language: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in SUPPORTED_STT_LANGUAGES:
        return candidate
    fallback = str(assistant_language or "en").strip().lower()
    return fallback if fallback in SUPPORTED_STT_LANGUAGES else "en"


def _normalize_endpointing_window(min_value: Any, max_value: Any) -> tuple[float, float]:
    def _coerce(value: Any, default: float) -> float:
        try:
            delay = float(value if value not in (None, "") else default)
        except (TypeError, ValueError):
            delay = default
        return min(6.0, max(0.1, delay))

    minimum = _coerce(min_value, 0.3)
    maximum = _coerce(max_value, 1.2)
    if maximum < minimum:
        maximum = minimum
    return minimum, maximum


def _supports_turn_handling() -> bool:
    try:
        return "turn_handling" in inspect.signature(AgentSession).parameters
    except Exception:
        return False


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
        return " ".join(parts).strip()
    return str(content)


def _event_text_payload(value: Any) -> str:
    if value is None:
        return ""
    transcript = getattr(value, "transcript", None)
    if isinstance(transcript, str):
        return transcript.strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    if hasattr(value, "content"):
        return _flatten_message_content(getattr(value, "content"))
    return _flatten_message_content(value)


def _history_messages(session: AgentSession) -> list[Any]:
    messages = session.history.messages() if callable(getattr(session.history, "messages", None)) else session.history.messages
    return list(messages or [])


def _normalize_attr_key(value: str) -> str:
    return value.lower().replace("-", "_").replace(".", "_")


def _participant_context(participant: Optional[rtc.RemoteParticipant]) -> dict[str, str]:
    if participant is None:
        return {}
    attrs: dict[str, str] = {}
    try:
        attrs.update({str(key): str(value) for key, value in dict(participant.attributes).items()})
    except Exception:
        pass
    metadata = getattr(participant, "metadata", "") or ""
    if metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                attrs.update({str(key): str(value) for key, value in parsed.items()})
        except Exception:
            pass
    return attrs


def _lookup_attr(attrs: dict[str, str], *candidates: str) -> str:
    normalized = {_normalize_attr_key(key): value for key, value in attrs.items()}
    for candidate in candidates:
        key = _normalize_attr_key(candidate)
        if key in normalized and normalized[key]:
            return normalized[key]
    return ""


def _best_effort_caller_id(room: rtc.Room) -> Optional[str]:
    try:
        for participant in room.remote_participants.values():
            if "sip" in (participant.identity or "").lower():
                return participant.identity
        for participant in room.remote_participants.values():
            return participant.identity
    except Exception:
        pass
    return None


async def _wait_for_remote_participant(room: rtc.Room, timeout_sec: float = 4.0) -> Optional[rtc.RemoteParticipant]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            for participant in room.remote_participants.values():
                if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                    return participant
            for participant in room.remote_participants.values():
                return participant
        except Exception:
            pass
        await asyncio.sleep(0.1)
    return None


async def _post_json(path: str, payload: Dict[str, Any]) -> None:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    timeout = httpx.Timeout(12.0, connect=10.0)
    headers = {"X-Internal-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        logger.info("[HTTP_POST] response path=%s status=%s", path, response.status_code)
        response.raise_for_status()


async def _post_json_and_read(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    timeout = httpx.Timeout(12.0, connect=10.0)
    headers = {"X-Internal-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        logger.info("[HTTP_POST] response path=%s status=%s", path, response.status_code)
        response.raise_for_status()
        return response.json()


def _build_transcript_payload(
    session: AgentSession,
    room: rtc.Room,
    shutdown_reason: str,
    *,
    tenant_id: str,
    tenant_slug: str,
    outgoing_call_id: str,
    call_sid: str,
) -> Dict[str, Any]:
    messages: list[Dict[str, Any]] = []
    lines: list[str] = []
    for msg in _history_messages(session):
        if msg.role not in ("user", "assistant"):
            continue
        text = _flatten_message_content(msg.content).strip()
        if not text:
            continue
        lines.append(f"{msg.role}: {text}")
        messages.append(
            {
                "role": msg.role,
                "text": text,
                "interrupted": bool(getattr(msg, "interrupted", False)),
                "created_at": getattr(msg, "created_at", None),
            }
        )
    return {
        "tenant_id": tenant_id,
        "tenant_slug": tenant_slug,
        "outgoing_call_id": outgoing_call_id,
        "call_sid": call_sid,
        "room_name": room.name if room else None,
        "shutdown_reason": shutdown_reason,
        "timestamp": int(time.time()),
        "transcript": "\n".join(lines),
        "messages": messages,
    }


async def _fetch_outgoing_session_config(ctx: JobContext) -> dict[str, Any]:
    participant = await _wait_for_remote_participant(ctx.room)
    attrs = _participant_context(participant)
    payload = {
        "tenant_id": _lookup_attr(attrs, "tenant_id", "x_tenant_id", "x-tenant-id"),
        "tenant_slug": _lookup_attr(attrs, "tenant_slug", "x_tenant_slug", "x-tenant-slug"),
        "outgoing_call_id": _lookup_attr(attrs, "outgoing_call_id", "x_outgoing_call_id", "x-outgoing-call-id"),
        "room_name": ctx.room.name,
        "call_sid": _lookup_attr(attrs, "parent_call_sid", "x_parent_call_sid", "x-parent-call-sid"),
    }
    response = await _post_json_and_read("/agent/outgoing-session-config", payload)
    if not response.get("ok"):
        raise RuntimeError("Outgoing session config fetch failed")
    return response


class OutgoingAssistant(Agent):
    def __init__(self, *, session_config: dict[str, Any], debug_logger: CallDebugLogger):
        self._session_config = session_config
        self._tenant = session_config["tenant"]
        self._config = session_config["config"]
        self._outgoing = session_config["outgoing"]
        self._call = session_config["call"]
        self._debug = debug_logger
        self._assistant_language = str(self._config.get("assistant_language") or "en")
        self._business_name = str(self._config.get("business_name") or self._tenant.get("display_name") or "the business")
        self._call_end_in_progress = False
        super().__init__(instructions=self._build_instructions())

    def _build_instructions(self) -> str:
        services = self._config.get("services") or []
        service_text = "\n".join(f"- {item}" for item in services if str(item).strip())
        faq_notes = str(self._config.get("faq_notes") or "").strip()
        tenant_prompt = str(self._config.get("tenant_prompt") or "").strip()
        outgoing_prompt = str(self._outgoing.get("system_prompt") or "").strip()
        target_name = str(self._call.get("target_name") or "").strip()
        target_number = str(self._call.get("target_number") or "").strip()
        language_label = LANGUAGE_LABELS.get(self._assistant_language, self._assistant_language)
        return "\n\n".join(
            [
                PLATFORM_RULES,
                OUTGOING_PLATFORM_RULES,
                f"Speak only in {language_label} unless the callee clearly switches language.",
                f"Business: {self._business_name}",
                f"Callee name: {target_name or 'unknown'}",
                f"Callee number: {target_number}",
                f"Tenant prompt:\n{tenant_prompt or '(none)'}",
                f"Outgoing prompt:\n{outgoing_prompt or '(none)'}",
                f"Services:\n{service_text or '(none)'}",
                f"FAQ / Notes:\n{faq_notes or '(none)'}",
                f"Opening phrase (already spoken at call start): {self._outgoing.get('opening_phrase') or ''}",
            ]
        )

    def _farewell_text(self) -> str:
        template = FAREWELL_BY_LANGUAGE.get(self._assistant_language, FAREWELL_BY_LANGUAGE["en"])
        return template.format(business_name=self._business_name)

    @function_tool
    async def finish_call(self, context: RunContext, notes: str = "") -> str:
        if self._call_end_in_progress:
            return "Ending the call now."
        self._call_end_in_progress = True
        self._debug.log("tool", "finish_call.start", notes=notes)
        try:
            ctx = get_job_context()
            session_obj = getattr(context, "session", None)
            if session_obj is not None:
                try:
                    await asyncio.wait_for(session_obj.say(self._farewell_text()), timeout=6.0)
                except Exception:
                    logger.exception("[OUTGOING_CALL_END] farewell failed")
            await ctx.room.disconnect()
            return "Ending the call now."
        finally:
            self._call_end_in_progress = False


try:
    server = AgentServer(num_idle_processes=AGENT_NUM_IDLE_PROCESSES, load_threshold=AGENT_LOAD_THRESHOLD)
    logger.info("[SERVER] configured num_idle_processes=%s load_threshold=%.2f", AGENT_NUM_IDLE_PROCESSES, AGENT_LOAD_THRESHOLD)
except TypeError:
    logger.warning("[SERVER] AgentServer() does not accept num_idle_processes/load_threshold on this SDK; using defaults")
    server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=OUTGOING_AGENT_NAME)
async def outgoing_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("[OUTGOING_CALL_START] room=%s", ctx.room.name)
    debug_logger = CallDebugLogger(path=Path(OUTGOING_AGENT_DEBUG_LOG_PATH))

    await ctx.connect()
    session_config = await _fetch_outgoing_session_config(ctx)
    tenant = session_config["tenant"]
    config = session_config["config"]
    outgoing = session_config["outgoing"]
    call = session_config["call"]
    tenant_id = str(tenant.get("id") or "")
    tenant_slug = str(tenant.get("slug") or "")
    call_sid = str(call.get("telnyx_call_control_id") or "")
    outgoing_call_id = str(call.get("id") or "")

    business_timezone = str(config.get("timezone") or DEFAULT_BUSINESS_TIMEZONE)
    llm_model = str(config.get("llm_model") or DEFAULT_LLM_MODEL)
    tts_voice = str(config.get("tts_voice") or DEFAULT_TTS_VOICE)
    tts_speed = _normalize_tts_speed(config.get("tts_speed"))
    assistant_language = str(config.get("assistant_language") or "en")
    stt_language = _normalize_stt_language(config.get("stt_language"), assistant_language)
    min_endpointing_delay, max_endpointing_delay = _normalize_endpointing_window(
        config.get("min_endpointing_delay"),
        config.get("max_endpointing_delay"),
    )
    interruption_mode = _normalize_interruption_mode(None)
    interruption_min_duration = _normalize_interruption_min_duration(None)
    false_interruption_timeout = _normalize_false_interruption_timeout(None)
    supports_turn_handling = _supports_turn_handling()

    debug_logger.log(
        "call",
        "session_started",
        room_name=ctx.room.name,
        tenant_slug=tenant_slug,
        outgoing_call_id=outgoing_call_id,
        business_timezone=business_timezone,
        assistant_language=assistant_language,
        stt_language=stt_language,
        llm_model=llm_model,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay,
        interruption_mode=interruption_mode,
        interruption_min_duration=interruption_min_duration,
        false_interruption_timeout=false_interruption_timeout,
    )
    debug_logger.log(
        "config",
        "runtime_snapshot",
        snapshot={
            "tenant_slug": tenant_slug,
            "outgoing_call_id": outgoing_call_id,
            "config_version": config.get("version"),
            "business_name": str(config.get("business_name") or tenant.get("display_name") or ""),
            "business_timezone": business_timezone,
            "assistant_language": assistant_language,
            "stt_language": stt_language,
            "llm_model": llm_model,
            "tts_voice": tts_voice,
            "tts_speed": tts_speed,
            "turn_detection_model": "MultilingualModel",
            "supports_turn_handling": supports_turn_handling,
            "min_endpointing_delay": min_endpointing_delay,
            "max_endpointing_delay": max_endpointing_delay,
            "interruption_mode": interruption_mode,
            "interruption_min_duration": interruption_min_duration,
            "false_interruption_timeout": false_interruption_timeout,
            "opening_phrase": outgoing.get("opening_phrase"),
            "target_name": call.get("target_name"),
            "target_number": call.get("target_number"),
        },
    )

    session_kwargs: Dict[str, Any] = {
        "stt": deepgram.STT(model="nova-3", language=stt_language),
        "llm": openai.LLM(model=llm_model),
        "tts": cartesia.TTS(model="sonic-3", voice=tts_voice, language=assistant_language, speed=tts_speed),
        "vad": ctx.proc.userdata["vad"],
    }
    if supports_turn_handling:
        session_kwargs["turn_handling"] = {
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": min_endpointing_delay, "max_delay": max_endpointing_delay},
            "interruption": {
                "mode": interruption_mode,
                "min_duration": interruption_min_duration,
                "resume_false_interruption": RESUME_FALSE_INTERRUPTION,
                "false_interruption_timeout": false_interruption_timeout,
            },
            "preemptive_generation": {"enabled": True},
        }
    else:
        session_kwargs.update(
            turn_detection=MultilingualModel(),
            min_endpointing_delay=min_endpointing_delay,
            max_endpointing_delay=max_endpointing_delay,
            min_interruption_duration=interruption_min_duration,
            false_interruption_timeout=false_interruption_timeout,
            resume_false_interruption=RESUME_FALSE_INTERRUPTION,
            preemptive_generation=True,
        )

    session = AgentSession(**session_kwargs)
    await session.start(
        agent=OutgoingAssistant(session_config=session_config, debug_logger=debug_logger),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        old_state = str(getattr(ev, "old_state", ""))
        new_state = str(getattr(ev, "new_state", ""))
        debug_logger.log("turn", "user_state_changed", old_state=old_state, new_state=new_state)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: Any) -> None:
        debug_logger.log("agent", "agent_state_changed", old_state=str(getattr(ev, "old_state", "")), new_state=str(getattr(ev, "new_state", "")))

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        text = _event_text_payload(ev)
        if text:
            debug_logger.log("transcript", "USER_FINAL" if bool(getattr(ev, "is_final", False)) else "USER_PARTIAL", text=text)

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: Any) -> None:
        item = getattr(ev, "item", None)
        role = str(getattr(item, "role", ""))
        text = _event_text_payload(item).strip()
        if role in ("user", "assistant") and text:
            debug_logger.log("transcript", "USER_COMMITTED" if role == "user" else "ASSISTANT_COMMITTED", text=text)

    @session.on("function_tools_executed")
    def _on_function_tools_executed(ev: Any) -> None:
        zipped = getattr(ev, "zipped", None)
        if callable(zipped):
            for idx, pair in enumerate(zipped(), start=1):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    debug_logger.log("tool", "function_tools_executed", index=idx, payload=pair)
                    continue
                function_call, function_output = pair
                debug_logger.log(
                    "tool",
                    "TOOL_EXECUTED",
                    index=idx,
                    name=str(getattr(function_call, "name", "")),
                    arguments=str(getattr(function_call, "arguments", "")),
                    output=str(getattr(function_output, "output", "")),
                    is_error=bool(getattr(function_output, "is_error", False)),
                )

    async def _send_transcript_on_shutdown(reason: str) -> None:
        debug_logger.log("call", "shutdown_started", room_name=ctx.room.name, reason=reason)
        try:
            payload = _build_transcript_payload(
                session=session,
                room=ctx.room,
                shutdown_reason=reason,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
                outgoing_call_id=outgoing_call_id,
                call_sid=call_sid,
            )
            await _post_json("/outgoing/events/transcript", payload)
        except Exception:
            logger.exception("[OUTGOING_CALL_END] transcript send failed room=%s reason=%s", ctx.room.name, reason)
        finally:
            debug_logger.log("call", "shutdown_finished", room_name=ctx.room.name, reason=reason)
            debug_logger.close(cleanup=True)

    ctx.add_shutdown_callback(_send_transcript_on_shutdown)

    opening_phrase = str(outgoing.get("opening_phrase") or "").strip()
    if opening_phrase:
        await session.say(opening_phrase)


if __name__ == "__main__":
    cli.run_app(server)
