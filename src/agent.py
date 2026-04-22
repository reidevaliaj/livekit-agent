import asyncio
import inspect
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

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

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "https://voice.code-studio.eu")
INTERNAL_API_KEY = (os.getenv("INTERNAL_API_KEY") or "").strip()
DEFAULT_TENANT_ID = os.getenv("TENANT_ID", "codestudio")
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
ENABLE_LLM_WARMUP = os.getenv("ENABLE_LLM_WARMUP", "false").strip().lower() == "true"
LLM_WARMUP_TIMEOUT_SEC = float(os.getenv("LLM_WARMUP_TIMEOUT_SEC", "3.5").strip() or "3.5")
LLM_WARMUP_MODEL = (os.getenv("LLM_WARMUP_MODEL", "gpt-4.1-nano").strip() or "gpt-4.1-nano")

PLATFORM_RULES = """
Rules:
- Speak in a warm, business-professional tone.
- Keep responses short and phone-friendly.
- If something in the sentence you recive does not make sense or seams not correct ask again kindly.
- Never offer prices or promise timelines unless the tenant context explicitly says to.
- Ask only the next needed question.
- Collect name.
- Use check_meeting_slot before confirming any meeting inside the booking horizon.
- Only say a meeting is booked or confirmed when check_meeting_slot returns that the slot is available.
- If check_meeting_slot says the slot is unavailable, busy, outside hours, or live calendar access is unavailable, clearly say the meeting is not booked yet.
- Never invent availability or time slots.
- If the caller says there is nothing else they need, or says goodbye/thanks after the conversation is complete, call call_end immediately.
- If you have enough information, ask if the caller needs anything else. If not, end politely and call call_end.
- For unrelated or persistent vendor calls, politely decline and end the call if they continue.
""".strip()

LANGUAGE_LABELS = {
    "en": "English",
    "it": "Italian",
    "de": "German",
}

FAREWELL_BY_LANGUAGE = {
    "en": "Thank you for calling {business_name}. Goodbye.",
    "it": "Grazie per aver chiamato {business_name}. Arrivederci.",
    "de": "Vielen Dank fuer Ihren Anruf bei {business_name}. Auf Wiedersehen.",
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


def _build_debug_runtime_snapshot(
    *,
    config: dict[str, Any],
    tenant: dict[str, Any],
    business_timezone: str,
    assistant_language: str,
    stt_language: str,
    llm_model: str,
    tts_voice: str,
    tts_speed: float,
    min_endpointing_delay: float,
    max_endpointing_delay: float,
    interruption_mode: str,
    interruption_min_duration: float,
    false_interruption_timeout: float | None,
    supports_turn_handling: bool,
) -> dict[str, Any]:
    return {
        "tenant_slug": tenant.get("slug"),
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
        "resume_false_interruption": RESUME_FALSE_INTERRUPTION,
        "preemptive_generation": True,
        "meeting_duration_minutes": int(config.get("meeting_duration_minutes") or 30),
        "booking_horizon_days": int(config.get("booking_horizon_days") or 14),
        "enabled_tools": dict(config.get("enabled_tools") or {}),
        "tenant_prompt_chars": len(str(config.get("tenant_prompt") or "")),
        "prompt_appendix_chars": len(str(config.get("prompt_appendix") or "")),
        "services_count": len(config.get("services") or []),
    }


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


async def _post_json(path: str, payload: Dict[str, Any]) -> None:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    timeout = httpx.Timeout(12.0, connect=10.0)
    headers = {"X-Internal-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else None
    logger.info("[HTTP_POST] sending path=%s url=%s payload_keys=%s", path, url, sorted(payload.keys()))
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
                transcript_value = item.get("transcript")
                if isinstance(transcript_value, str):
                    parts.append(transcript_value)
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
                continue
            transcript_attr = getattr(item, "transcript", None)
            if isinstance(transcript_attr, str):
                parts.append(transcript_attr)
                continue
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


async def _warmup_llm_once() -> None:
    if not ENABLE_LLM_WARMUP:
        return
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("[WARMUP] skipped: OPENAI_API_KEY not set")
        return

    payload = {
        "model": LLM_WARMUP_MODEL,
        "messages": [{"role": "system", "content": "Respond with exactly: ok"}],
        "max_completion_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(LLM_WARMUP_TIMEOUT_SEC, connect=2.0)
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        logger.info("[WARMUP] LLM warmup done model=%s elapsed=%.3fs", LLM_WARMUP_MODEL, time.monotonic() - started)
    except Exception:
        logger.exception("[WARMUP] LLM warmup failed model=%s timeout=%.2fs", LLM_WARMUP_MODEL, LLM_WARMUP_TIMEOUT_SEC)


def _history_messages(session: AgentSession) -> list[Any]:
    messages = session.history.messages() if callable(getattr(session.history, "messages", None)) else session.history.messages
    return list(messages or [])


def _build_transcript_payload(session: AgentSession, room: rtc.Room, shutdown_reason: str, tenant_id: str) -> Dict[str, Any]:
    messages: list[Dict[str, Any]] = []
    lines: list[str] = []
    for msg in _history_messages(session):
        if msg.role not in ("user", "assistant"):
            continue
        text = _flatten_message_content(msg.content).strip()
        if not text:
            continue
        lines.append(f"{msg.role}: {text}")
        messages.append({"role": msg.role, "text": text, "interrupted": bool(getattr(msg, "interrupted", False)), "created_at": getattr(msg, "created_at", None)})
    return {"tenant_id": tenant_id, "room_name": room.name if room else None, "caller_id": _best_effort_caller_id(room) if room else None, "shutdown_reason": shutdown_reason, "timestamp": int(time.time()), "transcript": "\n".join(lines), "messages": messages}


def _format_slot_for_voice(slot: Dict[str, Any]) -> str:
    start_raw = str(slot.get("start", ""))
    end_raw = str(slot.get("end", ""))
    if not start_raw or not end_raw:
        return ""
    try:
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
    except ValueError:
        return ""
    return f"{start.strftime('%A %d %B at %H:%M')} to {end.strftime('%H:%M')} ({start.tzname() or DEFAULT_BUSINESS_TIMEZONE})"


def _format_day_blocks_for_voice(day_blocks: list[Dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in day_blocks[:3]:
        day = str(block.get("day", "")).strip()
        ranges = block.get("ranges", [])
        if not day or not isinstance(ranges, list) or not ranges:
            continue
        lines.append(f"{day}: {', '.join(str(value) for value in ranges[:3])}")
    return " ; ".join(lines)


def _normalize_attr_key(value: str) -> str:
    return value.lower().replace("-", "_").replace(".", "_")


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


def _fallback_session_config(room_name: str, caller_id: str) -> dict[str, Any]:
    return {
        "tenant": {"id": DEFAULT_TENANT_ID, "slug": DEFAULT_TENANT_ID, "display_name": "Code Studio", "status": "active", "notes": "Legacy fallback session config"},
        "config": {"version": 1, "business_name": "Code Studio", "assistant_language": "en", "assistant_language_label": "English", "stt_language": "en", "timezone": DEFAULT_BUSINESS_TIMEZONE, "greeting": "Thanks for calling Code Studio. How may we help you today?", "tenant_prompt": "You are the receptionist for Code Studio. Help callers understand the business, answer with the configured services, and collect accurate lead details.", "services": ["Web Design", "WordPress, TYPO3, Shopify", "Headless CMS", "Web applications", "AI integration and agents creation", "SEO"], "faq_notes": "", "prompt_appendix": "", "business_hours": "09:00-17:00", "business_days": "1,2,3,4,5", "meeting_duration_minutes": 30, "booking_horizon_days": 14, "enabled_tools": {"email_summary": True, "meeting_creation": True, "case_creation": True, "calendar_lookup": True, "zoom_meetings": True}, "llm_model": DEFAULT_LLM_MODEL, "min_endpointing_delay": 0.3, "max_endpointing_delay": 1.2, "tts_voice": DEFAULT_TTS_VOICE, "tts_speed": DEFAULT_TTS_SPEED, "owner_name": "Rey", "owner_email": "info@code-studio.eu", "reply_to_email": "Rej Aliaj <info@code-studio.eu>", "from_email": "Code Studio <noreply@code-studio.eu>", "notification_targets": ["info@code-studio.eu"], "extra_settings": {"meeting_owner_email": "aliajrei@gmail.com"}},
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "room_name": room_name,
        "caller_id": caller_id,
        "called_number": "",
        "call_sid": "",
    }

async def _fetch_session_config(ctx: JobContext) -> dict[str, Any]:
    participant = await _wait_for_remote_participant(ctx.room)
    attrs = _participant_context(participant)
    caller_id = _best_effort_caller_id(ctx.room) or ""
    payload = {
        "tenant_id": _lookup_attr(attrs, "tenant_id", "x_tenant_id", "x-tenant-id"),
        "tenant_slug": _lookup_attr(attrs, "tenant_slug", "x_tenant_slug", "x-tenant-slug"),
        "config_version": int(_lookup_attr(attrs, "config_version", "x_config_version", "x-config-version") or 0) or None,
        "room_name": ctx.room.name,
        "caller_id": caller_id,
        "called_number": _lookup_attr(attrs, "called_number", "x_called_number", "x-called-number"),
        "call_sid": _lookup_attr(attrs, "parent_call_sid", "x_parent_call_sid", "x-parent-call-sid"),
    }
    try:
        response = await _post_json_and_read("/agent/session-config", payload)
        if response.get("ok"):
            return response
    except Exception:
        logger.exception("[SESSION_CONFIG] failed to fetch backend session config; using fallback")
    return _fallback_session_config(ctx.room.name, caller_id)


def _build_call_context_text(session_config: dict[str, Any]) -> str:
    config = session_config["config"]
    business_timezone = str(config.get("timezone") or DEFAULT_BUSINESS_TIMEZONE)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(business_timezone))
    horizon_end = now_local + timedelta(days=int(config.get("booking_horizon_days") or 14))
    return (
        f"Current UTC time: {now_utc.isoformat()}\n"
        f"Current business local time ({business_timezone}): {now_local.isoformat()}\n"
        f"Meeting booking horizon ends at ({business_timezone}): {horizon_end.isoformat()}"
    )


def _build_instructions(session_config: dict[str, Any], call_context_text: str) -> str:
    tenant = session_config["tenant"]
    config = session_config["config"]
    business_name = config.get("business_name") or tenant.get("display_name") or "the business"
    assistant_language = str(config.get("assistant_language") or "en").strip().lower() or "en"
    assistant_language_label = str(config.get("assistant_language_label") or LANGUAGE_LABELS.get(assistant_language, assistant_language.upper()))
    owner_name = config.get("owner_name") or "the responsible person"
    tenant_prompt = str(config.get("tenant_prompt") or "").strip()
    services = config.get("services") or []
    faq_notes = str(config.get("faq_notes") or "").strip()
    prompt_appendix = str(config.get("prompt_appendix") or "").strip()
    extra_settings = config.get("extra_settings") or {}
    meeting_owner_email = extra_settings.get("meeting_owner_email") or config.get("owner_email") or ""

    sections = [
        PLATFORM_RULES,
        f"Tenant business name: {business_name}",
        f"Tenant slug: {tenant.get('slug')}",
        f"Configured assistant language: {assistant_language_label} ({assistant_language})",
        f"Business timezone: {config.get('timezone')}",
        f"Business hours: {config.get('business_hours')} on weekdays {config.get('business_days')}",
        f"Meeting duration: {config.get('meeting_duration_minutes')} minutes",
        f"Booking horizon: {config.get('booking_horizon_days')} days",
        (
            f"You must speak in {assistant_language_label} for greetings, questions, confirmations, "
            f"meeting scheduling, and call wrap-up unless the tenant prompt explicitly requires otherwise."
        ),
        "Services offered:",
        "\n".join(f"- {service}" for service in services) if services else "- Use only the business notes provided.",
        f"Escalation owner: {owner_name}",
        f"Escalation email: {meeting_owner_email}",
    ]
    if tenant_prompt:
        sections.extend(["Mandatory tenant prompt:", tenant_prompt])
    if faq_notes:
        sections.extend(["Business notes:", faq_notes])
    if prompt_appendix:
        sections.extend(["Tenant prompt appendix:", prompt_appendix])
    sections.extend(["Call context:", call_context_text])
    return "\n\n".join(section for section in sections if section)


class Assistant(Agent):
    def __init__(self, session_config: dict[str, Any], call_context_text: str, debug_logger: Optional[CallDebugLogger] = None) -> None:
        self._call_end_in_progress = False
        self._debug_logger = debug_logger
        self._tenant_id = str(session_config["tenant"]["id"])
        self._business_name = str(session_config["config"].get("business_name") or session_config["tenant"].get("display_name") or "the business")
        self._assistant_language = str(session_config["config"].get("assistant_language") or "en").strip().lower() or "en"
        instructions = _build_instructions(session_config, call_context_text)
        super().__init__(instructions=instructions)

    def _debug_log(self, category: str, event: str, **fields: Any) -> None:
        if self._debug_logger is not None:
            self._debug_logger.log(category, event, **fields)

    def _farewell_text(self) -> str:
        template = FAREWELL_BY_LANGUAGE.get(self._assistant_language, FAREWELL_BY_LANGUAGE["en"])
        return template.format(business_name=self._business_name)

    @function_tool
    async def check_meeting_slot(self, context: RunContext, preferred_start_iso: str, duration_minutes: int = 30) -> str:
        try:
            self._debug_log("tool", "check_meeting_slot.start", preferred_start_iso=preferred_start_iso, duration_minutes=duration_minutes)
            payload = {
                "tenant_id": self._tenant_id,
                "preferred_start_iso": preferred_start_iso,
                "duration_minutes": duration_minutes,
                "alternatives_limit": 3,
            }
            data = await _post_json_and_read("/tools/check-meeting-slot", payload)
            self._debug_log("tool", "check_meeting_slot.backend_response", data=data)
            status = str(data.get("status", "")) if isinstance(data, dict) else ""
            if status == "free":
                confirmed = data.get("confirmed_slot", {}) if isinstance(data, dict) else {}
                spoken = _format_slot_for_voice(confirmed) if isinstance(confirmed, dict) else ""
                result_text = f"AVAILABLE. Confirm exactly this slot: {spoken}." if spoken else "AVAILABLE. Confirm the requested slot."
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text
            if status in ("busy", "outside_hours"):
                next_slots = data.get("next_slots", []) if isinstance(data, dict) else []
                lines: list[str] = []
                for idx, slot in enumerate(next_slots[:3], start=1):
                    spoken = _format_slot_for_voice(slot)
                    if spoken:
                        lines.append(f"{idx}) {spoken}")
                if lines:
                    result_text = ("NOT AVAILABLE. Do not confirm the requested slot. Offer only these alternatives: " + " ; ".join(lines) if status == "busy" else "OUTSIDE BUSINESS HOURS. Do not confirm the requested slot. Offer only these alternatives: " + " ; ".join(lines))
                    self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                    return result_text
                block_txt = _format_day_blocks_for_voice(data.get("day_blocks", []) if isinstance(data, dict) else [])
                if block_txt:
                    result_text = "NOT AVAILABLE. Do not confirm the requested slot. Available blocks are: " + block_txt
                    self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                    return result_text
                result_text = "NOT AVAILABLE. Do not confirm the requested slot. Ask for another time in the booking window."
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text
            if status == "outside_horizon":
                result_text = "OUTSIDE HORIZON. Do not confirm any meeting. Explain that the team will handle this request after the call."
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text
            if status == "unavailable":
                result_text = "CALENDAR UNAVAILABLE. Do not confirm or schedule any meeting. You may only note the caller's preferred time and explain that the team will confirm later."
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text
            result_text = "Please provide your preferred date and time in the booking window."
            self._debug_log("tool", "check_meeting_slot.result", result=result_text)
            return result_text
        except Exception as exc:
            logger.exception("check_meeting_slot failed")
            result_text = f"I could not check that slot right now: {exc}"
            self._debug_log("tool", "check_meeting_slot.error", error=str(exc), result=result_text)
            return result_text

    @function_tool
    async def call_end(self, context: RunContext, call_type: str, name: str = "", company: str = "", contact_email: str = "", contact_phone: str = "", topic: str = "", notes: str = "", urgency: str = "", preferred_time_window: str = "") -> str:
        if self._call_end_in_progress:
            logger.warning("[CALL_END_TOOL] call_end already in progress; ignoring duplicate request")
            self._debug_log("tool", "call_end.duplicate", call_type=call_type)
            return "Ending the call now."

        self._call_end_in_progress = True
        try:
            self._debug_log("tool", "call_end.start", call_type=call_type, name=name, company=company, contact_email=contact_email, contact_phone=contact_phone, topic=topic, notes=notes, urgency=urgency, preferred_time_window=preferred_time_window)
            ctx: JobContext = get_job_context()
            room = ctx.room
            session_obj = getattr(context, "session", None)
            transcript = ""
            if session_obj is not None:
                try:
                    history_payload = _build_transcript_payload(session_obj, room, "call_end_requested", self._tenant_id)
                    transcript = history_payload.get("transcript", "")
                except Exception:
                    logger.exception("[CALL_END_TOOL] failed to collect transcript context")

            decision = await _post_json_and_read("/tools/validate-call-end", {"tenant_id": self._tenant_id, "transcript": transcript})
            self._debug_log("tool", "call_end.validation_response", decision=decision)
            if not bool(decision.get("end_call", 0)):
                result_text = "I can keep helping. If you want to finish now, please say a clear ending like 'that's all, thank you' or 'goodbye'."
                self._debug_log("tool", "call_end.result", result=result_text)
                return result_text

            payload = {"tenant_id": self._tenant_id, "call_type": call_type, "name": name, "company": company, "contact_email": contact_email, "contact_phone": contact_phone, "topic": topic, "notes": notes, "urgency": urgency, "preferred_time_window": preferred_time_window, "room_name": room.name if room else None, "caller_id": _best_effort_caller_id(room) if room else None, "timestamp": int(time.time())}
            self._debug_log("tool", "call_end.event_payload", payload=payload)
            await _post_json("/events/call-end", payload)

            sip_identities: list[str] = []
            try:
                for participant in room.remote_participants.values():
                    identity = (participant.identity or "").strip()
                    is_sip = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP or "sip" in identity.lower()
                    if identity and is_sip:
                        sip_identities.append(identity)
                if sip_identities:
                    logger.info("[CALL_END_TOOL] sip participants queued for kick count=%s ids=%s", len(sip_identities), sip_identities)
            except Exception:
                logger.exception("[CALL_END_TOOL] failed to snapshot SIP participants")

            if session_obj is not None:
                try:
                    await asyncio.wait_for(session_obj.say(self._farewell_text()), timeout=6.0)
                    logger.info("[CALL_END_TOOL] farewell spoken before disconnect")
                except asyncio.TimeoutError:
                    logger.warning("[CALL_END_TOOL] farewell timed out; disconnecting anyway")
                except Exception:
                    logger.exception("[CALL_END_TOOL] farewell speak failed; disconnecting anyway")

            try:
                await ctx.room.disconnect()
                logger.info("[CALL_END_TOOL] room disconnected after call_end room=%s", room.name if room else None)
            except Exception:
                logger.exception("[CALL_END_TOOL] room.disconnect failed; trying shutdown")
                try:
                    ctx.shutdown(reason="call_end tool completed")
                    logger.info("[CALL_END_TOOL] ctx.shutdown called")
                except Exception:
                    logger.exception("[CALL_END_TOOL] ctx.shutdown failed")

            if sip_identities:
                room_api = getattr(getattr(ctx, "api", None), "room", None)
                remove_participant = getattr(room_api, "remove_participant", None)
                if callable(remove_participant):
                    try:
                        logger.info("[CALL_END_TOOL] remove_participant signature=%s", str(inspect.signature(remove_participant)))
                    except Exception:
                        logger.info("[CALL_END_TOOL] remove_participant signature=unavailable")
                    for identity in sip_identities:
                        kicked = False
                        room_name = room.name if room else ""
                        attempts = []
                        try:
                            from livekit.api import RoomParticipantIdentity  # type: ignore
                            attempts.append(lambda: remove_participant(RoomParticipantIdentity(room=room_name, identity=identity)))
                        except Exception:
                            pass
                        attempts.extend([
                            lambda: remove_participant(room=room_name, identity=identity),
                            lambda: remove_participant(room_name=room_name, identity=identity),
                            lambda: remove_participant(room=room_name, participant_identity=identity),
                            lambda: remove_participant(room_name=room_name, participant_identity=identity),
                        ])
                        for attempt in attempts:
                            try:
                                result = attempt()
                                if inspect.isawaitable(result):
                                    await result
                                logger.info("[CALL_END_TOOL] kicked SIP participant identity=%s", identity)
                                kicked = True
                                break
                            except TypeError:
                                continue
                            except Exception:
                                logger.exception("[CALL_END_TOOL] failed to kick SIP participant identity=%s", identity)
                                break
                        if not kicked:
                            logger.error("[CALL_END_TOOL] unable to kick SIP participant identity=%s", identity)
                else:
                    logger.warning("[CALL_END_TOOL] ctx.api.room.remove_participant unavailable; SIP kick skipped")

            result_text = "Saved. I sent the details to the team."
            self._debug_log("tool", "call_end.result", result=result_text)
            return result_text
        except Exception as exc:
            logger.exception("call_end failed")
            result_text = "I could not finalize that right now, but our team has your details."
            self._debug_log("tool", "call_end.error", error=str(exc), result=result_text)
            return result_text
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

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("[CALL_START] room=%s", ctx.room.name)
    debug_logger = CallDebugLogger()

    await ctx.connect()
    session_config = await _fetch_session_config(ctx)
    config = session_config["config"]
    tenant = session_config["tenant"]
    tenant_id = str(tenant["id"])
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
    call_context_text = _build_call_context_text(session_config)

    logger.info(
        "[SESSION_CONFIG] tenant=%s config_version=%s language=%s stt_language=%s llm_model=%s tts_speed=%s min_endpointing_delay=%.2f max_endpointing_delay=%.2f interruption_mode=%s interruption_min_duration=%.2f false_interruption_timeout=%s turn_detector=MultilingualModel preemptive_generation=%s",
        tenant.get("slug"),
        config.get("version"),
        assistant_language,
        stt_language,
        llm_model,
        tts_speed,
        min_endpointing_delay,
        max_endpointing_delay,
        interruption_mode,
        interruption_min_duration,
        false_interruption_timeout,
        True,
    )
    debug_logger.log("call", "session_started", room_name=ctx.room.name, tenant_slug=tenant.get("slug"), config_version=config.get("version"), business_timezone=business_timezone, assistant_language=assistant_language, stt_language=stt_language, llm_model=llm_model, tts_voice=tts_voice, tts_speed=tts_speed, min_endpointing_delay=min_endpointing_delay, max_endpointing_delay=max_endpointing_delay, interruption_mode=interruption_mode, interruption_min_duration=interruption_min_duration, false_interruption_timeout=false_interruption_timeout)
    debug_logger.log(
        "config",
        "runtime_snapshot",
        snapshot=_build_debug_runtime_snapshot(
            config=config,
            tenant=tenant,
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
            supports_turn_handling=supports_turn_handling,
        ),
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
            "endpointing": {
                "min_delay": min_endpointing_delay,
                "max_delay": max_endpointing_delay,
            },
            "interruption": {
                "mode": interruption_mode,
                "min_duration": interruption_min_duration,
                "resume_false_interruption": RESUME_FALSE_INTERRUPTION,
                "false_interruption_timeout": false_interruption_timeout,
            },
            "preemptive_generation": {
                "enabled": True,
            },
        }
    else:
        if interruption_mode == "adaptive":
            logger.warning("[SESSION_CONFIG] adaptive interruption requested but current SDK does not support turn_handling; falling back to legacy interruption handling")
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
        agent=Assistant(session_config=session_config, call_context_text=call_context_text, debug_logger=debug_logger),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC()),
            ),
        ),
    )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        old_state = str(getattr(ev, "old_state", ""))
        new_state = str(getattr(ev, "new_state", ""))
        debug_logger.log("turn", "user_state_changed", old_state=old_state, new_state=new_state)
        if new_state.lower().endswith("listening"):
            debug_logger.log("turn", "USER_STOPPED_SPEAKING", old_state=old_state, new_state=new_state)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: Any) -> None:
        debug_logger.log("agent", "agent_state_changed", old_state=str(getattr(ev, "old_state", "")), new_state=str(getattr(ev, "new_state", "")))

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        text = _event_text_payload(ev)
        if not text:
            return
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
                debug_logger.log("tool", "TOOL_EXECUTED", index=idx, name=str(getattr(function_call, "name", "")), arguments=str(getattr(function_call, "arguments", "")), output=str(getattr(function_output, "output", "")), is_error=bool(getattr(function_output, "is_error", False)))
            return
        debug_logger.log("tool", "TOOL_EXECUTED", payload=ev)

    async def _send_transcript_on_shutdown(reason: str) -> None:
        logger.info("[CALL_END] shutdown callback fired room=%s reason=%s", ctx.room.name, reason)
        debug_logger.log("call", "shutdown_started", room_name=ctx.room.name, reason=reason)
        try:
            payload = _build_transcript_payload(session=session, room=ctx.room, shutdown_reason=reason, tenant_id=tenant_id)
            logger.info("[CALL_END] transcript prepared room=%s messages=%s chars=%s", ctx.room.name, len(payload["messages"]), len(payload["transcript"]))
            if not payload["messages"] and not payload["transcript"]:
                logger.warning("[CALL_END] transcript payload is empty room=%s reason=%s", ctx.room.name, reason)
            await _post_json("/events/transcript", payload)
            logger.info("[CALL_END] transcript sent room=%s messages=%s", ctx.room.name, len(payload["messages"]))
        except Exception:
            logger.exception("[CALL_END] transcript send failed room=%s reason=%s", ctx.room.name, reason)
        finally:
            debug_logger.log("call", "shutdown_finished", room_name=ctx.room.name, reason=reason)
            debug_logger.close(cleanup=True)

    ctx.add_shutdown_callback(_send_transcript_on_shutdown)

    if ENABLE_LLM_WARMUP:
        asyncio.create_task(_warmup_llm_once())

    greeting = str(config.get("greeting") or f"Thanks for calling {config.get('business_name')}. How may we help you today?")
    await session.say(greeting)


if __name__ == "__main__":
    cli.run_app(server)
