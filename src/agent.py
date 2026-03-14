import os
import time
import logging
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
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
    get_job_context,
    cli,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero, deepgram, cartesia, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from call_debug import CallDebugLogger

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")


# -----------------------------
# Config
# -----------------------------
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "https://voice.code-studio.eu")
TENANT_ID = os.getenv("TENANT_ID", "codestudio")
BUSINESS_TIMEZONE = os.getenv("BUSINESS_TIMEZONE", "Europe/Budapest")
AGENT_NUM_IDLE_PROCESSES = int(os.getenv("AGENT_NUM_IDLE_PROCESSES", "1").strip() or "1")
AGENT_LOAD_THRESHOLD = float(os.getenv("AGENT_LOAD_THRESHOLD", "0.95").strip() or "0.95")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
ENABLE_LLM_WARMUP = os.getenv("ENABLE_LLM_WARMUP", "false").strip().lower() == "true"
LLM_WARMUP_TIMEOUT_SEC = float(os.getenv("LLM_WARMUP_TIMEOUT_SEC", "3.5").strip() or "3.5")
LLM_WARMUP_MODEL = os.getenv("LLM_WARMUP_MODEL", "gpt-4.1-nano").strip() or "gpt-4.1-nano"

def _best_effort_caller_id(room: rtc.Room) -> Optional[str]:
    try:
        for p in room.remote_participants.values():
            if "sip" in (p.identity or "").lower():
                return p.identity
        for p in room.remote_participants.values():
            return p.identity
    except Exception:
        pass
    return None


async def _post_json(path: str, payload: Dict[str, Any]) -> None:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    logger.info(
        "[HTTP_POST] sending path=%s url=%s payload_keys=%s",
        path,
        url,
        sorted(payload.keys()),
    )
    timeout = httpx.Timeout(12.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        logger.info("[HTTP_POST] response path=%s status=%s", path, r.status_code)
        r.raise_for_status()


async def _post_json_and_read(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    timeout = httpx.Timeout(12.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        logger.info("[HTTP_POST] response path=%s status=%s", path, r.status_code)
        r.raise_for_status()
        return r.json()


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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(LLM_WARMUP_TIMEOUT_SEC, connect=2.0)
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        logger.info(
            "[WARMUP] LLM warmup done model=%s elapsed=%.3fs",
            LLM_WARMUP_MODEL,
            time.monotonic() - started,
        )
    except Exception:
        logger.exception(
            "[WARMUP] LLM warmup failed model=%s timeout=%.2fs",
            LLM_WARMUP_MODEL,
            LLM_WARMUP_TIMEOUT_SEC,
        )


def _history_messages(session: AgentSession) -> list[Any]:
    messages = (
        session.history.messages()
        if callable(getattr(session.history, "messages", None))
        else session.history.messages
    )
    return list(messages or [])


def _build_transcript_payload(
    session: AgentSession, room: rtc.Room, shutdown_reason: str
) -> Dict[str, Any]:
    messages: list[Dict[str, Any]] = []
    lines: list[str] = []
    history_messages = _history_messages(session)
    for msg in history_messages:
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
        "tenant_id": TENANT_ID,
        "room_name": room.name if room else None,
        "caller_id": _best_effort_caller_id(room) if room else None,
        "shutdown_reason": shutdown_reason,
        "timestamp": int(time.time()),
        "transcript": "\n".join(lines),
        "messages": messages,
    }


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
    return (
        f"{start.strftime('%A %d %B at %H:%M')} to {end.strftime('%H:%M')} "
        f"({start.tzname() or BUSINESS_TIMEZONE})"
    )


def _format_day_blocks_for_voice(day_blocks: list[Dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in day_blocks[:3]:
        day = str(block.get("day", "")).strip()
        ranges = block.get("ranges", [])
        if not day or not isinstance(ranges, list) or not ranges:
            continue
        lines.append(f"{day}: {', '.join(str(r) for r in ranges[:3])}")
    return " ; ".join(lines)


class Assistant(Agent):
    def __init__(
        self,
        call_context_text: str,
        debug_logger: Optional[CallDebugLogger] = None,
    ) -> None:
        self._call_end_in_progress = False
        self._debug_logger = debug_logger
        super().__init__(
            instructions=(
                """
You are the voice assistant for Code Studio (web development + AI automation agency).
Your goal is to understand why the person called and capture the minimum details needed.

Call types:
1) Sales lead (website, e-commerce, automation, AI, voice AI)
2) Support issue (existing project / site problem)
3) Meeting request (wants a call with Rej Aliaj)
4) Vendor/sales solicitation (caller is trying to sell us a product/service)

Our services (only these):
- Web Design
- WordPress, TYPO3, Shopify
- Headless CMS
- Web applications
- AI integration and agents creation
- SEO

Rules:
- Speak in a warm, business-professional tone.
- Keep responses short (phone style).
- On the caller's first turn, answer in one short sentence, then ask one focused question.
- Ask only the needed questions.
- Do not promise prices or timelines.
- If sales lead: capture name, company, need, best email.
- If caller shows interest in any of our services:
  give brief, receptionist-level information only, then guide them toward scheduling a meeting with Rey which is main developer.
  Do not go into deep technical detail.
- If support: capture name, company, problem, best email.
- When collecting email:
  1) ask caller to spell it slowly if needed,
  2) convert spoken words like "at" -> "@" and "dot" -> ".",
  3) read the final email back and ask explicit confirmation ("Is this correct?").
- If email remains unclear after two tries no problem leave it at that we will use phone which we already have.
- If meeting: first ask the caller for their preferred date and time.
- Then call check_meeting_slot using ISO 8601 datetime with timezone offset.
- If caller asks for a meeting beyond 2 weeks, say that the responsible person will handle it after the call.
- Confirm a meeting slot only when check_meeting_slot returns status "free" and the caller explicitly agrees.
- If status is "busy" or "outside_hours", offer only the returned next_slots.
- Never invent availability or times not returned by the tool.
- If the caller is trying to sell us something (vendor/sales solicitation):
  say we are not interested right now, and that the responsible person will be notified.
  If there is interest later, we will contact them.
- If a vendor caller is pushy or repeats after refusal, politely end the call by calling call_end function.
- If caller asks for content unrelated to our business (e.g., weather, jokes, random trivia):
  politely decline and steer back to business-related requests only.
- If they keep pushing unrelated requests after refusal, politely end the call by calling call_end function.
- IF the call is finished you have got the infomation needed or the caller has got the information needed for them. Wich them a nice day if their response after that has no further requests call call_end function
- When you decide to call call_end, just call it.

Call context:
""".strip()
                + "\n"
                + call_context_text
            )
        )

    def _debug_log(self, category: str, event: str, **fields: Any) -> None:
        if self._debug_logger is None:
            return
        self._debug_logger.log(category, event, **fields)

    @function_tool
    async def check_meeting_slot(
        self,
        context: RunContext,
        preferred_start_iso: str,
        duration_minutes: int = 30,
    ) -> str:
        """
        Use after user provides a specific preferred meeting datetime.
        preferred_start_iso must be ISO 8601 with timezone offset.
        """
        try:
            self._debug_log(
                "tool",
                "check_meeting_slot.start",
                preferred_start_iso=preferred_start_iso,
                duration_minutes=duration_minutes,
            )
            payload = {
                "tenant_id": TENANT_ID,
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
                if spoken:
                    result_text = f"That time is available. We can confirm: {spoken}."
                    self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                    return result_text
                result_text = "That time is available. We can confirm it."
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
                    if status == "busy":
                        result_text = "That time is not available. Next available options are: " + " ; ".join(lines)
                        self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                        return result_text
                    result_text = "That time is outside business hours. Available options are: " + " ; ".join(lines)
                    self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                    return result_text

                day_blocks = data.get("day_blocks", []) if isinstance(data, dict) else []
                block_txt = _format_day_blocks_for_voice(day_blocks if isinstance(day_blocks, list) else [])
                if block_txt:
                    result_text = "I cannot use that exact time. Available blocks are: " + block_txt
                    self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                    return result_text
                result_text = "I cannot use that exact time right now. Please suggest another time in the next two weeks."
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text

            if status == "outside_horizon":
                result_text = (
                    "That request is outside the next two weeks. "
                    "The responsible person will handle it after this call."
                )
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text

            if status == "unavailable":
                result_text = (
                    "I cannot reach live calendar availability right now. "
                    "Please share your preferred time and our team will confirm after this call."
                )
                self._debug_log("tool", "check_meeting_slot.result", result=result_text)
                return result_text

            result_text = "Please provide your preferred date and time in the next two weeks."
            self._debug_log("tool", "check_meeting_slot.result", result=result_text)
            return result_text
        except Exception as e:
            logger.exception("check_meeting_slot failed")
            result_text = f"I could not check that slot right now: {e}"
            self._debug_log("tool", "check_meeting_slot.error", error=str(e), result=result_text)
            return result_text

    @function_tool
    async def call_end(
        self,
        context: RunContext,
        call_type: str,
        name: str = "",
        company: str = "",
        contact_email: str = "",
        contact_phone: str = "",
        topic: str = "",
        notes: str = "",
        urgency: str = "",
        preferred_time_window: str = "",
    ) -> str:
        """
        Use this tool when the call is finished or enough info is collected.
        """
        if self._call_end_in_progress:
            logger.warning("[CALL_END_TOOL] call_end already in progress; ignoring duplicate request")
            self._debug_log("tool", "call_end.duplicate", call_type=call_type)
            return "Ending the call now."

        self._call_end_in_progress = True
        try:
            self._debug_log(
                "tool",
                "call_end.start",
                call_type=call_type,
                name=name,
                company=company,
                contact_email=contact_email,
                contact_phone=contact_phone,
                topic=topic,
                notes=notes,
                urgency=urgency,
                preferred_time_window=preferred_time_window,
            )
            ctx: JobContext = get_job_context()
            room = ctx.room
            session_obj = getattr(context, "session", None)
            transcript = ""
            if session_obj is not None:
                try:
                    history_payload = _build_transcript_payload(
                        session=session_obj,
                        room=room,
                        shutdown_reason="call_end_requested",
                    )
                    transcript = history_payload.get("transcript", "")
                except Exception:
                    logger.exception("[CALL_END_TOOL] failed to collect transcript context")

            validation_payload = {
                "transcript": transcript,
            }
            decision = await _post_json_and_read("/tools/validate-call-end", validation_payload)
            self._debug_log("tool", "call_end.validation_response", decision=decision)
            end_call = bool(decision.get("end_call", 0))
            logger.info("[CALL_END_TOOL] validator end_call=%s", end_call)
            if not end_call:
                result_text = (
                    "I can keep helping. If you want to finish now, please say a clear ending like "
                    "'that's all, thank you' or 'goodbye'."
                )
                self._debug_log("tool", "call_end.result", result=result_text)
                return result_text

            payload = {
                "tenant_id": TENANT_ID,
                "call_type": call_type,
                "name": name,
                "company": company,
                "contact_email": contact_email,
                "contact_phone": contact_phone,
                "topic": topic,
                "notes": notes,
                "urgency": urgency,
                "preferred_time_window": preferred_time_window,
                "room_name": room.name if room else None,
                "caller_id": _best_effort_caller_id(room) if room else None,
                "timestamp": int(time.time()),
            }
            self._debug_log("tool", "call_end.event_payload", payload=payload)
            await _post_json("/events/call-end", payload)

            # Snapshot SIP participant identities so we can force-kick after agent disconnect.
            sip_identities: list[str] = []
            try:
                for p in room.remote_participants.values():
                    identity = (p.identity or "").strip()
                    if not identity:
                        continue
                    is_sip = (
                        p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                        or "sip" in identity.lower()
                    )
                    if is_sip:
                        sip_identities.append(identity)
                if sip_identities:
                    logger.info(
                        "[CALL_END_TOOL] sip participants queued for kick count=%s ids=%s",
                        len(sip_identities),
                        sip_identities,
                    )
            except Exception:
                logger.exception("[CALL_END_TOOL] failed to snapshot SIP participants")

            # Always play a short closing line before disconnecting the room.
            if session_obj is not None:
                try:
                    # Avoid long hangs here; if TTS stalls, we still disconnect promptly.
                    await asyncio.wait_for(
                        session_obj.say("Thank you for calling Code Studio. Goodbye."),
                        timeout=6.0,
                    )
                    logger.info("[CALL_END_TOOL] farewell spoken before disconnect")
                except asyncio.TimeoutError:
                    logger.warning("[CALL_END_TOOL] farewell timed out; disconnecting anyway")
                except Exception:
                    logger.exception("[CALL_END_TOOL] farewell speak failed; disconnecting anyway")

            # Explicitly end the live call once details are saved.
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

            # After disconnecting the agent, force-kick SIP participants if API access is available.
            if sip_identities:
                room_api = getattr(getattr(ctx, "api", None), "room", None)
                remove_participant = getattr(room_api, "remove_participant", None)
                if callable(remove_participant):
                    try:
                        logger.info(
                            "[CALL_END_TOOL] remove_participant signature=%s",
                            str(inspect.signature(remove_participant)),
                        )
                    except Exception:
                        logger.info("[CALL_END_TOOL] remove_participant signature=unavailable")

                    for identity in sip_identities:
                        kicked = False
                        room_name = room.name if room else ""
                        attempts = []

                        # Newer server-sdk style: remove_participant(RoomParticipantIdentity(...))
                        try:
                            from livekit.api import RoomParticipantIdentity  # type: ignore

                            attempts.append(lambda: remove_participant(RoomParticipantIdentity(room=room_name, identity=identity)))
                        except Exception:
                            pass

                        # Backward-compatible keyword styles.
                        attempts.extend(
                            [
                                lambda: remove_participant(room=room_name, identity=identity),
                                lambda: remove_participant(room_name=room_name, identity=identity),
                                lambda: remove_participant(room=room_name, participant_identity=identity),
                                lambda: remove_participant(room_name=room_name, participant_identity=identity),
                            ]
                        )

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
                                logger.exception(
                                    "[CALL_END_TOOL] failed to kick SIP participant identity=%s",
                                    identity,
                                )
                                break

                        if not kicked:
                            logger.error("[CALL_END_TOOL] unable to kick SIP participant identity=%s", identity)
                else:
                    logger.warning("[CALL_END_TOOL] ctx.api.room.remove_participant unavailable; SIP kick skipped")

            result_text = "Saved. I sent the details to the team."
            self._debug_log("tool", "call_end.result", result=result_text)
            return result_text
        except Exception as e:
            logger.exception("call_end failed")
            result_text = "I could not finalize that right now, but our team has your details."
            self._debug_log("tool", "call_end.error", error=str(e), result=result_text)
            return result_text
        finally:
            self._call_end_in_progress = False


try:
    server = AgentServer(
        num_idle_processes=AGENT_NUM_IDLE_PROCESSES,
        load_threshold=AGENT_LOAD_THRESHOLD,
    )
    logger.info(
        "[SERVER] configured num_idle_processes=%s load_threshold=%.2f",
        AGENT_NUM_IDLE_PROCESSES,
        AGENT_LOAD_THRESHOLD,
    )
except TypeError:
    # Backward-compatible fallback for older AgentServer signatures.
    logger.warning(
        "[SERVER] AgentServer() does not accept num_idle_processes/load_threshold on this SDK; using defaults"
    )
    server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("[CALL_START] room=%s", ctx.room.name)
    logger.info(
        "[SESSION_CONFIG] llm_model=%s turn_detector=MultilingualModel preemptive_generation=%s",
        LLM_MODEL,
        True,
    )

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(BUSINESS_TIMEZONE))
    two_weeks_local = now_local + timedelta(days=14)
    call_context_text = (
        f"Current UTC time: {now_utc.isoformat()}\n"
        f"Current business local time ({BUSINESS_TIMEZONE}): {now_local.isoformat()}\n"
        f"Meeting booking horizon ends at ({BUSINESS_TIMEZONE}): {two_weeks_local.isoformat()}"
    )
    debug_logger = CallDebugLogger()
    debug_logger.log(
        "call",
        "session_started",
        room_name=ctx.room.name,
        business_timezone=BUSINESS_TIMEZONE,
        llm_model=LLM_MODEL,
    )

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model=LLM_MODEL),
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(
            call_context_text=call_context_text,
            debug_logger=debug_logger,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        old_state = str(getattr(ev, "old_state", ""))
        new_state = str(getattr(ev, "new_state", ""))
        debug_logger.log(
            "turn",
            "user_state_changed",
            old_state=old_state,
            new_state=new_state,
        )
        if new_state.lower().endswith("listening"):
            debug_logger.log(
                "turn",
                "USER_STOPPED_SPEAKING",
                old_state=old_state,
                new_state=new_state,
            )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: Any) -> None:
        debug_logger.log(
            "agent",
            "agent_state_changed",
            old_state=str(getattr(ev, "old_state", "")),
            new_state=str(getattr(ev, "new_state", "")),
        )

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        text = _event_text_payload(ev)
        if not text:
            return
        debug_logger.log(
            "transcript",
            "USER_FINAL" if bool(getattr(ev, "is_final", False)) else "USER_PARTIAL",
            text=text,
        )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: Any) -> None:
        item = getattr(ev, "item", None)
        role = str(getattr(item, "role", ""))
        text = _event_text_payload(item).strip()
        if role in ("user", "assistant") and text:
            debug_logger.log(
                "transcript",
                "USER_COMMITTED" if role == "user" else "ASSISTANT_COMMITTED",
                text=text,
            )

    @session.on("function_tools_executed")
    def _on_function_tools_executed(ev: Any) -> None:
        zipped = getattr(ev, "zipped", None)
        if callable(zipped):
            for idx, pair in enumerate(zipped(), start=1):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    debug_logger.log("tool", "function_tools_executed", index=idx, payload=pair)
                    continue
                function_call, function_output = pair
                call_name = str(getattr(function_call, "name", ""))
                arguments = str(getattr(function_call, "arguments", ""))
                output_text = str(getattr(function_output, "output", ""))
                debug_logger.log(
                    "tool",
                    "TOOL_EXECUTED",
                    index=idx,
                    name=call_name,
                    arguments=arguments,
                    output=output_text,
                    is_error=bool(getattr(function_output, "is_error", False)),
                )
            return
        debug_logger.log("tool", "TOOL_EXECUTED", payload=ev)

    async def _send_transcript_on_shutdown(reason: str) -> None:
        logger.info("[CALL_END] shutdown callback fired room=%s reason=%s", ctx.room.name, reason)
        debug_logger.log("call", "shutdown_started", room_name=ctx.room.name, reason=reason)

        try:
            payload = _build_transcript_payload(
                session=session, room=ctx.room, shutdown_reason=reason
            )
            logger.info(
                "[CALL_END] transcript prepared room=%s messages=%s chars=%s",
                ctx.room.name,
                len(payload["messages"]),
                len(payload["transcript"]),
            )
            if not payload["messages"] and not payload["transcript"]:
                logger.warning(
                    "[CALL_END] transcript payload is empty room=%s reason=%s",
                    ctx.room.name,
                    reason,
                )
            await _post_json("/events/transcript", payload)
            logger.info(
                "[CALL_END] transcript sent room=%s messages=%s",
                ctx.room.name,
                len(payload["messages"]),
            )
        except Exception:
            logger.exception(
                "[CALL_END] transcript send failed room=%s reason=%s",
                ctx.room.name,
                reason,
            )
        finally:
            debug_logger.log("call", "shutdown_finished", room_name=ctx.room.name, reason=reason)
            debug_logger.close(cleanup=True)

    ctx.add_shutdown_callback(_send_transcript_on_shutdown)

    await ctx.connect()

    if ENABLE_LLM_WARMUP:
        asyncio.create_task(_warmup_llm_once())

    await session.say("Thanks for calling Code Studio. How may we help you today?")


if __name__ == "__main__":
    cli.run_app(server)
