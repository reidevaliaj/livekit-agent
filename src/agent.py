import os
import time
import logging
import asyncio
import random
import wave
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from pathlib import Path
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
    inference,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")


# -----------------------------
# Config
# -----------------------------
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "https://voice.code-studio.eu")
TENANT_ID = os.getenv("TENANT_ID", "codestudio")
BUSINESS_TIMEZONE = os.getenv("BUSINESS_TIMEZONE", "Europe/Budapest")

ENABLE_FILLER_BRIDGE = os.getenv("ENABLE_FILLER_BRIDGE", "true").strip().lower() == "true"
FILLER_DELAY_SECONDS = float(os.getenv("FILLER_DELAY_SECONDS", "1.0").strip() or "1.0")
FILLER_MAX_PER_CALL = int(os.getenv("FILLER_MAX_PER_CALL", "3").strip() or "3")
FILLER_PHRASES = [
    p.strip()
    for p in os.getenv("FILLER_PHRASES", "Alright.,Okay.,Sure.").split(",")
    if p.strip()
]
if not FILLER_PHRASES:
    FILLER_PHRASES = ["Alright.", "Okay.", "Sure."]

ENABLE_AMBIENCE_LOOP = os.getenv("ENABLE_AMBIENCE_LOOP", "true").strip().lower() == "true"
AMBIENCE_FILE = os.getenv("AMBIENCE_FILE", str(Path(__file__).parent / "assets" / "ambience_office.wav"))


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
    def __init__(self, call_context_text: str) -> None:
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
- Ask only the needed questions.
- Do not promise prices or timelines.
- If sales lead: capture name, company, need, best email/phone.
- If caller shows interest in any of our services:
  give brief, receptionist-level information only, then guide them toward scheduling a meeting with Founder Rey.
  Do not go into deep technical detail.
- If support: capture name, company, problem, best contact.
- When collecting email:
  1) ask caller to spell it slowly if needed,
  2) convert spoken words like "at" -> "@" and "dot" -> ".",
  3) read the final email back and ask explicit confirmation ("Is this correct?").
- If email remains unclear after two tries, ask for phone as fallback contact.
- If meeting: first ask the caller for their preferred date and time.
- Then call check_meeting_slot using ISO 8601 datetime with timezone offset.
- If caller asks for a meeting beyond 2 weeks, say that the responsible person will handle it after the call.
- Confirm a meeting slot only when check_meeting_slot returns status "free" and the caller explicitly agrees.
- If status is "busy" or "outside_hours", offer only the returned next_slots.
- Never invent availability or times not returned by the tool.
- If the caller is trying to sell us something (vendor/sales solicitation):
  say we are not interested right now, and that the responsible person will be notified.
  If there is interest later, we will contact them.
- If a vendor caller is pushy or repeats after refusal, politely end the call.
- If caller asks for content unrelated to our business (e.g., weather, jokes, random trivia):
  politely decline and steer back to business-related requests only.
- If they keep pushing unrelated requests after refusal, politely end the call.
- At the end (or when enough info), send call_end event.

Call context:
""".strip()
                + "\n"
                + call_context_text
            )
        )

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
            payload = {
                "tenant_id": TENANT_ID,
                "preferred_start_iso": preferred_start_iso,
                "duration_minutes": duration_minutes,
                "alternatives_limit": 3,
            }
            data = await _post_json_and_read("/tools/check-meeting-slot", payload)
            status = str(data.get("status", "")) if isinstance(data, dict) else ""

            if status == "free":
                confirmed = data.get("confirmed_slot", {}) if isinstance(data, dict) else {}
                spoken = _format_slot_for_voice(confirmed) if isinstance(confirmed, dict) else ""
                if spoken:
                    return f"That time is available. We can confirm: {spoken}."
                return "That time is available. We can confirm it."

            if status in ("busy", "outside_hours"):
                next_slots = data.get("next_slots", []) if isinstance(data, dict) else []
                lines: list[str] = []
                for idx, slot in enumerate(next_slots[:3], start=1):
                    spoken = _format_slot_for_voice(slot)
                    if spoken:
                        lines.append(f"{idx}) {spoken}")
                if lines:
                    if status == "busy":
                        return "That time is not available. Next available options are: " + " ; ".join(lines)
                    return "That time is outside business hours. Available options are: " + " ; ".join(lines)

                day_blocks = data.get("day_blocks", []) if isinstance(data, dict) else []
                block_txt = _format_day_blocks_for_voice(day_blocks if isinstance(day_blocks, list) else [])
                if block_txt:
                    return "I cannot use that exact time. Available blocks are: " + block_txt
                return "I cannot use that exact time right now. Please suggest another time in the next two weeks."

            if status == "outside_horizon":
                return (
                    "That request is outside the next two weeks. "
                    "The responsible person will handle it after this call."
                )

            if status == "unavailable":
                return (
                    "I cannot reach live calendar availability right now. "
                    "Please share your preferred time and our team will confirm after this call."
                )

            return "Please provide your preferred date and time in the next two weeks."
        except Exception as e:
            logger.exception("check_meeting_slot failed")
            return f"I could not check that slot right now: {e}"

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
        try:
            ctx: JobContext = get_job_context()
            room = ctx.room
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
            await _post_json("/events/call-end", payload)
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
            return "Saved. I sent the details to the team. Ending the call now."
        except Exception as e:
            logger.exception("call_end failed")
            return f"Failed to send details: {e}"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("[CALL_START] room=%s", ctx.room.name)

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(BUSINESS_TIMEZONE))
    two_weeks_local = now_local + timedelta(days=14)
    call_context_text = (
        f"Current UTC time: {now_utc.isoformat()}\n"
        f"Current business local time ({BUSINESS_TIMEZONE}): {now_local.isoformat()}\n"
        f"Meeting booking horizon ends at ({BUSINESS_TIMEZONE}): {two_weeks_local.isoformat()}"
    )

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(call_context_text=call_context_text),
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

    ambience_stop_event = asyncio.Event()
    ambience_task: Optional[asyncio.Task[Any]] = None

    if ENABLE_AMBIENCE_LOOP:
        async def _ambience_loop() -> None:
            ambience_path = Path(AMBIENCE_FILE)
            if not ambience_path.exists():
                logger.warning("[AMBIENCE] file not found: %s", ambience_path)
                return

            try:
                with wave.open(str(ambience_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    num_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    if sample_width != 2:
                        logger.warning(
                            "[AMBIENCE] unsupported sample width=%s in %s (expected 16-bit PCM)",
                            sample_width,
                            ambience_path,
                        )
                        return

                    source = rtc.AudioSource(sample_rate=sample_rate, num_channels=num_channels)
                    track = rtc.LocalAudioTrack.create_audio_track("office_ambience", source)
                    await ctx.room.local_participant.publish_track(track)
                    logger.info(
                        "[AMBIENCE] started loop file=%s sample_rate=%s channels=%s",
                        ambience_path,
                        sample_rate,
                        num_channels,
                    )

                    frame_samples = max(1, sample_rate // 50)  # ~20ms chunks
                    while not ambience_stop_event.is_set():
                        pcm = wf.readframes(frame_samples)
                        if not pcm:
                            wf.rewind()
                            continue

                        samples_per_channel = len(pcm) // (2 * num_channels)
                        if samples_per_channel <= 0:
                            continue

                        frame = rtc.AudioFrame(
                            data=pcm,
                            sample_rate=sample_rate,
                            num_channels=num_channels,
                            samples_per_channel=samples_per_channel,
                        )
                        await source.capture_frame(frame)
                        await asyncio.sleep(samples_per_channel / sample_rate)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[AMBIENCE] loop failed")

        ambience_task = asyncio.create_task(_ambience_loop())

    filler_stop_event = asyncio.Event()
    filler_task: Optional[asyncio.Task[Any]] = None

    if ENABLE_FILLER_BRIDGE:
        async def _filler_bridge_watchdog() -> None:
            last_user_key: Optional[tuple[float, str]] = None
            filler_sent_for_turn = False
            filler_count = 0
            while not filler_stop_event.is_set():
                try:
                    if filler_count >= FILLER_MAX_PER_CALL:
                        await asyncio.sleep(0.2)
                        continue

                    messages = _history_messages(session)
                    if not messages:
                        await asyncio.sleep(0.1)
                        continue

                    latest_msg = messages[-1]
                    if getattr(latest_msg, "role", None) != "user":
                        await asyncio.sleep(0.1)
                        continue

                    latest_user = latest_msg

                    user_ts = float(getattr(latest_user, "created_at", 0.0) or 0.0)
                    user_text = _flatten_message_content(getattr(latest_user, "content", "")).strip()
                    user_key = (user_ts, user_text)

                    if user_key != last_user_key:
                        last_user_key = user_key
                        filler_sent_for_turn = False

                    if not filler_sent_for_turn and user_ts > 0 and (time.time() - user_ts) >= FILLER_DELAY_SECONDS:
                        assistant_after_user = False
                        for msg in messages:
                            if getattr(msg, "role", None) != "assistant":
                                continue
                            assistant_ts = float(getattr(msg, "created_at", 0.0) or 0.0)
                            if assistant_ts >= user_ts:
                                assistant_after_user = True
                                break

                        if not assistant_after_user:
                            phrase = random.choice(FILLER_PHRASES)
                            logger.info(
                                "[FILLER] bridge phrase room=%s phrase=%s",
                                ctx.room.name,
                                phrase,
                            )
                            try:
                                await session.say(phrase)
                            except Exception:
                                logger.exception("[FILLER] failed to say bridge phrase")
                            filler_sent_for_turn = True
                            filler_count += 1

                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("[FILLER] watchdog failed")
                    await asyncio.sleep(0.2)

        filler_task = asyncio.create_task(_filler_bridge_watchdog())

    async def _send_transcript_on_shutdown(reason: str) -> None:
        logger.info("[CALL_END] shutdown callback fired room=%s reason=%s", ctx.room.name, reason)

        ambience_stop_event.set()
        if ambience_task and not ambience_task.done():
            ambience_task.cancel()
            try:
                await ambience_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("[AMBIENCE] cleanup failed")

        filler_stop_event.set()
        if filler_task and not filler_task.done():
            filler_task.cancel()
            try:
                await filler_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("[FILLER] cleanup failed")

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

    ctx.add_shutdown_callback(_send_transcript_on_shutdown)

    await ctx.connect()
    await asyncio.sleep(0.5)
    await session.say("Thanks for calling Code Studio How may we help you today?")


if __name__ == "__main__":
    cli.run_app(server)
