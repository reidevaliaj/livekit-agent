import os
import time
import logging
from typing import Optional, Dict, Any

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
TENANT_ID = os.getenv("TENANT_ID", "codestudio")  # later: dynamic per customer


def _best_effort_caller_id(room: rtc.Room) -> Optional[str]:
    """
    SIP calls typically show participant identities like:
      sip_+355....
    This is best-effort because exact fields can vary.
    """
    try:
        for p in room.remote_participants.values():
            if "sip" in (p.identity or "").lower():
                return p.identity
        # fallback: first remote participant identity
        for p in room.remote_participants.values():
            return p.identity
    except Exception:
        pass
    return None


async def _post_json(path: str, payload: Dict[str, Any]) -> None:
    url = FASTAPI_BASE_URL.rstrip("/") + path
    logger.info(
        "[TRANSCRIPT_POST] sending path=%s url=%s payload_keys=%s",
        path,
        url,
        sorted(payload.keys()),
    )
    timeout = httpx.Timeout(10.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        logger.info(
            "[TRANSCRIPT_POST] response path=%s status=%s",
            path,
            r.status_code,
        )
        r.raise_for_status()


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


def _build_transcript_payload(
    session: AgentSession, room: rtc.Room, shutdown_reason: str
) -> Dict[str, Any]:
    messages: list[Dict[str, Any]] = []
    lines: list[str] = []
    history_messages = (
        session.history.messages()
        if callable(getattr(session.history, "messages", None))
        else session.history.messages
    )
    for msg in history_messages:
        if msg.role not in ("user", "assistant"):
            continue
        text = _flatten_message_content(msg.content).strip()
        if not text:
            continue
        line = f"{msg.role}: {text}"
        lines.append(line)
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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are the voice assistant for Code Studio (web development + AI automation agency).
Your goal is to understand WHY the person called and capture the minimum details needed.

Call types:
1) Sales lead (website, e-commerce, automation, AI, voice AI)
2) Support issue (existing project / site problem)
3) Meeting request (wants a call with Rej Aliaj)

Rules:
- Speak in a warm, business-professional tone.
- Keep responses short (phone style).
- Ask only the needed questions.
- Do not promise prices or timelines.
- If it’s a sales lead: capture name, company, what they need, best email/phone.
- If support: capture name, company, problem, urgency, best contact.
- If meeting: capture name, company, topic, best contact and preferred time window.
- At the end (or when you have enough info), send a structured “call_end” event.
""".strip()
        )

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
        Use this tool when the call is finished or you have enough info.
        It sends a structured summary to our backend.
        """
        try:
            ctx: JobContext = get_job_context()
            room = ctx.room
            payload = {
                "tenant_id": TENANT_ID,
                "call_type": call_type,  # "sales" | "support" | "meeting" | "other"
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
            return "Saved. I sent the details to the team."
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
        agent=Assistant(),
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

    async def _send_transcript_on_shutdown(reason: str) -> None:
        logger.info(
            "[CALL_END] shutdown callback fired room=%s reason=%s",
            ctx.room.name,
            reason,
        )
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
    await session.say("Thanks for calling Code Studio. How may we help you today?")


if __name__ == "__main__":
    cli.run_app(server)
