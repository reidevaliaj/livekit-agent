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
    timeout = httpx.Timeout(10.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()


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
            ctx: JobContext = context.job_ctx  # LiveKit provides job context here
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

    await ctx.connect()
    await session.say("Thanks for calling Code Studio. How may we help you today?")


if __name__ == "__main__":
    cli.run_app(server)