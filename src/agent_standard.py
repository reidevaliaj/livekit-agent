from typing import Any, Dict

from livekit.agents import AgentSession
from livekit.plugins import cartesia, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel


def build_standard_session(
    *,
    vad_model: Any,
    assistant_language: str,
    stt_language: str,
    llm_model: str,
    tts_voice: str,
    tts_speed: float,
    supports_turn_handling: bool,
    min_endpointing_delay: float,
    max_endpointing_delay: float,
    interruption_mode: str,
    interruption_min_duration: float,
    interruption_min_words: int,
    false_interruption_timeout: float | None,
    preemptive_generation_enabled: bool,
    resume_false_interruption: bool,
) -> AgentSession:
    session_kwargs: Dict[str, Any] = {
        "stt": deepgram.STT(model="nova-3", language=stt_language),
        "llm": openai.LLM(model=llm_model),
        "tts": cartesia.TTS(
            model="sonic-3",
            voice=tts_voice,
            language=assistant_language,
            speed=tts_speed,
        ),
        "vad": vad_model,
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
                "min_words": interruption_min_words,
                "resume_false_interruption": resume_false_interruption,
                "false_interruption_timeout": false_interruption_timeout,
            },
            "preemptive_generation": {
                "enabled": preemptive_generation_enabled,
            },
        }
    else:
        session_kwargs.update(
            turn_detection=MultilingualModel(),
            min_endpointing_delay=min_endpointing_delay,
            max_endpointing_delay=max_endpointing_delay,
            min_interruption_duration=interruption_min_duration,
            false_interruption_timeout=false_interruption_timeout,
            resume_false_interruption=resume_false_interruption,
            preemptive_generation=preemptive_generation_enabled,
        )

    return AgentSession(**session_kwargs)


async def speak_standard_text(
    session: AgentSession,
    text: str,
    *,
    allow_interruptions: bool = True,
) -> None:
    await session.say(text, allow_interruptions=allow_interruptions)
