from livekit.agents import AgentSession
from livekit.plugins import openai
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES = {"en", "it", "de"}


def _realtime_transcription_language(stt_language: str, assistant_language: str) -> str | None:
    if stt_language in SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES:
        return stt_language
    if assistant_language in SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES:
        return assistant_language
    return None


def build_realtime_session(
    *,
    incoming_realtime_model: str,
    incoming_realtime_voice: str,
    stt_language: str,
    assistant_language: str,
    interruption_min_duration: float,
    interruption_min_words: int,
    false_interruption_timeout: float | None,
    resume_false_interruption: bool,
) -> AgentSession:
    realtime_transcription_kwargs: dict[str, object] = {
        "model": "gpt-4o-mini-transcribe",
    }
    realtime_language = _realtime_transcription_language(stt_language, assistant_language)
    if realtime_language:
        realtime_transcription_kwargs["language"] = realtime_language

    return AgentSession(
        llm=openai.realtime.RealtimeModel(
            model=incoming_realtime_model,
            voice=incoming_realtime_voice,
            input_audio_transcription=InputAudioTranscription(**realtime_transcription_kwargs),
            input_audio_noise_reduction="near_field",
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="high",
                create_response=True,
                interrupt_response=True,
            ),
        ),
        allow_interruptions=True,
        discard_audio_if_uninterruptible=True,
        min_interruption_duration=interruption_min_duration,
        min_interruption_words=interruption_min_words,
        false_interruption_timeout=false_interruption_timeout,
        resume_false_interruption=resume_false_interruption,
        preemptive_generation=False,
    )


async def speak_realtime_text(
    session: AgentSession,
    text: str,
    *,
    allow_interruptions: bool = True,
) -> None:
    handle = session.generate_reply(
        instructions=(
            "Speak to the caller now. Keep it short and natural. "
            f"Say this message faithfully in one turn: {text}"
        ),
        allow_interruptions=allow_interruptions,
        input_modality="text",
    )
    await handle
