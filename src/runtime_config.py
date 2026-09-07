"""Effective voice settings. There is deliberately one speech engine."""

from dataclasses import dataclass

REALTIME_MODEL = "gpt-realtime-2.1-mini"
VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


@dataclass(frozen=True)
class VoiceOptions:
    model: str = REALTIME_MODEL
    voice: str = "marin"
    language: str = "en"


def voice_options(snapshot: dict, direction: str) -> VoiceOptions:
    config = snapshot["config"]
    outgoing = snapshot.get("outgoing", {}) if direction == "outgoing" else {}
    extra = config.get("extra_settings") or {}
    voice = str(
        outgoing.get("openai_realtime_voice")
        or config.get("openai_realtime_voice")
        or extra.get("openai_realtime_voice")
        or "marin"
    ).lower()
    language = str(
        outgoing.get("assistant_language") or config.get("assistant_language") or "en"
    ).lower()
    return VoiceOptions(voice=voice if voice in VOICES else "marin", language=language)
