"""Shared realtime session, telephone lifecycle and privacy-safe observability."""

import asyncio
import contextlib
import json
import logging
import os
import time
from importlib.metadata import version
from pathlib import Path

from livekit import api, rtc
from livekit.agents import AgentSession, JobContext, room_io
from livekit.plugins import noise_cancellation, openai
from openai.types.realtime import AudioTranscription, RealtimeReasoning
from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad

from backend_client import BackendClient, validate_session_config
from diagnostics import CallDiagnostics
from receptionist import Receptionist
from runtime_config import voice_options

logger = logging.getLogger("receptionist")
ROOT = Path(__file__).resolve().parent.parent
FAREWELLS = {
    "en": "Thank you. Goodbye.",
    "it": "Grazie. Arrivederci.",
    "de": "Vielen Dank. Auf Wiederhören.",
}


def build_session(options) -> AgentSession:
    transcription = {"model": "gpt-4o-mini-transcribe"}
    if options.language in {"en", "it", "de"}:
        transcription["language"] = options.language
    return AgentSession(
        llm=openai.realtime.RealtimeModel(
            model=options.model,
            voice=options.voice,
            input_audio_transcription=AudioTranscription(**transcription),
            reasoning=RealtimeReasoning(effort="low"),
            turn_detection=SemanticVad(
                type="semantic_vad",
                eagerness="high",
                create_response=True,
                interrupt_response=True,
            ),
        ),
        turn_handling={
            "turn_detection": "realtime_llm",
            "interruption": {
                "enabled": True,
                "discard_audio_if_uninterruptible": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            },
            "preemptive_generation": {"enabled": False},
        },
        max_tool_steps=4,
        session_close_transcript_timeout=2.0,
    )


def attributes(participant) -> dict:
    if participant is None:
        return {}
    result = dict(participant.attributes)
    try:
        metadata = json.loads(participant.metadata or "{}")
        if isinstance(metadata, dict):
            result.update(metadata)
    except (ValueError, TypeError):
        pass
    return {
        str(k).lower().replace("-", "_").replace(".", "_"): v for k, v in result.items()
    }


def get_attr(attrs, key):
    return str(attrs.get(key) or attrs.get("x_" + key) or "")


def metadata(ctx):
    try:
        value = json.loads(ctx.job.metadata or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


async def fetch_config(ctx, backend, direction, dispatch):
    first = direction == "outgoing" and dispatch.get("mode") == "livekit_first"
    participant = None
    if not first:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=8.0)
    attrs = attributes(participant)
    payload = {
        "room_name": ctx.room.name,
        "tenant_id": get_attr(attrs, "tenant_id")
        or str(dispatch.get("tenant_id") or ""),
        "tenant_slug": get_attr(attrs, "tenant_slug")
        or str(dispatch.get("tenant_slug") or ""),
        "caller_id": participant.identity if participant else "",
        "called_number": get_attr(attrs, "called_number"),
        "call_sid": get_attr(attrs, "parent_call_sid")
        or str(dispatch.get("call_sid") or ""),
    }
    if direction == "outgoing":
        payload["outgoing_call_id"] = get_attr(attrs, "outgoing_call_id") or str(
            dispatch.get("outgoing_call_id") or ""
        )
        endpoint = "/agent/outgoing-session-config"
    else:
        raw_version = get_attr(attrs, "config_version")
        payload["config_version"] = int(raw_version) if raw_version.isdigit() else None
        endpoint = "/agent/session-config"
    snapshot = validate_session_config(
        await backend.post(endpoint, payload, budget=3.0)
    )
    if direction == "outgoing" and not snapshot.get("call", {}).get("id"):
        raise ValueError("Verified outgoing call is unavailable")
    return snapshot, participant


async def disconnect_sip(ctx, identity: str | None = None):
    participants = list(ctx.room.remote_participants.values())
    targets = [
        p.identity
        for p in participants
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        and (identity is None or p.identity == identity)
    ]
    for participant_identity in targets:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2.0):
                await ctx.api.room.remove_participant(
                    api.RoomParticipantIdentity(
                        room=ctx.room.name, identity=participant_identity
                    )
                )
    # JobContext closes the AgentSession and invokes on_session_end before it
    # disconnects the agent room, so transcript/control streams can drain.
    ctx.shutdown(reason="receptionist finished")


class CallRuntime:
    def __init__(self, ctx, backend, snapshot, direction, dispatch, participant):
        self.ctx, self.backend, self.snapshot = ctx, backend, snapshot
        self.direction, self.dispatch = direction, dispatch
        self.participant_identity = (
            participant.identity
            if participant
            else str(dispatch.get("participant_identity") or "")
        )
        self.options = voice_options(snapshot, direction)
        self.log = CallDiagnostics(
            Path(os.getenv("CALL_DIAGNOSTICS_DIR") or ROOT / "runtime" / "calls"),
            direction,
            str(snapshot["tenant"]["id"]),
            ctx.room.name,
        )
        self.session = build_session(self.options)
        self._finalized = False
        self._ending = False
        self._stop_at = None
        self._tasks = set()
        self.bind_events()

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def bind_events(self):
        @self.session.on("user_state_changed")
        def user_state(event):
            self.log.emit(
                "turn_state",
                actor="user",
                old_state=event.old_state,
                new_state=event.new_state,
            )
            if event.old_state == "speaking" and event.new_state == "listening":
                self._stop_at = time.monotonic()
            elif event.new_state == "speaking":
                self._stop_at = None

        @self.session.on("agent_state_changed")
        def agent_state(event):
            self.log.emit(
                "turn_state",
                actor="agent",
                old_state=event.old_state,
                new_state=event.new_state,
            )
            if event.new_state == "speaking" and self._stop_at is not None:
                self.log.emit(
                    "response_timing",
                    user_stop_to_agent_speaking_ms=(time.monotonic() - self._stop_at)
                    * 1000,
                    measurement="software_state_proxy_not_pstn_audio",
                )
                self._stop_at = None

        @self.session.on("conversation_item_added")
        def item_added(event):
            metrics = getattr(event.item, "metrics", None)
            if isinstance(metrics, dict):
                self.log.emit("per_turn_metrics", metrics=metrics)

        # AgentSession forwards metrics from its RealtimeSession; RealtimeModel
        # is a factory and is not an event emitter.
        @self.session.on("metrics_collected")
        def realtime_metrics(event):
            metrics = event.metrics
            self.log.emit(
                "per_turn_metrics",
                metrics={
                    k: getattr(metrics, k, None)
                    for k in (
                        "ttft",
                        "duration",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                },
            )

        @self.session.on("error")
        def session_error(event):
            recoverable = bool(getattr(event.error, "recoverable", False))
            self.log.emit(
                "session_error",
                error_type=type(event.error).__name__,
                recoverable=recoverable,
            )
            if not recoverable:
                self.spawn(self.finish(speak=False))

    async def start(self):
        config = self.snapshot["config"]
        self.log.emit(
            "session_config",
            model=self.options.model,
            voice=self.options.voice,
            language=self.options.language,
            config_version=config.get("version"),
            sdk_version=version("livekit-agents"),
            turn_detection="semantic_vad",
            eagerness="high",
            reasoning_effort="low",
        )
        self.log.emit("lifecycle", stage="session_starting")
        assistant = Receptionist(
            self.snapshot,
            self.direction,
            self.ctx.room.name,
            self.backend,
            self.log,
            self.finish,
        )
        options = room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                )
            ),
        )
        if self.participant_identity:
            options.participant_identity = self.participant_identity
        async with asyncio.timeout(12.0):
            await self.session.start(
                agent=assistant, room=self.ctx.room, room_options=options, record=False
            )
        self.log.emit("lifecycle", stage="session_ready")
        # Model/WebSocket session is prepared before placing a LiveKit-first call.
        if (
            self.direction == "outgoing"
            and self.dispatch.get("mode") == "livekit_first"
            and not await self.dial()
        ):
            await self.finish(speak=False)
            return
        greeting = (
            self.snapshot.get("outgoing", {}).get("opening_phrase")
            if self.direction == "outgoing"
            else config.get("greeting")
        )
        if greeting:
            self.log.emit("lifecycle", stage="greeting_requested")
            await self.speak(str(greeting), interruptible=True)

    async def speak(self, text: str, *, interruptible: bool):
        if not interruptible:
            # This is the terminal farewell only. Disable server turn taking
            # before requesting non-interruptible speech; normal calls keep VAD.
            self.session.llm.update_options(turn_detection=None)
        async with asyncio.timeout(30.0 if interruptible else 8.0):
            handle = self.session.generate_reply(
                instructions=f"Speak exactly this message in the configured language, with no additions: {text}",
                # Realtime turn taking controls interruption. It is disabled
                # above for the farewell and remains active for the greeting.
                allow_interruptions=True,
                tool_choice="none",
                input_modality="text",
            )
            await handle.wait_for_playout()

    def outgoing_payload(self):
        call = self.snapshot.get("call", {})
        return {
            "tenant_id": str(self.snapshot["tenant"]["id"]),
            "tenant_slug": str(self.snapshot["tenant"].get("slug") or ""),
            "outgoing_call_id": str(call.get("id") or ""),
            "call_sid": str(
                call.get("provider_call_id")
                or call.get("telnyx_call_control_id")
                or call.get("twilio_call_sid")
                or ""
            ),
            "room_name": self.ctx.room.name,
            "timestamp": int(time.time()),
        }

    async def status(self, **fields):
        try:
            await self.backend.post(
                "/outgoing/calls/livekit-status",
                {**self.outgoing_payload(), **fields},
                budget=3.0,
            )
        except Exception as error:
            self.log.emit(
                "delivery_error", stage="sip_status", error_type=type(error).__name__
            )

    async def dial(self):
        data = self.dispatch
        if not all(
            data.get(k)
            for k in (
                "phone_number",
                "sip_trunk_id",
                "participant_identity",
                "outgoing_call_id",
            )
        ):
            self.log.emit(
                "session_error", error_type="InvalidDispatchMetadata", recoverable=False
            )
            return False
        attrs = {
            "tenant_id": str(self.snapshot["tenant"]["id"]),
            "tenant_slug": str(self.snapshot["tenant"].get("slug") or ""),
            "outgoing_call_id": str(self.snapshot["call"]["id"]),
            "call_direction": "outgoing",
            "call_provider": str(data.get("provider") or "telnyx"),
        }
        request = api.CreateSIPParticipantRequest(
            room_name=self.ctx.room.name,
            sip_trunk_id=str(data["sip_trunk_id"]),
            sip_call_to=str(data["phone_number"]),
            sip_number=str(data.get("from_number") or ""),
            participant_identity=self.participant_identity,
            participant_name=str(data.get("participant_name") or "Callee"),
            participant_metadata=json.dumps(attrs),
            participant_attributes=attrs,
            play_dialtone=False,
            krisp_enabled=True,
            wait_until_answered=True,
            display_name=str(data.get("caller_display_name") or ""),
        )
        self.log.emit("lifecycle", stage="dial_started")
        try:
            async with asyncio.timeout(75.0):
                result = await self.ctx.api.sip.create_sip_participant(request)
            async with asyncio.timeout(8.0):
                await self.ctx.wait_for_participant(identity=self.participant_identity)
        except Exception as error:
            self.log.emit(
                "session_error", error_type=type(error).__name__, recoverable=False
            )
            details = getattr(error, "metadata", {}) or {}
            await self.status(
                status="failed",
                participant_identity=self.participant_identity,
                sip_status_code=str(details.get("sip_status_code") or ""),
                error="SIP dial did not complete",
            )
            return False
        self.log.emit("lifecycle", stage="callee_answered")
        # Status reporting cannot delay the first spoken greeting.
        self.spawn(
            self.status(
                status="bridged",
                provider_call_sid=result.sip_call_id,
                participant_identity=self.participant_identity,
            )
        )
        return True

    async def finish(self, *, speak=True):
        if self._ending:
            return
        self._ending = True
        try:
            if speak:
                with contextlib.suppress(Exception):
                    await self.speak(
                        FAREWELLS.get(self.options.language, FAREWELLS["en"]),
                        interruptible=False,
                    )
            if self.direction == "outgoing":
                try:
                    await self.backend.post(
                        "/outgoing/calls/end",
                        {**self.outgoing_payload(), "reason": "assistant_goodbye"},
                        budget=3.0,
                    )
                except Exception as error:
                    self.log.emit(
                        "delivery_error",
                        stage="provider_hangup",
                        error_type=type(error).__name__,
                    )
        finally:
            await disconnect_sip(self.ctx, self.participant_identity or None)

    async def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        try:
            # on_session_end runs after the voice pipeline closes and transcript drain completes.
            history = self.session.history.messages()
            messages = []
            for message in history:
                if message.role not in {"user", "assistant"}:
                    continue
                text = message.text_content
                if text:
                    messages.append(
                        {
                            "role": message.role,
                            "text": text,
                            "interrupted": bool(message.interrupted),
                            "created_at": message.created_at,
                        }
                    )
            payload = {
                "tenant_id": str(self.snapshot["tenant"]["id"]),
                "agent_call_id": self.log.call_id,
                "room_name": self.ctx.room.name,
                "shutdown_reason": "session_ended",
                "timestamp": int(time.time()),
                "messages": messages,
                "transcript": "\n".join(f"{m['role']}: {m['text']}" for m in messages),
            }
            if self.direction == "outgoing":
                payload.update(self.outgoing_payload())
                path = "/outgoing/events/transcript"
            else:
                payload["caller_id"] = self.participant_identity
                path = "/events/transcript"
            # Retry the identical event, never regenerate its timestamp or identity.
            # Both attempts fit inside the worker's bounded finalization window.
            for attempt in range(2):
                try:
                    result = await self.backend.post(path, payload, budget=2.5)
                    if (
                        result.get("ok") is not True
                        or result.get("accepted") is not True
                    ):
                        raise ValueError("Transcript event was not durably accepted")
                    break
                except Exception:
                    if attempt == 1:
                        raise
                    await asyncio.sleep(0.2)
            self.log.emit(
                "session_ended",
                reason="completed",
                transcript_accepted=bool(result.get("accepted")),
            )
        except Exception as error:
            self.log.emit(
                "session_ended",
                reason="transcript_delivery_failed",
                error_type=type(error).__name__,
                transcript_accepted=False,
            )
        finally:
            if self._tasks:
                _, pending = await asyncio.wait(self._tasks, timeout=1.0)
                for task in pending:
                    task.cancel()
            await self.backend.aclose()
            await self.session.llm.aclose()
            self.log.close()


async def run_call(ctx: JobContext, direction: str):
    backend = BackendClient(
        os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000"),
        os.getenv("AGENT_API_TOKEN", ""),
    )
    runtime = None
    try:
        await ctx.connect()
        dispatch = metadata(ctx)
        snapshot, participant = await fetch_config(ctx, backend, direction, dispatch)
        runtime = CallRuntime(ctx, backend, snapshot, direction, dispatch, participant)
        ctx.proc.userdata["call_runtime"] = runtime
        await runtime.start()
    except Exception as error:
        # Never log exception strings: HTTP/provider exceptions can contain secrets or caller content.
        logger.error("Call startup failed: %s", type(error).__name__)
        if runtime:
            runtime.log.emit(
                "session_error", error_type=type(error).__name__, recoverable=False
            )
        await disconnect_sip(ctx, runtime.participant_identity if runtime else None)
        if not runtime:
            await backend.aclose()


async def on_session_end(ctx: JobContext):
    runtime = ctx.proc.userdata.pop("call_runtime", None)
    if runtime:
        await runtime.finalize()
