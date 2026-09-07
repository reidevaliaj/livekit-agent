# AI receptionist voice workers

Incoming and outgoing calls share one OpenAI Realtime speech engine:
`gpt-realtime-2.1-mini`, low reasoning effort, semantic VAD with high eagerness,
and interruptible speech. The backend supplies business context, language and a
supported OpenAI voice (Marin by default). There are no alternative runtime modes.
The auxiliary `gpt-4o-mini-transcribe` stream supplies call history; responses do
not wait for that transcript. No Deepgram, Cartesia, Silero or downloaded turn
detector models are used.

## Setup and run

Use Python 3.12+ and uv. Copy `.env.example` to `.env.local` and supply the existing
LiveKit/OpenAI credentials plus the backend's dedicated `AGENT_API_TOKEN`.

```sh
uv sync --locked
uv run --locked python src/agent.py start
uv run --locked python src/outgoing_agent.py start
```

Run each worker in its own managed process. Incoming dispatch remains `my-agent`;
outgoing defaults to `outgoing-agent`. Existing SIP identities and Twilio/Telnyx
backend routing contracts remain supported. LiveKit-first outbound prepares the
voice session before dialing and requests its opening phrase after answer.

When deploying into a release directory, set `CALL_DIAGNOSTICS_DIR` to the shared
application path read by the administration UI, usually
`/home/rei/apps/livekit-agent/runtime/calls`. Set `FASTAPI_BASE_URL` to the private
backend address. Authentication uses `X-Agent-Token`, not the legacy internal key.
Dependency updates must regenerate `uv.lock` and pass the offline suite before a
new release replaces the running worker. Keep the previous release and its
environment for rollback; never update a running environment in place.

## Modules

* `agent.py` / `outgoing_agent.py`: worker entrypoints and dispatch names.
* `voice_runtime.py`: shared realtime session, SIP lifecycle, transcript delivery.
* `receptionist.py`: business instructions, knowledge, availability, booking and end-call tools.
* `backend_client.py`: pooled authenticated HTTP with whole-operation deadlines.
* `runtime_config.py`: fixed model and effective language/voice validation.
* `diagnostics.py`: bounded asynchronous diagnostic writing to unique call files.

Availability is not a booking. `check_meeting_slot` always reports `booked=false`.
`book_meeting` confirms only a backend `status=booked`, `ok=true` result containing
an event ID. Its stable idempotency key binds tenant, call room, normalized UTC
slot and duration. Unknown/pending/timeouts never produce a booking confirmation.
Disabled tools are omitted from the model and checked again at execution.

For an incoming holiday-booking demo, set the tenant's
`extra_settings.booking_mode` to `simulation` and explicitly disable
`calendar_lookup`, `meeting_creation`, `zoom_meetings`, `email_summary` and
`case_creation` in its versioned backend configuration. Supply a fictional
catalogue and demo availability in the tenant prompt/FAQ. The agent confirms
only test reservations after caller consent, exposes no calendar tools, and
waits for the caller to finish before `call_end` records an internal summary.
The mode has no effect on outgoing calls; removing it restores ordinary booking
rules. This mode does not create reservations in any external system.

Session configuration failures terminate the affected call; the worker never
uses another business as a fallback. Goodbye and SIP cleanup are bounded, even
when event delivery or provider hangup fails. Final transcripts use the backend's
durable event acceptance endpoint after the voice pipeline drains. A delivery
failure is visible as `transcript_delivery_failed`; an accepted event means queued,
not that every downstream action has finished.

## Diagnostics

Each call writes `runtime/calls/<UUID>.jsonl`. The envelope contains:
`ts`, `elapsed_ms`, `call_id`, `direction` (`incoming`/`outgoing`), `tenant_id`,
hashed `room_id`, and `event`. Fields are explicitly allowlisted. Caller speech,
phone numbers, tool arguments/results, and credentials are not diagnostic fields.
Files use restrictive permissions. The queue is bounded and drops diagnostics
instead of stalling the voice pipeline when full. Apply normal retention to this
directory at deployment; files are separated per call and never overwrite another
active call.

| Event | Fields |
|---|---|
| `session_config` | model, voice, language, config_version, sdk_version, turn_detection, eagerness, reasoning_effort |
| `lifecycle` | stage: session_starting, session_ready, dial_started, callee_answered, greeting_requested |
| `turn_state` | actor, old_state, new_state |
| `response_timing` | user_stop_to_agent_speaking_ms, measurement=software_state_proxy_not_pstn_audio |
| `per_turn_metrics` | metrics: numeric SDK latency/token fields only |
| `tool_completed` | tool_name, duration_ms, outcome |
| `session_error` / `delivery_error` | error_type, recoverable or stage |
| `session_ended` | reason, transcript_accepted |

State gaps and SDK metrics are software measurements. They do not directly
measure first sound at a telephone receiver. Realtime has no separate text-LLM
and TTS stage timings. Compare the same metric across like call modes and retain
telephone recordings from controlled tests when validating perceived latency.

Cloud session recording is explicitly disabled by this worker (`record=False`).
Existing backend call histories still receive transcripts. Provider/account
retention policies are managed separately.

## Offline validation

```sh
uv run --locked pytest tests -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

Tests use mocked HTTP/SIP and provider-free SDK construction. They cover deadlines,
auth, tenant failure, booking truthfulness/idempotency, tool disablement, diagnostic
privacy, outbound start ordering and teardown failures. They do not make calls or
paid AI requests and cannot establish real voice quality or production latency.
Perform a controlled test call and a limited rollout after deployment.

Relevant API references: [OpenAI Realtime plugin](https://docs.livekit.io/agents/models/realtime/plugins/openai/),
[LiveKit data hooks](https://docs.livekit.io/deploy/observability/data/),
[turn handling](https://docs.livekit.io/reference/agents/turn-handling-options/).
