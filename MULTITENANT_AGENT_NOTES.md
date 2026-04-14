# Agent Multi-Tenant Notes

## What Changed
- The shared agent no longer relies on a hardcoded Code Studio prompt or a global `TENANT_ID` for live calls.
- At call start, the agent now asks the backend for `/agent/session-config` and uses that snapshot to build the instructions, greeting, timezone context, model choice, and voice choice.
- Tool calls now send the resolved tenant id from the session snapshot.
- Transcript uploads now carry the resolved tenant id from the same snapshot.
- If backend config lookup fails, the agent falls back to the legacy Code Studio defaults so the call can still proceed.

## New Runtime Assumptions
- The backend should expose `/agent/session-config` and accept the internal API key.
- Best case: LiveKit SIP participant attributes include the tenant headers from Twilio / SIP trunk header mapping.
- Fallback: if those attributes are missing, the backend still tries to resolve the tenant from recent call events and caller id.

## Environment Variables Used
- `FASTAPI_BASE_URL`
- `INTERNAL_API_KEY`
- `TENANT_ID` only as fallback
- `BUSINESS_TIMEZONE` only as fallback
- `LLM_MODEL` only as fallback
- `DEFAULT_TTS_VOICE` only as fallback

## Debug Logging
- Existing live debug logging remains active.
- `session_started` now also includes tenant slug and config version when available.
