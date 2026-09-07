"""Outgoing telephone worker entrypoint."""

import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, JobContext, cli

from voice_runtime import on_session_end, run_call

load_dotenv(".env.local")
server = AgentServer(
    host="127.0.0.1",
    num_idle_processes=int(os.getenv("AGENT_NUM_IDLE_PROCESSES", "1")),
    load_threshold=float(os.getenv("AGENT_LOAD_THRESHOLD", "0.95")),
    port=int(os.getenv("OUTGOING_AGENT_HTTP_PORT", "8082")),
    session_end_timeout=20.0,
)


@server.rtc_session(
    agent_name=os.getenv("OUTGOING_AGENT_NAME", "outgoing-agent"),
    on_session_end=on_session_end,
)
async def outgoing_agent(ctx: JobContext):
    await run_call(ctx, "outgoing")


if __name__ == "__main__":
    cli.run_app(server)
