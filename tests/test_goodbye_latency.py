import asyncio
from types import SimpleNamespace

import pytest

from receptionist import Receptionist


@pytest.mark.asyncio
async def test_goodbye_starts_while_call_details_are_being_saved():
    save_started = asyncio.Event()
    goodbye_started = asyncio.Event()
    release_save = asyncio.Event()

    async def post(*args, **kwargs):
        save_started.set()
        await release_save.wait()
        return {"ok": True, "accepted": True}

    async def finish():
        goodbye_started.set()

    agent = Receptionist(
        {
            "tenant": {"id": "test"},
            "config": {"business_name": "Example", "timezone": "UTC"},
        },
        "incoming",
        "room",
        SimpleNamespace(post=post),
        SimpleNamespace(emit=lambda *args, **kwargs: None),
        finish,
    )
    task = asyncio.create_task(agent.call_end())
    try:
        await asyncio.wait_for(save_started.wait(), timeout=1)
        await asyncio.wait_for(goodbye_started.wait(), timeout=0.2)
    finally:
        release_save.set()
        result = await task
    assert result["details_saved"] is True
