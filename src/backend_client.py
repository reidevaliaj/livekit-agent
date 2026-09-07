"""One authenticated connection pool per call, with whole-operation deadlines."""

import asyncio
from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str, token: str, *, transport=None):
        if not token.strip():
            raise ValueError("AGENT_API_TOKEN is required")
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-Agent-Token": token.strip()},
            timeout=httpx.Timeout(4.0, connect=1.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.aclose()

    async def aclose(self):
        await self.http.aclose()

    async def post(
        self, path: str, payload: dict[str, Any], *, budget: float = 4.0
    ) -> dict:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Backend path must be relative to the configured origin")
        async with asyncio.timeout(budget):
            response = await self.http.post(path, json=payload, timeout=budget)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Backend response must be an object")
            return data


def validate_session_config(data: dict) -> dict:
    tenant = data.get("tenant")
    config = data.get("config")
    if (
        data.get("ok") is not True
        or not isinstance(tenant, dict)
        or not tenant.get("id")
    ):
        raise ValueError("Verified tenant session configuration is unavailable")
    if not isinstance(config, dict) or not config.get("business_name"):
        raise ValueError("Business session configuration is unavailable")
    return data
