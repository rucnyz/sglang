"""aginfer GET /aginfer/state response cache — self-contained (#251).

http_server.py holds one ``AginferStateCache`` instance and the ``/aginfer/state``
endpoint is a thin ``.get(tokenizer_manager)`` hook. A background task refreshes
the serialized dump at the configured cadence (per #160: 50ms @ ~5ms dump cost is
<1% scheduler overhead), so the hot endpoint serves cached bytes.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class AginferStateCache:
    def __init__(self, refresh_ms: float):
        self._refresh_s = float(refresh_ms) / 1000.0
        self._lock = threading.Lock()
        self._body: Optional[bytes] = None
        self._media: str = "application/json"
        self._task: Optional[asyncio.Task] = None
        self._started: bool = False

    @staticmethod
    async def _serialise(responses) -> tuple:
        """Single-rank / per_rank branch; returns (body_bytes, media_type)."""
        if len(responses) == 1:
            r = responses[0]
            if r.state_bytes is not None:
                return r.state_bytes, "application/json"
            import orjson
            return orjson.dumps(r.state), "application/json"
        import orjson
        per_rank = [
            orjson.loads(r.state_bytes) if r.state_bytes is not None else r.state
            for r in responses
        ]
        return orjson.dumps({"per_rank": per_rank}), "application/json"

    async def refresh_one(self, tokenizer_manager) -> None:
        """Single refresh tick: fetch from scheduler, update cache. Exceptions
        are logged + swallowed; the cache holds stale data until the next tick."""
        try:
            responses = await tokenizer_manager.get_aginfer_state()
            body, media = await self._serialise(responses)
            with self._lock:
                self._body = body
                self._media = media
        except Exception:  # noqa: BLE001
            logger.exception("aginfer state refresh failed; keeping stale cache")

    async def _refresh_loop(self, tokenizer_manager) -> None:
        while True:
            await self.refresh_one(tokenizer_manager)
            await asyncio.sleep(self._refresh_s)

    async def get(self, tokenizer_manager) -> tuple:
        """Returns (body_bytes, media_type). Lazy-starts the background refresh on
        the first call (single-flight; Python attr write is atomic), with a
        synchronous cold-start fetch so the first response is fresh."""
        if not self._started:
            self._started = True
            await self.refresh_one(tokenizer_manager)
            self._task = asyncio.create_task(
                self._refresh_loop(tokenizer_manager), name="aginfer-state-refresh",
            )
        with self._lock:
            body = self._body
            media = self._media
        if body is None:
            # refresh raised + left cache empty — synchronous fallback so the
            # caller doesn't get a confusing 500.
            responses = await tokenizer_manager.get_aginfer_state()
            body, media = await self._serialise(responses)
        return body, media
