"""
Phase 2e.4.d — KVArenaActuator.

Wraps a (MHATokenToKVPool, BaseTokenToKVPoolAllocator) pair and exposes a
single `set_capacity_tokens(n)` method that resizes both in lockstep.
The actuator is policy-agnostic; the BudgetAgent (or a test harness)
calls it with target capacities computed by some upstream policy.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KVArenaActuator:
    def __init__(self, pool, allocator) -> None:
        self.pool = pool
        self.allocator = allocator
        if not hasattr(pool, "_kv_arena"):
            raise RuntimeError(
                "KVArenaActuator requires a MHATokenToKVPool created with SGLANG_KV_ARENA=1"
            )
        # Cache initial capacities (= max).
        self.max_tokens: int = pool.size
        self.live_tokens: int = pool.size
        logger.info(
            "KVArenaActuator: max_tokens=%d page_size=%d", self.max_tokens, pool.page_size
        )

    def set_capacity_tokens(self, n_tokens: int) -> int:
        """Resize KV pool + allocator to back exactly `n_tokens` of capacity.

        Returns the engine-visible capacity (clamped to allocator.size,
        rounded internally to chunk granularity by the arena).
        """
        n_tokens = max(self.pool.page_size, min(n_tokens, self.max_tokens))
        if n_tokens == self.live_tokens:
            return self.live_tokens

        # Arena rounds up to chunk granularity; we want at LEAST n_tokens
        # backed but the engine's view stays clamped to allocator.size.
        self.pool.set_capacity_tokens(n_tokens)
        # Allocator uses page-id space: clamp to allocator.size.
        page_size = max(1, self.pool.page_size)
        n_pages = min(n_tokens // page_size, self.allocator.size)
        self.allocator.set_capacity_pages(n_pages)
        self.live_tokens = n_pages * page_size
        logger.info(
            "KVArenaActuator: capacity -> %d tokens (%d pages); allocator.size=%d",
            self.live_tokens, n_pages, self.allocator.size,
        )
        return self.live_tokens

    def shrink_fraction(self, frac: float) -> int:
        return self.set_capacity_tokens(int(self.max_tokens * frac))
