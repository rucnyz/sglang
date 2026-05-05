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

    @property
    def n_pages(self) -> int:
        """Total physical page count for this pool (= number of 2 MiB cuMem
        handles owned). Higher layers reason in pages, not token-slots."""
        return int(self.pool.size) // self._tokens_per_page()

    def _tokens_per_page(self) -> int:
        """Internal: how many SGLang allocator token-slots live in one
        physical page. Used only by `expand_pages_to_token_slots` and
        `chunks_for_pages` — never exposed to the planner."""
        arena = getattr(self.pool, "_kv_arena", None)
        if arena is None:
            raise RuntimeError(
                "KVArenaActuator: pool has no _kv_arena (SGLANG_KV_ARENA / "
                "SGLANG_ARENA_SHARED not set?)."
            )
        tpc = getattr(arena, "tokens_per_chunk", None)
        if tpc is None:
            raise RuntimeError(
                "KVArenaActuator: _kv_arena lacks tokens_per_chunk attribute."
            )
        return int(tpc)

    def expand_pages_to_token_slots(self, page_ids):
        """Translate page-ids (one per 2 MiB cuMem handle) to the
        token-slot ids the SGLang allocator uses. Page p contains
        token-slots [p * tps + 1, (p+1) * tps + 1), where tps =
        tokens-per-page; slot 0 is the null sentinel.
        """
        tps = self._tokens_per_page()
        out = []
        for p in page_ids:
            out.extend(range(p * tps + 1, (p + 1) * tps + 1))
        return out

    def page_is_fully_free(self, page_id: int, free_token_set: set) -> bool:
        """Check whether every token-slot in `page_id` is in
        `free_token_set`. Used by OwnerProvider to compute fully-free
        pages."""
        tps = self._tokens_per_page()
        for s in range(page_id * tps + 1, (page_id + 1) * tps + 1):
            if s not in free_token_set:
                return False
        return True

    def live_capacity_tokens(self) -> int:
        """Phase 2e.5.6.3: uniform getter so CrossPoolTransferActuator can
        use either KVArenaActuator or MambaArenaActuator interchangeably.
        """
        return self.live_tokens

    def cap_allocator_only(self, n_tokens: int) -> int:
        """Phase 2e.5.6.3: cap ONLY the allocator's free-page list,
        without touching the underlying MultiTensorArena's physical
        chunk mapping. Used by `CrossPoolTransferActuator` which
        orchestrates arena shrink/grow itself via `cross_arena_transfer`
        (calling the regular `set_capacity_tokens` here would shrink
        the arena a second time, leaking handles to the shared free
        pool).
        """
        n_tokens = max(self.pool.page_size, min(n_tokens, self.max_tokens))
        page_size = max(1, self.pool.page_size)
        n_pages = min(n_tokens // page_size, self.allocator.size)
        self.allocator.set_capacity_pages(n_pages)
        self.live_tokens = n_pages * page_size
        return self.live_tokens
