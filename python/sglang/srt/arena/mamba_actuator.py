"""
Phase 2e.5.6.3 — MambaArenaActuator.

Mirror of `KVArenaActuator` for `MambaPool`. Wraps a single MambaPool
instance (built with `SGLANG_MAMBA_ARENA=1`, optionally with
`SGLANG_ARENA_SHARED=1`) and exposes a single `set_capacity_tokens(n)`
that resizes both the underlying `MultiTensorArena` and MambaPool's
internal slot allocator (`free_slots` / `_capped_slots`) in lockstep.

The actuator is policy-agnostic: an upstream policy (planner /
budgeter / unit test) decides the target capacity. The cross-pool
actuator wires the KV and mamba actuators together.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class MambaArenaActuator:
    def __init__(self, pool) -> None:
        self.pool = pool
        if not hasattr(pool, "_mamba_temporal_arena"):
            raise RuntimeError(
                "MambaArenaActuator requires a MambaPool created with "
                "SGLANG_MAMBA_ARENA=1 (or SGLANG_ARENA_SHARED=1)."
            )
        if not hasattr(pool, "set_capacity_slots"):
            raise RuntimeError(
                "MambaPool is missing set_capacity_slots — Phase 2e.5.6.3 "
                "wiring not present in this build."
            )
        # Cache initial slot capacity (= MambaPool.size).
        self.max_slots: int = pool.size
        self.live_slots: int = pool.size
        # Tokens-per-slot: 1 in MambaPool's accounting (each slot stores
        # one full sequence's mamba state). The arena's tokens_per_chunk
        # is in slot units too.
        self.tokens_per_slot: int = 1
        logger.info(
            "MambaArenaActuator: max_slots=%d, tokens_per_slot=%d",
            self.max_slots, self.tokens_per_slot,
        )

    def set_capacity_tokens(self, n_tokens: int) -> int:
        """Resize MambaPool's live capacity to `n_tokens` slots.

        Returns the post-resize live capacity (clamped to max_slots).
        Wraps two operations in lockstep:
          1. `MambaPool.set_capacity_slots(n)`: caps the slot allocator
             so future allocs don't return ids in the soon-to-be-unmapped
             tail.
          2. The arena's `set_capacity_tokens` would change the physical
             chunk mapping, but for cross-pool transfer the chunk move
             happens via `cross_arena_transfer`, NOT via this actuator.
             So we don't call arena.set_capacity_tokens here. The cross-
             pool actuator orchestrates the order: cap allocator → do
             the cross-arena chunk move → (or grow) → uncap allocator.
        """
        n_tokens = max(1, min(n_tokens, self.max_slots))
        if n_tokens == self.live_slots:
            return self.live_slots
        n_slots = self.pool.set_capacity_slots(n_tokens)
        self.live_slots = n_slots
        return self.live_slots

    def live_capacity_tokens(self) -> int:
        return self.live_slots

    @property
    def n_pages(self) -> int:
        """Total physical page count for this pool. Mamba's recurrent
        state is one slot per req per layer, so one page typically holds
        one slot — but the same accessor convention matches KVArenaActuator."""
        return int(self.pool.size) // self._tokens_per_page()

    def _tokens_per_page(self) -> int:
        """Internal: tokens-per-page for the mamba arena (== slots-per-
        chunk; typically 1 under T1 page-grain VMM)."""
        arena = getattr(self.pool, "_mamba_temporal_arena", None)
        if arena is None:
            raise RuntimeError(
                "MambaArenaActuator: pool has no _mamba_temporal_arena "
                "(SGLANG_ARENA_SHARED?)."
            )
        tpc = getattr(arena, "tokens_per_chunk", None)
        if tpc is None:
            raise RuntimeError(
                "MambaArenaActuator: _mamba_temporal_arena lacks tokens_per_chunk."
            )
        return int(tpc)

    def expand_pages_to_token_slots(self, page_ids):
        """Translate page-ids → token-slot ids for the mamba allocator.
        See KVArenaActuator.expand_pages_to_token_slots."""
        tps = self._tokens_per_page()
        out = []
        for p in page_ids:
            out.extend(range(p * tps + 1, (p + 1) * tps + 1))
        return out

    def page_is_fully_free(self, page_id: int, free_token_set: set) -> bool:
        tps = self._tokens_per_page()
        for s in range(page_id * tps + 1, (page_id + 1) * tps + 1):
            if s not in free_token_set:
                return False
        return True

    def cap_allocator_only(self, n_tokens: int) -> int:
        """Phase 2e.5.6.3: same contract as KVArenaActuator.cap_allocator_only.
        For MambaPool this is identical to set_capacity_tokens because
        MambaPool.set_capacity_slots already touches only the slot
        free-list — it does not call into the underlying MultiTensorArena.
        Provided as a separate name so the cross-pool actuator can call
        the same method on either side.
        """
        return self.set_capacity_tokens(n_tokens)
