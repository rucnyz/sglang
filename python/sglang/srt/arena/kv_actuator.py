"""
KVArenaActuator: KV-side per-pool actuator for the cross-pool fire path.

Wraps a (MHATokenToKVPool, BaseTokenToKVPoolAllocator) pair and exposes
the per-pool surface XPoolActuator dispatches during a fire:
`expand_pages_to_token_slots` (page-id -> token-slot translation),
`unmark_token_slots` (ID-flow dst grow restore), `migrate_slot`
(Stage-3 slot relocation), and the `n_pages` view. The actuator is
policy-agnostic; the upstream planner decides which pages move.
"""

from __future__ import annotations
import logging

import torch

logger = logging.getLogger(__name__)


class KVArenaActuator:
    def __init__(self, pool, allocator) -> None:
        self.pool = pool
        self.allocator = allocator
        if pool._kv_arena is None:
            raise RuntimeError(
                "KVArenaActuator requires a MHATokenToKVPool created with SGLANG_KV_ARENA=1"
            )
        # Growable ceiling = the ARENA's max_tokens (VA-reserved,
        # physical-on-demand), NOT the boot pool.size. Capping at pool.size
        # would hard-cap KV at boot, so a mamba-to-kv fire (grow KV by
        # shrinking mamba) would shrink mamba (a real cache loss) while KV
        # could never grow: pure regression. The KV allocator carries the
        # matching `[boot, arena_max]` dynamic-cap headroom; actual growth
        # past boot is gated downstream by available shared physical handles
        # (mamba donations). `_kv_arena` is guaranteed non-None by the
        # __init__ invariant above.
        self.max_tokens: int = int(pool._kv_arena.max_tokens)
        logger.info(
            "KVArenaActuator: max_tokens=%d (arena ceiling) page_size=%d",
            self.max_tokens, pool.page_size,
        )

    @property
    def n_pages(self) -> int:
        """Total physical page count for this pool (= number of 2 MiB cuMem
        handles owned). Higher layers reason in pages, not token-slots."""
        return int(self.pool.size) // self._tokens_per_page()

    def grow_headroom_pages(self) -> int:
        """Physical page headroom before the KV allocator hits its id-space
        ceiling (max_size). A cross-fire dst grant must be clamped to this so
        dst chunk ids never expand past CappedFreeList.size. Symmetric with
        MambaArenaActuator.grow_headroom_pages."""
        alloc = self.allocator
        headroom_slots = max(0, alloc.max_size - alloc.live_size)
        return headroom_slots // self._tokens_per_page()

    def _tokens_per_page(self) -> int:
        """Internal: how many SGLang allocator token-slots live in one
        physical page. Consumed by `expand_pages_to_token_slots`,
        `page_is_fully_free`, and the `n_pages` property; never exposed to
        the planner.

        `pool._kv_arena` is guaranteed non-None by `__init__`'s
        invariant check; `tokens_per_chunk` is part of MultiTensorArena's
        public contract. Direct access; any AttributeError here is a
        structural break and should crash loudly.
        """
        return int(self.pool._kv_arena.tokens_per_chunk)

    def expand_pages_to_token_slots(self, page_ids):
        """Translate page-ids (one per 2 MiB cuMem handle) to the
        token-slot ids the SGLang allocator uses. Page p physically
        backs slot range [p*tps, (p+1)*tps), where tps =
        tokens-per-page.

        Page 0 is rejected loudly: chunk 0 carries padded slot 0 (see
        design.md §"Per-unit sizes") and must remain mapped.
        `_compute_fully_free_pages` upstream already filters page 0
        out of the candidate set, so any page_id == 0 reaching here
        is a structural break.
        """
        tps = self._tokens_per_page()
        out = []
        for p in page_ids:
            if int(p) == 0:
                raise ValueError(
                    "expand_pages_to_token_slots: page 0 carries "
                    "padded slot 0 (see design.md §\"Per-unit sizes\"); "
                    "unmapping chunk 0 corrupts the padded-output "
                    "target. Caller selected page 0; fix the planner "
                    "/ OwnerProvider."
                )
            out.extend(range(p * tps, (p + 1) * tps))
        return out

    def page_is_fully_free(self, page_id: int, free_token_set: set) -> bool:
        """Check whether every token-slot in `page_id` is in
        `free_token_set`. Page p backs slots [p*tps, (p+1)*tps).
        Page 0 always returns False: chunk 0 carries padded slot 0 and
        must never be unmapped (design.md §"Production invariants on the
        FREE↔CAPPED split").

        A loud per-page page-0 guard / unit-test reference; production
        fully-free computation is the vectorized
        `SchedulerOwnerProvider._compute_fully_free_pages`, not a per-page
        loop."""
        if page_id == 0:
            return False
        tps = self._tokens_per_page()
        for s in range(page_id * tps, (page_id + 1) * tps):
            if s not in free_token_set:
                return False
        return True

    def unmark_token_slots(self, token_slots) -> None:
        """ID-based dst grow restore. Drop the given token slots from
        ``allocator._capped_pages`` so subsequent ``alloc()`` can hand
        them out again. The IDs come from
        ``expand_pages_to_token_slots(arena.grow_returned_chunk_ids)``,
        so they are exactly the slots whose backing chunks were just
        cuMemMap'd by the cross-pool actuator. Uniform dispatch surface
        with ``MambaArenaActuator.unmark_token_slots``.

        For KV, allocator state is the source of truth: the engine-visible
        cap stays at ``pool.size``, and freshly available capacity is
        tracked by ``allocator.live_size`` = size − (the implicit
        reserved-headroom tail + the cross-fire ``marks``); the tail is
        never materialized on the hot path.
        """
        if not token_slots:
            return
        ids = torch.tensor(
            list(token_slots), dtype=torch.int64, device=self.allocator.device,
        )
        self.allocator.unmark_pages_capped(ids)
        new_live = self.allocator.live_size
        if new_live > self.pool.size:
            self.pool.size = new_live

    def migrate_slot(self, src: int, dst: int) -> bool:
        """Uniform Stage-3 migration surface, matching
        ``MambaArenaActuator.migrate_slot`` so ``XPoolActuator._run_stage0``
        is pool-agnostic. For KV the slot state lives on the allocator (the
        pool holds only the k/v buffers), so this delegates to
        ``TokenToKVPoolAllocator.migrate_slot`` — which relocates the slot's
        bytes (via the kvcache's ``move_kv_cache``) and swaps the free/cap
        state. The owning request's ``req_to_token`` pointer rewrite is the
        caller's job (Stage-0 handler), same as the mamba side."""
        return self.allocator.migrate_slot(int(src), int(dst))

