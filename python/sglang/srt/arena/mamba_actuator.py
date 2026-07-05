"""
MambaArenaActuator: mirror of `KVArenaActuator` for `MambaPool`. Wraps a
single MambaPool instance (built with `SGLANG_MAMBA_ARENA=1`, optionally
with `SGLANG_ARENA_SHARED=1`) and exposes the per-pool fire surface
(`expand_pages_to_token_slots`, `unmark_token_slots`, `migrate_slot`,
`n_pages`) plus the cross-fire `.allocator`.

The actuator is policy-agnostic: an upstream policy (planner /
budgeter) decides the target capacity. The cross-pool actuator wires
the KV and mamba actuators together.

Mamba allocator surface: for symmetric mamba-to-kv fire support, the
actuator exposes `.allocator` with the same `mark_pages_capped` /
`unmark_pages_capped` / `.device` API the KV allocator
(`TokenToKVPoolAllocator`) carries. Without it `XPoolActuator.cap_barrier`
would reject a mamba-source fire (`src allocator missing
mark_pages_capped`) and the budgeter's mamba-to-kv path would never fire.
"""
from __future__ import annotations
import logging

import torch

logger = logging.getLogger(__name__)


class _MambaCapAllocator:
    """Cap-allocator surface for MambaPool, mirroring the KV-side
    `TokenToKVPoolAllocator.mark_pages_capped/unmark_pages_capped`
    contract. The pool's slot ids ARE the "pages" here (mamba is one
    slot per request per layer, no page-grain rebatching).

    This parallel surface is what lets direction-symmetric fires work
    end-to-end: a mamba-source (mamba-to-kv) fire needs the source
    allocator to expose `mark_pages_capped`, exactly like the KV side.

    Slot bookkeeping reuses `MambaPool._capped_slots` (already
    maintained by `MambaSlotAllocator.mark`/`unmark`/`set_capacity`
    via CappedFreeList). Mark and unmark delegate directly to the
    allocator.

    Mamba pool size is O(hundreds) of slots, so mutating
    `pool.free_slots` per mark/unmark is cheap (unlike KV, which defers
    via a `_capped_pages` filter to avoid a 19 MB realloc). We keep the
    API symmetric but use the natural mutate-in-place form for mamba.
    """

    def __init__(self, pool) -> None:
        self.pool = pool
        # Device is taken from pool.free_slots (always lives on the
        # pool's compute device). Same convention as KV allocator.
        self.device = pool._allocator.device

    def mark_pages_capped(self, slot_indices: "torch.Tensor") -> int:
        """Delegate to MambaSlotAllocator.mark (CappedFreeList)."""
        if slot_indices is None or slot_indices.numel() == 0:
            return 0
        return self.pool._allocator.mark(slot_indices.to(self.device).to(torch.int64))

    def unmark_pages_capped(self, slot_indices: "torch.Tensor") -> int:
        """Delegate to MambaSlotAllocator.unmark (CappedFreeList)."""
        if slot_indices is None or slot_indices.numel() == 0:
            return 0
        return self.pool._allocator.unmark(slot_indices.to(self.device).to(torch.int64))

    def count_reachable_capped(self, cap_t: "torch.Tensor") -> int:
        """Delegate to MambaSlotAllocator.count_reachable_capped."""
        return self.pool._allocator.count_reachable_capped(
            cap_t.to(self.device).to(torch.int64)
        )

    def count_referenced(self, cap_t: "torch.Tensor") -> int:
        """How many target slots are NOT free (still backing live/cached state)?"""
        target = cap_t.to(self.device).to(torch.int64)
        if target.numel() == 0:
            return 0
        free = self.pool._allocator.free_slots
        if free.numel() == 0:
            return int(target.numel())
        return int((~torch.isin(target, free)).sum().item())


class MambaArenaActuator:
    def __init__(self, pool) -> None:
        self.pool = pool
        if pool._mamba_temporal_arena is None:
            raise RuntimeError(
                "MambaArenaActuator requires a MambaPool created with "
                "SGLANG_MAMBA_ARENA=1 (or SGLANG_ARENA_SHARED=1)."
            )
        if not hasattr(getattr(pool, "_allocator", None), "set_capacity"):
            raise RuntimeError(
                "MambaPool._allocator is missing set_capacity — "
                "CappedFreeList wiring not present in this build."
            )
        # Cap actuator at the ARENA's max possible chunks (VA upper
        # bound), not MambaPool's current size. Otherwise cross-pool
        # fires can't grow mamba past init even though there's
        # reserved VA waiting. The arena pre-reserves up to
        # SGLANG_ARENA_MAMBA_HEADROOM_BYTES (default 80 GiB).
        # MambaPool itself is allocated with conv_state + temporal_state
        # at max_size so it can address the new slots.
        arena = pool._mamba_temporal_arena
        self.max_slots: int = int(
            arena.max_chunks_per_pool * arena.tokens_per_chunk
        )
        # Tokens-per-slot: 1 in MambaPool's accounting (each slot stores
        # one full sequence's mamba state). The arena's tokens_per_chunk
        # is in slot units too.
        self.tokens_per_slot: int = 1
        # Cap-allocator surface for mamba-source fires. Exposes the same
        # mark/unmark contract `XPoolActuator.cap_barrier` calls on the KV
        # side; required for a mamba-to-kv fire to clear cap_barrier.
        self.allocator = _MambaCapAllocator(pool)
        logger.info(
            "MambaArenaActuator: max_slots=%d, tokens_per_slot=%d, "
            "allocator=_MambaCapAllocator (mamba-source cap_barrier supported)",
            self.max_slots, self.tokens_per_slot,
        )

    @property
    def n_pages(self) -> int:
        """Total physical page count for this pool. Mamba's recurrent
        state is one slot per req per layer, so one page typically holds
        one slot — but the same accessor convention matches KVArenaActuator."""
        return int(self.pool.size) // self._tokens_per_page()

    def _tokens_per_page(self) -> int:
        """Internal: tokens-per-page for the mamba arena (== slots-per-
        chunk; typically 1 under page-grain VMM).

        `pool._mamba_temporal_arena` is guaranteed non-None by the
        `__init__` invariant check; `tokens_per_chunk` is part of
        MultiTensorArena's public contract. Direct access; any
        AttributeError is a structural break, crash loudly.
        """
        return int(self.pool._mamba_temporal_arena.tokens_per_chunk)

    def expand_pages_to_token_slots(self, page_ids):
        """Translate page-ids → token-slot ids for the mamba allocator.
        Page p backs slots [p*tps, (p+1)*tps).

        Page 0 is rejected loudly: chunk 0 carries padded slot 0 (see
        design.md §"Per-unit sizes"); unmapping it corrupts the
        padded-output target. The bug is especially severe for mamba
        where tps==1 makes `expand([0])` return the empty list, so the
        actuator would silently unmap chunk 0 without marking any slot.
        See KVArenaActuator counterpart.
        """
        tps = self._tokens_per_page()
        out = []
        for p in page_ids:
            if int(p) == 0:
                raise ValueError(
                    "expand_pages_to_token_slots: page 0 carries "
                    "padded slot 0 (see design.md §\"Per-unit sizes\"); "
                    "unmapping chunk 0 corrupts the padded-output "
                    "target. With mamba tps=1, this also silently "
                    "dropped chunk 0 entirely."
                )
            out.extend(range(p * tps, (p + 1) * tps))
        return out

    def page_is_fully_free(self, page_id: int, free_token_set: set) -> bool:
        if page_id == 0:
            return False  # chunk 0 carries padded slot 0; never unmap it.
        tps = self._tokens_per_page()
        for s in range(page_id * tps, (page_id + 1) * tps):
            if s not in free_token_set:
                return False
        return True

    def unmark_token_slots(self, token_slots) -> None:
        """ID-based dst grow restore. Routes through the allocator
        (``_MambaCapAllocator.unmark_pages_capped`` -> ``CappedFreeList.unmark``
        -> ``_grow_into_tail``) so the restored slots raise
        ``mamba_allocator.live_size`` — the single source of truth the admission
        cap reads (``BudgetAgent._maybe_update_admission_cap``). The engine-visible
        bound ``pool.size`` is then reconciled from the allocator.

        Byte-symmetric with ``KVArenaActuator.unmark_token_slots``; the only
        difference is that ``_MambaCapAllocator`` carries no ``live_size``
        property, so the new bound is read from ``self.pool.live_size`` (which
        delegates to the same ``_allocator``).
        """
        if not token_slots:
            return
        ids = torch.tensor(
            list(token_slots), dtype=torch.int64, device=self.allocator.device,
        )
        self.allocator.unmark_pages_capped(ids)
        new_live = self.pool.live_size
        if new_live > self.pool.size:
            self.pool.size = new_live

    def migrate_slot(self, src: int, dst: int) -> bool:
        """Uniform Stage-3 migration surface, matching
        ``KVArenaActuator.migrate_slot`` so ``XPoolActuator._run_stage0`` is
        pool-agnostic. For mamba the slot state lives on the pool
        (MambaPool is pool+allocator combined), so this delegates to
        ``MambaPool.migrate_slot`` (relocates the recurrent state + swaps
        free/capped state). The owning request's ssm_state_indices pointer
        rewrite is the caller's job (Stage-0 handler)."""
        return self.pool.migrate_slot(int(src), int(dst))

