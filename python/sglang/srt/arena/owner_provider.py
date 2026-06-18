"""Page ownership map for cross-pool transfer planning.

In our design "page" = one 2 MiB cuMem physical handle (the unmap unit).
The OwnerMap is a per-page summary of which pages can be safely unmapped:

  - free: every token-slot inside this page is in the allocator's free
          list. Page can be unmapped without losing data.
  - non-free: at least one token-slot is held by a tree-cached prefix or
          a live request. Page cannot be unmapped without first evicting
          or migrating those tokens (cost-driven extension; not used by
          the current planner which only picks free pages).

This module exposes the `OwnerMap` shape and the `OwnerProvider` Protocol.
The scheduler-side implementation that walks the engine state lives in
`sglang.srt.budgeter.scheduler_owner_provider`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Set, Tuple, runtime_checkable


@dataclass
class OwnerMap:
    """Per-page summary for one pool.

    `n_pages` and `free_pages` are at *page granularity* — one entry per
    2 MiB cuMem handle. Token-slot bookkeeping happens inside the
    OwnerProvider before this map is built.
    """

    pool_name: str
    """Identifier for log correlation, e.g. 'kv' or 'mamba'."""

    n_pages: int
    """Total page count in this pool (== number of physical handles)."""

    free_pages: Set[int] = field(default_factory=set)
    """Page-ids currently fully free (every constituent token-slot is in
    the allocator's free list)."""

    cached_pages_in_cost_order: Optional[List[int]] = None
    """Stage-2 (Drain-expansion, design.md §"Page selection") candidate
    pages: CACHED page-ids in sglang's ACTIVE eviction order (cheapest
    victim first — LRU `last_access_time` or LPB `ℓ(b)`). Populated ONLY
    when the planner asks for drain expansion (`allow_drain=True`);
    `None` for Stage-1-only callers so the radix-tree walk stays
    zero-cost on the common free-only path."""

    live_pages_in_cost_order: Optional[
        List[Tuple[int, Tuple[Tuple[int, int], ...]]]
    ] = None
    """Stage-3 (Migration-expansion, design.md §"Page selection")
    candidates, ordered by ASCENDING per-page migration cost `c_m`. NOTE:
    unlike `cached_pages_in_cost_order` (plain page-ids), each entry is
    `(freed_page_id, ((src_slot, dst_slot), ...))` — the page plus the
    concrete byte-exact slot relocations that empty it, so Stage-0 needs no
    destination guessing (`fire_planner.build` unpacks `for pid, moves in
    live`). Populated ONLY when the planner asks for migrate expansion
    (`allow_migrate=True`); `None` otherwise so the common path pays
    nothing."""


@runtime_checkable
class OwnerProvider(Protocol):
    """Scheduler-side wrapper. Called by the planner once per fire
    decision; constructs an `OwnerMap` consistent with the engine's
    current state.

    Implementations MUST construct the OwnerMap inside the scheduler
    lock window — i.e., between two forward steps — so the snapshot is
    consistent.
    """

    def build_kv_owner_map(
        self, *, allow_drain: bool = False, allow_migrate: bool = False,
        max_drain_pages: Optional[int] = None,
    ) -> OwnerMap:
        """`allow_drain` / `allow_migrate` request the Stage-2 / Stage-3
        cost-order candidate lists (`cached_pages_in_cost_order` /
        `live_pages_in_cost_order`). When both False (the common free-
        only fire path) the cost-order radix walk is skipped — the
        returned lists stay None so Stage-1-only callers pay nothing.

        `max_drain_pages` bounds the Drain-expansion victim
        materialization to that many fully-covered pages (the cost-order
        prefix the planner consumes for an `n_pages_target`-page fire), so
        the per-fire scheduler-thread walk scales with the fire magnitude,
        not the whole evictable set. None ⇒ the full cost-ordered list
        (cost-pricing callers that need the complete victim sequence)."""
        ...

    def build_mamba_owner_map(
        self, *, allow_drain: bool = False, allow_migrate: bool = False,
        max_drain_pages: Optional[int] = None,
    ) -> Optional[OwnerMap]:
        """Returns None if the engine has no mamba pool. See
        `build_kv_owner_map` for the `allow_drain` / `allow_migrate` /
        `max_drain_pages` expansion flags."""
        ...

    def n_free_source_pages(self, direction: str) -> int:
        """Count of fully-free SOURCE pages a fire in `direction` can harvest
        free-first (Stage-1), without the cost-order walk — the source pool is
        KV for `kv_to_mamba`, mamba for `mamba_to_kv`. The single free-supply
        signal the drain-cost pricing and the actuator's Stage-1 selection
        share, so a free-harvest fire is priced as the zero-drain it
        executes. 0 when the direction's source pool is absent."""
        ...
