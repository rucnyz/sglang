"""T8 — Owner-map provider protocol.

The cross-pool fire planner needs to know, for every page-id in a pool,
which of three things owns it:

  - FREE: page-id is in the allocator's free list. Safe to unmap.
  - TREE: page-id is referenced by a radix-tree node (cached prefix).
          Drainable — evict the tree entry, page returns to free.
  - ACTIVE: page-id is held by an active request (running, paused, or
            already-prefilled-in-queue). Must be migrated, not unmapped.

This module defines the abstract `OwnerProvider` interface plus the
concrete `OwnerMap` data structure. The scheduler-side implementation
lives in `sglang.srt.budgeter.scheduler_owner_provider`; tests use a
synthetic implementation that doesn't need a live scheduler.

Why a Protocol and not a direct import: the actuator + planner sit
under `arena/` and `budgeter/`, both of which must remain importable
without pulling in `managers.scheduler`. The scheduler implements the
interface and injects an instance at BudgetAgent construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Set, Tuple, runtime_checkable


@dataclass
class OwnerMap:
    """Snapshot of page ownership across one pool's page-id space.

    Built once per fire by walking running batch + tree cache + free list.
    The planner consumes it read-only; never mutated after construction.

    Coverage invariant (planner asserts before emitting a FirePlan):
        len(free_pages) + len(tree_pages) + len(active_pages) == n_pages
    """

    pool_name: str
    """Identifier for log correlation, e.g. 'kv' or 'mamba'."""

    n_pages: int
    """Total page-id count in this pool's allocator (== allocator.size)."""

    free_pages: Set[int]
    """Page-ids currently in `allocator.free_pages` ∪ `allocator.release_pages`.
    These are immediately available for new allocs; safe to drain."""

    tree_pages: Dict[int, "TreeNodeRef"] = field(default_factory=dict)
    """Page-id → tree node reference. Eviction of the tree node frees
    the page. Multiple pages share a TreeNodeRef when they belong to the
    same node's `value` list."""

    active_pages: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    """Page-id → (req_pool_idx, slot_in_req). Walking
    `req_to_token_pool.req_to_token[req_pool_idx, :seqlen]` and assigning
    `slot_in_req = enumerate index` reproduces the location the executor
    must update during a migrate."""

    capped_pages: Set[int] = field(default_factory=set)
    """Page-ids currently in `allocator._capped_pages` — held out by a
    previous fire's cap-barrier or by `set_capacity_pages`. Non-empty
    only when a fire is mid-flight; planner refuses to build a new plan
    while any pages are capped."""

    def coverage(self) -> int:
        return (
            len(self.free_pages)
            + len(self.tree_pages)
            + len(self.active_pages)
            + len(self.capped_pages)
        )

    def assert_complete(self) -> None:
        """Ground-truth check: every page-id is owned by exactly one of
        free / tree / active / capped. If this fails, the planner refuses
        to emit a plan rather than risk unmapping pages it cannot account for."""
        cov = self.coverage()
        if cov != self.n_pages:
            raise RuntimeError(
                f"OwnerMap[{self.pool_name}] coverage broken: "
                f"free={len(self.free_pages)} tree={len(self.tree_pages)} "
                f"active={len(self.active_pages)} capped={len(self.capped_pages)} "
                f"(sum={cov}) != n_pages={self.n_pages}. "
                f"Refusing to fire — fix the owner walker before unmapping anything."
            )


@dataclass(frozen=True)
class TreeNodeRef:
    """Opaque handle to a radix-tree node, plus the start offset of the
    page-list slice that owns the page in question. The executor passes
    this back into `tree_cache.evict_node(...)` (or equivalent) to free
    the corresponding pages.

    We intentionally do not import `RadixCache` types here — a node ref
    is whatever the scheduler-side provider hands us, treated as opaque
    by the planner and by the executor's drain step.
    """

    node: object
    """The actual tree node object (RadixCache.Node, BradixNode, ...)."""

    page_offset: int
    """Index into the node's page-id array where this page sits, so the
    executor can locate sibling pages owned by the same node."""


@runtime_checkable
class OwnerProvider(Protocol):
    """Implemented by the scheduler-side wrapper. Called by the planner
    once per fire decision (hundreds of ms apart, not in the forward
    critical path).

    Implementations MUST construct the OwnerMap inside the scheduler
    lock window — i.e. between two forward steps — so the snapshot is
    consistent. Calling outside the lock is a correctness bug.
    """

    def build_kv_owner_map(self) -> OwnerMap: ...

    def build_mamba_owner_map(self) -> Optional[OwnerMap]:
        """Returns None if the build was constructed without a mamba pool
        (pure-attention models). The planner skips mamba-side work in
        that case."""
        ...
