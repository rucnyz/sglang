"""T8 — Cross-pool fire plan types.

A `FirePlan` is the unit of work the cross-pool actuator executes during a
single fire (e.g. shrink KV by N pages, grow mamba by M chunks). Every
decision the actuator might have made — which chunks to unmap, which tree
refs to drain, which active-req pages to migrate — is resolved by the
scheduler-side planner before the actuator is invoked. The actuator's job
becomes a deterministic sequence of physical ops with no fallbacks.

This module defines the types only. The planner that produces them lives
in `sglang.srt.budgeter.fire_planner`; the executor that consumes them
lives in `sglang.srt.arena.cross_pool_actuator.CrossPoolActuator.execute`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class MigrateOp:
    """One active-req page that must be relocated before its host chunk
    can be unmapped.

    The planner has already reserved `dst_page` from the head (un-capped)
    region of the same allocator. The executor:
      1. D2D-copies the KV slice from `src_page` to `dst_page`.
      2. Atomically writes `dst_page` into `req_to_token[req_pool_idx, slot_in_req]`.
      3. Returns `src_page` to the allocator's free list (page is still
         capped, so it cannot be re-allocated before the unmap step).
    """

    src_page: int
    dst_page: int
    req_pool_idx: int
    slot_in_req: int


@dataclass(frozen=True)
class FirePlan:
    """Fully-specified single-direction transfer.

    All page-id and chunk-id values are in the SOURCE pool's space; the
    destination pool only needs to know how many chunks to grow.

    Invariants the planner must guarantee at construction time, the
    executor may `assert` at execute time:
      - `capped_page_range[0] <= every page in chunks_to_unmap_src*tpc < capped_page_range[1]`
      - `set(pages_to_drain) ∩ set(p.src_page for p in pages_to_migrate) == ∅`
      - every page in [pages_to_drain ∪ pages_to_migrate] lies inside
        `capped_page_range`
      - every active-req-owned page in `capped_page_range` is covered by
        exactly one `MigrateOp`
      - `expected_unmap_pages == sum(chunk.tpc for chunk in chunks_to_unmap_src)`
    """

    direction: str
    """One of 'kv_to_mamba', 'mamba_to_kv'."""

    capped_page_range: Tuple[int, int]
    """[low, high) page-id interval to mark as capped on the source allocator
    BEFORE drain/migrate runs. This is the cap-barrier — once set, no new
    alloc can land in the range, eliminating the alloc/unmap race."""

    chunks_to_unmap_src: List[int]
    """Chunk ids on the source arena to physically unmap once drain+migrate
    have emptied them. The planner has verified each chunk's pages will all
    be in the free-set by the time the executor reaches the unmap step."""

    pages_to_drain: List[int]
    """Page-ids in `capped_page_range` currently owned by radix-tree nodes
    (cached prefix). The executor evicts the referencing tree entries; the
    pages return to the (capped) free set. No physical copy."""

    pages_to_migrate: List[MigrateOp]
    """Page-ids in `capped_page_range` currently owned by an active req.
    Each entry tells the executor where to copy KV state and which req
    slot to update so the req keeps running uninterrupted."""

    chunks_to_map_dst: int
    """Number of chunks to grow on the destination arena AFTER source unmap
    completes. The destination subpool name is implied by `direction`."""

    expected_unmap_pages: int
    """Sanity check: the executor asserts that the actual page count freed
    by `arena.shrink_explicit(chunks_to_unmap_src)` matches this. If they
    diverge, something has changed pool state behind the planner's back —
    the executor aborts and the scheduler picks a fresh plan next tick."""

    plan_seq: int
    """Monotonic counter for log correlation. The planner increments it
    once per FirePlan emitted; the executor stamps it into the
    budgeter.jsonl row so a crash log can be matched back to the plan."""


@dataclass
class FirePlanResult:
    """What `execute(plan)` returns. Captured into budgeter.jsonl."""

    plan_seq: int
    direction: str
    unmapped_pages: int
    granted_chunks: int
    drained_pages: int
    migrated_pages: int
    cap_barrier_us: int
    drain_us: int
    migrate_us: int
    unmap_us: int
    map_us: int
    total_us: int
    aborted: bool = False
    abort_reason: str = ""
