"""Cross-pool fire plan types.

A `FirePlan` is the unit of work the cross-pool actuator executes during a
single fire (e.g. shrink KV by N pages, grow mamba by M pages). Every
decision the actuator might make --- which pages to unmap --- is resolved
by the scheduler-side planner before the actuator is invoked. The actuator
becomes a deterministic sequence of physical ops with no fallbacks.

In our design "page" denotes one 2 MiB cuMem physical handle (the unmap
unit). The underlying SGLang allocator's per-token-slot details are hidden
inside the actuator's allocator-translation layer; the planner, the plan,
and the executor's main loop reason in pages exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FirePlan:
    """Fully-specified single-direction transfer.

    All page-ids are in the source pool's page space (one entry per
    2 MiB physical handle). The destination pool only needs to know how
    many pages to grow.

    Invariants the planner guarantees:
      - every page in `pages_to_unmap` is currently in the source's
        free-page set (no in-flight token references it) AFTER Stage-0
        pre-conditioning (`drains` evicted CACHED→FREE, `migrations`
        relocated LIVE→FREE);
      - `len(pages_to_unmap) == pages_to_map_dst` (handle conservation).
    """

    direction: str
    """One of 'kv_to_mamba', 'mamba_to_kv'."""

    pages_to_unmap: List[int]
    """Source-pool page-ids to physically unmap. These are 2 MiB cuMem
    handles; the executor calls `arena.shrink_explicit(name, ids)` and
    `cuMemUnmap` on each."""

    pages_to_map_dst: int
    """Number of pages to grow on the destination pool after unmap."""

    plan_seq: int
    """Monotonic counter for log correlation."""

    drains: Tuple[int, ...] = ()
    """Stage-2 (Drain-expansion) pages: CACHED page-ids the actuator's
    Stage-0 must evict (CACHED→FREE) before cap-barrier so they become
    harvestable. Empty for free-only plans (default) → Stage-0 skipped,
    byte-identical to the anywhere-free path (design.md §"Transfer
    protocol": Stage 0 is skipped when only FREE pages were selected)."""

    migrations: Tuple[Tuple[int, int], ...] = ()
    """Stage-3 (Migration-expansion) MOVES: `(src_slot, dst_slot)` pairs
    the actuator's Stage-0 must relocate (LIVE→FREE via byte-exact
    `MambaPool.migrate_slot(src, dst)` + `ssm_state_indices` rewrite)
    before cap-barrier. The planner assigns each `dst_slot` from the
    SCATTERED free slots on KEPT pages (never a page in `pages_to_unmap`,
    never a whole-free transfer page), so Stage-0 runs the exact
    relocation with no destination guessing. Completing every move for a
    freed page empties it; that page is in `pages_to_unmap`. Empty for
    free-only / drain-only plans (default)."""


@dataclass
class FirePlanResult:
    """What `execute(plan)` returns. Captured into budgeter.jsonl."""

    plan_seq: int
    direction: str
    unmapped_pages: int
    granted_pages: int
    cap_barrier_us: int
    unmap_us: int
    map_us: int
    total_us: int
    aborted: bool = False
    abort_reason: str = ""
