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
from typing import List


@dataclass(frozen=True)
class FirePlan:
    """Fully-specified single-direction transfer.

    All page-ids are in the source pool's page space (one entry per
    2 MiB physical handle). The destination pool only needs to know how
    many pages to grow.

    Invariants the planner guarantees:
      - every page in `pages_to_unmap` is currently in the source's
        free-page set (no in-flight token references it);
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
