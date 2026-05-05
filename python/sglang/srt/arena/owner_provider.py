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
from typing import Optional, Protocol, Set, runtime_checkable


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


@runtime_checkable
class OwnerProvider(Protocol):
    """Scheduler-side wrapper. Called by the planner once per fire
    decision; constructs an `OwnerMap` consistent with the engine's
    current state.

    Implementations MUST construct the OwnerMap inside the scheduler
    lock window — i.e., between two forward steps — so the snapshot is
    consistent.
    """

    def build_kv_owner_map(self) -> OwnerMap: ...

    def build_mamba_owner_map(self) -> Optional[OwnerMap]:
        """Returns None if the engine has no mamba pool."""
        ...
