"""Scheduler-side OwnerProvider.

Builds a per-pool `OwnerMap` at *page granularity* (one entry per
2 MiB cuMem physical handle). The token-slot bookkeeping in SGLang's
allocator is translated into page state inside this provider; the
planner above only sees pages.

A page is "fully free" when every constituent token-slot is in the
allocator's free list (the SGLang allocator hands out one token-slot
at a time; one page contains many slots). Other pages — those with
any tree-cached or live token — are not considered for the current
free-only planner; cost-driven extensions that handle them are out of
scope here.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.arena.owner_provider import OwnerMap, OwnerProvider

logger = logging.getLogger(__name__)


class SchedulerOwnerProvider:
    """Implements `OwnerProvider` against a live `Scheduler`."""

    def __init__(self, scheduler, kv_actuator, mamba_actuator=None) -> None:
        self._scheduler = scheduler
        self._kv_actuator = kv_actuator
        self._mamba_actuator = mamba_actuator

    # ------------------------------------------------------------------

    def build_kv_owner_map(self) -> OwnerMap:
        kv_act = self._kv_actuator
        allocator = self._scheduler.token_to_kv_pool_allocator
        n_pages = int(kv_act.n_pages)
        free_token_set = self._free_token_set(allocator)
        free_pages = {
            p for p in range(n_pages)
            if kv_act.page_is_fully_free(p, free_token_set)
        }
        return OwnerMap(pool_name="kv", n_pages=n_pages, free_pages=free_pages)

    def build_mamba_owner_map(self) -> Optional[OwnerMap]:
        mamba_act = self._mamba_actuator
        if mamba_act is None:
            return None
        mamba_pool = mamba_act.pool
        n_pages = int(mamba_act.n_pages)
        free_token_set = self._tensor_to_set(getattr(mamba_pool, "free_slots", None))
        capped_set = self._tensor_to_set(getattr(mamba_pool, "_capped_slots", None))
        # Pages mid-fire (any token-slot in capped) are not "fully free".
        if capped_set:
            free_token_set = free_token_set - capped_set
        free_pages = {
            p for p in range(n_pages)
            if mamba_act.page_is_fully_free(p, free_token_set)
        }
        return OwnerMap(pool_name="mamba", n_pages=n_pages, free_pages=free_pages)

    # ------------------------------------------------------------------

    @staticmethod
    def _free_token_set(allocator) -> set:
        free = set()
        free |= SchedulerOwnerProvider._tensor_to_set(getattr(allocator, "free_pages", None))
        free |= SchedulerOwnerProvider._tensor_to_set(getattr(allocator, "release_pages", None))
        # Pages the actuator already capped should NOT count as free for
        # planning (they're held out mid-fire by a previous tick).
        return free

    @staticmethod
    def _tensor_to_set(t: Optional[torch.Tensor]) -> set:
        if t is None or t.numel() == 0:
            return set()
        return {int(x) for x in t.cpu().tolist()}


# Structural Protocol check (cheap, runtime).
assert isinstance(  # noqa: S101
    SchedulerOwnerProvider.__new__(SchedulerOwnerProvider), OwnerProvider
)
