"""Builds concrete ``FirePlan`` objects from the high-level "shrink
pool A by N pages, grow pool B" intent that ``XPoolPlanner.decide``
already produced.

The planner is ground-truth aware: it consults an ``OwnerProvider`` so
the plan never references an active-req page without an accompanying
migrate. Allocator placement bias keeps live → head, free → tail, so
tail chunks are the common case; this module picks pages from the
free-page set returned by ``OwnerMap``.
"""

from __future__ import annotations

import logging
from typing import Optional

from sglang.srt.arena.fire_plan import FirePlan
from sglang.srt.arena.owner_provider import OwnerProvider

logger = logging.getLogger(__name__)


class XPoolFirePlanner:
    """Builds FirePlans. Stateless beyond a monotonic plan_seq counter.

    Construction:
      planner = XPoolFirePlanner(kv_act, mamba_act, owner_provider)

    Usage (called by BudgetAgent at fire time, scheduler lock held):
      plan = planner.build(direction="kv_to_mamba", target_drop_pages=4096,
                           dst_grant_chunks=30)
      if plan is None:  # not buildable this tick
          return
      result = actuator.execute(plan)
    """

    def __init__(
        self,
        kv_actuator,
        mamba_actuator,
        owner_provider: OwnerProvider,
    ) -> None:
        self.kv = kv_actuator
        self.mamba = mamba_actuator
        self.owner_provider = owner_provider
        self._seq = 0
        # design.md §"Page selection": when all three expansion stages
        # exhaust without reaching `n`, the planner emits a refuse and
        # increments this counter — the observable signal for "anywhere-
        # free + Drain + Migration exhausted". Surfaced in the budgeter
        # JSONL snapshot (agent.py) so a sustained non-zero rate flags
        # pool-sizing / workload-composition drift.
        self.refuse_count = 0
        logger.info(
            "XPoolFirePlanner init: kv=%s mamba=%s owner_provider=%s",
            type(kv_actuator).__name__,
            type(mamba_actuator).__name__ if mamba_actuator else "None",
            type(owner_provider).__name__,
        )

    def build(
        self,
        direction: str,
        n_pages_target: int,
        *,
        allow_drain: bool = False,
        allow_migrate: bool = False,
    ) -> Optional[FirePlan]:
        """Produce a FirePlan that physically transfers `n_pages_target`
        2 MiB pages from the source pool to the destination.

        Three-stage knapsack page selection (design.md §"Page
        selection: anywhere-free, Drain-expansion, Migration-
        expansion"). The candidate set is expanded in increasing cost
        order until `n` pages are reached; the cheapest stage that meets
        `n` wins:

          Stage 1 (anywhere-free):   FREE pages, descending page-id.
          Stage 2 (Drain-expansion): CACHED pages in active-eviction
                                     order, only when `allow_drain`.
          Stage 3 (Migration-exp.):  LIVE pages by ascending per-page
                                     `c_m`, only when `allow_migrate`.

        `allow_drain` / `allow_migrate` gate the (non-free) stages; both
        default False so the common Budgeter fire path stays free-only
        (and the OwnerProvider's cost-order radix walk stays zero-cost).
        The Admitter's `cross_evict` / `cross_migrate` candidates pass
        them True.

        Sets `plan.drains` (Stage-2 CACHED page-ids the actuator's
        Stage-0 must evict) and `plan.migrations` (Stage-3 LIVE slot-ids
        the actuator's Stage-0 must relocate). When only FREE pages are
        selected both stay empty → the plan is byte-identical to the
        the prior free-only plan and Stage-0 is skipped.

        Returns None and increments `refuse_count` when free + (drain) +
        (migrate) candidates can't reach `n`. Caller degrades to defer.
        """
        if direction not in ("kv_to_mamba", "mamba_to_kv"):
            raise ValueError(f"unknown direction: {direction!r}")

        # Bound the Drain-expansion victim materialization to the fire
        # magnitude: the planner consumes at most `n` cached pages in
        # Stage-2, so the OwnerProvider need only surface that many — its
        # cost-order radix walk otherwise materializes the WHOLE evictable
        # set on the scheduler thread every fire. Only meaningful when
        # draining; None keeps the full walk for any cost-pricing caller.
        max_drain_pages = max(1, n_pages_target) if allow_drain else None
        if direction == "kv_to_mamba":
            owner_map = self.owner_provider.build_kv_owner_map(
                allow_drain=allow_drain, allow_migrate=allow_migrate,
                max_drain_pages=max_drain_pages,
            )
        else:
            owner_map = self.owner_provider.build_mamba_owner_map(
                allow_drain=allow_drain, allow_migrate=allow_migrate,
                max_drain_pages=max_drain_pages,
            )
            if owner_map is None:
                logger.warning(
                    "XPoolFirePlanner.build: mamba_to_kv but owner "
                    "provider returned None — refusing plan."
                )
                return None

        n = max(1, n_pages_target)
        if n >= owner_map.n_pages:
            logger.warning(
                "XPoolFirePlanner: target=%d would exhaust pool "
                "(n_pages=%d); refusing.",
                n_pages_target, owner_map.n_pages,
            )
            return None

        # --- Stage 1: anywhere-free ----------------------------------
        # Highest-id free pages first so under the allocator's placement
        # bias (paper §3.2.1; live → head, free → tail) the picked pages
        # cluster at the tail and the head stays densely packed for live
        # allocation. Without placement bias, any K free pages work
        # equally well — page selection has no contiguity requirement.
        free_selected = sorted(owner_map.free_pages, reverse=True)[:n]
        selected: list[int] = list(free_selected)
        drains: list[int] = []
        migrations: list[int] = []

        # --- Stage 2: Drain-expansion (CACHED → FREE) ----------------
        if len(selected) < n and allow_drain:
            cached = owner_map.cached_pages_in_cost_order or []
            for pid in cached:
                if len(selected) >= n:
                    break
                drains.append(pid)
                selected.append(pid)

        # --- Stage 3: Migration-expansion (LIVE → FREE) --------------
        # `live_pages_in_cost_order` is a list of (freed_page_id, moves)
        # where moves is a tuple of (src_slot, dst_slot) relocations that
        # empty that page. We take whole pages (each counts 1 toward `n`)
        # and accumulate their concrete moves into `migrations`.
        n_migrate_pages = 0
        if len(selected) < n and allow_migrate:
            live = owner_map.live_pages_in_cost_order or []
            for pid, moves in live:
                if len(selected) >= n:
                    break
                migrations.extend(moves)
                selected.append(pid)
                n_migrate_pages += 1

        if len(selected) < n:
            self.refuse_count += 1
            logger.info(
                "XPoolFirePlanner: page selection short "
                "(need=%d got=%d: free=%d drain=%d migrate=%d); refusing "
                "plan — refuse_count=%d.",
                n, len(selected), len(free_selected), len(drains),
                n_migrate_pages, self.refuse_count,
            )
            return None

        # Sorted so the actuator's tail-evict matches the cap_barrier'd
        # range; this preserves the byte-identical-to-Stage-1 invariant
        # for free-only plans (the prior impl also sorted ascending).
        pages_to_unmap = sorted(selected)

        self._seq += 1
        plan = FirePlan(
            direction=direction,
            pages_to_unmap=pages_to_unmap,
            pages_to_map_dst=n,
            plan_seq=self._seq,
            drains=tuple(drains),
            migrations=tuple(migrations),
        )
        logger.info(
            "XPoolFirePlanner.build: seq=%d dir=%s n_pages=%d "
            "(free=%d drain=%d migrate_pages=%d migrate_moves=%d)",
            plan.plan_seq, plan.direction, n,
            len(free_selected), len(drains), n_migrate_pages, len(migrations),
        )
        return plan
