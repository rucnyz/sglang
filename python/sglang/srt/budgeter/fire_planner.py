"""T8 step 2 — XPoolFirePlanner.

Turns the high-level "shrink KV by N pages, grow mamba by M chunks"
intent (already produced by `CrossPoolPlanner.decide`) into a concrete
`FirePlan` with explicit drain/migrate/unmap steps. The planner is
ground-truth aware: it consults an `OwnerProvider` so the plan never
references an active-req page without an accompanying migrate.

Tail-only for now (T2 placement bias keeps live → head, free → tail, so
tail chunks are the common case). A future revision may pick chunks from
anywhere in the page-id space using `allocator.free_page_mask` — the
FirePlan shape supports it; only the planner's selection step changes.

Flag: SGLANG_T8_PLANNER=1 turns the new path on. While off, this module
just sits idle — no other code imports it yet.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from sglang.srt.arena.fire_plan import FirePlan
from sglang.srt.arena.owner_provider import OwnerMap, OwnerProvider

logger = logging.getLogger(__name__)


def _flag_enabled() -> bool:
    return os.environ.get("SGLANG_T8_PLANNER", "0") == "1"


def _executor_flag_enabled() -> bool:
    return os.environ.get("SGLANG_T8_EXECUTE", "0") == "1"


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
    ) -> Optional[FirePlan]:
        """Produce a FirePlan that physically transfers `n_pages_target`
        2 MiB pages from the source pool to the destination.

        Returns None when no such plan is achievable from the current
        free-page pool. Caller skips the fire and retries on the next
        budgeter tick.
        """
        if direction not in ("kv_to_mamba", "mamba_to_kv"):
            raise ValueError(f"unknown direction: {direction!r}")

        if direction == "kv_to_mamba":
            owner_map = self.owner_provider.build_kv_owner_map()
        else:
            owner_map = self.owner_provider.build_mamba_owner_map()
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

        free_pages = owner_map.free_pages
        if len(free_pages) < n:
            logger.info(
                "XPoolFirePlanner: insufficient free pages "
                "(need=%d have=%d); refusing plan.",
                n, len(free_pages),
            )
            return None

        # Highest-id free pages first so under T2 placement bias the
        # picked pages cluster at the tail (the head stays densely packed
        # for live allocation). Without placement bias, any K free pages
        # work equally well — page selection has no contiguity requirement.
        pages_to_unmap = sorted(sorted(free_pages, reverse=True)[:n])

        self._seq += 1
        plan = FirePlan(
            direction=direction,
            pages_to_unmap=pages_to_unmap,
            pages_to_map_dst=n,
            plan_seq=self._seq,
        )
        logger.info(
            "XPoolFirePlanner.build: seq=%d dir=%s n_pages=%d "
            "(n_free_in_pool=%d)",
            plan.plan_seq, plan.direction, n, len(free_pages),
        )
        return plan


def is_planner_enabled() -> bool:
    """Public flag check. BudgetAgent calls this to decide whether to
    construct a planner and route fires through it."""
    return _flag_enabled()


def is_executor_enabled() -> bool:
    """Public flag check for the plan-based executor path
    (CrossPoolTransferActuator.execute(plan)). Both planner and executor
    flags must be set for T8 fires to dispatch."""
    return _executor_flag_enabled()
