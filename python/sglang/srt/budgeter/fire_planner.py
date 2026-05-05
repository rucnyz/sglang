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

from sglang.srt.arena.fire_plan import FirePlan, MigrateOp
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
        target_drop_pages: int,
        dst_grant_chunks: int,
    ) -> Optional[FirePlan]:
        """Produce a FirePlan that physically drops `target_drop_pages`
        page-equivalents from the source pool and grants `dst_grant_chunks`
        chunks to the destination.

        Returns None when the plan can't be built safely (e.g., not enough
        free dst pages to migrate every active page out of the tail).
        Caller should either retry with a smaller delta next tick or skip
        the fire entirely.
        """
        if direction not in ("kv_to_mamba", "mamba_to_kv"):
            raise ValueError(f"unknown direction: {direction!r}")

        if direction == "kv_to_mamba":
            src_act = self.kv
            owner_map = self.owner_provider.build_kv_owner_map()
        else:
            if self.mamba is None:
                logger.warning(
                    "XPoolFirePlanner.build: mamba_to_kv requested but no "
                    "mamba actuator wired — refusing plan."
                )
                return None
            src_act = self.mamba
            mamba_om = self.owner_provider.build_mamba_owner_map()
            if mamba_om is None:
                logger.warning(
                    "XPoolFirePlanner.build: mamba_to_kv requested but "
                    "owner provider returned None for mamba — refusing plan."
                )
                return None
            owner_map = mamba_om

        # Coverage check: planner refuses to fire if owner walker missed
        # any pages. Better to log and skip than to unmap a page we can't
        # account for (which is exactly the T7 v3 crash class).
        try:
            owner_map.assert_complete()
        except RuntimeError as e:
            logger.error("XPoolFirePlanner: owner map incomplete — %s", e)
            return None

        tpc = src_act.tokens_per_chunk
        n_pages = owner_map.n_pages
        if tpc <= 0 or n_pages <= 0:
            logger.warning(
                "XPoolFirePlanner: invalid pool state tpc=%d n_pages=%d", tpc, n_pages
            )
            return None

        n_to_drop = max(1, target_drop_pages)
        if n_to_drop >= n_pages:
            logger.warning(
                "XPoolFirePlanner: target_drop_pages=%d would exhaust pool "
                "(%d pages); refusing.",
                target_drop_pages, n_pages,
            )
            return None

        # Anywhere-free selection. The cost model of \S3.1 prices per-page
        # actions; here we instantiate the cheapest case --- pages already
        # in `free` state (cost 0) --- by picking $n_to_drop$ free pages
        # from anywhere in $\\mathcal{P}_i$. We sort descending so high-id
        # pages go first; under T2 placement bias these cluster at the
        # tail naturally, but the planner does not require them to.
        free_pages_sorted = sorted(owner_map.free_pages, reverse=True)
        if len(free_pages_sorted) < n_to_drop:
            logger.info(
                "XPoolFirePlanner: insufficient free pages "
                "(need=%d have=%d); refusing plan. Caller may retry "
                "after pressure subsides or with a smaller target.",
                n_to_drop, len(free_pages_sorted),
            )
            return None

        pages_to_unmap = sorted(free_pages_sorted[:n_to_drop])
        # Chunk = page in T1 page-grain VMM. Page-id layout is 1-indexed
        # (page 0 is the null sentinel), so chunk c owns page c+1.
        chunks_to_unmap = [p - 1 for p in pages_to_unmap]
        # capped_page_range is the bounding interval over the picked pages
        # (logging only; the executor uses chunks_to_unmap_src as the
        # source of truth for which specific pages to cap).
        capped_low = pages_to_unmap[0]
        capped_high = pages_to_unmap[-1] + 1

        self._seq += 1
        plan = FirePlan(
            direction=direction,
            capped_page_range=(capped_low, capped_high),
            chunks_to_unmap_src=chunks_to_unmap,
            pages_to_drain=[],
            pages_to_migrate=[],
            chunks_to_map_dst=int(dst_grant_chunks),
            expected_unmap_pages=n_to_drop,
            plan_seq=self._seq,
        )
        logger.info(
            "XPoolFirePlanner.build: seq=%d dir=%s n_unmap=%d "
            "(span=[%d,%d), n_free_in_pool=%d) grant_chunks=%d",
            plan.plan_seq, plan.direction, n_to_drop,
            capped_low, capped_high, len(free_pages_sorted),
            plan.chunks_to_map_dst,
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
