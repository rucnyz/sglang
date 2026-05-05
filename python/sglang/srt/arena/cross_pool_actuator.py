"""
T8 — CrossPoolTransferActuator: pure mechanical executor.

Wraps two MultiTensorArena instances (KV + mamba) sharing one
SharedHandlePool. Exposes a single decision-free entrypoint:

  execute(plan: FirePlan, drain_callback, migrate_callback) -> FirePlanResult

The plan is built upstream by `sglang.srt.budgeter.fire_planner` against
ground-truth ownership state from `sglang.srt.budgeter.scheduler_owner_provider`.
The executor does cap-barrier → drain → migrate → verify → unmap+map →
uncap-dst with no fallbacks — every page in the unmap range has been
classified as free / drainable / migratable by the planner.

See `dev/T8_xpool_layering_refactor/README.md` for the full design and
the legacy `_do_transfer` / `kv_to_mamba_chunks` family that this
replaces.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.arena.fire_plan import FirePlan, FirePlanResult


if TYPE_CHECKING:
    from sglang.srt.arena.multi_tensor_arena import MultiTensorArena
    from sglang.srt.arena.chunk_arena import SharedHandlePool


logger = logging.getLogger(__name__)


class CrossPoolTransferActuator:
    """KV ↔ mamba chunk migration over a shared handle pool."""

    def __init__(
        self,
        kv_arena: "MultiTensorArena",
        mamba_arena: "MultiTensorArena",
        shared_pool: "SharedHandlePool",
        kv_actuator=None,
        mamba_actuator=None,
    ) -> None:
        self.kv = kv_arena
        self.mamba = mamba_arena
        self.shared = shared_pool
        # Per-pool actuators are required for plan-based execute(); this
        # was optional in the legacy "raw chunk move only" path. T8 made
        # them mandatory because cap-barrier+migrate need allocator access.
        self.kv_actuator = kv_actuator
        self.mamba_actuator = mamba_actuator

        if kv_arena._arena._external_pool is not shared_pool:
            raise ValueError("kv_arena does not use the provided shared_pool")
        if mamba_arena._arena._external_pool is not shared_pool:
            raise ValueError("mamba_arena does not use the provided shared_pool")

        self.n_kv_subpools = kv_arena.n_layers * kv_arena.n_kinds
        self.n_mamba_subpools = mamba_arena.n_layers * mamba_arena.n_kinds

        logger.info(
            "CrossPoolTransferActuator: kv_subpools=%d, mamba_subpools=%d, "
            "shared_handles=%d, free=%d",
            self.n_kv_subpools, self.n_mamba_subpools,
            self.shared.total_count(), self.shared.free_count(),
        )

    # ------------------------------------------------------------------

    def _all_subpool_names(self, mta: "MultiTensorArena") -> list[str]:
        n = mta.n_layers * mta.n_kinds
        return [mta._pool_name(i) for i in range(n)]

    def _src_actuator(self, src: "MultiTensorArena"):
        return self.kv_actuator if src is self.kv else self.mamba_actuator

    def _dst_actuator(self, dst: "MultiTensorArena"):
        return self.kv_actuator if dst is self.kv else self.mamba_actuator

    # ---- Plan-based execution ----------------------------------------

    def execute(self, plan: "FirePlan") -> "FirePlanResult":
        """Execute a planner-produced FirePlan. Three steps, no callbacks:

          1. cap-barrier — translate page-ids to allocator token-slots and
                           mark them off-limits, blocking concurrent allocs;
          2. verify      — confirm no target page leaked back into free;
          3. unmap + map — physically `cuMemUnmap` source pages, `cuMemMap`
                           the freed handles into the destination pool.

        The planner has already guaranteed every page in `pages_to_unmap`
        is currently fully free. The verify step is a sanity check that
        should never trip in steady state.
        """
        if plan.direction == "kv_to_mamba":
            src, dst = self.kv, self.mamba
            src_act, dst_act = self.kv_actuator, self.mamba_actuator
        elif plan.direction == "mamba_to_kv":
            src, dst = self.mamba, self.kv
            src_act, dst_act = self.mamba_actuator, self.kv_actuator
        else:
            raise ValueError(f"unknown plan.direction: {plan.direction!r}")

        if src_act is None or dst_act is None:
            raise RuntimeError(
                "execute(plan) requires both per-pool actuators wired "
                "(via __init__ kv_actuator + mamba_actuator)."
            )

        result = FirePlanResult(
            plan_seq=plan.plan_seq,
            direction=plan.direction,
            unmapped_pages=0,
            granted_pages=0,
            cap_barrier_us=0,
            unmap_us=0,
            map_us=0,
            total_us=0,
        )
        t_start = time.monotonic_ns()

        # --- Step 1: cap-barrier --------------------------------------
        cap_t0 = time.monotonic_ns()
        alloc = getattr(src_act, "allocator", None)
        if alloc is None or not hasattr(alloc, "mark_pages_capped"):
            raise RuntimeError(
                "execute(plan): src allocator missing mark_pages_capped."
            )
        # Translation page-id → allocator token-slot ids is hidden in the
        # actuator; the planner above never sees token slots.
        cap_slots = src_act.expand_pages_to_token_slots(plan.pages_to_unmap)
        cap_t = torch.tensor(cap_slots, dtype=torch.int64, device=alloc.device)
        moved_to_capped = alloc.mark_pages_capped(cap_t)
        result.cap_barrier_us = (time.monotonic_ns() - cap_t0) // 1000
        logger.info(
            "execute[seq=%d] cap-barrier: pages=%d slots=%d marked=%d in %d us",
            plan.plan_seq, len(plan.pages_to_unmap),
            len(cap_slots), moved_to_capped, result.cap_barrier_us,
        )

        # --- Step 2: verify -------------------------------------------
        free_pages_t = getattr(alloc, "free_pages", None)
        if free_pages_t is not None and free_pages_t.numel() > 0:
            in_target = torch.isin(free_pages_t, cap_t)
            n_violations = int(in_target.sum().item())
            if n_violations > 0:
                alloc.unmark_pages_capped(cap_t)
                result.aborted = True
                result.abort_reason = (
                    f"verify failed: {n_violations} of {len(cap_slots)} "
                    f"target slots still in free_pages"
                )
                result.total_us = (time.monotonic_ns() - t_start) // 1000
                logger.error(
                    "execute[seq=%d] ABORT after verify: %s",
                    plan.plan_seq, result.abort_reason,
                )
                return result

        # --- Step 3: unmap + map --------------------------------------
        src_names = self._all_subpool_names(src)
        dst_names = self._all_subpool_names(dst)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        unmap_t0 = time.monotonic_ns()
        unmapped_total = 0
        for name in src_names:
            unmapped_total += src._arena.shrink_explicit(name, plan.pages_to_unmap)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        result.unmap_us = (time.monotonic_ns() - unmap_t0) // 1000
        result.unmapped_pages = unmapped_total

        map_t0 = time.monotonic_ns()
        granted_total = 0
        for name in dst_names:
            granted_total += dst._arena.grow(name, plan.pages_to_map_dst)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        result.map_us = (time.monotonic_ns() - map_t0) // 1000
        result.granted_pages = granted_total

        # --- Step 4: uncap dst ----------------------------------------
        # Translate granted pages to token-slots for the dst allocator's
        # capacity bump. Reuse the same actuator-internal translation.
        dst_grow_slots = len(dst_act.expand_pages_to_token_slots(
            list(range(plan.pages_to_map_dst))
        ))
        new_dst_cap = dst_act.live_capacity_tokens() + dst_grow_slots
        dst_act.cap_allocator_only(new_dst_cap)

        result.total_us = (time.monotonic_ns() - t_start) // 1000
        logger.info(
            "execute[seq=%d] DONE dir=%s unmapped=%d granted=%d "
            "cap=%dus unmap=%dus map=%dus total=%dus",
            plan.plan_seq, plan.direction, result.unmapped_pages,
            result.granted_pages, result.cap_barrier_us,
            result.unmap_us, result.map_us, result.total_us,
        )
        return result

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
            "shared_total_handles": self.shared.total_count(),
            "shared_free_handles": self.shared.free_count(),
        }
