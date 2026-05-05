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

from sglang.srt.arena.fire_plan import FirePlan, FirePlanResult, MigrateOp


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

    # ---- Balanced (leftover-free) wrappers ---------------------------

    # ---- T8 plan-based execution -------------------------------------

    def execute(
        self,
        plan: "FirePlan",
        drain_callback=None,
        migrate_callback=None,
    ) -> "FirePlanResult":
        """T8 step 3: execute a planner-produced FirePlan.

        The plan already encodes which chunks to unmap, which tree refs
        to drain, which active-req pages to migrate. The executor runs
        a fixed 6-step protocol with no decisions:

          1. cap-barrier   — pull every page in capped_page_range out of
                             allocator.free_pages (no new alloc lands there)
          2. drain         — call `drain_callback(pages_to_drain)` so the
                             scheduler evicts referencing tree nodes; the
                             pages return through allocator.free into the
                             capped set (allocator handles this transition).
          3. migrate       — call `migrate_callback(pages_to_migrate)` so
                             the scheduler D2D-copies KV slices to dst_page
                             and atomically updates req.kv_indices[slot].
          4. verify        — assert every page in capped_page_range is now
                             in capped state (free_pages disjoint from
                             [low,high)). Aborts the fire if not.
          5. unmap+map     — physically shrink_explicit on src, grow on dst.
          6. uncap dst     — raise dst allocator's capacity to expose the
                             newly-mapped chunks to alloc.

        Callbacks are required when their corresponding lists are
        non-empty. We refuse to silently no-op a non-empty list — that
        was exactly the T7 v3 failure mode.

        Returns a `FirePlanResult` capturing per-step timings + counts.
        """
        if plan.pages_to_drain and drain_callback is None:
            raise RuntimeError(
                f"FirePlan seq={plan.plan_seq} has {len(plan.pages_to_drain)} "
                f"pages_to_drain but no drain_callback provided. Refusing "
                f"to execute — silent skip would unmap pages still owned "
                f"by tree nodes."
            )
        if plan.pages_to_migrate and migrate_callback is None:
            raise RuntimeError(
                f"FirePlan seq={plan.plan_seq} has {len(plan.pages_to_migrate)} "
                f"pages_to_migrate but no migrate_callback provided. Refusing "
                f"to execute — silent skip would unmap pages still owned "
                f"by active reqs."
            )

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
                "(via __init__ kv_actuator + mamba_actuator). The plan-based "
                "path does not have an idle-window-only fallback — the "
                "scheduler-side wiring is mandatory."
            )

        result = FirePlanResult(
            plan_seq=plan.plan_seq,
            direction=plan.direction,
            unmapped_pages=0,
            granted_chunks=0,
            drained_pages=0,
            migrated_pages=0,
            cap_barrier_us=0,
            drain_us=0,
            migrate_us=0,
            unmap_us=0,
            map_us=0,
            total_us=0,
        )
        t_start = time.monotonic_ns()

        # --- Step 1: cap-barrier --------------------------------------
        # The planner picked specific page-ids (chunk_id + 1 under T1's
        # 1-indexed page layout). We cap exactly those pages; any
        # intermediate page-ids inside `capped_page_range` that the
        # planner did NOT select are left alone.
        cap_t0 = time.monotonic_ns()
        alloc = getattr(src_act, "allocator", None)
        if alloc is None or not hasattr(alloc, "mark_pages_capped"):
            raise RuntimeError(
                "execute(plan): src allocator missing mark_pages_capped — "
                "the plan-based path requires T3 cap-barrier API."
            )
        cap_pages = [c + 1 for c in plan.chunks_to_unmap_src]
        cap_t = torch.tensor(cap_pages, dtype=torch.int64, device=alloc.device)
        moved_to_capped = alloc.mark_pages_capped(cap_t)
        result.cap_barrier_us = (time.monotonic_ns() - cap_t0) // 1000
        logger.info(
            "execute[seq=%d] cap-barrier: pages=%d marked=%d "
            "(free→capped) in %d us",
            plan.plan_seq, len(cap_pages), moved_to_capped,
            result.cap_barrier_us,
        )

        # --- Step 2: drain --------------------------------------------
        drain_t0 = time.monotonic_ns()
        if plan.pages_to_drain:
            result.drained_pages = drain_callback(plan.pages_to_drain)
        result.drain_us = (time.monotonic_ns() - drain_t0) // 1000

        # --- Step 3: migrate ------------------------------------------
        mig_t0 = time.monotonic_ns()
        if plan.pages_to_migrate:
            result.migrated_pages = migrate_callback(plan.pages_to_migrate)
        result.migrate_us = (time.monotonic_ns() - mig_t0) // 1000

        # --- Step 4: verify -------------------------------------------
        # Every page the planner selected must now be in capped state, not
        # free. (Cap-barrier moved them out of free_pages in step 1; any
        # callbacks should have left them capped.) If any selected page
        # leaked back into free_pages, abort and restore the cap state.
        free_pages_t = getattr(alloc, "free_pages", None)
        if free_pages_t is not None and free_pages_t.numel() > 0:
            in_target = torch.isin(free_pages_t, cap_t)
            n_violations = int(in_target.sum().item())
            if n_violations > 0:
                alloc.unmark_pages_capped(cap_t)
                result.aborted = True
                result.abort_reason = (
                    f"verify failed: {n_violations} of "
                    f"{len(cap_pages)} target pages still in free_pages"
                )
                result.total_us = (time.monotonic_ns() - t_start) // 1000
                logger.error(
                    "execute[seq=%d] ABORT after verify: %s",
                    plan.plan_seq, result.abort_reason,
                )
                return result

        # --- Step 5: unmap + map --------------------------------------
        src_names = self._all_subpool_names(src)
        dst_names = self._all_subpool_names(dst)

        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        unmap_t0 = time.monotonic_ns()
        unmapped_total = 0
        for name in src_names:
            unmapped_total += src._arena.shrink_explicit(name, plan.chunks_to_unmap_src)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        result.unmap_us = (time.monotonic_ns() - unmap_t0) // 1000
        result.unmapped_pages = unmapped_total

        # Sanity: did we actually unmap what we expected?
        # Per-subpool count of unmapped pages should equal
        # len(chunks_to_unmap_src) * tpc; total is summed across subpools.
        expected_total = plan.expected_unmap_pages * len(src_names)
        if unmapped_total != expected_total:
            logger.warning(
                "execute[seq=%d] unmap count mismatch: got=%d expected=%d "
                "(per-subpool %d * %d subpools). chunks may have been "
                "skipped by shrink_explicit's bounds check.",
                plan.plan_seq, unmapped_total, expected_total,
                plan.expected_unmap_pages, len(src_names),
            )

        map_t0 = time.monotonic_ns()
        granted_total = 0
        for name in dst_names:
            granted_total += dst._arena.grow(name, plan.chunks_to_map_dst)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        result.map_us = (time.monotonic_ns() - map_t0) // 1000
        result.granted_chunks = granted_total

        # --- Step 6: uncap dst ----------------------------------------
        dst_grow_tokens = plan.chunks_to_map_dst * dst.tokens_per_chunk
        new_dst_cap = dst_act.live_capacity_tokens() + dst_grow_tokens
        dst_act.cap_allocator_only(new_dst_cap)

        result.total_us = (time.monotonic_ns() - t_start) // 1000
        logger.info(
            "execute[seq=%d] DONE dir=%s unmapped=%d granted=%d "
            "drained=%d migrated=%d cap=%dus drain=%dus migrate=%dus "
            "unmap=%dus map=%dus total=%dus",
            plan.plan_seq, plan.direction, result.unmapped_pages,
            result.granted_chunks, result.drained_pages, result.migrated_pages,
            result.cap_barrier_us, result.drain_us, result.migrate_us,
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
