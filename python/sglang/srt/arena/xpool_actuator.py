"""XPoolActuator: pure mechanical executor.

Wraps two MultiTensorArena instances (KV + mamba) sharing one
SharedHandlePool. Exposes a single decision-free entrypoint:

  execute(plan: FirePlan) -> FirePlanResult

The plan is built upstream by `sglang.srt.budgeter.fire_planner` against
ground-truth ownership state from `sglang.srt.budgeter.scheduler_owner_provider`.
The executor does cap-barrier → drain → migrate → verify → unmap+map →
uncap-dst with no fallbacks — every page in the unmap range has been
classified as free / drainable / migratable by the planner.

The flow is also exposed in two phases for async use by BudgetAgent:
  - cap_barrier(plan) -> FireToken    (synchronous, scheduler thread)
  - execute_async(token) -> FirePlanResult  (worker thread)

The phase split lets the scheduler resume work immediately after the
cap-barrier mark (which is what makes the to-be-unmapped pages safe from
fresh allocations); the expensive cuMemUnmap + cuMemMap + cap-bump runs
off-thread.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.arena.fire_plan import FirePlan, FirePlanResult


if TYPE_CHECKING:
    from sglang.srt.arena.multi_tensor_arena import MultiTensorArena
    from sglang.srt.arena.chunk_arena import SharedHandlePool


logger = logging.getLogger(__name__)


@dataclass
class _PendingApply:
    """Deferred pool-state update produced by the fire worker, applied on
    the scheduler thread by ``apply_pending_fires()``.  Separates the
    CUDA driver work (cuMemUnmap/Map, worker thread) from allocator
    metadata (pool.size, unmark, scheduler thread)."""

    plan_seq: int
    src_pool: object
    dst_pool: object
    src_tpc: int
    dst_tpc: int
    shrunk_pages: int
    grown_pages: int
    dst_token_slots: list


@dataclass
class FireToken:
    """Opaque handoff from cap_barrier (sync) to execute_async (worker).

    Carries everything execute_async needs so it doesn't have to re-read
    actuator state after the scheduler resumes — the pages have already
    been removed from the allocator's free list by cap_barrier, so the
    worker can do cuMemUnmap on them without worrying about a race with
    fresh allocations.
    """

    plan: "FirePlan"
    src: "MultiTensorArena"
    dst: "MultiTensorArena"
    src_act: object
    dst_act: object
    cap_t: torch.Tensor          # tensor of capped token-slots (for rollback if needed)
    cap_slots_count: int
    cap_barrier_us: int
    t_start_ns: int
    aborted: bool = False
    abort_reason: str = ""
    unmapped_total: int = 0
    per_src: int = 0
    granted_in_barrier: list | None = None


class XPoolActuator:
    """KV ↔ mamba chunk migration over a shared handle pool."""

    def __init__(
        self,
        kv_arena: "MultiTensorArena",
        mamba_arena: "MultiTensorArena",
        shared_pool: "SharedHandlePool",
        kv_actuator=None,
        mamba_actuator=None,
        stage0_handler=None,
        kv_serving_floor_tokens: int = 0,
    ) -> None:
        self.kv = kv_arena
        self.mamba = mamba_arena
        self.shared = shared_pool
        # Serving floor for k2m fires: a fire may never shrink the KV
        # allocator's AVAILABLE tokens below this — the pool must always
        # admit one full prefill chunk plus a decode step, else
        # alloc_token_slots raises "Out of memory" with nothing evictable
        # and the scheduler dies (observed: 9B dynamic t6@128, available
        # 6408 < chunk 8192 after the Admitter's k2m drained the pool;
        # the cap_barrier fast path made fires cheap enough to outrun
        # request completion, unmasking the missing invariant).
        # 0 disables (test/back-compat); the BudgetAgent wires
        # 2*chunked_prefill_size + max_running_requests.
        self.kv_serving_floor_tokens = int(kv_serving_floor_tokens)
        # Per-pool actuators are mandatory: cap-barrier and migrate need
        # allocator access, which only the per-pool actuator exposes.
        self.kv_actuator = kv_actuator
        self.mamba_actuator = mamba_actuator
        # Stage-0 collaborator (design.md §"Transfer protocol": Stage 0).
        # Performs the scheduler-coupled Drain (CACHED→FREE radix
        # evict) and Migration (LIVE→FREE slot relocation + ssm-state-
        # index rewrite) pre-conditioning so the planner-selected
        # `drains` / `migrations` pages become FREE before cap-barrier.
        # `None` for free-only deployments (the actuator never invokes it
        # unless a plan carries non-empty drains/migrations). Interface:
        #   evict_pages(direction: str, drains: tuple[int,...]) -> None
        #   rewrite_ssm_state_indices(src_slot, dst_slot) -> None
        # (migration dsts are planner-assigned in plan.migrations as
        #  (src_slot, dst_slot) pairs — no dst allocation here.)
        self.stage0_handler = stage0_handler

        if kv_arena._arena._external_pool is not shared_pool:
            raise ValueError("kv_arena does not use the provided shared_pool")
        if mamba_arena._arena._external_pool is not shared_pool:
            raise ValueError("mamba_arena does not use the provided shared_pool")

        self.n_kv_subpools = kv_arena.n_layers * kv_arena.n_kinds
        self.n_mamba_subpools = mamba_arena.n_layers * mamba_arena.n_kinds

        # Serialize the body of execute_async() so the Budgeter's
        # _fire_worker_loop and the Admitter's synchronous cross-* fires
        # can't both touch SharedHandlePool._free_handles at the same time.
        # The SharedHandlePool itself has no internal lock (chunk_arena.py);
        # without this serialization the worker and the Admitter would race
        # on the free-handle Python list and double-allocate or lose handles.
        # See dev/interlayer/2_admitter/README.md.
        self._fire_inflight = threading.Lock()
        self._pending_applies: list[_PendingApply] = []
        self._pending_lock = threading.Lock()
        # Freeze the shared pool so any runtime caller of grow() / extend
        # ops raises instead of silently racing fires. The arenas have
        # already done their init-time growth above; no more legitimate
        # growth happens until shutdown.
        self.shared.freeze()

        logger.info(
            "XPoolActuator: kv_subpools=%d, mamba_subpools=%d, "
            "shared_handles=%d, free=%d",
            self.n_kv_subpools, self.n_mamba_subpools,
            self.shared.total_count(), self.shared.free_count(),
        )

    # ------------------------------------------------------------------

    @property
    def lcm_pages(self) -> int:
        """Atomic per-fire page granularity: lcm(n_kv_subpools, n_mamba_subpools).

        `execute()` rounds the per-fire `target_total` DOWN to a
        multiple of this value (see the `total = (target_total // lcm_n)
        * lcm_n` line) — a request below one LCM-unit transfers zero
        pages. The Admitter rounds UP to the same multiple when pricing
        c_xfer so its cost reflects what the actuator will actually
        fire; without this matching the rounding, cross-* is
        under-priced by an LCM factor and biases selection toward
        cross-* over defer.
        """
        return math.lcm(self.n_kv_subpools, self.n_mamba_subpools)

    def _all_subpool_names(self, mta: "MultiTensorArena") -> list[str]:
        n = mta.n_layers * mta.n_kinds
        return [mta._pool_name(i) for i in range(n)]

    def _resolve_direction(self, plan: "FirePlan"):
        if plan.direction == "kv_to_mamba":
            return self.kv, self.mamba, self.kv_actuator, self.mamba_actuator
        elif plan.direction == "mamba_to_kv":
            return self.mamba, self.kv, self.mamba_actuator, self.kv_actuator
        else:
            raise ValueError(f"unknown plan.direction: {plan.direction!r}")

    # ---- Stage 0: Drain / Migration pre-conditioning (sync) ----

    def _run_stage0(self, plan: "FirePlan", src_act) -> None:
        """Pre-condition `plan.drains` (CACHED→FREE) and
        `plan.migrations` (LIVE→FREE) so those pages are genuinely free
        before the cap-barrier (design.md §"Transfer protocol": Stage 0).

        Drain: evict the cached blocks backing each `drains` page via
        sglang's own radix-cache eviction, delegated to the
        scheduler-coupled `stage0_handler.evict_pages` (the actuator
        has no radix-cache reference).

        Migration: for each `migrations` mamba slot, relocate its
        recurrent state into a free dst slot via the byte-exact
        `MambaPool.migrate_slot` (property A2,
        dev/interlayer/0_page_state_machine/step2_migrate_slot_replay_invariant)
        AND rewrite the owning in-flight req's
        `ssm_state_indices` to the new slot (mandated by the
        `migrate_slot` docstring — the pool moves bytes, the caller
        moves the pointer). The dst slot and the req-side rewrite are
        scheduler-coupled, so both go through `stage0_handler`.

        After Stage-0 every drained/migrated page is FREE and joins
        `plan.pages_to_unmap` for the cap-barrier below.
        """
        handler = self.stage0_handler
        if handler is None:
            raise RuntimeError(
                f"cap_barrier[seq={plan.plan_seq}]: plan carries "
                f"drains={len(plan.drains)} migrations={len(plan.migrations)} "
                f"but no stage0_handler is wired. A Drain/Migration plan "
                f"needs the scheduler-coupled radix-evict + ssm-rewrite "
                f"collaborator (XPoolActuator(stage0_handler=...))."
            )

        if plan.drains:
            # CACHED → FREE via sglang's eviction (the radix cache frees
            # the slots, returning the page to the allocator's free list).
            handler.evict_pages(plan.direction, plan.drains)

        # LIVE → FREE. Pool-agnostic: the source per-pool actuator
        # exposes a uniform `migrate_slot(src, dst)` — mamba routes to
        # `MambaPool.migrate_slot`, KV to `TokenToKVPoolAllocator.migrate_slot`.
        # The planner already assigned each `(src_slot, dst_slot)`
        # move — `dst_slot` is a SCATTERED free slot on a KEPT page (never a
        # page in `pages_to_unmap`), so Stage-0 runs the exact relocation
        # with no destination guessing. The owning req's pointer rewrite is
        # dispatched per source pool (the byte contents are already
        # relocated by migrate_slot).
        is_mamba_src = plan.direction == "mamba_to_kv"
        # Validate-then-apply: a migration mutates TWO
        # places — the slot's bytes (`migrate_slot`) and the owning req's
        # pointer (`rewrite_*`). If a later move's rewrite raised AFTER its
        # `migrate_slot` already freed `src` and moved its bytes, that req
        # would read a freed slot and `dst` would be live-but-orphaned (and
        # the raise is swallowed upstream by `agent.tick`). So confirm EVERY
        # source has a live owner BEFORE relocating any bytes — an invalid
        # plan aborts here with zero mutation. (dst-availability is re-checked
        # by `migrate_slot` per move, which fails before moving bytes.)
        for src_slot, dst_slot in plan.migrations:
            if not handler.has_live_owner(plan.direction, int(src_slot)):
                raise RuntimeError(
                    f"cap_barrier[seq={plan.plan_seq}]: migration src slot "
                    f"{int(src_slot)} has no live owner in the running batch "
                    f"— refusing to relocate any bytes (validate-then-apply; "
                    f"Migration targets LIVE slots, no half-applied state)."
                )
        # Sync contract (migrate_slot docstring): before relocating bytes,
        # drain in-flight kernels still READING the source slots, so the copy
        # into dst can't clobber a slot the current step is mid-read on. This
        # matters most for KV — its all-layer move races the live decode —
        # whereas mamba is safe by between-step quiescence; the barrier is
        # correct for both and migrations are rare (Stage-3, cooldown-gated).
        if plan.migrations and torch.cuda.is_available():
            torch.cuda.synchronize()
        for src_slot, dst_slot in plan.migrations:
            src_slot, dst_slot = int(src_slot), int(dst_slot)
            ok = src_act.migrate_slot(src_slot, dst_slot)
            if not ok:
                raise RuntimeError(
                    f"cap_barrier[seq={plan.plan_seq}]: migrate_slot("
                    f"{src_slot}->{dst_slot}) failed (dst not free or "
                    f"src==dst). Stage-0 cannot pre-condition the plan."
                )
            if is_mamba_src:
                # mamba pointer: scalar req.mamba_pool_idx (A2 proves the
                # captured-graph replay reads the new slot correctly).
                handler.rewrite_ssm_state_indices(src_slot, dst_slot)
            else:
                # KV pointer: req_to_token[req, pos] = dst — the handler
                # locates the in-flight req holding `src` and rewrites that
                # position. The attention backend re-reads
                # req_to_token into kv_indices each replay, so this
                # propagates to the captured graph.
                handler.rewrite_kv_token_indices(src_slot, dst_slot)
            logger.debug(
                "stage0[seq=%d] migrated %s slot %d -> %d (src now free)",
                plan.plan_seq, "mamba" if is_mamba_src else "kv",
                src_slot, dst_slot,
            )
        if plan.migrations and torch.cuda.is_available():
            # Make the relocated bytes AND the owning-req pointer rewrite
            # visible before the next graph replay reads the dst slots.
            torch.cuda.synchronize()
        if plan.migrations:
            logger.info(
                "stage0[seq=%d] dir=%s drained=%d migrated=%d pages "
                "pre-conditioned to FREE",
                plan.plan_seq, plan.direction, len(plan.drains),
                len(plan.migrations),
            )

    # ---- Phase 1: cap-barrier (sync; scheduler thread) ----

    def cap_barrier(self, plan: "FirePlan") -> "FireToken":
        """Synchronous phase: mark target pages off-limits in the
        allocator's free list so the scheduler can resume admitting
        requests without racing the worker's cuMemUnmap.

        Returns a FireToken; if cap_barrier hit an unrecoverable
        condition (allocator missing API, verify violation), the
        token carries aborted=True and the worker should skip it.
        """
        src, dst, src_act, dst_act = self._resolve_direction(plan)
        if src_act is None or dst_act is None:
            raise RuntimeError(
                "cap_barrier(plan) requires both per-pool actuators wired "
                "(via __init__ kv_actuator + mamba_actuator)."
            )

        t_start = time.monotonic_ns()
        cap_t0 = t_start

        # Stage-0 (design.md §"Transfer protocol"): pre-condition
        # the planner-selected CACHED (drains) / LIVE (migrations) pages
        # to FREE BEFORE the free-page cap-barrier below, so they join
        # `plan.pages_to_unmap` as genuinely-free pages. Guarded on a
        # non-empty plan so the free-only path skips Stage-0 entirely when
        # only FREE pages were selected.
        #
        # Thread-safety: this runs on the SCHEDULER thread (Budgeter tick /
        # Admitter arrival — both serialized on that one thread, as is all
        # radix-cache eviction and src-pool alloc/free). The fire WORKER
        # thread only does cuMemUnmap/Map + the DST allocator's unmark/
        # cap-bump (under that allocator's own `_alloc_lock`), and there is a
        # single worker consuming a queue, so workers serialize. A cross-fire
        # touches the two pools in DISJOINT roles (src shrinks, dst grows),
        # so the scheduler-thread Stage-0 (src radix evict + src
        # mark_pages_capped) never shares mutable state with a concurrent
        # worker. The caller therefore does NOT hold a coarse allocator lock
        # across this call — and must not: Stage-0's `tree_cache.evict` frees
        # slots through `allocator.free()`, which re-acquires the (non-
        # reentrant) `_alloc_lock`, so a caller-held lock would self-deadlock.
        # Per-op locks on each mark/unmark/free mutation are the actual
        # mechanism; scheduler-single-threadedness is the rest.
        alloc = src_act.allocator

        # ---- FAST PATH: free-only plan (no drains, no migrations) ----
        # Clamp the grant BEFORE expanding/marking. The planner routinely
        # offers far more pages than the dst headroom admits (observed: 80
        # offered, ~5 granted on Nemotron-3 case3); the legacy path expanded
        # and marked ALL offered pages (~655K token-slot ids through a
        # Python list -> CUDA tensor, ~200+ ms of SCHEDULER-THREAD time per
        # fire) and then unmarked the ~94% surplus. Pure integer clamping
        # first + vectorized expansion of only the kept pages brings the
        # cap-barrier to <1 ms. Drain/migration plans keep the legacy
        # mark-all-then-restore path below (Stage-0 pre-conditioning is
        # entangled with the full offered set).
        if not plan.drains and not plan.migrations:
            n_src = len(self._all_subpool_names(src))
            n_dst = len(self._all_subpool_names(dst))
            target_src_total = n_src * len(plan.pages_to_unmap)
            # k2m serving floor: never shrink the KV allocator's available
            # tokens below the floor (one prefill chunk + decode headroom).
            # per_src pages remove per_src * tokens-per-page allocatable
            # slots, so cap the src-side page count accordingly.
            if plan.direction == "kv_to_mamba" and self.kv_serving_floor_tokens:
                tps_src = src_act._tokens_per_page()
                avail = int(src_act.allocator.available_size())
                shrinkable_pages = max(
                    0, (avail - self.kv_serving_floor_tokens) // tps_src
                )
                target_src_total = min(target_src_total, n_src * shrinkable_pages)
            target_dst_total = n_dst * plan.pages_to_map_dst
            target_dst_total = min(
                target_dst_total, n_dst * dst_act.grow_headroom_pages()
            )
            target_total = min(target_src_total, target_dst_total)
            lcm_n = self.lcm_pages
            total = (target_total // lcm_n) * lcm_n
            per_src = total // n_src if n_src else 0
            unmapped_total = per_src * n_src
            kept_cap_t = src_act.expand_pages_to_token_slots_tensor(
                plan.pages_to_unmap[:per_src], alloc.device
            )
            moved_to_capped = alloc.mark_pages_capped(kept_cap_t)
            _p5 = time.monotonic_ns()
            cap_barrier_us = (_p5 - cap_t0) // 1000
            logger.debug(
                "cap_barrier[seq=%d] fast-path: offered=%d kept=%d pages "
                "slots=%d marked=%d in %d us",
                plan.plan_seq, len(plan.pages_to_unmap), per_src,
                kept_cap_t.numel(), moved_to_capped, cap_barrier_us,
            )
            return FireToken(
                plan=plan, src=src, dst=dst,
                src_act=src_act, dst_act=dst_act,
                cap_t=kept_cap_t, cap_slots_count=int(kept_cap_t.numel()),
                cap_barrier_us=cap_barrier_us,
                t_start_ns=t_start,
                unmapped_total=unmapped_total,
                per_src=per_src,
                granted_in_barrier=None,
            )

        # Legacy path — reached only for drain/migration plans (the fast
        # path above returned for free-only plans).
        self._run_stage0(plan, src_act)
        _p1 = time.monotonic_ns()
        cap_slots = src_act.expand_pages_to_token_slots(plan.pages_to_unmap)
        _p2 = time.monotonic_ns()
        cap_t = torch.tensor(cap_slots, dtype=torch.int64, device=alloc.device)
        _p3 = time.monotonic_ns()
        # Free-only-cap invariant: a cross-fire may donate only pages that
        # are GENUINELY FREE. If the Drain selection offered a page whose
        # eviction did not actually free it (a slot still backs a live or
        # evictable radix node), capping + unmapping it would STRAND that
        # cache — cuMemUnmap drops the bytes while the radix node still
        # references the page, so eviction can never reclaim it,
        # `available_size` collapses, and a long-context fill OOMs at low
        # apparent usage. Verify free-ness BEFORE the irreversible mark +
        # unmap and abort the WHOLE fire if violated — fail closed: grant 0
        # this tick, retry when the pages are free. Runs pre-mark because the
        # cap convention leaves capped pages in `free_pages`, so only a
        # pre-mark snapshot tells 'free' from 'referenced'. The one small GPU
        # sync per fire is acceptable (fires are ~1/s) and only over the
        # cap-target slots.
        n_referenced = alloc.count_referenced(cap_t)
        if n_referenced > 0:
            logger.error(
                "cap_barrier[seq=%d] ABORT: %d of %d cap targets "
                "are not free (still backing live/evictable slots) — Drain "
                "selection over-reached; refusing to strand cache.",
                plan.plan_seq, n_referenced, len(cap_slots),
            )
            return FireToken(
                plan=plan, src=src, dst=dst,
                src_act=src_act, dst_act=dst_act,
                cap_t=cap_t, cap_slots_count=len(cap_slots),
                cap_barrier_us=(time.monotonic_ns() - cap_t0) // 1000,
                t_start_ns=t_start, aborted=True,
                abort_reason=(
                    f"{n_referenced} of {len(cap_slots)} cap targets not "
                    f"free (strand guard)"
                ),
            )
        moved_to_capped = alloc.mark_pages_capped(cap_t)
        _p4 = time.monotonic_ns()
        cap_barrier_us = (_p4 - cap_t0) // 1000
        logger.debug(
            "cap_barrier[seq=%d] pages=%d slots=%d marked=%d in %d us "
            "(expand=%dus tensor=%dus mark=%dus)",
            plan.plan_seq, len(plan.pages_to_unmap),
            len(cap_slots), moved_to_capped, cap_barrier_us,
            (_p2-_p1)//1000, (_p3-_p2)//1000, (_p4-_p3)//1000,
        )

        # Compute the LCM-rounded transfer count. cuMemUnmap/Map moved to
        # execute_async (worker thread): free pages are safe to unmap with
        # in-flight kernels (no kernel accesses free VA). Pool metadata
        # (pool.size, unmark) deferred to apply_pending_fires on the next
        # scheduler tick.
        n_src = len(self._all_subpool_names(src))
        n_dst = len(self._all_subpool_names(dst))
        target_src_total = n_src * len(plan.pages_to_unmap)
        target_dst_total = n_dst * plan.pages_to_map_dst
        # Clamp the dst grant to the allocator's physical id-space headroom
        # (max_size - live_size). The arena's chunk-id space is far larger than
        # CappedFreeList.size, so an unclamped grant expands to chunk ids past
        # the ceiling and unmark_token_slots fail-fasts. Feeding the clamped
        # value into `total = min(src, dst)` reduces per_src here in lockstep
        # with per_dst (execute_async), so the src pool is never shrunk more
        # than the dst grows (surplus src pages are restored below).
        target_dst_total = min(
            target_dst_total, n_dst * dst_act.grow_headroom_pages()
        )
        target_total = min(target_src_total, target_dst_total)
        lcm_n = self.lcm_pages
        total = (target_total // lcm_n) * lcm_n
        per_src = total // n_src if n_src else 0
        unmapped_total = per_src * n_src
        self._restore_src_surplus(src_act, plan.pages_to_unmap[per_src:])
        kept_slots = src_act.expand_pages_to_token_slots(
            plan.pages_to_unmap[:per_src]
        )
        kept_cap_t = torch.tensor(
            kept_slots, dtype=torch.int64, device=alloc.device
        )
        _p5 = time.monotonic_ns()
        cap_barrier_us = (_p5 - cap_t0) // 1000

        return FireToken(
            plan=plan, src=src, dst=dst,
            src_act=src_act, dst_act=dst_act,
            cap_t=kept_cap_t, cap_slots_count=len(kept_slots),
            cap_barrier_us=cap_barrier_us,
            t_start_ns=t_start,
            unmapped_total=unmapped_total,
            per_src=per_src,
            granted_in_barrier=None,
        )

    # ---- Phase 2: unmap + map + cap-bump (async; worker thread) ----

    def execute_async(self, token: "FireToken") -> "FirePlanResult":
        """Worker phase: physically cuMemUnmap source pages and cuMemMap
        the freed handles into the destination pool, then bump dst cap.

        Safe to call off-scheduler-thread because (a) the source pages
        are off the allocator's free list (cap_barrier removed them),
        and (b) chunk_arena ctypes calls release the GIL during the
        actual cuMem* syscalls.

        The body acquires `self._fire_inflight` so the Budgeter worker
        thread and the Admitter's sync fire path can't simultaneously
        mutate SharedHandlePool._free_handles. Token state is per-fire and
        already on the stack; only the shared-pool ops need the lock.
        """
        with self._fire_inflight:
            return self._execute_async_locked(token)

    def _execute_async_locked(self, token: "FireToken") -> "FirePlanResult":
        plan = token.plan
        result = FirePlanResult(
            plan_seq=plan.plan_seq,
            direction=plan.direction,
            unmapped_pages=0,
            granted_pages=0,
            cap_barrier_us=token.cap_barrier_us,
            unmap_us=0,
            map_us=0,
            total_us=0,
        )

        if token.aborted:
            result.aborted = True
            result.abort_reason = token.abort_reason
            result.total_us = (time.monotonic_ns() - token.t_start_ns) // 1000
            return result

        src, dst = token.src, token.dst
        src_act, dst_act = token.src_act, token.dst_act

        # Worker-side verify. The `int(...).item()` GPU sync runs here so
        # it doesn't stall the scheduler. If a target page leaked back into
        # the src allocator's free list, roll back the mark and abort before
        # any cuMemUnmap.
        src_alloc = src_act.allocator
        _v1 = time.monotonic_ns()
        # "Did any capped target leak back into the source allocator's
        # free list after cap-barrier?" Delegated to the allocator's
        # own `count_reachable_capped` so this stays agnostic to the
        # KV page-tensor vs mamba slot-tensor representation — the
        # latter is what makes `mamba_to_kv` (mamba source) fires work
        # without reading `free_pages` directly here.
        n_violations = src_alloc.count_reachable_capped(token.cap_t)
        _v2 = time.monotonic_ns()
        logger.debug(
            "execute_async[seq=%d] verify count_reachable_capped = %d us",
            plan.plan_seq, (_v2 - _v1) // 1000,
        )
        if n_violations > 0:
            src_alloc.unmark_pages_capped(token.cap_t)
            reason = (
                f"verify failed: {n_violations} of "
                f"{token.cap_slots_count} target slots still reachable "
                f"in the source free list"
            )
            logger.error(
                "execute_async[seq=%d] ABORT after verify: %s",
                plan.plan_seq, reason,
            )
            result.aborted = True
            result.abort_reason = reason
            result.total_us = (
                time.monotonic_ns() - token.t_start_ns
            ) // 1000
            return result

        # --- unmap + map (fully async on worker thread) ---
        # cuMemUnmap on FREE pages is safe with in-flight GPU kernels: no
        # kernel accesses free-slot VA (verified by count_reachable above).
        # cuMemMap adds new VA (never touches existing pages). Pool metadata
        # (pool.size, unmark_dst) is deferred to apply_pending_fires on the
        # scheduler thread.
        src_names = self._all_subpool_names(src)
        dst_names = self._all_subpool_names(dst)
        n_src = len(src_names)
        n_dst = len(dst_names)
        per_src = token.per_src
        unmapped_total = token.unmapped_total
        lcm_n = self.lcm_pages
        target_src_total = n_src * len(plan.pages_to_unmap)
        target_dst_total = n_dst * plan.pages_to_map_dst
        # Re-clamp to the dst allocator's CURRENT physical headroom (live_size
        # may have advanced since cap_barrier via a prior fire's apply_pending
        # on this thread). Authoritative bound: guarantees granted chunk ids
        # never exceed CappedFreeList.size (mirrors the cap_barrier clamp).
        target_dst_total = min(
            target_dst_total, n_dst * dst_act.grow_headroom_pages()
        )
        target_total = min(target_src_total, target_dst_total)
        total = (target_total // lcm_n) * lcm_n
        per_dst = total // n_dst if n_dst else 0

        unmap_t0 = time.monotonic_ns()
        if per_src > 0 and token.granted_in_barrier is None:
            for name in src_names:
                src._arena.shrink_explicit(name, plan.pages_to_unmap[:per_src])
        result.unmap_us = (time.monotonic_ns() - unmap_t0) // 1000
        result.unmapped_pages = unmapped_total

        map_t0 = time.monotonic_ns()
        granted_ids_per_subpool: list[list[int]] = []
        granted_per_subpool: list[int] = []
        if token.granted_in_barrier is not None:
            granted_ids_per_subpool = token.granted_in_barrier
            granted_per_subpool = [len(ids) for ids in granted_ids_per_subpool]
        else:
            for name in dst_names:
                ids = dst._arena.grow(name, per_dst)
                granted_ids_per_subpool.append(ids)
                granted_per_subpool.append(len(ids))
        result.map_us = (time.monotonic_ns() - map_t0) // 1000
        granted_total = sum(granted_per_subpool)
        result.granted_pages = granted_total

        # `dst._arena.grow(name, per_dst)` returns the ACTUAL chunk IDs
        # granted, which may be empty or shorter than `per_dst` when
        # SharedHandlePool partially exhausts. The exposed cap must
        # reflect the actual number, not the intended one — otherwise
        # the next alloc returns an unmapped slot ID and the next
        # forward pass crashes with CUDA illegal access.
        #
        # Atomicity invariant: every dst sub-pool must hold the same
        # chunk IDs for the new slot range to be safely allocatable.
        # Use `min(granted_per_subpool)` as the safe count, and take
        # the first `actual_per_dst` IDs from sub-pool 0's list as the
        # common set (under the lockstep invariant — all sub-pools
        # grew via `first_free_slot`, so the prefixes match — this is
        # the same set every sub-pool actually mapped).
        actual_per_dst = min(granted_per_subpool) if granted_per_subpool else 0
        common_chunk_ids = (
            granted_ids_per_subpool[0][:actual_per_dst]
            if granted_ids_per_subpool else []
        )
        # Lockstep check: all sub-pools' first `actual_per_dst` IDs must
        # match. chunk_arena.grow allocates at `first_free_slot` positions
        # which line up across sub-pools when the SharedHandlePool free
        # list pops uniformly; if that invariant ever breaks we'd silently
        # expose a slot whose chunk is unmapped in some sub-pool.
        for sub_idx, sub_ids in enumerate(granted_ids_per_subpool):
            if sub_ids[:actual_per_dst] != common_chunk_ids:
                raise RuntimeError(
                    f"execute_async[seq={plan.plan_seq}]: lockstep "
                    f"invariant broken — sub-pool 0 granted IDs "
                    f"{common_chunk_ids} but sub-pool {sub_idx} granted "
                    f"{sub_ids[:actual_per_dst]}. chunk_arena.grow should "
                    f"map at first_free_slot positions, which match across "
                    f"sub-pools under lockstep."
                )
        # Cleanup over-granted chunks. When `granted_per_subpool`
        # is uneven (SharedHandlePool partial exhaustion), the sub-pools
        # that granted more than `actual_per_dst` have cuMemMap'd chunks
        # that are NOT exposed via the cap restore — those handles would
        # leak permanently. Unmap them so handles return to
        # `SharedHandlePool._free_handles`. After this, every sub-pool
        # has exactly `actual_per_dst` more chunks than at fire start.
        for sub_idx, sub_ids in enumerate(granted_ids_per_subpool):
            if len(sub_ids) > actual_per_dst:
                extra = sub_ids[actual_per_dst:]
                dst._arena.shrink_explicit(dst_names[sub_idx], extra)
        # Pipe the actual mapped IDs straight to the dst actuator's uniform
        # ID-based restore API: a single `dst_act.unmark_token_slots(
        # token_slots)` call. KV dispatch routes to
        # `allocator.unmark_pages_capped`; mamba dispatch routes to
        # `pool.unmark_slots`.
        token_slots = dst_act.expand_pages_to_token_slots(common_chunk_ids)
        dst_grow_slots = len(token_slots)
        if actual_per_dst < per_dst:
            logger.warning(
                "execute_async[seq=%d]: granted_per_subpool=%s, min=%d, "
                "per_dst=%d intended — exposing only %d slots to avoid "
                "unmapped-slot crash",
                plan.plan_seq, granted_per_subpool, actual_per_dst,
                per_dst, dst_grow_slots,
            )
        if token.granted_in_barrier:
            logger.info(
                "execute_async[seq=%d] dst restore: %d slots (done in cap_barrier)",
                plan.plan_seq, dst_grow_slots,
            )
        else:
            src_tpc = src_act._tokens_per_page()
            dst_tpc = dst_act._tokens_per_page()
            pending = _PendingApply(
                plan_seq=plan.plan_seq,
                src_pool=src_act.pool,
                dst_pool=dst_act.pool,
                src_tpc=src_tpc,
                dst_tpc=dst_tpc,
                shrunk_pages=per_src,
                grown_pages=actual_per_dst,
                dst_token_slots=list(token_slots),
            )
            with self._pending_lock:
                self._pending_applies.append(pending)
            logger.info(
                "execute_async[seq=%d] dst restore: %d slots (deferred to apply_pending)",
                plan.plan_seq, dst_grow_slots,
            )
        result.total_us = (time.monotonic_ns() - token.t_start_ns) // 1000
        logger.info(
            "execute_async[seq=%d] DONE dir=%s unmapped=%d granted=%d "
            "cap=%dus unmap=%dus map=%dus total=%dus",
            plan.plan_seq, plan.direction, result.unmapped_pages,
            result.granted_pages, result.cap_barrier_us,
            result.unmap_us, result.map_us, result.total_us,
        )
        return result

    def _restore_src_surplus(self, src_act, surplus_pages) -> None:
        """Return the cap-barrier mark on un-transferred src pages to the src
        allocator's free list.

        cap_barrier caps every page in `plan.pages_to_unmap`; the LCM-floor in
        `_execute_async_locked` then unmaps only the first `per_src` of them.
        `surplus_pages` (`pages_to_unmap[per_src:]`) were never unmapped, so
        their chunks stay mapped and their slots stay allocatable. Without this
        restore they linger in the src pool's capped set forever, eroding
        `live_size` by `len(surplus_pages) * slots_per_page` per fire.

        Runs under `_fire_inflight` (the caller holds it). Routes through the
        src allocator's uniform `unmark_pages_capped`: KV → `CappedFreeList`
        unmark, mamba → `_MambaCapAllocator.unmark_pages_capped`.
        """
        if not surplus_pages:
            return
        surplus_slots = src_act.expand_pages_to_token_slots(surplus_pages)
        if not surplus_slots:
            return
        src_alloc = src_act.allocator
        restore_t = torch.tensor(
            surplus_slots, dtype=torch.int64, device=src_alloc.device,
        )
        src_alloc.unmark_pages_capped(restore_t)

    # ---- Deferred apply (scheduler thread) -----------------------------

    def apply_pending_fires(self) -> int:
        """Apply pool metadata from completed async fires.

        Called on the scheduler thread (e.g. in BudgetAgent.tick).
        Updates pool.size and unmarkes dst slots for each completed fire.
        Returns the number of pending entries applied.
        """
        with self._pending_lock:
            batch = self._pending_applies
            self._pending_applies = []
        if not batch:
            return 0
        for p in batch:
            shrunk_tokens = p.shrunk_pages * p.src_tpc
            grown_tokens = p.grown_pages * p.dst_tpc
            p.src_pool.size = max(0, p.src_pool.size - shrunk_tokens)
            p.dst_pool.size += grown_tokens
            dst_act = self._resolve_dst_act(p.dst_pool)
            dst_act.unmark_token_slots(p.dst_token_slots)
            logger.info(
                "apply_pending[seq=%d] src.size-=%d dst.size+=%d unmarked=%d",
                p.plan_seq, shrunk_tokens, grown_tokens, len(p.dst_token_slots),
            )
        return len(batch)

    def _resolve_dst_act(self, dst_pool):
        if self.kv_actuator is not None and self.kv_actuator.pool is dst_pool:
            return self.kv_actuator
        return self.mamba_actuator

    # ---- Plan-based execution ----------------------------------------

    def execute(self, plan: "FirePlan") -> "FirePlanResult":
        """Execute a planner-produced FirePlan synchronously. Three steps,
        no callbacks:

          1. cap-barrier — translate page-ids to allocator token-slots and
                           mark them off-limits, blocking concurrent allocs;
          2. verify      — confirm no target page leaked back into free;
          3. unmap + map — physically `cuMemUnmap` source pages, `cuMemMap`
                           the freed handles into the destination pool.

        Equivalent to `execute_async(cap_barrier(plan))` PLUS an inline
        `apply_pending_fires()`: this is the SYNCHRONOUS entrypoint, so the
        transfer must be fully complete (physical map + metadata) when it
        returns. Its callers -- the on-demand grows (`_grow_kv_from_mamba` /
        `_grow_mamba_from_kv`, invoked from `alloc_token_slots` when the live
        KV cap is exhausted) and the Admitter -- retry `allocator.alloc()`
        IMMEDIATELY on return; if the dst unmark were left deferred to the next
        Budgeter tick, the retry would still see the pre-grow capacity and raise
        a spurious 'Out of memory' even though the grow physically succeeded
        (observed as the rep2 OOM on a repeated KV-bound burst). The async
        Budgeter-worker path (cap_barrier + execute_async, applied on the tick)
        is unaffected.
        """
        token = self.cap_barrier(plan)
        result = self.execute_async(token)
        self.apply_pending_fires()
        return result

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
            "shared_total_handles": self.shared.total_count(),
            "shared_free_handles": self.shared.free_count(),
        }
