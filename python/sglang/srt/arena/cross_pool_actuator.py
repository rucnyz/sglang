"""
Phase 2e.5.6 — CrossPoolTransferActuator: KV ↔ mamba physical-handle migration.

Sits on top of two MultiTensorArena instances that share one
SharedHandlePool (Phase 2e.5.6.1). Exposes:

  - kv_to_mamba_chunks(n_per_kv_subpool):
        Shrinks each KV sub-pool by `n` chunks (frees `n * n_kv_subpools`
        handles into the shared pool), then grows each mamba sub-pool
        by `floor(n_freed / n_mamba_subpools)` chunks. Any remainder
        stays in the shared pool's free list and is available for the
        next call (or the reverse direction).

  - mamba_to_kv_chunks(n_per_mamba_subpool):
        Symmetric.

The asymmetry — KV has `n_kv_layers * 2` sub-pools (k, v per layer),
mamba has `n_mamba_layers * 1` (temporal per layer) — is handled by
keeping per-call grow rounding to the floor. Engineering rationale:
the planner's "budget" is in tokens-of-capacity per pool, which maps
to "live capacity = min mapped chunks * tokens_per_chunk" inside each
MultiTensorArena. The min-across-subpools requirement is what forces
us to grow (or shrink) all sub-pools by the same amount.

The planner is policy-side; this actuator only handles the mechanical
"move these many chunks from arena A to arena B" operation.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.arena.chunk_arena import cross_arena_transfer


def _select_drainable_chunks(
    src_act, n_chunks: int, tokens_per_chunk: int
) -> list[int]:
    """T3 (paper §3.2.2): pick `n_chunks` chunk indices on `src_act` whose
    ALL pages are currently free in the allocator's free-page mask.

    Returns the highest-indexed such chunks (preferring tail under T2's
    placement bias). May return fewer than `n_chunks` if not enough chunks
    are fully drainable.

    Returns an empty list and logs at INFO if no `select_drain_pages` API
    is available (e.g., when the allocator is not the BaseTokenToKVPoolAllocator).
    """
    alloc = getattr(src_act, "allocator", None) if src_act is not None else None
    if alloc is None or not hasattr(alloc, "free_page_mask"):
        return []
    if tokens_per_chunk <= 0:
        return []
    mask = alloc.free_page_mask()
    # Reshape: mask[1:1 + n_chunks * tokens_per_chunk] groups consecutive
    # `tokens_per_chunk` pages into one chunk. A chunk is drainable iff
    # all its pages are free.
    n_pages = alloc.size
    n_total_chunks = n_pages // tokens_per_chunk
    if n_total_chunks == 0:
        return []
    pages = mask[1:1 + n_total_chunks * tokens_per_chunk]
    chunks = pages.view(n_total_chunks, tokens_per_chunk).all(dim=1)
    drainable = torch.where(chunks)[0].tolist()
    # Highest-indexed chunks first → contiguous tail unmap.
    drainable.sort(reverse=True)
    return drainable[:n_chunks]

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
        kv_actuator=None,        # Optional[KVArenaActuator]
        mamba_actuator=None,     # Optional[MambaArenaActuator]
        kv_live_slot_inspector=None,    # Optional[Callable[[int], int]]
        mamba_live_slot_inspector=None, # Optional[Callable[[int], int]]
    ) -> None:
        self.kv = kv_arena
        self.mamba = mamba_arena
        self.shared = shared_pool
        # Phase 2e.5.6.3: when both per-pool actuators are provided, the
        # cross-pool actuator coordinates capacity changes with the
        # allocators so the engine respects the new capacities (live-
        # traffic safe, modulo the busy-engine gate at the budgeter level).
        # When omitted, falls back to the 2e.5.6.2 "raw chunk move only"
        # behavior — safe only in idle windows.
        self.kv_actuator = kv_actuator
        self.mamba_actuator = mamba_actuator

        # Rigorous drain check: in-flight slot ids held by running reqs
        # are NOT in any allocator free buffer. Without scheduler-side
        # introspection, _drain_complete's accounting can return True
        # while reqs still hold ids > new_cap → cuMemUnmap kills active
        # KV → cudaErrorIllegalAddress on next kernel. Each inspector is
        # callable(new_cap_pages: int) -> int, returning the count of
        # in-flight slot ids ≥ new_cap_pages on its respective pool.
        # When > 0, drain is incomplete regardless of accounting.
        self._kv_live_slot_inspector = kv_live_slot_inspector
        self._mamba_live_slot_inspector = mamba_live_slot_inspector

        # Drain protocol state (paper §design-l2-actuator). When a fire
        # decides to shrink src, we cap src's allocator to new_cap so no
        # new req can be admitted to the tail [new_cap, old_cap]. The
        # actual cuMemUnmap+cuMemMap is deferred until in-flight reqs
        # holding tail slots have completed naturally — drain check
        # compares len(src_allocator._capped_pages) to (size - new_cap).
        # While pending, no new fire is admitted; planner.decide is
        # bypassed in agent.py.
        self._pending: dict | None = None

        if kv_arena._arena._external_pool is not shared_pool:
            raise ValueError("kv_arena does not use the provided shared_pool")
        if mamba_arena._arena._external_pool is not shared_pool:
            raise ValueError("mamba_arena does not use the provided shared_pool")

        self.n_kv_subpools = kv_arena.n_layers * kv_arena.n_kinds
        self.n_mamba_subpools = mamba_arena.n_layers * mamba_arena.n_kinds

        # Balanced unit (lcm-aware): the smallest dst_per_subpool that
        # makes the transfer leftover-free (= no handles accumulate in
        # the shared pool after a round-trip). For dst with n_dst sub-pools
        # and src with n_src, we need n_per_dst * n_dst == n_per_src * n_src
        # to balance. The smallest n_per_dst integer satisfying this is
        # n_src // gcd(n_src, n_dst); the corresponding n_per_src is
        # n_dst // gcd(n_src, n_dst).
        g = math.gcd(self.n_kv_subpools, self.n_mamba_subpools)
        # kv → mamba: dst=mamba (count=n_mamba), src=kv (count=n_kv).
        self.balanced_unit_kv_to_mamba_dst = self.n_kv_subpools // g
        self.balanced_unit_kv_to_mamba_src = self.n_mamba_subpools // g
        # mamba → kv: dst=kv, src=mamba.
        self.balanced_unit_mamba_to_kv_dst = self.n_mamba_subpools // g
        self.balanced_unit_mamba_to_kv_src = self.n_kv_subpools // g

        logger.info(
            "CrossPoolTransferActuator: kv_subpools=%d, mamba_subpools=%d, "
            "shared_handles=%d, free=%d, balanced_unit_kv2m=(dst=%d,src=%d), "
            "balanced_unit_m2kv=(dst=%d,src=%d)",
            self.n_kv_subpools, self.n_mamba_subpools,
            self.shared.total_count(), self.shared.free_count(),
            self.balanced_unit_kv_to_mamba_dst,
            self.balanced_unit_kv_to_mamba_src,
            self.balanced_unit_mamba_to_kv_dst,
            self.balanced_unit_mamba_to_kv_src,
        )

    # ------------------------------------------------------------------

    def _all_subpool_names(self, mta: "MultiTensorArena") -> list[str]:
        n = mta.n_layers * mta.n_kinds
        return [mta._pool_name(i) for i in range(n)]

    def _src_actuator(self, src: "MultiTensorArena"):
        return self.kv_actuator if src is self.kv else self.mamba_actuator

    def _dst_actuator(self, dst: "MultiTensorArena"):
        return self.kv_actuator if dst is self.kv else self.mamba_actuator

    def _drain_complete(self, src_act, new_cap_tokens: int) -> bool:
        """Drain protocol check (paper §design-l2-actuator L184).

        Returns True iff no in-flight request still references a slot
        with id > new_cap. The check is computed by accounting:

          in_use_above = (size - new_cap)
                       - capped_above
                       - release_pages_above
                       - free_group_above
                       - free_pages_above

        At drain completion, in_use_above == 0, equivalently the right-
        hand side accumulators >= (size - new_cap). The previous version
        checked only `capped_above` — but SGLang's allocator can hold
        pages > new_cap in three other places that aren't in_use:

        1. `_capped_pages` — explicit tail buffer (cap-aware free routes
           freed-above-cap entries here)
        2. `release_pages` — reqs that freed via `is_not_in_free_group=True`
           with `need_sort=True` go through release_pages first, then
           merge into free_pages later. After cap, release entries can
           still hold ids > new_cap.
        3. `free_group` (a Python list) — batched frees pending flush via
           `free_group_end`. Same situation.
        4. `free_pages` — after cap, set_capacity_pages drops above-cap
           ids out of free, but if some were in release at cap time and
           later flush back, they re-enter free_pages.

        Counting all four covers the cases where pages > new_cap have
        been fully released by their owning requests but haven't yet
        landed in `_capped_pages`. If any slot id > new_cap is genuinely
        in_use (held by a still-running req), the right-hand side falls
        short and drain is not complete.
        """
        if src_act is None:
            return True  # No allocator coord — no drain needed.
        # KV: src_act.allocator with .size, ._capped_pages, .release_pages, .free_group, .free_pages
        # Mamba: src_act.pool with .size, ._capped_slots, .free_slots
        import torch  # local import to avoid module-level dep cycles
        # Pick the inspector that matches the source pool. We can't tell
        # KV vs mamba just from src_act, so use identity on the cached
        # actuators we were given at init.
        if src_act is self.kv_actuator:
            live_inspector = self._kv_live_slot_inspector
        elif src_act is self.mamba_actuator:
            live_inspector = self._mamba_live_slot_inspector
        else:
            live_inspector = None
        alloc = getattr(src_act, "allocator", None)
        if alloc is not None:
            page_size = max(1, src_act.pool.page_size)
            new_cap_pages = new_cap_tokens // page_size
            new_cap_pages = min(new_cap_pages, alloc.size)
            expected = alloc.size - new_cap_pages
            if expected <= 0:
                return True

            # Rigorous live-slot check: walk in-flight reqs first. If any
            # holds a KV slot id > new_cap_pages, drain is not complete —
            # those slots' physical handles must not be unmapped.
            if live_inspector is not None:
                try:
                    live_above = int(live_inspector(new_cap_pages))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "live_slot_inspector raised %r — treating as "
                        "drain-not-complete (conservative)", e,
                    )
                    return False
                if live_above > 0:
                    logger.info(
                        "_drain_complete: ok=False (live in-flight slots "
                        "above new_cap_pages=%d count=%d)",
                        new_cap_pages, live_above,
                    )
                    return False

            def _count_above(t, threshold):
                if t is None or t.numel() == 0:
                    return 0
                return int((t > threshold).sum().item())

            capped_above = _count_above(
                getattr(alloc, "_capped_pages", None), new_cap_pages
            )
            release_above = _count_above(
                getattr(alloc, "release_pages", None), new_cap_pages
            )
            # free_pages should have NO ids > new_cap after cap_allocator_only,
            # but a subsequent merge_and_sort_free can reintroduce them from
            # release_pages. Re-count to be safe.
            free_above = _count_above(
                getattr(alloc, "free_pages", None), new_cap_pages
            )
            # free_group is a list[Tensor]; iterate and count.
            free_group_above = 0
            free_group = getattr(alloc, "free_group", None)
            if free_group:
                for t in free_group:
                    free_group_above += _count_above(t, new_cap_pages)
            total_above_freed = (
                capped_above + release_above + free_above + free_group_above
            )
            in_use_above = expected - total_above_freed
            ok = total_above_freed >= expected
            logger.info(
                "_drain_complete: ok=%s expected=%d total_freed_above=%d "
                "(capped=%d release=%d free=%d free_group=%d) in_use_above=%d "
                "(alloc.size=%d new_cap_pages=%d, free_pages.numel=%d, "
                "release.numel=%d, capped.numel=%d)",
                ok, expected, total_above_freed,
                capped_above, release_above, free_above, free_group_above,
                in_use_above, alloc.size, new_cap_pages,
                getattr(getattr(alloc, "free_pages", None), "numel", lambda: 0)(),
                getattr(getattr(alloc, "release_pages", None), "numel", lambda: 0)(),
                getattr(getattr(alloc, "_capped_pages", None), "numel", lambda: 0)(),
            )
            return ok

        pool = getattr(src_act, "pool", None)
        if pool is not None:
            new_cap_slots = min(new_cap_tokens, pool.size)
            expected = pool.size - new_cap_slots
            if expected <= 0:
                return True

            # Same rigorous live-slot check as KV side: walk in-flight reqs
            # holding mamba slots > new_cap. Paper §190 ("DeltaNet/SSM slot
            # shrink: mark slots above the new cap as 'not-for-reuse';
            # release as owning requests complete").
            if live_inspector is not None:
                try:
                    live_above = int(live_inspector(new_cap_slots))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "mamba live_slot_inspector raised %r — treating as "
                        "drain-not-complete (conservative)", e,
                    )
                    return False
                if live_above > 0:
                    logger.info(
                        "_drain_complete (mamba): ok=False (live in-flight "
                        "slots above new_cap_slots=%d count=%d)",
                        new_cap_slots, live_above,
                    )
                    return False

            capped = getattr(pool, "_capped_slots", None)
            free_slots = getattr(pool, "free_slots", None)

            def _count_above(t, threshold):
                if t is None:
                    return 0
                # MambaPool free_slots can be torch.Tensor; _capped_slots same.
                if hasattr(t, "numel"):
                    if t.numel() == 0:
                        return 0
                    return int((t > threshold).sum().item())
                # Fallback for list/other.
                return sum(1 for v in t if v > threshold)

            capped_above = _count_above(capped, new_cap_slots)
            free_above = _count_above(free_slots, new_cap_slots)
            return (capped_above + free_above) >= expected
        # Unknown actuator shape; conservative: not drained.
        return False

    def _do_transfer(
        self,
        src: "MultiTensorArena",
        dst: "MultiTensorArena",
        n_per_dst_subpool: int,
        direction_label: str,
    ) -> dict:
        """Grow every dst sub-pool by `n_per_dst_subpool` chunks; this
        requires shrinking each src sub-pool by
        `ceil(n_per_dst_subpool * n_dst_subpools / n_src_subpools)`. Any
        leftover unmapped handles stay in the shared pool's free list and
        are available for the next call.

        Why dst-anchored (not src-anchored):
          live capacity of an MTA is min mapped chunks across its
          sub-pools. If we shrank src by 1 chunk per src-subpool but
          src has more sub-pools than dst, dst would only grow by
          floor(n_src/n_dst) per dst-subpool, which is 0 when n_src <
          n_dst (e.g., KV's 20 sub-pools < mamba's 30). Making the
          caller specify dst-side guarantees the transfer always
          actually grows dst.

        Returns: stats dict.
        """
        if n_per_dst_subpool <= 0:
            raise ValueError(
                f"n_per_dst_subpool={n_per_dst_subpool} must be > 0"
            )

        src_names = self._all_subpool_names(src)
        dst_names = self._all_subpool_names(dst)
        n_src = len(src_names)
        n_dst = len(dst_names)

        # Phase 2e.5.6.3.c bug fix: bail when dst is already at max OR
        # src is already at min. Without this, we'd shrink src for nothing
        # (dst can't grow, all the freed handles get stranded in the
        # shared free pool, src capacity collapses toward 0).
        dst_min_mapped = min(
            dst._arena.pool_mapped_chunks(name) for name in dst_names
        )
        if dst_min_mapped + n_per_dst_subpool > dst.max_chunks_per_pool:
            return {
                "direction": direction_label,
                "n_per_src_subpool": 0,
                "n_per_dst_subpool": n_per_dst_subpool,
                "src_subpools": n_src,
                "dst_subpools": n_dst,
                "unmapped_total": 0,
                "granted_total": 0,
                "free_before": self.shared.free_count(),
                "free_after_shrink": self.shared.free_count(),
                "free_after_grow": self.shared.free_count(),
                "kv_capacity_tokens": self.kv.current_capacity_tokens(),
                "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
                "skipped": "dst_at_max",
            }

        # How many chunks must each src sub-pool shed to free enough
        # handles for the dst grow? Subtract whatever's already free in
        # the shared pool first — under the mobile-soft split, that's
        # where (init − static_min) chunks per arena live initially, so
        # in many cases dst can grow purely from mobile soft without
        # touching src. Whatever remains is divided equally across src
        # sub-pools (ceil so dst grows fully).
        needed = n_per_dst_subpool * n_dst
        shared_free = self.shared.free_count()
        needed_from_src = max(0, needed - shared_free)
        n_per_src_subpool = (needed_from_src + n_src - 1) // n_src if n_src > 0 else 0

        src_min_mapped = min(
            src._arena.pool_mapped_chunks(name) for name in src_names
        )
        # Static-min floor (paper §design-l2-actuator L184): the actuator
        # refuses any shrink that would drop a sub-pool below static_min.
        # When shared_arena=True, memory_pool sets static_min=1 chunk per
        # sub-pool, leaving (init - 1) chunks per sub-pool transferable
        # via drain protocol. When shared_arena=False, static_min=init and
        # the actuator can't shrink (engine behaves identically to non-L2
        # baseline).
        static_min = src.static_min_chunks_per_pool
        if src_min_mapped - n_per_src_subpool < static_min:
            return {
                "direction": direction_label,
                "n_per_src_subpool": 0,
                "n_per_dst_subpool": n_per_dst_subpool,
                "src_subpools": n_src,
                "dst_subpools": n_dst,
                "unmapped_total": 0,
                "granted_total": 0,
                "free_before": self.shared.free_count(),
                "free_after_shrink": self.shared.free_count(),
                "free_after_grow": self.shared.free_count(),
                "kv_capacity_tokens": self.kv.current_capacity_tokens(),
                "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
                "skipped": f"src_at_static_min (would drop {src_min_mapped} → {src_min_mapped - n_per_src_subpool} below static_min={static_min})",
            }

        # Phase 2e.5.6.3: if per-pool actuators are wired, coordinate
        # with the allocators. The contract:
        #   1. BEFORE shrinking the src arena physically, cap the src
        #      pool's allocator to its new capacity (= live - shrink
        #      tokens). New requests are immediately refused tail slots;
        #      already-allocated slots in the soon-to-be-unmapped tail
        #      MUST have been drained by the busy-engine gate at the
        #      budgeter level (or, for the demo, we just refuse to fire
        #      while engine is busy).
        #   2. Do the chunk move (src.shrink + dst.grow).
        #   3. AFTER the dst arena has new physical chunks, raise the
        #      dst pool's allocator cap so the new slots become
        #      allocatable.
        src_act = self._src_actuator(src)
        dst_act = self._dst_actuator(dst)
        # Translate chunk counts to token counts for the actuator API.
        src_tokens_per_chunk = src.tokens_per_chunk
        dst_tokens_per_chunk = dst.tokens_per_chunk
        src_shrink_tokens = n_per_src_subpool * src_tokens_per_chunk
        dst_grow_tokens = n_per_dst_subpool * dst_tokens_per_chunk

        # Paper §design-l2-actuator drain protocol. When src needs to
        # shrink, in-flight reqs may currently hold slots above the new
        # cap. We cap the allocator (no new admit to tail) and then
        # check drain status: all pages above new_cap should be in
        # `_capped_pages` (the cap-aware free path puts there). If
        # drain is not yet complete, abort this fire (un-cap) and the
        # gate retries on a future tick (after cooldown).
        if src_act is not None and n_per_src_subpool > 0:
            old_src_cap = src_act.live_capacity_tokens()
            new_src_cap = max(1, old_src_cap - src_shrink_tokens)
            src_act.cap_allocator_only(new_src_cap)
            if not self._drain_complete(src_act, new_src_cap):
                # Restore allocator cap; gate will retry next admissible tick.
                src_act.cap_allocator_only(old_src_cap)
                return {
                    "direction": direction_label,
                    "n_per_src_subpool": 0,
                    "n_per_dst_subpool": n_per_dst_subpool,
                    "src_subpools": n_src,
                    "dst_subpools": n_dst,
                    "unmapped_total": 0,
                    "granted_total": 0,
                    "free_before": self.shared.free_count(),
                    "free_after_shrink": self.shared.free_count(),
                    "free_after_grow": self.shared.free_count(),
                    "kv_capacity_tokens": self.kv.current_capacity_tokens(),
                    "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
                    "skipped": "drain_pending",
                }

        # Stage 1 actuator-cost instrumentation: wall-time the cuMemUnmap
        # (src.shrink) and cuMemMap (dst.grow) operations so gate config's
        # chunk_cost_us can be calibrated against real measurements rather
        # than the conservative paper-default 50ms/chunk × 60 = 3s.
        # cuda.synchronize bracket so the wall time reflects GPU work, not
        # just CPU enqueue.
        free_before = self.shared.free_count()
        unmapped_total = 0
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        shrink_t0 = time.monotonic_ns()
        # T3 (paper §3.2.2): SGLANG_SMART_OVERCAP=1 picks chunks whose
        # all pages are free in the allocator (anywhere in the pool, not
        # just tail). Default path keeps legacy "shrink tail" semantic.
        smart_overcap = os.environ.get("SGLANG_SMART_OVERCAP", "0") == "1"
        smart_chunks = []
        if smart_overcap and src_act is not None:
            tpc = getattr(src_act, "tokens_per_chunk", None)
            if tpc is None:
                pool = getattr(src_act, "pool", None)
                if pool is not None:
                    tpc = getattr(pool, "tokens_per_chunk", 1)
            if tpc and tpc > 0:
                smart_chunks = _select_drainable_chunks(
                    src_act, n_per_src_subpool, int(tpc)
                )
                logger.info(
                    "T3 smart over-cap selection: tokens_per_chunk=%d, "
                    "requested=%d, drainable=%d (%s)",
                    tpc, n_per_src_subpool, len(smart_chunks),
                    "tail" if all(c >= len(src._arena.pools[src_names[0]].mapped) - n_per_src_subpool
                                  for c in smart_chunks) else "non-tail",
                )
        if smart_overcap and len(smart_chunks) >= n_per_src_subpool:
            for name in src_names:
                unmapped_total += src._arena.shrink_explicit(name, smart_chunks)
        else:
            # Default tail-shrink path. Falls back here when smart_overcap
            # is off, when the allocator can't supply enough drainable
            # chunks, or when the source actuator lacks the API.
            for name in src_names:
                unmapped_total += src._arena.shrink(name, n_per_src_subpool)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        shrink_t1 = time.monotonic_ns()
        free_after_shrink = self.shared.free_count()

        # Grow every dst sub-pool by exactly n_per_dst_subpool. Anything
        # we couldn't grant (because src didn't free enough handles) is
        # tracked in the stats; the leftover stays in the shared free
        # list for the next call.
        granted_total = 0
        grow_t0 = time.monotonic_ns()
        for name in dst_names:
            granted_total += dst._arena.grow(name, n_per_dst_subpool)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        grow_t1 = time.monotonic_ns()

        free_after_grow = self.shared.free_count()
        shrink_us = (shrink_t1 - shrink_t0) // 1000
        grow_us = (grow_t1 - grow_t0) // 1000

        if dst_act is not None:
            new_dst_cap = dst_act.live_capacity_tokens() + dst_grow_tokens
            # Same reasoning as above — only un-cap the allocator side;
            # the dst arena was already grown above by `dst._arena.grow`.
            dst_act.cap_allocator_only(new_dst_cap)

        stats = {
            "direction": direction_label,
            "n_per_src_subpool": n_per_src_subpool,
            "n_per_dst_subpool": n_per_dst_subpool,
            "src_subpools": n_src,
            "dst_subpools": n_dst,
            "unmapped_total": unmapped_total,
            "granted_total": granted_total,
            "free_before": free_before,
            "free_after_shrink": free_after_shrink,
            "free_after_grow": free_after_grow,
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
            "shrink_us": shrink_us,
            "grow_us": grow_us,
            "fire_total_us": shrink_us + grow_us,
        }
        logger.info(
            "CrossPoolTransferActuator.%s: shrank %d/src=%d → freed %d (%.1f ms), "
            "grew %d/dst=%d → consumed %d (%.1f ms), total %.1f ms, "
            "leftover free %d → KV cap=%d tok, mamba cap=%d tok",
            direction_label,
            n_per_src_subpool, n_src, unmapped_total, shrink_us / 1000.0,
            n_per_dst_subpool, n_dst, granted_total, grow_us / 1000.0,
            (shrink_us + grow_us) / 1000.0,
            free_after_grow,
            stats["kv_capacity_tokens"], stats["mamba_capacity_tokens"],
        )
        return stats

    # ------------------------------------------------------------------

    def kv_to_mamba_chunks(self, n_per_mamba_subpool: int) -> dict:
        """Grow mamba by `n` chunks per mamba sub-pool, sourcing handles
        from KV via the shared pool. KV sheds
        `ceil(n * n_mamba_subpools / n_kv_subpools)` chunks per KV
        sub-pool (rounded up so dst grows fully). Any leftover handles
        stay in the shared free list.
        """
        return self._do_transfer(
            src=self.kv, dst=self.mamba,
            n_per_dst_subpool=n_per_mamba_subpool,
            direction_label="kv_to_mamba",
        )

    def mamba_to_kv_chunks(self, n_per_kv_subpool: int) -> dict:
        """Symmetric: grow KV by `n` chunks per KV sub-pool, sourcing from
        mamba. See `kv_to_mamba_chunks`.
        """
        return self._do_transfer(
            src=self.mamba, dst=self.kv,
            n_per_dst_subpool=n_per_kv_subpool,
            direction_label="mamba_to_kv",
        )

    # ---- Balanced (leftover-free) wrappers ---------------------------

    def balanced_kv_to_mamba(self, multiplier: int = 1) -> dict:
        """`kv_to_mamba_chunks` at the balanced unit, scaled by `multiplier`.

        Balanced means the source-shrink and destination-grow consume
        exactly the same number of handles, so the shared pool's free
        count doesn't drift. Use this in oscillator-style demos where
        every kv_to_mamba is matched by a balanced_mamba_to_kv: round-trip
        leaves both pools and the free pool at their starting state.
        """
        return self.kv_to_mamba_chunks(
            self.balanced_unit_kv_to_mamba_dst * multiplier
        )

    def balanced_mamba_to_kv(self, multiplier: int = 1) -> dict:
        """Symmetric balanced wrapper. See `balanced_kv_to_mamba`."""
        return self.mamba_to_kv_chunks(
            self.balanced_unit_mamba_to_kv_dst * multiplier
        )

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
            "shared_total_handles": self.shared.total_count(),
            "shared_free_handles": self.shared.free_count(),
        }
