"""T8 step 4 — Scheduler-side OwnerProvider.

Walks the live engine's data structures to build a per-pool `OwnerMap`:

  - free / release pages from `BaseTokenToKVPoolAllocator.free_pages` /
    `release_pages`.
  - capped pages from `_capped_pages` (non-empty only when a fire is
    mid-flight; planner refuses to fire while any pages are capped).
  - active pages by walking `running_batch.reqs` and `waiting_queue`
    (the latter restricted to reqs that have already been assigned a
    `req_pool_idx`, i.e. already started prefill).
  - tree pages by DFS over `tree_cache.root_node` collecting each
    node's `value` tensor — but skipping any page that is also active
    (sglang's prefix lock means tree-cached pages owned by the
    matching req's first segment are double-listed; we resolve to ACTIVE
    because migrating them is the only safe op).

The provider does NOT import `sglang.srt.managers.scheduler` — instead it
takes a `scheduler` instance ducktyped to the attributes we need
(`running_batch`, `waiting_queue`, `tree_cache`, `req_to_token_pool`,
`token_to_kv_pool_allocator`). This keeps the budgeter package importable
without pulling the scheduler module.

Construction is the only side effect; calls are cheap (~1ms on H200 conc=800
based on req_count × seqlen) and run between forward steps.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.arena.owner_provider import OwnerMap, OwnerProvider, TreeNodeRef

logger = logging.getLogger(__name__)


class SchedulerOwnerProvider:
    """Implements `OwnerProvider` against a live `Scheduler`.

    Construction:
      provider = SchedulerOwnerProvider(scheduler)

    Usage (called by XPoolFirePlanner.build, scheduler lock held):
      kv_om = provider.build_kv_owner_map()
      # kv_om.assert_complete() will raise if coverage is broken.
    """

    def __init__(self, scheduler) -> None:
        # Ducktyped — we need running_batch, waiting_queue, tree_cache,
        # req_to_token_pool, token_to_kv_pool_allocator.
        self._scheduler = scheduler

    # ------------------------------------------------------------------

    def build_kv_owner_map(self) -> OwnerMap:
        sched = self._scheduler
        allocator = sched.token_to_kv_pool_allocator
        n_pages = int(allocator.size)

        # --- 1. free / release / capped pages -----------------------------
        free_set = self._tensor_to_set(getattr(allocator, "free_pages", None))
        release_set = self._tensor_to_set(getattr(allocator, "release_pages", None))
        capped_set = self._tensor_to_set(getattr(allocator, "_capped_pages", None))
        free_set |= release_set  # both count as available for drain

        # --- 2. active pages (running batch + already-prefilled queue) ---
        active_pages: dict[int, tuple[int, int]] = {}
        req_to_token = sched.req_to_token_pool.req_to_token
        max_ctx = req_to_token.shape[1]

        # Reqs to walk: running batch always; waiting queue only when the
        # req has been admitted enough to claim a req_pool_idx slot
        # (chunked-prefill continuation, paused with KV preserved, etc.).
        reqs: list = list(getattr(sched.running_batch, "reqs", []) or [])
        for r in getattr(sched, "waiting_queue", []) or []:
            if getattr(r, "req_pool_idx", None) is not None:
                reqs.append(r)

        # Batch the gather: build (req_pool_idxs, seqlens) and slice
        # req_to_token in one go. For conc=800 reqs × seqlen=4K this is
        # ~3M int32 reads = sub-ms even with tolist() overhead.
        for req in reqs:
            idx = getattr(req, "req_pool_idx", None)
            if idx is None:
                continue
            seqlen = self._req_seqlen(req, max_ctx)
            if seqlen <= 0:
                continue
            # .cpu().tolist() is the bottleneck; we do one-row-at-a-time
            # to avoid building a large CPU-host buffer for the whole pool.
            row = req_to_token[idx, :seqlen].cpu().tolist()
            for slot, p in enumerate(row):
                if p == 0:
                    continue  # 0 is the null-sentinel page
                active_pages[int(p)] = (int(idx), slot)

        # --- 3. tree pages (DFS, skip pages already active) --------------
        tree_pages: dict[int, TreeNodeRef] = {}
        tree_cache = getattr(sched, "tree_cache", None)
        root = getattr(tree_cache, "root_node", None) if tree_cache is not None else None
        if root is not None:
            self._walk_tree(root, tree_pages, active_pages)

        return OwnerMap(
            pool_name="kv",
            n_pages=n_pages,
            free_pages=free_set,
            tree_pages=tree_pages,
            active_pages=active_pages,
            capped_pages=capped_set,
        )

    # ------------------------------------------------------------------

    def build_mamba_owner_map(self) -> Optional[OwnerMap]:
        """Walk the mamba pool's slot-id space.

        Mamba uses slots, not page-ids: 1 slot = 1 chunk at page-grain
        VMM (T1), `n_pages == mamba_pool.size`. Slot ownership:
          - free: `MambaPool.free_slots` tensor.
          - capped: `MambaPool._capped_slots` tensor (mid-fire holdouts).
          - active: each req's `mamba_pool_idx` (1-element tensor — one
            mamba slot per req).
          - tree: skipped — mamba radix caches (e.g. MambaRadixCache)
            keep their own bookkeeping; for now the OwnerMap simply
            classifies anything not free/active/capped as live (held by
            something we can't migrate). Coverage assert will catch a
            non-empty residual and the planner will refuse, which is
            the safe behaviour until per-cache walkers land.
        """
        sched = self._scheduler
        mamba_pool = self._find_mamba_pool(sched)
        if mamba_pool is None:
            return None
        n_pages = int(mamba_pool.size)

        free_set = self._tensor_to_set(getattr(mamba_pool, "free_slots", None))
        capped_set = self._tensor_to_set(getattr(mamba_pool, "_capped_slots", None))

        active_pages: dict[int, tuple[int, int]] = {}
        reqs: list = list(getattr(sched.running_batch, "reqs", []) or [])
        for r in getattr(sched, "waiting_queue", []) or []:
            if getattr(r, "mamba_pool_idx", None) is not None:
                reqs.append(r)
        for req in reqs:
            mp = getattr(req, "mamba_pool_idx", None)
            if mp is None:
                continue
            try:
                slot = int(mp[0]) if hasattr(mp, "__getitem__") else int(mp)
            except (TypeError, IndexError):
                continue
            if slot == 0:
                continue
            req_idx = int(getattr(req, "req_pool_idx", -1))
            active_pages[slot] = (req_idx, 0)

        # Residual = total - (free + capped + active). If non-empty,
        # those slots are held by some bookkeeping we don't yet model
        # (mamba prefix cache typically). Refuse to invent ownership;
        # planner will refuse via assert_complete.
        return OwnerMap(
            pool_name="mamba",
            n_pages=n_pages,
            free_pages=free_set,
            tree_pages={},
            active_pages=active_pages,
            capped_pages=capped_set,
        )

    @staticmethod
    def _find_mamba_pool(sched):
        # The mamba pool is reachable via several paths depending on the
        # engine config (HiMambaRadixCache, MambaRadixCache, plain pool).
        for attr in ("mamba_pool", "_mamba_pool"):
            p = getattr(sched, attr, None)
            if p is not None:
                return p
        # Try via the per-pool actuator (set up by BudgetAgent).
        ba = getattr(sched, "budget_agent", None)
        if ba is not None:
            xpa = getattr(ba, "_xpool_actuator", None)
            if xpa is not None:
                ma = getattr(xpa, "mamba_actuator", None)
                if ma is not None:
                    return getattr(ma, "pool", None)
        return None

    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_set(t: Optional[torch.Tensor]) -> set:
        if t is None or t.numel() == 0:
            return set()
        return {int(x) for x in t.cpu().tolist()}

    @staticmethod
    def _req_seqlen(req, max_ctx: int) -> int:
        """Best-effort seqlen — Req has many overlapping length fields
        depending on prefill stage / chunked / spec-decoding. We want the
        count of pages currently bound to this req in req_to_token, which
        is the longer of (input prefill written so far, full seqlen)
        clamped to max_ctx.
        """
        # `seqlen` property on Req returns len(origin_input_ids)+len(output_ids);
        # for chunked prefill mid-flight, fill_ids is the more accurate
        # written-so-far count.
        candidates: list[int] = []
        for attr in ("seqlen", "fill_ids"):
            v = getattr(req, attr, None)
            if v is None:
                continue
            if attr == "seqlen":
                try:
                    candidates.append(int(v))
                except TypeError:
                    pass  # property that needs a call?
            else:
                try:
                    candidates.append(len(v))
                except TypeError:
                    pass
        if not candidates:
            return 0
        return min(max(candidates), max_ctx)

    @classmethod
    def _walk_tree(cls, root, out: dict, active_pages: dict) -> None:
        """DFS the radix tree, pushing every non-evicted node's pages
        into `out`. Pages already in `active_pages` are skipped — they
        belong to a req via prefix-lock, and migrating them (not evicting
        the tree node) is the only safe op.
        """
        # Iterative DFS to avoid Python recursion-depth blow-up on long
        # prefix chains.
        stack = [root]
        while stack:
            node = stack.pop()
            children = getattr(node, "children", None)
            if children:
                for child in children.values():
                    stack.append(child)
            value = getattr(node, "value", None)
            if value is None:
                continue
            try:
                pages = value.cpu().tolist()
            except AttributeError:
                continue  # non-tensor value (shouldn't happen for radix)
            for offset, p in enumerate(pages):
                pi = int(p)
                if pi == 0 or pi in active_pages:
                    continue
                # First-writer wins — if a page is referenced by multiple
                # nodes (shouldn't happen in radix but defensive), we
                # remember the first one.
                if pi not in out:
                    out[pi] = TreeNodeRef(node=node, page_offset=offset)


# Sanity: the class is structurally an OwnerProvider.
assert isinstance(  # noqa: S101 — module-level structural guard
    SchedulerOwnerProvider.__new__(SchedulerOwnerProvider), OwnerProvider
)
