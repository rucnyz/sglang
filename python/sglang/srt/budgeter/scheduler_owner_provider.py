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
from sglang.srt.mem_cache.memory_pool import kv_live_migration_enabled

logger = logging.getLogger(__name__)


class SchedulerOwnerProvider:
    """Implements `OwnerProvider` against a live `Scheduler`."""

    def __init__(self, scheduler, kv_actuator, mamba_actuator=None) -> None:
        self._scheduler = scheduler
        self._kv_actuator = kv_actuator
        self._mamba_actuator = mamba_actuator
        # One-time warning latches for the migration fail-closed gates (the
        # "env on but pool can't migrate" KV gate, and the ungated-tps>1 mamba
        # gate). Initialized unconditionally so the attributes always exist —
        # no getattr-None state probes.
        self._warned_kv_migrate_incapable = False
        self._warned_mamba_migrate_ungated = False

    # ------------------------------------------------------------------

    def build_kv_owner_map(
        self, *, allow_drain: bool = False, allow_migrate: bool = False,
        max_drain_pages: Optional[int] = None,
    ) -> OwnerMap:
        kv_act = self._kv_actuator
        n_pages = int(kv_act.n_pages)
        free_pages = self._kv_free_pages()
        cached, live = self._expansion_lists(
            "kv", allow_drain=allow_drain, allow_migrate=allow_migrate,
            max_drain_pages=max_drain_pages,
        )
        return OwnerMap(
            pool_name="kv", n_pages=n_pages, free_pages=free_pages,
            cached_pages_in_cost_order=cached,
            live_pages_in_cost_order=live,
        )

    def mamba_tokens_per_page(self) -> int:
        """Slots per VMM chunk for the mamba pool (`tokens_per_chunk`).
        1 ⇒ atomic layout (one SSM slot fills a chunk; tp=1/fp32);
        >=2 ⇒ fragmentable. 0 if no mamba actuator is wired."""
        mamba_act = self._mamba_actuator
        if mamba_act is None:
            return 0
        return int(mamba_act._tokens_per_page())

    def kv_tokens_per_page(self) -> int:
        """KV tokens per VMM chunk (`tokens_per_chunk`). 0 if no KV
        actuator is wired. Used to bound the per-fire grant size when
        pricing the KV grow benefit."""
        kv_act = self._kv_actuator
        if kv_act is None:
            return 0
        return int(kv_act._tokens_per_page())

    def build_mamba_owner_map(
        self, *, allow_drain: bool = False, allow_migrate: bool = False,
        max_drain_pages: Optional[int] = None,
    ) -> Optional[OwnerMap]:
        mamba_act = self._mamba_actuator
        if mamba_act is None:
            return None
        mamba_pool = mamba_act.pool
        n_pages = int(mamba_act.n_pages)
        # Snapshot the mamba pool state under its own `_alloc_lock`
        # so `free_slots`, `_capped_slots`, and the live/cached walks see a
        # consistent view: the BudgetAgent fire worker reassigns these
        # tensors under the SAME lock (alloc / free / migrate_slot /
        # set_capacity_slots). decide_for_req calls this while holding the
        # KV `_alloc_lock`; the mamba lock nests INSIDE (KV-outer,
        # mamba-inner) — none of the pure-read callees re-acquire either
        # lock, so this cannot deadlock or re-enter.
        with mamba_pool._alloc_lock:
            free_pages = self._mamba_free_pages_locked(mamba_act, mamba_pool)
            cached, live = self._expansion_lists(
                "mamba", allow_drain=allow_drain, allow_migrate=allow_migrate,
                max_drain_pages=max_drain_pages,
            )
        return OwnerMap(
            pool_name="mamba", n_pages=n_pages, free_pages=free_pages,
            cached_pages_in_cost_order=cached,
            live_pages_in_cost_order=live,
        )

    # ------------------------------------------------------------------

    def _kv_free_pages(self):
        """Fully-free KV SOURCE pages (Stage-1 free-harvest set). Reused by
        `build_kv_owner_map` and `n_free_source_pages`."""
        allocator = self._scheduler.token_to_kv_pool_allocator
        return self._compute_fully_free_pages(
            n_pages=int(self._kv_actuator.n_pages),
            tps=self._kv_actuator._tokens_per_page(),
            free_slot_tensors=[
                allocator.free_pages,
                allocator.release_pages,
            ],
            # Pages mid-fire (any slot in `_capped_pages`) are NOT fully free:
            # `mark_pages_capped` leaves `free_pages` untouched, so without this
            # the planner re-selects already-unmapped capped pages → wasted/
            # short fires + c^xfer EWMA polluted by near-zero-work samples.
            exclude_slot_tensor=allocator._capped_pages,
        )

    def _mamba_free_pages_locked(self, mamba_act, mamba_pool):
        """Fully-free mamba SOURCE pages. Caller MUST hold the allocator lock."""
        mamba_allocator = mamba_pool._allocator
        return self._compute_fully_free_pages(
            n_pages=int(mamba_act.n_pages),
            tps=mamba_act._tokens_per_page(),
            free_slot_tensors=[mamba_allocator.free_slots],
            exclude_slot_tensor=mamba_allocator._capped_slots,
        )

    def n_free_source_pages(self, direction: str) -> int:
        """Count of fully-free SOURCE pages a fire in `direction` can harvest
        free-first (Stage-1), WITHOUT the cost-order radix walk. The source is
        KV for `kv_to_mamba`, mamba for `mamba_to_kv`.

        Single source of truth for the free supply: the planner's drain-cost
        pricing (`BudgetAgent`) and the actuator's Stage-1 selection
        (`XPoolFirePlanner.build`) both read it, so a free-harvest fire is
        priced as the zero-drain it actually executes. Returns 0 when the
        direction's source pool is absent."""
        if direction == "kv_to_mamba":
            return len(self._kv_free_pages())
        if direction == "mamba_to_kv":
            mamba_act = self._mamba_actuator
            if mamba_act is None:
                return 0
            mamba_pool = mamba_act.pool
            with mamba_pool._alloc_lock:
                return len(self._mamba_free_pages_locked(mamba_act, mamba_pool))
        raise ValueError(f"unknown direction: {direction!r}")

    def _expansion_lists(
        self, pool_name: str, *, allow_drain: bool, allow_migrate: bool,
        max_drain_pages: Optional[int] = None,
    ):
        """Stage-2 / Stage-3 candidate page lists (design.md §"Page
        selection: anywhere-free, Drain-expansion, Migration-expansion"),
        populated ONLY when the planner requests them.

        Returns `(cached_pages_in_cost_order, live_pages_in_cost_order)`.
        Both default `None` on the common free-only path so the radix-
        tree cost-order walk stays zero-cost (the `allow_*` gates exist
        precisely so Stage-1-only callers pay nothing).

        - `cached_pages_in_cost_order` (Drain): the radix-cache
          eviction-order victims, mapped to page-ids, in the SAME order
          sglang's own eviction would pop them (LRU `last_access_time`
          or LPB `ℓ(b)`). Reuses the single-source-of-truth victim walk
          that `c^evict` consumes — `MambaRadixCache._plan_full_eviction`
          on the hybrid cache, `RadixCache._iter_evict_victims` on a
          KV-only cache — so the priced (`c^evict`) set, the planner's
          drain set, and the actuator's Stage-0 evicted set are
          byte-identical by construction. Only FULLY-covered pages (every
          constituent slot freed by a victim or already free) are
          emitted, deduped, in first-fully-covered order.
        - `live_pages_in_cost_order` (Migration): mamba LIVE slots not
          held by any cached node, mapped to pages, ordered by ascending
          per-page `c_m`. `c_m` is a per-slot CONSTANT (design.md
          §"Shared cost model": `c_m(X) ≈ X/side_stream_bw + per-slot
          const`), so this degenerates to a stable ascending-slot-id
          order (the tie-break) — an LRU-ish FIFO over the live slot
          space. Only pages whose every live slot can migrate to a free
          dst slot are emitted.

        Both walks are pure-read (no tree/pool mutation); the caller
        (planner.build) runs them inside the scheduler lock window so the
        snapshot is consistent (`OwnerProvider` Protocol contract)."""
        if not allow_drain and not allow_migrate:
            return None, None
        cached = (
            self._cached_pages_in_cost_order(pool_name, max_pages=max_drain_pages)
            if allow_drain else None
        )
        live = self._live_pages_in_cost_order(pool_name) if allow_migrate else None
        return cached, live

    # -- Stage 2: Drain-expansion (CACHED → FREE, cost-order) ----------

    def _cached_pages_in_cost_order(
        self, pool_name: str, max_pages: Optional[int] = None
    ) -> list:
        """Radix-cache eviction-order victims → page-ids (design.md
        §"Drain-expansion"). Walks the SAME pure-read victim selector
        `c^evict` and the real eviction consume, maps each victim's freed
        slots to page-ids, and emits a page the first time it becomes
        FULLY covered (every slot free).

        `pool_name='kv'`: victim's KV token-slots (`node.value`).
        `pool_name='mamba'`: victim's mamba snapshot slots
        (`node.mamba_value`) — full eviction frees both buffers
        (`MambaRadixCache._evict_leaf_node`), so the same `_plan_full_
        eviction` walk supplies the cost-ordered mamba slots too.
        """
        tree_cache = self._scheduler.tree_cache
        if tree_cache is None:
            return []
        act = self._kv_actuator if pool_name == "kv" else self._mamba_actuator
        tps = act._tokens_per_page()
        already_free = self._free_slot_set(pool_name)

        victims = self._iter_drain_victims(tree_cache, pool_name)
        return self._slots_to_fully_covered_pages(
            victims, tps, already_free, max_pages=max_pages
        )

    def _iter_drain_victims(self, tree_cache, pool_name: str):
        """Yield (in eviction cost order) the slot-id tensors each victim
        frees from `pool_name`. Routes to the cache's own single-source-
        of-truth victim selector so the order is byte-identical to a real
        evict — no parallel policy implementation."""
        # Hybrid cache (MambaRadixCache): full eviction frees BOTH KV
        # (`value`) and mamba (`mamba_value`) per leaf; `_plan_full_
        # eviction` is the shared selector. A big token budget exposes
        # the full cost-ordered victim sequence (pure-read; no mutation).
        if hasattr(tree_cache, "_plan_full_eviction"):
            # `_plan_full_eviction` counts FULL (KV) tokens regardless of
            # which buffer we harvest, so the budget is always the full
            # evictable count (+1 so the stop condition never truncates
            # the last victim). Feeding the mamba count here would stop
            # the walk early (mamba_evictable < full_evictable).
            big = int(tree_cache.full_evictable_size()) + 1
            victims, swept = tree_cache._plan_full_eviction(big)
            for node in victims:
                slots = node.value if pool_name == "kv" else node.mamba_value
                if slots is not None and slots.numel() > 0:
                    yield slots
            # Swept tombstones free KV only (mamba already gone); they
            # contribute to the KV drain set but never the mamba one.
            if pool_name == "kv":
                for node in swept:
                    if node.value is not None and node.value.numel() > 0:
                        yield node.value
            return
        # KV-only cache (RadixCache): one buffer per node (`value`).
        if pool_name != "kv":
            raise ValueError(
                f"_iter_drain_victims: KV-only RadixCache has no {pool_name!r} "
                f"side (tree_cache={type(tree_cache).__name__})"
            )
        if not hasattr(tree_cache, "_iter_evict_victims"):
            raise RuntimeError(
                f"_iter_drain_victims: tree_cache "
                f"{type(tree_cache).__name__} exposes neither "
                f"_plan_full_eviction nor _iter_evict_victims — cannot "
                f"build the cost-ordered Drain set."
            )
        # KV-only RadixCache counts KV tokens; +1 so the stop condition
        # never truncates the last victim.
        big = int(tree_cache.evictable_size()) + 1
        for node in tree_cache._iter_evict_victims(big):
            if node.value is not None and node.value.numel() > 0:
                yield node.value

    # -- Stage 3: Migration-expansion (LIVE → FREE, cost-order) --------

    def _live_pages_in_cost_order(self, pool_name: str) -> list:
        """Migration moves (LIVE→FREE consolidation), cost-ordered
        (design.md §"Migration-expansion"). Migration is mamba-only (KV
        has no `migrate_slot` primitive), so a KV-pool request returns [].

        Returns `[(freed_page_id, ((src_slot, dst_slot), ...)), ...]`: each
        entry is a fully-live source page plus the byte-exact relocations
        that empty it so Stage-1 can transfer it. The data structure
        carries the CONCRETE src→dst slot moves (not page-ids) so Stage-0
        executes `migrate_slot` on the right slots and never has to guess a
        destination — fixing the page-id-as-slot-id and arbitrary-dst
        hazards.

        Destinations are SCATTERED free slots — free slots on KEPT pages
        (partially-live pages that stay mapped). They are NEVER:
          - on a page being freed (no self-destination), nor
          - on a WHOLE-free page (those are Stage-1's transfer payload),
        which holds by construction because **source pages are FULLY-LIVE**
        (all `tps` slots live-uncached, hence zero free slots) while
        **donor slots come only from partial pages** — the two sets are
        disjoint. Each donor slot is assigned to exactly one move (no
        double-spend).

        Cost order: `c_m` is a per-slot constant, so ascending page-id is
        the cost order (an LRU-ish FIFO over the live slot space).

        ATOMIC pools (`tps == 1`) yield []: a page is either fully-live
        (1 live slot, a source needing 1 donor) or whole-free (1 free
        slot) — there are no partial pages, so no donor exists. Only a
        fragmentable layout (`tps >= 2`) produces scattered donors.
        """
        if pool_name == "kv":
            return self._kv_live_pages_in_cost_order()
        if pool_name != "mamba":
            return []
        mamba_act = self._mamba_actuator
        if mamba_act is None:
            return []
        pool = mamba_act.pool
        tps = mamba_act._tokens_per_page()
        n_pages = int(mamba_act.n_pages)

        # Fail-closed gate (audit). Mamba live-slot migration is atomic-inert
        # today only because mamba runs tps==1 (no partial pages, no donors)
        # — an unasserted runtime accident, not a gate. Unlike the KV
        # side (SGLANG_XPOOL_KV_MIGRATE + can_migrate_slot + proven replay
        # proof), the mamba direction has NO opt-in flag or captured-graph
        # proof. A fragmentable layout (tps>=2, anticipated for TP/bf16 ssm)
        # would otherwise silently relocate LIVE recurrent state. Refuse until
        # a mamba-side gate + proof exist, so inertness is GATED not incidental.
        if tps != 1:
            if not self._warned_mamba_migrate_ungated:
                logger.warning(
                    "mamba live-slot migration requested with tps=%d (>1) but "
                    "it is ungated and unproven (no replay proof, no opt-in "
                    "flag) — refusing. Mamba cross-fire stays free-only/drain.",
                    tps,
                )
                self._warned_mamba_migrate_ungated = True
            return []

        live = set(self._mamba_live_uncached_slots(pool))
        if not live:
            return []
        free = self._free_slot_set("mamba")
        capped = pool._capped_slots
        capped_pages = (
            {int(s) // tps for s in capped.cpu().tolist()}
            if capped is not None and capped.numel() > 0
            else set()
        )
        # Classify each WHOLE page p ∈ [1, n_pages) (page 0 is padded,
        # capped pages are mid-fire). `n_pages = size // tps`,
        # so when `size` is not a multiple of `tps` the trailing partial
        # page `[n_pages*tps, size]` is INTENTIONALLY excluded: a partial
        # page can never be a fully-live SOURCE (it has < tps slots), and
        # dropping its free slots as donors only UNDER-counts migration
        # capacity (conservative — never over-selects, never mis-maps).
        # (We don't assert `size % tps == 0`: the mamba actuator marks the
        # real slots from `expand_pages_to_token_slots`, so a non-divisible
        # size is valid; a hard assert would crash legitimate fragmentable
        # configs. The conservative drop is the safe, documented choice.)
        #   - whole-free  (all tps free): Stage-1 payload — neither source
        #     nor donor.
        #   - fully-live  (all tps live-uncached): a migration SOURCE.
        #   - partial     (has >=1 free, not whole-free): its free slots
        #     are scattered DONOR dsts; the page stays mapped.
        donor_slots: list = []
        source_pages: list = []  # (pid, sorted live src slots)
        for pid in range(1, n_pages):
            if pid in capped_pages:
                continue
            base = pid * tps
            free_here = [s for s in range(base, base + tps) if s in free]
            if len(free_here) == tps:
                continue  # whole-free
            live_here = [s for s in range(base, base + tps) if s in live]
            if len(live_here) == tps:
                source_pages.append((pid, live_here))
            elif free_here:
                donor_slots.extend(free_here)
        if not source_pages or not donor_slots:
            return []
        donor_slots.sort()
        source_pages.sort()  # ascending page-id == ascending c_m
        out = []
        di = 0
        for pid, srcs in source_pages:
            if di + tps > len(donor_slots):
                break  # not enough disjoint donor slots left to empty it
            moves = tuple(
                (srcs[k], donor_slots[di + k]) for k in range(tps)
            )
            di += tps
            out.append((pid, moves))
        return out

    def _mamba_live_uncached_slots(self, pool) -> list:
        """LIVE mamba slots = allocated slots NOT free, NOT capped, NOT
        held as a cached `mamba_value` snapshot, excluding padded slot 0.

        Cached snapshots are forked/cloned copies pinned by radix nodes;
        they are harvested by Drain, not Migration (migrating a cached
        slot would orphan the node's pointer). LIVE slots are owned by an
        in-flight req via `req.mamba_pool_idx` — exactly the slots whose
        `ssm_state_indices` Stage-0 rewrites after `migrate_slot`."""

        size = int(pool.size)
        if size <= 0:
            return []
        device = pool.free_slots.device
        # all candidate slot ids in the live range [1..size].
        allocated = torch.ones(size + 1, dtype=torch.bool, device=device)
        allocated[0] = False  # padded slot 0 never migrates.
        for t in (pool.free_slots, pool._capped_slots):
            if t is not None and t.numel() > 0:
                tl = t.long()
                valid = (tl >= 0) & (tl <= size)
                allocated[tl[valid]] = False
        # Exclude cached mamba snapshots (held by radix nodes).
        for s in self._cached_mamba_slots():
            if 0 <= s <= size:
                allocated[s] = False
        return allocated.nonzero(as_tuple=True)[0].cpu().tolist()

    def _cached_mamba_slots(self) -> set:
        """Set of mamba slot ids pinned by radix-tree `mamba_value`
        snapshots (the Drain harvest set). Empty for a KV-only cache."""
        tree_cache = self._scheduler.tree_cache
        if tree_cache is None or not hasattr(tree_cache, "mamba_lru_list"):
            return set()
        out: set = set()
        for node in tree_cache.mamba_lru_list.cache.values():
            mv = node.mamba_value
            if mv is not None and mv.numel() > 0:
                out.update(int(x) for x in mv.cpu().tolist())
        return out

    def _kv_live_pages_in_cost_order(self) -> list:
        """KV Stage-3 Migration: consolidate scattered LIVE-
        UNCACHED KV slots so a k2m fire can free a whole arena chunk. A KV
        "page" is a chunk of `tps = tokens_per_chunk` token-slots (large), so
        KV is highly fragmentable. SOURCE = a fully-live-uncached page (all
        tps slots live); DONORS = free slots on PARTIAL (kept) pages;
        whole-free pages are Stage-1 payload (not donors), capped pages are
        mid-fire, page 0 is the padded sentinel — all excluded. Cached slots
        are EXCLUDED (Drain harvests those; migrating a cached/shared slot
        would orphan the radix node + other reqs' rows — audit H2).

        Returns `[(freed_page_id, ((src,dst),...)), ...]`, each source slot
        paired to a distinct donor, ascending page-id (== ascending per-page
        c_m, a stable cost order). Classification is vectorized (tps is large
        — a per-slot Python loop would be ~n_pages·tps ops on the scheduler
        thread)."""
        # Fail-closed enable gate, shared with boot-time KV-pool
        # construction (memory_pool.kv_live_migration_enabled) so the walk and
        # enable_kv_cache_copy can never disagree. The execution path (Stage-0
        # migrate_slot + req_to_token rewrite) is wired and the captured-graph
        # replay is proven (flashinfer); until per-backend coverage lands
        # this returns no moves, so a cross_migrate(kv_to_mamba)
        # candidate degrades to free-only/drain even when the planner passes
        # allow_migrate=True. Flip SGLANG_XPOOL_KV_MIGRATE=1 to enable.
        if not kv_live_migration_enabled():
            return []
        kv_act = self._kv_actuator
        if kv_act is None:
            return []
        allocator = self._scheduler.token_to_kv_pool_allocator
        # The env gate (read per-walk) and enable_kv_cache_copy (read once at
        # boot) can disagree if the env is flipped on AFTER the pool was built,
        # and an MLA/NPU hybrid may never have a usable move_kv_cache at all.
        # Verify the pool can ACTUALLY migrate before emitting any move, so an
        # incapable pool refuses here (zero mutation) instead of asserting
        # inside migrate_slot mid-fire (where agent.tick would swallow it).
        if not allocator.can_migrate_slot():
            if not self._warned_kv_migrate_incapable:
                logger.warning(
                    "SGLANG_XPOOL_KV_MIGRATE is on but the KV pool cannot "
                    "migrate slots (no enable_kv_cache_copy / MLA pool) — "
                    "refusing KV migration; set enable_kv_cache_copy at boot. "
                    "KV cross-fire degrades to free-only/drain."
                )
                self._warned_kv_migrate_incapable = True
            return []
        tps = int(kv_act._tokens_per_page())
        n_pages = int(kv_act.n_pages)
        # tps<2: atomic — every page is whole-live or whole-free, no partial
        # donors (mirrors mamba atomic-inert). n_pages<2: nothing to do.
        if tps < 2 or n_pages < 2:
            return []
        size_m = n_pages * tps
        device = allocator.free_pages.device

        def _ids_in_range(t):
            if t is None or t.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            tl = t.to(device).long()
            return tl[(tl >= 0) & (tl < size_m)]

        capped_ids = _ids_in_range(allocator._capped_pages)
        free_ids = _ids_in_range(allocator.free_pages)
        rel_ids = _ids_in_range(allocator.release_pages)

        # truly-free = (free_pages ∪ release_pages) − capped. Convention A:
        # capped ids STAY in free_pages but are NOT allocatable, so they are
        # neither donors nor whole-free payload.
        free_mask = torch.zeros(size_m, dtype=torch.bool, device=device)
        free_mask[free_ids] = True
        free_mask[rel_ids] = True
        if capped_ids.numel():
            free_mask[capped_ids] = False

        # live-uncached = allocated (not free, not capped) ∧ not cached,
        # excluding padded slot 0.
        live_mask = torch.ones(size_m, dtype=torch.bool, device=device)
        live_mask[0] = False
        live_mask[free_ids] = False
        if rel_ids.numel():
            live_mask[rel_ids] = False
        if capped_ids.numel():
            live_mask[capped_ids] = False
        cached = self._cached_kv_slots()
        if cached:
            cidx = torch.tensor(
                [s for s in cached if 0 <= s < size_m],
                dtype=torch.long, device=device,
            )
            if cidx.numel():
                live_mask[cidx] = False

        # capped PAGES are mid-fire; page 0 is the sentinel — exclude both
        # from source/donor selection.
        excluded_page = torch.zeros(n_pages, dtype=torch.bool, device=device)
        excluded_page[0] = True
        if capped_ids.numel():
            excluded_page[capped_ids // tps] = True

        fm = free_mask.view(n_pages, tps)
        lm = live_mask.view(n_pages, tps)
        free_count = fm.sum(1)
        live_count = lm.sum(1)
        is_source = (live_count == tps) & (~excluded_page)
        is_partial = (free_count > 0) & (free_count < tps) & (~excluded_page)

        source_pids = is_source.nonzero(as_tuple=True)[0].cpu().tolist()
        if not source_pids:
            return []
        donor_mask = fm & is_partial.view(n_pages, 1)
        donor_slots = donor_mask.view(-1).nonzero(as_tuple=True)[0].cpu().tolist()
        if not donor_slots:
            return []

        out = []
        di = 0
        for pid in source_pids:  # ascending page-id == ascending c_m
            if di + tps > len(donor_slots):
                break  # not enough disjoint donors left to empty this page
            base = pid * tps
            moves = tuple(
                (base + k, int(donor_slots[di + k])) for k in range(tps)
            )
            di += tps
            out.append((pid, moves))
        return out

    def _cached_kv_slots(self) -> set:
        """EVERY KV slot id pinned by a radix-tree node (LOCKED and evictable
        alike) — EXCLUDED from Migration, which targets LIVE-UNCACHED slots
        only. A cached slot may back a shared prefix held by MULTIPLE running
        reqs; migrating it would rewrite just one owner's `req_to_token` and
        leave every co-sharer pointing at the freed slot, orphaning the radix
        node (audit H2).

        This must be the FULL cached set, NOT the cost-order Drain victims: a
        LOCKED shared-prefix node (`lock_ref > 0`) is never an eviction victim
        (`_iter_evict_victims` skips locked parents), so a victim-walk-based
        set would miss exactly the most-shared — most-dangerous-to-migrate —
        slots. We walk the whole tree from the root, unioning each node's KV
        `value` (every TreeNode, hybrid or KV-only, stores KV slots there)."""
        tree_cache = self._scheduler.tree_cache
        if tree_cache is None:
            return set()
        out: set = set()
        stack = list(tree_cache.root_node.children.values())
        while stack:
            node = stack.pop()
            value = node.value
            if value is not None and value.numel() > 0:
                out.update(int(x) for x in value.cpu().tolist())
            stack.extend(node.children.values())
        return out

    # -- shared slot→page mapping --------------------------------------

    def _free_slot_set(self, pool_name: str) -> set:
        """Currently-free slot ids for `pool_name` (the slots a page can
        already count as covered without any Drain/Migration)."""

        if pool_name == "kv":
            allocator = self._scheduler.token_to_kv_pool_allocator
            tensors = [
                allocator.free_pages,
                allocator.release_pages,
            ]
        else:
            pool = self._mamba_actuator.pool
            tensors = [pool.free_slots]
        out: set = set()
        for t in tensors:
            if t is not None and t.numel() > 0:
                out.update(int(x) for x in t.cpu().tolist())
        return out

    @staticmethod
    def _slots_to_fully_covered_pages(
        slot_tensors,
        tps: int,
        already_free: set,
        max_pages: Optional[int] = None,
    ) -> list:
        """Map an ordered stream of freed slot-id tensors to the page-ids
        that become FULLY covered, preserving the input order, deduped.
        Used by Drain-expansion (`_cached_pages_in_cost_order`): a page is
        emitted the first tick every one of its slots is accounted for —
        either already free, or freed by a victim seen in the stream.
        Page 0 is never emitted (chunk 0 carries padded slot 0).
        (Migration-expansion has its own slot-level assignment in
        `_live_pages_in_cost_order`.)

        `max_pages`: stop after that many fully-covered pages are
        emitted — the cost-order PREFIX the planner consumes for an
        `n_pages_target`-page fire. Bounds the per-victim `.cpu().tolist()`
        + per-slot coverage loop (the scheduler-thread cost the cross-fire
        drain pays each fire) to the fire magnitude instead of the whole
        evictable set. None ⇒ walk the full stream (cost-pricing callers)."""
        emitted: list = []
        emitted_set: set = set()
        # Track, per page, which slots are covered. Seed with already-free.
        page_seen: dict = {}

        def _page_full(pid: int) -> bool:
            need = set(range(pid * tps, (pid + 1) * tps))
            seen = page_seen.get(pid, set())
            for s in need:
                if s in already_free or s in seen:
                    continue
                return False
            return True

        for t in slot_tensors:
            ids = [int(x) for x in (t.cpu().tolist() if hasattr(t, "cpu") else t)]
            for s in ids:
                pid = s // tps
                if pid == 0:
                    continue  # page 0 excluded
                if pid in emitted_set:
                    continue
                page_seen.setdefault(pid, set()).add(s)
                if not _page_full(pid):
                    continue
                emitted.append(pid)
                emitted_set.add(pid)
                if max_pages is not None and len(emitted) >= max_pages:
                    # cost-order prefix reached; the planner takes the first
                    # n_pages_target of these, so stop materializing victims.
                    return emitted
        return emitted

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fully_free_pages(
        n_pages: int,
        tps: int,
        free_slot_tensors: list,
        exclude_slot_tensor: Optional[torch.Tensor],
    ) -> set:
        """Vectorized: page p is fully free iff every slot in
        [p*tps, (p+1)*tps) is in the union of `free_slot_tensors`
        and NOT in `exclude_slot_tensor`. Page 0 is unconditionally
        excluded: chunk 0 carries padded slot 0 (see design.md
        §"Per-unit sizes"); the padded-output writer touches slot 0
        at every forward pass, so chunk 0's backing VA must stay
        mapped. Previously, the impl
        marked slot 0 as "free unconditionally" so page 0 could enter
        the result set — opening a path where the planner picked
        page 0, the actuator unmapped chunk 0, and the next padded
        write hit cudaErrorIllegalAddress (`expand_pages_to_token_slots`
        with tps=1 also silently dropped chunk 0 in the same path).

        Cost: O(n_slots) GPU ops + one device→host transfer of <= n_pages
        ints. Replaces a Python loop that did O(n_pages × tps) set lookups
        per fire (~2.5M for KV pool at sglang's default sizes; ~200 ms).
        """
        if n_pages <= 0 or tps <= 0:
            return set()

        # Pick device from whichever free tensor we have.
        device = None
        for t in free_slot_tensors:
            if t is not None and t.numel() > 0:
                device = t.device
                break
        if device is None and exclude_slot_tensor is not None and exclude_slot_tensor.numel() > 0:
            device = exclude_slot_tensor.device
        if device is None:
            return set()  # nothing free, nothing to do

        n_slots = n_pages * tps
        is_free = torch.zeros(n_slots, dtype=torch.bool, device=device)
        for t in free_slot_tensors:
            if t is None or t.numel() == 0:
                continue
            t_long = t.long() if t.dtype != torch.int64 else t
            # Filter to in-range indices to be safe (capped slots, growth state).
            valid = (t_long >= 0) & (t_long < n_slots)
            if bool(valid.all().item()):
                is_free[t_long] = True
            else:
                is_free[t_long[valid]] = True
        if exclude_slot_tensor is not None and exclude_slot_tensor.numel() > 0:
            ex = exclude_slot_tensor
            ex_long = ex.long() if ex.dtype != torch.int64 else ex
            valid = (ex_long >= 0) & (ex_long < n_slots)
            if bool(valid.all().item()):
                is_free[ex_long] = False
            else:
                is_free[ex_long[valid]] = False

        # Reshape into (n_pages, tps); a page is fully-free iff its row is all True.
        fully_free_mask = is_free.view(n_pages, tps).all(dim=1)
        # Page 0 unconditionally excluded — chunk 0 carries
        # padded slot 0 and must never be unmapped. Slot 0 isn't in
        # any `free_slot_tensors` (per allocator init), so its row is
        # already False at index 0 → page 0 is naturally not fully-
        # free. Mask the bit anyway to make the invariant explicit and
        # survive a future allocator change that ever admits slot 0 to
        # a free list. Index always valid: n_pages >= 1 here because
        # the early-return at the top of the method handled n_pages<=0.
        fully_free_mask[0] = False
        return set(fully_free_mask.nonzero(as_tuple=True)[0].cpu().tolist())


# Structural Protocol check (cheap, runtime).
assert isinstance(  # noqa: S101
    SchedulerOwnerProvider.__new__(SchedulerOwnerProvider), OwnerProvider
)
