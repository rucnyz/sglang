"""Scheduler-coupled Stage-0 handler for cross-pool fires.

`XPoolActuator._run_stage0` is a pure mechanical executor: it knows how
to call `MambaPool.migrate_slot` (byte-exact slot relocation) but NOT
how to drive sglang's radix-cache eviction, allocate a free dst slot, or
rewrite the in-flight request pointer that names the migrated slot. Those
three operations are scheduler-coupled (they touch `tree_cache`,
`mamba_pool`, and the running batch's `Req` objects), so the actuator
delegates them to this collaborator (design.md §"Transfer protocol":
Stage 0).

Interface consumed by `XPoolActuator._run_stage0`:
  - `evict_pages(direction, drains)`  — CACHED → FREE: evict the radix
        blocks backing each `drains` page via sglang's own eviction.
  - `rewrite_ssm_state_indices(src, dst)` — after `migrate_slot` moved the
        bytes, repoint the in-flight req that owns `src` at `dst` (both
        `req.mamba_pool_idx` AND the `HybridReqToTokenPool`'s
        `req_index_to_mamba_index_mapping`, which the attention backend
        reads via `get_mamba_indices`).

The migration DESTINATION is no longer chosen here: the planner assigns
each `(src_slot, dst_slot)` move (dst = a scattered free slot on a kept
page) so Stage-0 has no destination to guess.

Fail-loud throughout: a missing tree_cache or an un-owned migrated slot
raises rather than silently corrupting state.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams

logger = logging.getLogger(__name__)


class SchedulerStage0Handler:
    """Drives the scheduler-coupled half of a cross-pool fire's Stage-0.

    Constructed once per scheduler (alongside the actuator chain) with a
    `Scheduler` reference and the per-pool actuators. All methods run on
    the scheduler thread under the source allocator's lock window (the
    actuator's `cap_barrier` caller holds it).
    """

    def __init__(self, scheduler, kv_actuator, mamba_actuator=None) -> None:
        self._scheduler = scheduler
        self._kv_actuator = kv_actuator
        self._mamba_actuator = mamba_actuator

    # -- Drain: CACHED → FREE via sglang's own eviction ----------------

    def evict_pages(self, direction: str, drains) -> None:
        """Evict the radix-cache blocks backing each `drains` page so the
        page returns to the allocator's free list (CACHED → FREE).

        `drains` are page-ids in the SOURCE pool of `direction`. They were
        chosen by `SchedulerOwnerProvider._cached_pages_in_cost_order` from
        the cache's own cost-order victim walk, so evicting that many
        tokens in the same order frees exactly (a superset prefix of)
        these pages — the priced/selected/evicted sets stay byte-identical
        by construction (design.md §"Why exact c^evict").

        We translate the target pages back to token-slots and request the
        cache evict that many tokens from the matching side. Mamba-source
        drains go through `mamba_num`; KV-source drains through
        `num_tokens` (the hybrid `MambaRadixCache` routes each to
        `evict_mamba` / `evict_full`)."""
        if not drains:
            return
        tree_cache = self._scheduler.tree_cache
        if tree_cache is None:
            raise RuntimeError(
                "Stage-0 evict_pages: scheduler has no tree_cache — "
                "cannot drive the CACHED→FREE eviction the plan requires."
            )
        src_pool = direction.split("_to_")[0]
        act = self._kv_actuator if src_pool == "kv" else self._mamba_actuator
        tps = act._tokens_per_page()
        n_tokens = len(drains) * tps
        if src_pool == "kv":
            params = EvictParams(num_tokens=n_tokens)
        else:
            params = EvictParams(num_tokens=0, mamba_num=n_tokens)
        result = tree_cache.evict(params)
        logger.info(
            "stage0 evict_pages dir=%s drains=%d pages (%d tokens) -> "
            "kv_evicted=%d mamba_evicted=%d",
            direction, len(drains), n_tokens,
            getattr(result, "num_tokens_evicted", 0),
            getattr(result, "mamba_num_evicted", 0),
        )

    # -- Migration req-pointer rewrite ---------------------------------

    def rewrite_ssm_state_indices(self, src_slot: int, dst_slot: int) -> None:
        """Repoint the in-flight request that owns mamba slot `src_slot`
        at `dst_slot`, after `migrate_slot` relocated the bytes.

        `migrate_slot` moves the recurrent-state tensor contents; the
        OWNING-request pointer is the caller's job (its docstring).
        Production stores that pointer in TWO mirrored places:
          - `req.mamba_pool_idx` (per-Req scalar int tensor), and
          - `HybridReqToTokenPool.req_index_to_mamba_index_mapping[
              req.req_pool_idx]` (what the attention backend reads via
            `get_mamba_indices`).
        Both must move or the next decode reads the stale slot. Fails loud
        if no running req owns `src_slot` (Migration only targets LIVE,
        i.e. running-batch-owned, slots)."""
        owner = self._find_running_req_for_slot(src_slot)
        if owner is None:
            raise RuntimeError(
                f"Stage-0 rewrite_ssm_state_indices: no running req owns "
                f"migrated mamba slot {src_slot} — Migration must only "
                f"target LIVE slots held by an in-flight req."
            )
        rtp = self._scheduler.req_to_token_pool
        # Move the per-Req pointer (keep it a 0-dim/1-elem int tensor on
        # the same device/dtype the engine expects).
        old = owner.mamba_pool_idx
        new_idx = torch.tensor(
            dst_slot, dtype=old.dtype, device=old.device
        ).reshape(old.shape)
        owner.mamba_pool_idx = new_idx
        # Move the mirrored mapping the attention backend reads.
        rtp.req_index_to_mamba_index_mapping[owner.req_pool_idx] = dst_slot
        logger.debug(
            "stage0 rewrite ssm: req=%s slot %d -> %d "
            "(req_pool_idx=%s)",
            getattr(owner, "rid", "?"), src_slot, dst_slot,
            int(owner.req_pool_idx),
        )

    def rewrite_kv_token_indices(self, src_slot: int, dst_slot: int) -> None:
        """Repoint the in-flight request that holds KV token-slot `src_slot`
        at `dst_slot`, after `migrate_slot` relocated the slot's k/v bytes
        (the KV analog of `rewrite_ssm_state_indices`).

        The KV pointer lives in `req_to_token_pool.req_to_token[req_pool_idx,
        pos]` — the table the attention backend re-reads into `kv_indices`
        every decode replay (the spike's part B), so rewriting it there
        propagates to the captured graph. A LIVE KV slot is held at exactly
        one (req, pos); the search is bounded to each req's live length
        `[0, seqlen)` so a stale value lingering in the row's unused tail
        can't be a false match. Fails loud if no running req holds `src_slot`
        (Migration only targets LIVE, i.e. running-batch-owned, slots)."""
        owner = self._find_running_req_for_kv_slot(int(src_slot))
        if owner is None:
            raise RuntimeError(
                f"Stage-0 rewrite_kv_token_indices: no running req holds "
                f"migrated KV slot {int(src_slot)} — Migration must only "
                f"target LIVE slots held by an in-flight req."
            )
        req, idx, hits = owner
        self._scheduler.req_to_token_pool.req_to_token[idx, hits] = int(dst_slot)
        logger.debug(
            "stage0 rewrite kv: req=%s slot %d -> %d at pos %s "
            "(req_pool_idx=%d)",
            getattr(req, "rid", "?"), int(src_slot), dst_slot,
            hits.tolist(), idx,
        )

    def has_live_owner(self, direction: str, src_slot: int) -> bool:
        """Read-only: does a running req own the migration source `src_slot`
        (KV `req_to_token` for kv_to_mamba, scalar `mamba_pool_idx` for
        mamba_to_kv)? `XPoolActuator._run_stage0` checks this for EVERY move
        BEFORE relocating any bytes, so an invalid plan aborts with zero
        mutation instead of half-applying (migrate `src`'s bytes, then raise
        because the pointer rewrite finds no owner — leaving the req reading a
        freed slot and `dst` live-but-orphaned; audit HIGH-2). Mirrors the
        per-direction dispatch in `_run_stage0`'s rewrite step."""
        if direction == "mamba_to_kv":
            return self._find_running_req_for_slot(int(src_slot)) is not None
        return self._find_running_req_for_kv_slot(int(src_slot)) is not None

    def _find_running_req_for_kv_slot(self, src_slot: int):
        """The running-batch req holding KV token-slot `src_slot`, as
        `(req, req_pool_idx, hit_positions)`, or None. Bounded to each req's
        live token range `[0, seqlen)` — a slot id beyond `seqlen` is a stale
        leftover in the row's unused tail, not a live pointer. `Req.seqlen` is
        a guaranteed property; access it directly so a schema change fails
        loud (no silent full-row fallback that would reintroduce the
        stale-tail false match)."""
        req2tok = self._scheduler.req_to_token_pool.req_to_token
        rb = self._scheduler.running_batch
        reqs = list(getattr(rb, "reqs", []) or []) if rb is not None else []
        src = int(src_slot)
        for req in reqs:
            idx = int(req.req_pool_idx)
            row = req2tok[idx, : int(req.seqlen)]
            hits = (row == src).nonzero(as_tuple=True)[0]
            if hits.numel() > 0:
                return req, idx, hits
        return None

    def _find_running_req_for_slot(self, slot: int):
        """The running-batch req whose `mamba_pool_idx == slot`, or None."""
        rb = self._scheduler.running_batch
        reqs = list(getattr(rb, "reqs", []) or []) if rb is not None else []
        for req in reqs:
            mpi = getattr(req, "mamba_pool_idx", None)
            if mpi is None:
                continue
            if int(mpi.item() if hasattr(mpi, "item") else mpi) == int(slot):
                return req
        return None
