"""Per-arrival cost-driven admission decision (Admitter).

Implements design.md §"Admitter — per-arrival cost decision":

    On each request arrival, evaluate the seven candidates below and pick the
    cheapest. The destination pool i_dst is whichever pool blocks the arrival
    (symmetric, see `decide_for_req`); the seven candidates are:

        own-free:      cost = 0
        own-evict:     cost = c^evict_dst(X)
        own-migrate:   cost = c_m_dst(X)            (inert in dst=kv direction)
        cross-free:    cost = c^xfer(X)
        cross-evict:   cost = c^xfer(X) + c^evict_src(X)
        cross-migrate: cost = c^xfer(X) + c^evict_src(X) + c_m_src(X)
        defer:         cost = Q · w_q

Tie-break (strict, see `_TIE_BREAK_ORDER`): own-free > cross-free >
own-evict > cross-evict > own-migrate > cross-migrate > defer. Lowest
cost wins; on equal cost the tier with lower number wins.

Three entry points:
- `decide(...)` is the pure-function core over numeric inputs (no
  scheduler), reusable from tests.
- `decide_for_req(req, scheduler)` is the scheduler-side adapter: it
  derives the admission state, routes symmetrically (grow whichever pool
  blocks the arrival), calls `decide`, and logs the result.
- `execute_decision(...)` applies a cross-* decision by building a
  FirePlan via the planner and firing the actuator.

A hybrid arrival needs room in BOTH pools (x_tokens of KV and a mamba
state slot). `decide_for_req` detects which pool blocks the arrival and
grows THAT one: `dst_pool='kv'` grows KV from mamba (m2k); `dst_pool=
'mamba'` grows mamba from KV (k2m, the burst that otherwise crashes
`cache_unfinished_req`'s fork). Both pools scarce → defer (the two
grows are opposite directions; neither can serve). This is what lets the
Budgeter drain mamba aggressively (low static floor) while the Admitter
restores capacity on demand: design.md's "burst safety via the Admitter,
not pre-reserving capacity".
JSONL per-arrival logging requires `SGLANG_HIMA_ADMITTER_LOG=path`.

See dev/interlayer/2_admitter/README.md for the test suite + rationale.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sglang.srt.budgeter.cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass
class AdmitterDecision:
    """Result of one Admitter.decide() call. Carries the chosen action
    plus the full candidate cost vector (for logging / D6n test gates).

    On a successful cross_* fire, `fire_result` holds the actuator's
    `FirePlanResult`. The scheduler does NOT reserve dst-pool tokens at
    admission time — the request joins the waiting queue and
    PrefillAdder's normal `alloc` consumes the freshly-bumped capacity
    on the next iteration.
    """

    action: str  # own_free|own_evict|cross_free|cross_evict|own_migrate|cross_migrate|defer
    reason: str
    candidate_costs_us: dict = field(default_factory=dict)
    # Cross-fire direction: the pool this decision would GROW (`dst_pool`) by
    # draining `src_pool`. A hybrid arrival can be blocked by either pool, so
    # the Admitter is symmetric — `dst_pool='mamba'` grows mamba from KV (k2m,
    # the burst that crashes the cache fork), `dst_pool='kv'` grows KV from mamba (m2k).
    # The scheduler hook reads these to fire the actuator in the right
    # direction. Own/defer actions leave them at the queried-pool default.
    dst_pool: str = "kv"
    src_pool: str = "mamba"
    # The dst demand this decision was priced for, in KV-token-equiv PAGES ×
    # tokens_per_page (i.e. the `x` that fed `decide`). `execute_decision`
    # sizes the fire from THIS, not from the req's KV input length — so a
    # mamba grow transfers the mamba state the arrival needs (ceil(need /
    # mamba_tokens_per_chunk) chunks), not the req's KV-input worth of pages.
    fire_x_tokens: Optional[int] = None
    # Populated by execute_decision() when a sync fire is triggered.
    fire_result: Optional[object] = None


# Tie-break preference order. Lower index = preferred when costs are tied.
# Matches design.md §"Page selection" cost order: own_free > cross_free >
# own_evict > cross_evict > own_migrate > cross_migrate > defer. Migration
# (a side-stream slot copy + index rewrite) is the costliest harvest, so it
# ranks just above defer — picked only when free + evict are exhausted.
_TIE_BREAK_ORDER = {
    "own_free":      0,
    "cross_free":    1,
    "own_evict":     2,
    "cross_evict":   3,
    "own_migrate":   4,
    "cross_migrate": 5,
    "defer":         6,
}


class Admitter:
    """Per-arrival cost-driven admission decision.

    Constructed once per scheduler with a `CostModel` reference. Each
    arrival calls `decide(...)` which returns an `AdmitterDecision`. The
    caller (the scheduler-side hook) then applies the action:
    - own-free / own-evict / own-migrate / defer: no extra fire; sglang's
      existing allocator + radix-cache eviction handles it
    - cross-free / cross-evict / cross-migrate: trigger a synchronous
      actuator fire
    """

    def __init__(
        self,
        cost_model: CostModel,
        actuator: Optional[Any] = None,
        planner: Optional[Any] = None,
        lcm_pages: int = 1,
    ):
        self.cost_model = cost_model
        # Warn-once guard for the calibration-profile model-mismatch check,
        # run on the first decide (where the scheduler — hence the running
        # model — is in scope). See cost_model.warn_on_model_mismatch.
        self._model_checked = False
        # Fire-path collaborators. None until the scheduler wires them in.
        # decide() does not need them (pure function); execute_decision() does.
        self.actuator = actuator
        self.planner = planner
        # Page-state ground truth (`SchedulerOwnerProvider`). decide_for_req
        # reads the SAME page-level feasibility quantities the planner
        # selects from — fully-free src pages (cross_free) and the pages
        # Migration can consolidate to free (cross_migrate) — so the
        # Admitter's feasibility and the planner's page selection agree by
        # construction. None until `BudgetAgent._wire_admitter` pushes it.
        self.owner_provider = None
        # Async fire submission hook. BudgetAgent wires this to its
        # `_submit_admitter_fire` (the SAME cap_barrier + shared-worker-queue
        # path the Budgeter tick uses), so an Admitter cross-* fire does NOT
        # run the 10-30ms cuMemUnmap/Map synchronously on the scheduler
        # thread. None until wired (tests / pre-tick boot) -> execute_decision
        # falls back to the legacy synchronous actuator.execute.
        self._fire_submit = None
        # LCM of (n_kv_subpools, n_mamba_subpools). The actuator rounds
        # any planner-requested page count up to a multiple of this LCM
        # to keep per-sub-pool growth uniform; cost decisions must use
        # the same rounded count or cross-* is under-priced.
        # Default 1 = no rounding (used by tests and pre-actuator-wired
        # boot). BudgetAgent overwrites this with `actuator.lcm_pages`
        # the moment the actuator chain is built.
        self.lcm_pages = max(1, int(lcm_pages))
        # Per-arrival mamba demand (slots): 1 active SSM slot (held while
        # running) + 1 fork slot (cache_unfinished_req allocates a new slot
        # via _alloc_mamba_slot to donate/copy the prefix state). This is a
        # sglang-internal physical constraint, not a tuning knob.
        self._mamba_arrival_need_slots = 2
        # JSONL log of per-arrival decisions. Bound at construction
        # so a runtime env-flip doesn't surprise long-lived loggers. Path
        # is opened in append mode (line-buffered) so multiple scheduler
        # processes (e.g. TP > 1) can share a file — JSONL lines are
        # well under PIPE_BUF so writes are atomic. The log grows
        # unbounded; operators are expected to use logrotate (the "a"
        # mode handles truncation gracefully).
        self._log_path: Optional[str] = os.environ.get("SGLANG_HIMA_ADMITTER_LOG") or None
        self._log_fp = None
        if self._log_path:
            self._log_fp = open(self._log_path, "a", buffering=1)
            logger.info("Admitter: per-arrival JSONL log → %s", self._log_path)

    def decide(
        self,
        *,
        x_tokens: int,
        dst_pool: str,
        dst_free: int,
        dst_evictable: int,
        src_pool: str,
        src_free: int,
        src_evictable: int,
        queue_len: int,
        c_evict_dst_us: float,
        c_evict_src_us: float,
        c_xfer_per_page_us: float,
        dst_migratable: int = 0,
        src_migratable: int = 0,
        c_migrate_dst_us: float = float("inf"),
        c_migrate_src_us: float = float("inf"),
        tokens_per_page: int = 1024,
        lcm_pages: Optional[int] = None,
    ) -> AdmitterDecision:
        """Pure-function decision over numeric inputs.

        Args (all explicit so we never lie about what state was queried):
            x_tokens:           demanded tokens for dst pool
            dst_pool, src_pool: 'kv' or 'mamba' labels (for logging)
            dst_free:           free tokens already available in dst pool
            dst_evictable:      tokens evictable from dst pool's cache
            src_free:           free tokens in src pool (for cross-free)
            src_evictable:      evictable tokens in src pool (for cross-evict)
            queue_len:          current waiting_queue length (Q)
            c_evict_dst_us:     pre-computed c^evict_dst(X) in µs
            c_evict_src_us:     pre-computed c^evict_src(X) in µs
            c_xfer_per_page_us: per-page transfer cost (cost_model.c_xfer_us(1))
            tokens_per_page:    pool's tokens-per-physical-page (planner sizing)
            lcm_pages:          actuator's LCM(n_kv_subpools, n_mamba_subpools).
                                Overrides `self.lcm_pages` for this call.
                                ``c_xfer_total`` uses the page count rounded
                                up to a multiple of this LCM so cost
                                matches what the actuator will actually
                                fire.

        Returns:
            AdmitterDecision with the chosen action + full cost vector.
        """
        if lcm_pages is None:
            lcm_pages = self.lcm_pages
        lcm_pages = max(1, int(lcm_pages))
        # Number of pages required for X tokens, rounded up to the
        # actuator's per-fire LCM (xpool_actuator transfers whole LCM
        # units). `x_eff` is the token-equivalent of that rounded harvest:
        # a cross fire actually frees `n_pages_rounded` whole pages
        # (≥ X), so CROSS feasibility and the drain/migrate shortfall must
        # be checked against `x_eff`, not raw `x_tokens` — otherwise the
        # Admitter can pass feasibility against X while the planner refuses
        # for want of the rounded count, or under-price the harvest by the
        # rounded-up remainder. OWN candidates allocate natively (no LCM
        # transfer), so they stay gated on raw `x_tokens`.
        n_pages_rounded = _round_up_pages(x_tokens, tokens_per_page, lcm_pages)
        x_eff = n_pages_rounded * tokens_per_page

        # Compute the seven candidate costs.
        # own-free is feasible iff dst already has >= X tokens free.
        if dst_free >= x_tokens:
            own_free_cost = 0.0
        else:
            own_free_cost = float("inf")

        # own-evict is feasible iff dst has >= X tokens of evictable cache.
        if dst_evictable >= x_tokens:
            own_evict_cost = c_evict_dst_us
        else:
            own_evict_cost = float("inf")

        # own-migrate (intra-pool defrag: LIVE → FREE via slot relocation):
        # feasible iff dst has >= X tokens of migratable LIVE state. Not a
        # cross-pool fire, so it is NOT subject to the cross-* cold-start
        # gate. `c_migrate_dst_us` is +inf for dst='kv' (no KV migrate
        # primitive), so own_migrate is structurally inert in the dst=kv
        # direction — self-gating, ready for the dst=mamba direction.
        if dst_migratable >= x_tokens:
            own_migrate_cost = c_migrate_dst_us
        else:
            own_migrate_cost = float("inf")

        c_xfer_total = float(n_pages_rounded) * c_xfer_per_page_us
        # CUMULATIVE free → drain → migrate, mirroring the planner's
        # Stage 1→2→3 fill (design.md §"Admitter — per-arrival cost
        # decision"). Each cross candidate uses FREE first, then layers
        # its own mechanism for the shortfall; feasibility is the
        # running sum and the cost charges each mechanism only for the
        # pages it actually harvests (`c_evict_src_us` / `c_migrate_src_
        # us` are pre-priced for the shortfall by decide_for_req). All X
        # pages transfer, so c_xfer_total is paid by every candidate.
        # Feasibility is against `x_eff` (the rounded harvest the
        # planner actually frees), not raw `x_tokens`.
        if src_free >= x_eff:
            cross_free_cost = c_xfer_total
        else:
            cross_free_cost = float("inf")
        # cross-evict: FREE + Drain. feasible iff free+evictable cover X.
        if src_free + src_evictable >= x_eff:
            cross_evict_cost = c_xfer_total + c_evict_src_us
        else:
            cross_evict_cost = float("inf")
        # cross-migrate: FREE + Drain + Migration (the planner allows
        # Drain on a cross_migrate fire). feasible iff free+evictable+
        # migratable cover X. cost = transfer + the drain part + the
        # side-stream slot-copy for the migrate part. When migration is
        # not actually needed (free, or free+drain, already reach X) the
        # migrate part is 0 ⇒ c_migrate_src_us=0 and this ties the
        # cheaper candidate, which wins on tie-break order.
        if src_free + src_evictable + src_migratable >= x_eff:
            cross_migrate_cost = (
                c_xfer_total + c_evict_src_us + c_migrate_src_us
            )
        else:
            cross_migrate_cost = float("inf")

        # defer: always feasible. cost = Q · w_q.
        defer_cost = float(queue_len) * self.cost_model.w_q_us()

        candidate_costs = {
            "own_free":      own_free_cost,
            "own_evict":     own_evict_cost,
            "cross_free":    cross_free_cost,
            "cross_evict":   cross_evict_cost,
            "own_migrate":   own_migrate_cost,
            "cross_migrate": cross_migrate_cost,
            "defer":         defer_cost,
        }

        # Pick min cost; on tie, use the static tie-break order.
        best_action = min(
            candidate_costs,
            key=lambda a: (candidate_costs[a], _TIE_BREAK_ORDER[a]),
        )
        reason = (
            f"min cost @ {best_action} = "
            f"{candidate_costs[best_action]:.1f}us "
            f"(x_tokens={x_tokens}, dst={dst_pool}, Q={queue_len})"
        )

        return AdmitterDecision(
            action=best_action,
            reason=reason,
            candidate_costs_us=candidate_costs,
            dst_pool=dst_pool,
            src_pool=src_pool,
            fire_x_tokens=int(x_tokens),
        )

    # ------------------------------------------------------------------
    # Sync fire path
    # ------------------------------------------------------------------

    def execute_decision(
        self,
        decision: AdmitterDecision,
        *,
        x_tokens: int,
        src_pool: str,
        dst_pool: str,
        tokens_per_page: int = 1024,
    ) -> AdmitterDecision:
        """Apply a decision's side-effects.

        For cross-* actions: build a FirePlan via the planner with
        min-LCM page rounding and call ``actuator.execute(plan)`` (the
        actuator's ``_fire_inflight`` serializes against the Budgeter
        worker). The fresh dst capacity is left in the allocator;
        PrefillAdder's normal ``alloc`` consumes it on the next
        scheduler iteration. No reservation is taken at admission time.

        The chosen action selects how far the planner's three-stage page
        selection (design.md §"Page selection") may expand:
          - ``cross_free``    → Stage 1 only (anywhere-free).
          - ``cross_evict``   → Stages 1-2 (free + Drain-expansion);
                                ``allow_drain=True``.
          - ``cross_migrate`` → Stages 1-3 (free + Drain + Migration-
                                expansion); ``allow_drain=allow_migrate=
                                True``. Migration is the costliest harvest,
                                so a cross_migrate decision permits Drain
                                too (a cheaper page found mid-walk is still
                                preferred — the planner stops at the first
                                stage that reaches ``n``).

        For own_free / own_evict / own_migrate / defer: this is a no-op.

        On abort: fall back to action='defer' so the caller queues the
        request instead of trying to admit it.
        """
        if decision.action not in ("cross_free", "cross_evict", "cross_migrate"):
            return decision

        # --- Page rounding -------------------------------------------
        # The actuator's atomic cross-pool LCM logic rounds the per-fire
        # total DOWN to a multiple of lcm(n_src_subpools, n_dst_subpools).
        # Anything below one LCM-unit rounds to zero and no transfer
        # happens. Size to at least one LCM-unit, with X tokens worth of
        # pages rounded UP.
        # Size the fire from the decision's PRICED dst demand, not the req's
        # KV input length: for a mamba grow `decide` was fed
        # `ceil(need / mamba_tokens_per_chunk) · tokens_per_page` (so n_pages
        # below is the mamba CHUNK count the arrival needs), and for a KV grow
        # it equals x_tokens. Fall back to x_tokens only for decisions built
        # outside `decide` (none fire today).
        fire_x = decision.fire_x_tokens if decision.fire_x_tokens is not None else x_tokens
        n_pages_needed = max(1, (fire_x + tokens_per_page - 1) // tokens_per_page)
        lcm_n = self.actuator.lcm_pages
        if lcm_n <= 0:
            decision.action = "defer"
            decision.reason = "lcm of subpool counts is non-positive"
            return decision
        n_pages_rounded = max(
            lcm_n,
            ((n_pages_needed + lcm_n - 1) // lcm_n) * lcm_n,
        )

        # --- Plan + fire ---------------------------------------------
        # Map the chosen cross-* action to the planner's expansion gates.
        # cross_migrate permits Drain too (the planner stops at the first
        # stage that reaches `n`, so allowing the cheaper Drain stage is
        # never worse). cross_evict permits Drain only; cross_free neither.
        allow_drain = decision.action in ("cross_evict", "cross_migrate")
        allow_migrate = decision.action == "cross_migrate"
        direction = f"{src_pool}_to_{dst_pool}"
        plan = self.planner.build(
            direction, n_pages_rounded,
            allow_drain=allow_drain, allow_migrate=allow_migrate,
        )
        if plan is None:
            # design.md §"Page selection": a build→None means free +
            # Drain + Migration expansion all failed to reach `n`. The
            # planner already incremented `refuse_count`; surface it in
            # the decision reason so the per-arrival JSONL records WHY
            # the cross-* degraded to defer (and the refuse-rate the
            # budgeter snapshot also tracks).
            refuse_count = self.planner.refuse_count
            decision.action = "defer"
            decision.reason = (
                f"planner.build({direction}, {n_pages_rounded}) returned "
                f"None (page selection exhausted; refuse_count={refuse_count})"
            )
            return decision

        if self._fire_submit is not None:
            # Async path (default): cap_barrier runs inline on the scheduler
            # thread (reserving the transfer so the request can be admitted),
            # and the 10-30ms cuMemUnmap/Map is handed to the shared fire
            # worker. The dst capacity materializes at the next scheduler
            # iteration's apply_pending_fires — before the re-queued request
            # is served by PrefillAdder, so no immediate alloc retry is
            # needed here (the scheduler appends the req to waiting_queue
            # after this returns; it never allocs synchronously on arrival).
            # The worker updates the c^xfer EWMA on completion, so the cost
            # model still warms from Admitter fires. sync_result is non-None
            # only on the legacy synchronous fallback (async disabled).
            aborted, sync_result = self._fire_submit(plan)
            if aborted:
                decision.action = "defer"
                decision.reason = "fire aborted (submit)"
                return decision
            if sync_result is not None:
                decision.fire_result = sync_result
                if sync_result.granted_pages > 0:
                    self.cost_model.update_xfer(
                        total_us=float(sync_result.total_us),
                        n_chunks=int(sync_result.granted_pages),
                    )
            return decision

        # Legacy direct path (no BudgetAgent wired — unit tests / pre-tick
        # boot): full synchronous execute on the calling thread.
        result = self.actuator.execute(plan)
        decision.fire_result = result
        if result.aborted:
            decision.action = "defer"
            decision.reason = f"fire aborted: {result.abort_reason}"
            return decision

        # Producer-side EWMA wiring: Admitter's sync fires also update
        # `c^xfer` so the cost model warms up from Admitter-driven
        # transfers (not just Budgeter's). Without this, the Admitter is
        # open-loop and depends entirely on Budgeter fires to warm the
        # EWMA — but Budgeter and Admitter can race or one can be
        # disabled. Closed-loop wiring keeps the cost model accurate.
        if result.granted_pages > 0:
            self.cost_model.update_xfer(
                total_us=float(result.total_us),
                n_chunks=int(result.granted_pages),
            )

        return decision

    # ------------------------------------------------------------------
    # Scheduler hook + JSONL log
    # ------------------------------------------------------------------

    def decide_for_req(
        self,
        req: Any,
        scheduler: Any,
        tokens_per_page: Optional[int] = None,
    ) -> Optional[AdmitterDecision]:
        """Scheduler-side adapter: derive admission state from `scheduler`,
        call `decide(...)`, and log the result.

        Returns the AdmitterDecision, or None if the scheduler is not in
        NULL disagg mode (Admitter is NULL-disagg only; the disagg-mode
        arrival path is out of scope, see design.md §"Admitter — per-arrival
        cost decision").

        `tokens_per_page` defaults to `owner_provider.kv_tokens_per_page()`
        (the arch/dtype-dependent KV tokens per VMM chunk), so the pricing
        and feasibility basis matches the fire path `execute_decision` sizes
        against. The param is an explicit override for provider-less unit
        tests; production passes nothing.

        Symmetric routing: a hybrid arrival needs room in BOTH pools
        (x_tokens of KV AND a mamba state slot). We detect which pool blocks
        the arrival and grow that one:
          - mamba scarce (free+evictable < `_mamba_arrival_need_slots`) and KV
            can donate → GROW MAMBA from KV (dst='mamba', src='kv';
            `_decide_grow_mamba`). This is the k2m direction a mamba-pressure
            burst needs; without it the req is admitted into a pool that can't
            fork its cache slot and crashes.
          - KV scarce (or neither) → the dst='kv'/src='mamba' path below
            (grow KV from mamba, or own_*/defer).
          - both scarce → defer (the two grows are opposite cross directions;
            neither can serve, so fall back to sglang's normal back-pressure).

        State derivation:
          - x_tokens: `len(req.origin_input_ids)` (the only demand
            available at the `_add_request_to_queue` hook point).
          - dst_free: `token_to_kv_pool_allocator.available_size()`.
          - dst_evictable: `tree_cache.evictable_size()`.
          - src_free: `mamba_allocator.available_size()`.
          - src_evictable: derived from the tree cache via
            ``_evictable_size_mamba``.
          - queue_len: `len(scheduler.waiting_queue)`.

        Caller (scheduler hook) is responsible for honoring the
        returned action — own/defer = no-op (normal scheduler path
        handles them); cross_* = caller invokes ``execute_decision()``
        which fires the actuator and leaves the freshly-bumped dst
        capacity in the allocator for PrefillAdder's normal alloc.
        """
        _t0 = time.perf_counter()
        # Pricing/feasibility basis: the actuator transfers whole VMM chunks
        # of `kv_tokens_per_page()` KV tokens each (arch/dtype-dependent),
        # so price against THAT to match what `execute_decision` fires.
        if tokens_per_page is None:
            if self.owner_provider is None:
                # The BudgetAgent wires owner_provider (and the actuator chain)
                # on its FIRST tick; a request can arrive in the sub-tick window
                # before that, especially a co-arriving swarm that hits the
                # queue the instant the server is ready. No cross-pool option
                # exists until the chain is built, so defer to normal admission
                # here (consistent with _maybe_admitter_fire's observational
                # degradation). Cross-pool admission kicks in once wired.
                return None
            tokens_per_page = int(self.owner_provider.kv_tokens_per_page())

        # One-shot: warn if a stale calibration profile (SGLANG_CSIGMA_MODEL)
        # doesn't match the running model — the κ_i/c^xfer/c_m curves would
        # silently mis-price every decision.
        if not self._model_checked:
            self._model_checked = True
            from sglang.srt.budgeter.cost_model import check_model_mismatch
            sa = getattr(scheduler, "server_args", None)
            check_model_mismatch(getattr(sa, "model_path", None))

        kv_alloc = scheduler.token_to_kv_pool_allocator

        # ── Fast path: both pools have ample headroom → own_free ──
        # Lockless approximate reads (GIL-protected integers). A stale
        # value at worst sends one arrival through the full path, which
        # sglang's normal alloc/evict handles correctly. Skips cost
        # model, lock acquisition, and JSONL log: eliminates the
        # per-arrival tax when neither pool is pressured.
        x_tokens_fast = len(getattr(req, "origin_input_ids", []) or [])
        kv_free_approx = int(kv_alloc.available_size())
        if kv_free_approx >= x_tokens_fast:
            mamba_pool_fast = _get_mamba_pool(scheduler)
            mamba_alloc_fast = getattr(
                getattr(scheduler, "req_to_token_pool", None),
                "mamba_allocator", mamba_pool_fast,
            )
            mamba_free_approx = (
                int(mamba_alloc_fast.available_size())
                if mamba_pool_fast is not None else 999
            )
            if mamba_free_approx >= self._mamba_arrival_need_slots:
                return AdmitterDecision(
                    action="own_free",
                    reason="fast: headroom",
                    candidate_costs_us={"own_free": 0.0},
                )

        # Hold the destination allocator's `_alloc_lock` across the
        # capacity snapshot + c^evict prediction + decision (design.md
        # §"Why exact c^evict"). The BudgetAgent worker
        # thread can call `set_capacity_pages` / the actuator can
        # alloc/free on this same allocator concurrently; without the
        # lock the predicted own_evict set could be priced against a
        # capacity that shifts before the scheduler acts on the
        # decision. `available_size` / `evictable_size` are pure reads
        # (no nested `_alloc_lock`), so this is not re-entrant.
        with kv_alloc._alloc_lock:
            # Pool capacity reads.
            kv_free = int(kv_alloc.available_size())
            tree_cache = getattr(scheduler, "tree_cache", None)
            kv_evictable = _evictable_size_kv(tree_cache)
            mamba_pool = _get_mamba_pool(scheduler)
            mamba_allocator = getattr(
                getattr(scheduler, "req_to_token_pool", None),
                "mamba_allocator", mamba_pool,
            )
            # Mamba-source page-level feasibility for cross_free /
            # cross_migrate, read from the SAME SchedulerOwnerProvider page
            # computation the planner selects from — so the Admitter's
            # feasibility and the planner's page selection agree by
            # construction (no slot-vs-page mismatch):
            #   mamba_free     = (#fully-free mamba pages) · tokens_per_page
            #   src_migratable = (#pages Migration can consolidate to free)
            #                    · tokens_per_page    (KV-token-equiv)
            # An ATOMIC mamba layout (tokens_per_chunk == 1 — one SSM slot
            # fills a whole 2 MiB VMM chunk, e.g. tp=1 / fp32 ssm) yields 0
            # migratable pages: every free slot is its own whole-free page,
            # so Migration only swaps which chunk is free and cross_migrate
            # self-gates to +inf. Only a fragmentable layout (tp ≥ 2 or
            # bf16 ssm → tokens_per_chunk ≥ 2) gives Migration scattered
            # free slots to consolidate into whole transferable chunks.
            mamba_free, src_migratable = self._mamba_feasibility(
                scheduler, mamba_pool, int(tokens_per_page)
            )
            mamba_evictable = _evictable_size_mamba(tree_cache, scheduler)
            # Mamba evictable is a raw SLOT count; the cross-evict/cross-
            # migrate shortfall split below works in KV-token-equiv (like
            # `mamba_free`, `src_migratable`, and `x_eff`). Convert slots ->
            # whole chunks -> KV-equiv (mirror `_decide_grow_mamba`): one
            # transferable chunk = `mamba_tps` slots = `tokens_per_page`
            # KV-equiv. The raw-slot count stays for the mamba-scarcity check,
            # which is measured in slots against `_mamba_arrival_need_slots`.
            mamba_tps = self._mamba_tokens_per_chunk(scheduler, mamba_pool)
            mamba_evictable_kv = (
                (int(mamba_evictable) // mamba_tps) * int(tokens_per_page)
            )

            queue_len = len(getattr(scheduler, "waiting_queue", []) or [])
            x_tokens = len(getattr(req, "origin_input_ids", []) or [])

            # Cross-candidate Drain/Migration inputs, priced for the
            # CUMULATIVE free → drain → migrate fill the planner runs
            # (fire_planner Stage 1→2→3): each mechanism is charged ONLY for
            # the shortfall it covers, so the Admitter's predicted cost
            # equals the fire's actual byte cost (design.md §"Admitter —
            # per-arrival cost decision"). `src_*` are mamba (the cross
            # source); `mamba_free` came from the owner-provider page truth.
            # Size the shortfall against the ROUNDED harvest `x_eff` the
            # planner actually frees (n_pages_rounded·tps), so the drain/
            # migrate split matches what the fire pays — same basis as the
            # cross feasibility checks in decide().
            x_eff = (
                _round_up_pages(x_tokens, tokens_per_page, self.lcm_pages)
                * int(tokens_per_page)
            )
            shortfall_src = max(0, x_eff - mamba_free)
            # ONE c^evict walk at this target serves BOTH cross_evict (drains
            # the whole shortfall — feasible only when shortfall ≤ evictable,
            # where the target IS the shortfall) and cross_migrate (drains
            # all evictable, then migrates the rest). They coincide at the
            # boundary, so a single walk suffices.
            evict_target_src = min(shortfall_src, mamba_evictable_kv)
            migrate_tokens_src = max(0, shortfall_src - mamba_evictable_kv)
            n_migrate_slots = (
                (migrate_tokens_src + tokens_per_page - 1) // tokens_per_page
            )
            # `dst_migratable=0`: KV has no migrate primitive. own_migrate
            # stays inert via c_migrate_dst_us=+inf regardless, but pass 0 so
            # the candidate is structurally infeasible (units-honest).
            dst_migratable = 0

            # Cost-curve reads — c^evict walks the radix tree under the
            # lock so the priced set is byte-identical to a subsequent
            # evict().
            c_xfer_per_page_us = float(self.cost_model.c_xfer_us(1))
            # own_evict prices c^evict for the full X, amortized: the spot
            # cost of one eviction understates the true cost because eviction
            # is a chain reaction when the pool is full (each re-insert evicts
            # another entry). The amortization factor approximates the number
            # of downstream evictions one allocation triggers under pressure.
            c_evict_dst_us = float(self.cost_model.c_evict_us("kv", x_tokens))
            # cross Drain part: c^evict for the shortfall-targeted drain.
            # A zero target costs nothing AND must bypass the cache-None
            # +inf path (no drain ⇒ no eviction cost).
            c_evict_src_us = (
                float(self.cost_model.c_evict_us("mamba", evict_target_src))
                if evict_target_src > 0
                else 0.0
            )
            # cross Migrate part: c_m for the slots beyond free+drain. 'kv'
            # is +inf (no KV migrate primitive); 'mamba' is +inf during
            # cold-start (probe unseeded) — fail-closed — but c_m(0)=0, so a
            # no-migrate fill never trips the cold-start gate.
            c_migrate_src_us = float(
                self.cost_model.c_migrate_us("mamba", n_migrate_slots)
            )
            c_migrate_dst_us = float(
                self.cost_model.c_migrate_us("kv", n_migrate_slots)
            )

            # --- Symmetric routing ---------------------------------------
            # A hybrid arrival needs BOTH a KV slice (x_tokens) and a mamba
            # state slot. Detect which pool blocks THIS arrival and grow that
            # one: the existing dst=kv path grows KV from mamba (m2k); the new
            # dst=mamba path grows mamba from KV (k2m — the burst that crashes
            # where mamba can't fork a cache slot). If BOTH are scarce
            # the two grows are opposite cross directions (each is the other's
            # source) and neither can be served, so defer to sglang's normal
            # back-pressure — never fire a doomed cross or admit into a pool
            # that will crash.
            mamba_free_slots = (
                int(mamba_allocator.available_size()) if mamba_pool is not None else 0
            )
            kv_scarce = (kv_free + kv_evictable) < x_tokens
            mamba_scarce = (
                mamba_pool is not None
                and (mamba_free_slots + int(mamba_evictable))
                < self._mamba_arrival_need_slots
            )
            if mamba_scarce and kv_scarce:
                # Both pools blocked: a hybrid req needs room in both, and the
                # two grows are opposite cross directions (each is the other's
                # source), so neither grow can serve it. Defer to sglang's
                # normal back-pressure. Carry the full candidate vector (all
                # grows structurally infeasible) so logging / D6n gates see a
                # complete decision, same shape as `decide()`.
                inf = float("inf")
                decision = AdmitterDecision(
                    action="defer",
                    reason=(
                        f"both pools scarce (kv {kv_free}+{kv_evictable}<{x_tokens}; "
                        f"mamba {mamba_free_slots}+{int(mamba_evictable)}<"
                        f"{self._mamba_arrival_need_slots}) -> defer"
                    ),
                    candidate_costs_us={
                        "own_free": inf, "own_evict": inf,
                        "cross_free": inf, "cross_evict": inf,
                        "own_migrate": inf, "cross_migrate": inf,
                        "defer": float(queue_len) * self.cost_model.w_q_us(),
                    },
                    dst_pool="kv",
                    src_pool="mamba",
                )
            elif mamba_scarce:
                decision = self._decide_grow_mamba(
                    mamba_free_slots=mamba_free_slots,
                    mamba_evictable_slots=int(mamba_evictable),
                    mamba_tokens_per_chunk=mamba_tps,
                    kv_free=kv_free,
                    kv_evictable=kv_evictable,
                    queue_len=queue_len,
                    tokens_per_page=int(tokens_per_page),
                )
            else:
                decision = self.decide(
                    x_tokens=x_tokens,
                    dst_pool="kv",
                    dst_free=kv_free,
                    dst_evictable=kv_evictable,
                    src_pool="mamba",
                    src_free=mamba_free,
                    src_evictable=mamba_evictable_kv,
                    queue_len=queue_len,
                    c_evict_dst_us=c_evict_dst_us,
                    c_evict_src_us=c_evict_src_us,
                    c_xfer_per_page_us=c_xfer_per_page_us,
                    dst_migratable=dst_migratable,
                    src_migratable=src_migratable,
                    c_migrate_dst_us=c_migrate_dst_us,
                    c_migrate_src_us=c_migrate_src_us,
                    tokens_per_page=tokens_per_page,
                )

        decide_us = (time.perf_counter() - _t0) * 1e6
        if self._log_fp is not None:
            self._log_decision(
                decision,
                x_tokens=x_tokens,
                queue_len=queue_len,
                decide_us=decide_us,
            )
        return decision

    def _mamba_tokens_per_chunk(self, scheduler, mamba_pool) -> int:
        """Mamba slots per arena chunk — the transfer granularity for a k2m
        grow (one transferred page = one chunk = this many mamba slots). The
        owner provider is the production ground truth (same source as
        `_mamba_feasibility`'s `mamba_tokens_per_page`); tests use a scheduler
        seam; otherwise read the arena directly. Defaults to 1 (atomic layout)
        when undeterminable — safe for the grow decision (an atomic assumption
        only ever over-grows mamba, never under-grows it)."""
        if self.owner_provider is not None:
            return max(1, int(self.owner_provider.mamba_tokens_per_page()))
        if hasattr(scheduler, "get_mamba_tokens_per_chunk"):
            return max(1, int(scheduler.get_mamba_tokens_per_chunk()))
        arena = getattr(mamba_pool, "_mamba_temporal_arena", None)
        if arena is not None:
            return max(1, int(arena.tokens_per_chunk))
        return 1

    def _decide_grow_mamba(
        self,
        *,
        mamba_free_slots: int,
        mamba_evictable_slots: int,
        mamba_tokens_per_chunk: int,
        kv_free: int,
        kv_evictable: int,
        queue_len: int,
        tokens_per_page: int,
    ) -> AdmitterDecision:
        """Price growing mamba from KV (k2m) for a mamba-scarce arrival — the
        symmetric mirror of the dst=kv path. KV is the cross SOURCE; mamba is
        the dst we grow. Everything is in KV-token-equiv (the actuator
        transfers whole pages, `tokens_per_page` KV-equiv each), so `decide`'s
        feasibility/cost comparison stays unit-consistent.

        `dst_migratable=0` / `src_migratable=0`: KV has no consolidation-
        migrate primitive as a cross source, and mamba own-migrate is inert;
        both stay structurally infeasible via +inf migrate costs.
        """
        need = self._mamba_arrival_need_slots
        tps = max(1, int(mamba_tokens_per_chunk))
        # The transfer unit is the arena CHUNK: one transferred page maps one
        # chunk = `tps` mamba slots. So growing mamba by `need` SLOTS takes
        # `ceil(need / tps)` chunks; expressing the demand in chunks (× the KV
        # tokens_per_page so c_xfer/feasibility stay in KV-token-equiv) makes
        # `execute_decision` size the fire in mamba chunks, not the req's KV
        # input. Mamba own-capacity is likewise whole free/evictable chunks.
        need_chunks = (need + tps - 1) // tps
        mamba_free_chunks = mamba_free_slots // tps
        mamba_evictable_chunks = mamba_evictable_slots // tps
        x_grow = need_chunks * int(tokens_per_page)
        # mamba cache chunks evictable toward the need (own_evict sizing).
        short_chunks = max(0, need_chunks - mamba_free_chunks)

        c_xfer_per_page_us = float(self.cost_model.c_xfer_us(1))
        # own_evict on mamba (dst): evict mamba cache to self-serve the chunks.
        c_evict_dst_us = float(
            self.cost_model.c_evict_us("mamba", min(short_chunks, mamba_evictable_chunks))
        )
        # cross drain on KV (src): evict KV cache only for the shortfall the
        # free pages can't cover (a zero target costs nothing and must bypass
        # the cache-None +inf path — mirrors the dst=kv branch).
        kv_shortfall = max(0, x_grow - kv_free)
        kv_drain_target = min(kv_shortfall, kv_evictable)
        c_evict_src_us = (
            float(self.cost_model.c_evict_us("kv", kv_drain_target))
            if kv_drain_target > 0
            else 0.0
        )
        return self.decide(
            x_tokens=x_grow,
            dst_pool="mamba",
            dst_free=mamba_free_chunks * int(tokens_per_page),
            dst_evictable=mamba_evictable_chunks * int(tokens_per_page),
            src_pool="kv",
            src_free=kv_free,
            src_evictable=kv_evictable,
            queue_len=queue_len,
            c_evict_dst_us=c_evict_dst_us,
            c_evict_src_us=c_evict_src_us,
            c_xfer_per_page_us=c_xfer_per_page_us,
            dst_migratable=0,
            src_migratable=0,
            c_migrate_dst_us=float("inf"),
            c_migrate_src_us=float("inf"),
            tokens_per_page=int(tokens_per_page),
        )

    def _mamba_feasibility(self, scheduler, mamba_pool, tokens_per_page):
        """Mamba-source `(mamba_free, src_migratable)` in KV-token-equiv,
        at PAGE granularity (design.md §"Admitter — per-arrival cost
        decision" feasibility quantities `m_src^free` and `migratable_src`).
        `tokens_per_page` is the KV-token-equiv per transferred chunk.

        Production reads the SAME `SchedulerOwnerProvider` page computation
        the planner selects from, so Admitter feasibility and planner page
        selection can never disagree:
          - `mamba_free`     = (#fully-free mamba pages) · tokens_per_page
          - `src_migratable` = (#pages Migration can consolidate to free,
            scattered-free-slot budgeted) · tokens_per_page

        Cost discipline (the values only ever feed cross_free /
        cross_migrate):
          - When `cross_fire` is disabled, `decide` gates cross-* to +inf
            regardless, so we return (0, 0) without touching the pool —
            the common default path stays free of any owner-map work.
          - On an ATOMIC mamba layout (`tokens_per_chunk == 1`, the
            tp=1/fp32 single-GPU bench corner) every free slot is its own
            whole-free chunk and Migration consolidates nothing, so
            `src_migratable == 0` and `mamba_free` is just the free-slot
            count — computed CHEAPLY without building the owner map
            (avoids a per-arrival GPU sync under the KV lock). Only a
            fragmentable layout (`tokens_per_chunk >= 2`, TP / bf16 ssm)
            pays for the page walk.

        Unit tests that need migratable pages wire a MockOwnerProvider
        on `self.owner_provider` with `mamba_tokens_per_page() > 1`
        (fragmentable) and a `build_mamba_owner_map` returning free/live
        pages. Without a wired provider (boot temporal gap), the fallback
        reports free slots as KV-equiv and 0 migratable.
        """
        tps = int(tokens_per_page)
        mamba_allocator = getattr(mamba_pool, "_allocator", mamba_pool) if mamba_pool is not None else None
        provider = self.owner_provider
        if provider is not None:
            # Provider is the production ground truth.
            mamba_tps = provider.mamba_tokens_per_page()
            if mamba_tps <= 1:
                # Atomic: free slots == whole-free chunks, no migration.
                # Cheap path — no owner-map build / GPU sync on the arrival.
                free_slots = (
                    int(mamba_allocator.available_size())
                    if mamba_pool is not None
                    else 0
                )
                return free_slots * tps, 0
            owner_map = provider.build_mamba_owner_map(allow_migrate=True)
            if owner_map is not None:
                free_pages = len(owner_map.free_pages)
                mig_pages = len(owner_map.live_pages_in_cost_order or [])
                return free_pages * tps, mig_pages * tps
            return 0, 0

        # No provider yet (boot → first BudgetAgent tick, ~1 s). The actuator
        # is also None in this window, so cross-* can't fire; a free-slot
        # approximation is safe (own-free/own-evict/defer only).
        if mamba_pool is not None:
            return int(mamba_allocator.available_size()) * tps, 0
        return 0, 0

    def _log_decision(self, decision: AdmitterDecision, *,
                      x_tokens: int, queue_len: int,
                      decide_us: Optional[float] = None) -> None:
        """Emit one JSON line for this decision. Schema:
          ts, action, reason, dst_pool, src_pool, x_tokens, fire_x_tokens,
          queue_len, candidate_costs_us, decide_us,
          (optional) fire_granted_pages, fire_total_us, fire_aborted.

        `decide_us` is the wall time of the whole `decide_for_req` call
        (scheduler-side state derivation + the pure `decide`), i.e. the
        per-arrival scheduler-thread cost of the Admitter.

        `dst_pool` is the GROW direction — "mamba" marks the symmetric k2m
        grow, so a reader can count how often the Admitter grew mamba from KV
        vs the original m2k. `fire_x_tokens` is the demand the fire was
        sized for (mamba-chunk-equiv for a k2m grow), distinct from x_tokens
        (the req's KV input).
        """
        if self._log_fp is None:
            return
        # JSON has no Infinity; map infeasible candidates to JSON `null`.
        # Consumers should treat null as "infeasible / cost = +inf".
        entry = {
            "ts": round(time.time(), 6),
            "action": decision.action,
            "reason": decision.reason,
            "dst_pool": decision.dst_pool,
            "src_pool": decision.src_pool,
            "x_tokens": int(x_tokens),
            "fire_x_tokens": (
                int(decision.fire_x_tokens)
                if decision.fire_x_tokens is not None
                else None
            ),
            "queue_len": int(queue_len),
            "candidate_costs_us": {
                k: (None if v == float("inf") else round(v, 1))
                for k, v in decision.candidate_costs_us.items()
            },
            "decide_us": (
                round(decide_us, 1) if decide_us is not None else None
            ),
        }
        fr = decision.fire_result
        if fr is not None:
            entry["fire_granted_pages"] = int(fr.granted_pages)
            entry["fire_total_us"] = int(fr.total_us)
            entry["fire_aborted"] = bool(fr.aborted)
        self._log_fp.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        """Flush + close the JSONL log. Called by the scheduler at
        shutdown. Idempotent."""
        if self._log_fp is not None:
            self._log_fp.flush()
            self._log_fp.close()
            self._log_fp = None


def _round_up_pages(x_tokens: int, tokens_per_page: int, lcm_pages: int) -> int:
    """Pages needed for `x_tokens` (round up), then rounded up to the
    actuator's per-fire LCM. Single source of truth shared by `decide`
    (cost + cross feasibility) and `decide_for_req` (drain/migrate
    shortfall), so they agree on the rounded harvest the planner builds."""
    lcm = max(1, int(lcm_pages))
    tps = max(1, int(tokens_per_page))
    n_pages = max(1, (int(x_tokens) + tps - 1) // tps)
    return ((n_pages + lcm - 1) // lcm) * lcm


def _disagg_label(mode: Any) -> str:
    """Normalize disagg-mode enum or string to its string label."""
    # Enum-shaped: has a `.name` (e.g. DisaggregationMode.NULL.name == "NULL")
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(mode, str):
        return mode
    return str(mode)


def _evictable_size_kv(tree_cache: Any) -> int:
    """Return KV evictable tokens. Hybrid SSM models use a
    `MambaRadixCache` whose plain `evictable_size()` raises
    NotImplementedError — they expose `full_evictable_size()` instead
    for the KV slice. Plain RadixCache has only `evictable_size()`.
    Pick whichever is available.
    """
    if tree_cache is None:
        return 0
    # Prefer the hybrid-specific accessor when present.
    if hasattr(tree_cache, "full_evictable_size"):
        try:
            return int(tree_cache.full_evictable_size())
        except NotImplementedError:
            pass
    if hasattr(tree_cache, "evictable_size"):
        try:
            return int(tree_cache.evictable_size())
        except NotImplementedError:
            return 0
    return 0


def _evictable_size_mamba(tree_cache: Any, scheduler: Any) -> int:
    """Return mamba evictable tokens. MambaRadixCache exposes
    `mamba_evictable_size()`. Tests can override via
    `scheduler.get_mamba_evictable()`.
    """
    if hasattr(scheduler, "get_mamba_evictable"):
        return int(scheduler.get_mamba_evictable())
    if tree_cache is None:
        return 0
    if hasattr(tree_cache, "mamba_evictable_size"):
        try:
            return int(tree_cache.mamba_evictable_size())
        except NotImplementedError:
            return 0
    return 0


def _get_mamba_pool(scheduler: Any) -> Optional[Any]:
    """Locate the ``MambaPool`` (design.md §"Page ownership state") on
    a sglang Scheduler.

    Production access path: ``scheduler.token_to_kv_pool_allocator
    .get_kvcache().mamba_pool``. Test stubs may instead expose a
    ``get_mamba_pool()`` accessor, which takes precedence.
    """
    if hasattr(scheduler, "get_mamba_pool"):
        return scheduler.get_mamba_pool()
    kv_alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if kv_alloc is None:
        return None
    if hasattr(kv_alloc, "get_kvcache"):
        kv_cache = kv_alloc.get_kvcache()
        return getattr(kv_cache, "mamba_pool", None)
    return None
