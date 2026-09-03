"""Aginfer daemon kv_scheduler (T7).

For each paper §4 event arriving on the EventRouter, this module:

1. Fetches a fresh ``/aginfer/state`` snapshot from sglang.
2. Builds ``D_t`` — the decision_set per paper §4's table.
3. Calls the SHARED ``OursGreedyPolicy.decide(...)`` (same instance the
   inline scorer is calibrated against; lives in ``baselines/ours_greedy.py``).
4. Translates ``Action.assignments`` (a list of ``(unit_id, Tier)`` pairs)
   to the on-the-wire form ``[{"hash": ..., "target_tier": "HBM"|...}, ...]``
   and POSTs to ``POST /aginfer/migrate``.

The handler is registered on the EventRouter for every paper §4 event
kind via :func:`attach_kv_scheduler`.  ``memory_pressure`` /
``pressure_resolved`` are also routed here (they ALSO need to drive
admission_controller — T8 — but the kv_scheduler still owns the
migrate part of the response).

Design contract (verify/t7/README.md):

* ``decide()`` is called with a FRESH state on every event; never
  cached.  The EventRouter's serial-worker contract guarantees no
  two ``decide()`` calls overlap.
* ``decision_set`` is BUILT, not "everything" — paper §4 promised
  D_t is event-scoped.  For ``memory_pressure`` D_t is bounded by
  ``top_k`` (default 256) to keep ``decide()`` < 50 ms regardless
  of total tree size.
* λ_ACTING is a calibrated constant (default 1/5; mean tool call is
  ~5 s on terminus-2's swebenchpro).  λ_REASONING is derived from
  ``hits / age`` exactly as the inline scorer does
  (``baselines/sglang_adapter.py`` :func:`_node_to_unit`).
* Idempotent: re-receiving the same event produces the same
  migrate-set (modulo state drift between fetches).

Lambda calibration justification: see verify/t7/README.md §CALIBRATION
and the sensitivity sweep in verify/t7/verify.py [step_lambda_sweep].
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from baselines.base import (
    Action,
    ReuseUnit,
    Scope,
    SchedulerState,
    Tier,
    TierUsage,
    UnitType,
)
from baselines.costs import default_costs
from baselines.ours_greedy import OursGreedyPolicy

from ._fatal import fatal
from .action_timeline import PromoteAction
from .events import Event, EventKind
from .program_tracker import ProgramTracker, State

# #251 increment 4: build_paper_state + its closure moved to the in-engine package
# (sglang.srt.mem_cache.aginfer.state_builder) so the ENGINE builds its own s_t in-process.
# Re-imported here (single source) so the daemon's call sites + the verify suite that import
# these names from daemon.kv_scheduler keep working unchanged.
from sglang.srt.mem_cache.aginfer.state_builder import (  # noqa: E402,F401
    _clamp_lambda_acting,
    _estimate_load_back_s,
    _tier_from_string,
    _log_unknown_tier_once,
    _flatten_per_rank,
    build_paper_state,
    _units_for_session,
    _shared_prefix_units,
    _top_k_by_regret,
    _build_decision_set,
    hints_from_state,
    _env_float,
    _env_int,
    _LAMBDA_ACTING_FLOOR,
    _LAMBDA_ACTING_CEIL,
    _DEFAULT_LAMBDA_ACTING,
    _CONST_VU,
    _PHAT_REUSE_ALPHA,
    _PHAT_BOOTSTRAP_DT,
    _p_access_holder,
    _DEFAULT_MEMORY_PRESSURE_TOPK,
    _PROMOTE_FALLBACK_BW_BPS,
    LINK_IDLE_SECONDS,
    LINK_PAIRS,
    _TIER_LABEL_MAP,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- calibration

# λ for a unit owned by a program in ACTING state.  Default 1/5: mean
# tool call on terminus-2's swebenchpro is ~5 s (range 1–30).  Clamped
# to [1/30, 1/1] per audit #15 — see verify/t7/README.md WORST CASE.







# #208 const-V_u isolation arm: neutralise the reuse-prediction signal
# (p_hat / lambda → constant) in build_paper_state so the daemon's migrate /
# admission ranking is reuse-blind.  Matches sglang_adapter._CONST_VU.
# DESIGN §7: p_hat is the PROBABILITY a unit is reused before eviction.  It must
# be estimated from DEMONSTRATED reuse (hit_count), not from holder liveness — a
# live holder is not a reuse guarantee.  p_hat = 1 - exp(-alpha * reuses), where
# reuses = hit_count - 1 (observed repeats): a never-reused one-shot (hits<=1) ->
# 0 (its save term collapses, hold tax evicts it first); a heavily-reused prefix
# -> ~1 (kept).  Monotone in reuse count, recency-decoupled (no /age), never
# saturates for one-shot units.  (Reuse-interval IMMINENCE decay is T11 future.)
if _CONST_VU:
    logging.getLogger(__name__).warning(
        "AGINFER_CONST_VU active — daemon V_u reuse signal neutralised "
        "(p_hat=lambda=1.0) (#208)")

# Top-k cap on the memory_pressure decision_set.  Paper §7.1.  256 is
# enough to materially affect HBM occ on a B300 (~half a percent per
# unit at 2 KB/token × 4 k tokens/unit).

# #223: per-hash TOCTOU evict cooldown (seconds).  A device-leaf at dump time
# can gain a device-KV child before the migrate applies (a concurrent request
# extends that prefix — common for popular shared-prefix nodes), so a
# correctly-proposed remove-HBM is rejected at apply (remove_hbm_not_device_
# leaf).  The dump keeps reporting it as a leaf, so it is re-proposed every
# event → a systematic reject storm (956/cycle observed, migrate_applied=0).
# On a leaf-reject APPLY_FAILED we cool the hash down for this long; the
# dispatch skips re-proposing a REMOVE for a cooled hash until it expires (by
# then the node has usually gained a stable host backup or been evicted by
# sglang itself).  Backoff, not value math — wall-clock is fine here.
_EVICT_COOLDOWN_S = _env_float("AGINFER_EVICT_COOLDOWN_S", "5.0")

# #215 resume dedup: suppress a re-fire for this many dump generations after
# the clearing PUT is sent (covers the overlay lag), then allow a re-fire so a
# genuinely lost PUT still recovers.  2 generations comfortably exceeds the
# normal PUT round-trip while bounding the worst-case re-starve to ≤2 events.
_RESUME_DEDUP_WINDOW = _env_int("AGINFER_RESUME_DEDUP_WINDOW", "2")

# #240 do-no-harm SATURATION YIELD.  The daemon's explicit demote of a caller's
# idle tail races sglang's OWN lock on the same node at apply time (in-flight
# write_through of the just-finished turn, or sglang's V_u-guided reactive
# eviction under pressure) → `remove_hbm_not_device_leaf:locked`, migrate never
# lands.  Re-dispatching it every event only churns (the 5× do-no-harm regression
# at occ≈0.99).  We SELF-MEASURE whether recent demotes actually landed (the
# hash's residence in the next fresh dump no longer has HBM); below this apply-
# rate EMA the relief phase YIELDS the explicit demote.  This is value-optimal,
# not a workaround (DESIGN §9 saturation yield): the room is freed by sglang's
# reactive eviction regardless, so the proactive migrate buys nothing it can land
# — withholding it (no lock-race churn) strictly dominates, and the daemon keeps
# its unique lever, the predictive promote.
_DEMOTE_YIELD_EMA = _env_float("AGINFER_DEMOTE_YIELD_EMA", "0.4")

# DESIGN §3/§7 predictive-promote (action-timeline) scheduling constants.
# The promote-back of a tool-bound agent's idle tail is scheduled for
# ``T_start + tool_ETA − load_back_latency − margin`` so it lands in HBM a
# touch BEFORE the resume prefill rather than racing the transfer.  The
# margin absorbs daemon→sglang dispatch + apply jitter.  The bandwidth
# fallback is used ONLY when no σ→HBM ``bw_free`` sample exists yet (cold
# start); a conservative DRAM-class rate keeps the lead time from collapsing
# to zero (which would degenerate to a TOOL_CALL_END-time promote).
_PROMOTE_SAFETY_MARGIN_S = _env_float("AGINFER_PROMOTE_MARGIN_S", "0.05")
# #241: the predictive promote now executes as a prefill-only WARM (#238) — a
# full /generate(max_new_tokens=0), not a pure KV load_back.  Its completion
# cost is therefore a PREFILL (storage-load + any recompute + scheduler queue),
# ~seconds under load, not the ~0.1s of a DRAM/DISK transfer.  So the promote
# must be scheduled this many seconds before the resume (a FLOOR on the lead) so
# the warm completes in time; the data-transfer ``load_back_s`` is only a lower
# bound.  Win-preserving to overshoot (prefix simply HBM-resident a bit earlier).
_WARM_LEAD_S = _env_float("AGINFER_WARM_LEAD_S", "2.5")


def _filter_cooled_evicts(plan: List[Any], cooldown: Dict[str, float],
                          now: float) -> List[Any]:
    """#223 / #251 Stage B: the cooldown filter moved into the in-engine driver
    (single implementation).  This thin delegator keeps the daemon's call path and
    verify/kv_scheduler_value_rule working unchanged."""
    from sglang.srt.mem_cache.aginfer.scheduler_driver import filter_cooled_evicts
    return filter_cooled_evicts(plan, cooldown, now)


# DESIGN §7 bw_free branch: link is "cold-idle" iff
# time_since_last_sample_s > LINK_IDLE_SECONDS.  Public so
# verify/t13/ imports it instead of redeclaring (audit #175 —
# drift between the local-shadow constant in verify/ and the
# production constant would let either side change silently).

# The 4 transfer directions the daemon's bw_free vector covers
# (DESIGN §7 4 link channels).  Public for the same reason as
# LINK_IDLE_SECONDS above.






# ----------------------------------------------------------------- adapter












# ----------------------------------------------------------------- D_t builders










# ----------------------------------------------------------------- dispatch


# #251 Stage B: the migrate wire translators are single-sourced in the in-engine
# driver (so the in-process apply path and this daemon HTTP path emit byte-identical
# wire). Re-exported here so existing `from daemon.kv_scheduler import assignments_to_wire`
# callers (verify/kv_scheduler_value_rule, verify/t36, the legacy probe) keep working.
from sglang.srt.mem_cache.aginfer.scheduler_driver import (  # noqa: E402
    tier_to_wire as _tier_to_wire,
    assignments_to_wire,
)




# ----------------------------------------------------------------- handler


class KvScheduler:
    """Thin holder: shared policy + migrate dispatcher.

    The handler closure binds the EventRouter; we keep a class so the
    verify tests can inspect ``policy`` / replace the migrate URL / etc.
    """

    def __init__(
        self,
        *,
        tracker: ProgramTracker,
        sglang_base_url: str,
        policy: Optional[OursGreedyPolicy] = None,
        lambda_acting: float = _DEFAULT_LAMBDA_ACTING,
        observability=None,  # daemon._observability.DaemonObservability
        outbound=None,       # daemon.outbound.OutboundQueue
    ) -> None:
        self.tracker = tracker
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self.policy = policy or OursGreedyPolicy(default_costs())
        self.lambda_acting = lambda_acting
        # T11 / §1 online ETA estimator: learns the per-tool-call ETA from
        # observed TOOL_CALL_START→END intervals at a FINE granularity (a call
        # signature — `ls` vs `sleep` within the `bash` tool), so the §7 demote
        # value-gate uses a SELF-LEARNED ETA instead of an externally-fed
        # constant.  Cold start falls back to the event-provided `tool_eta_s`.
        from .eta_estimator import ETAEstimator
        self.eta_estimator = ETAEstimator()
        # T42: optional injection from main.py (router.observability).
        # When set, ``_record_skips`` and other failure paths can bump
        # the per-reason counter.  Left None for unit tests that only
        # care about the policy / wire path.
        self.observability = observability
        # T36: injection of the fire-and-forget outbound queue.
        # ``_dispatch_migrate`` REQUIRES this; constructing a
        # KvScheduler without it is a wiring bug (DESIGN §6 B4 makes
        # outbound the only valid production dispatch path).  Tests
        # that only exercise ``handle()`` up to the dispatch point may
        # leave it None; tests that call ``_dispatch_migrate`` must
        # provide one.
        self.outbound = outbound
        # DESIGN §9 (#194): when True, joint_decide includes the
        # program-level Pause/Resume candidates (the full union action
        # space).  When False, the joint decision is migrate-only (the
        # kv-only ablation arm — Run K).  main.py sets it from the
        # --admission-controller flag.
        self.admission_enabled: bool = False
        # Telemetry for tests.
        self.decisions: int = 0
        self.migrate_calls: int = 0
        self.pause_calls: int = 0    # #194: program pauses dispatched
        self.resume_calls: int = 0   # #194: program resumes dispatched
        # DESIGN §3/§7 predictive promote (action-timeline) telemetry.
        self.promotes_scheduled: int = 0     # promote actions placed on heap
        self.promotes: int = 0               # promote migrates dispatched
        self.promotes_skipped_stale: int = 0  # belief-invalidated at fire
        self.hint_calls: int = 0  # T40 (#184): hint PUTs enqueued
        # #230 Tier-2 characterization knob: artificially defer hint
        # DELIVERY by this many ms (the hint is computed now but reaches
        # sglang stale).  0 = off (production default).  Used ONLY by the
        # hint-latency-budget e2e arms to measure the freshness knee on the
        # real stack; never set in production.
        self._hint_delay_s: float = _env_float(
            "AGINFER_HINT_DELAY_MS", "0") / 1000.0
        self.hint_delayed_calls: int = 0
        self.last_action: Optional[Action] = None
        self.last_plan: Optional[List[Any]] = None
        self.last_decision_set_size: int = 0
        # #251 Stage B: the post-joint_decide decision subsystem (saturation-yield
        # apply-rate EMA + #223 cooldown filter) lives in the IN-ENGINE driver now;
        # the daemon delegates to it.  Behaviour-identical to the former inline
        # _pending_demote/_demote_apply_ema/_dump_gen + handle() block — this is the
        # first piece of the decision brain relocated in-process (the #251 blocker).
        from sglang.srt.mem_cache.aginfer.scheduler_driver import AginferDriver
        self.driver = AginferDriver()
        # Audit round-2 R2-N2: per-instance unknown-tier log set so
        # cross-test / cross-restart state doesn't leak.
        self._unknown_tier_log: set = set()
        # #215 resume double-fire dedup lives in the program_tracker (the
        # single authority for program lifecycle), reconciled against the
        # fresh dump each event — see ProgramTracker.reconcile_resume_acks /
        # resume_in_flight.  No parallel structure here.

    async def handle(self, event: Event, router) -> None:  # noqa: ANN001
        """Single entry point for all paper §4 events.

        Always refetches state at entry (per design contract), builds
        D_t, runs ``decide()``, POSTs any migrate actions.  Errors
        downstream of the policy do NOT propagate — paper §9 promises
        the inline scorer is a safety net.
        """
        # DESIGN §3 belief plane: program lifecycle (REASONING/ACTING) is
        # EVENT-sourced.  Drive the tracker from the event kind so a daemon fed
        # events directly (the S1 token-space driver POSTs to /aginfer/event,
        # bypassing the proxy that normally calls observe_arrival/completion)
        # tracks state the same as the proxy-observed path.  Idempotent + same
        # target states as the proxy's observe_* calls, so double-driving
        # (proxy + event, when both are active) converges harmlessly.
        self._apply_belief_transition(event)
        # T11 §1: drive the online ETA estimator from the tool-call lifecycle.
        # On START, replace the demote value-gate's ETA with the LEARNED one once
        # the call's fine-grained signature (tool + command-token; `ls` vs `sleep`
        # within `bash`) has enough observations — so the §7 gate keys on a
        # self-learned ETA, not an external constant.  Cold start keeps the
        # event-provided `tool_eta_s` as a bootstrap.  On END, record the observed
        # interval so future calls of the same signature predict from it.
        if event.kind == EventKind.TOOL_CALL_START and event.session:
            _tn = event.payload.get("tool_name")
            _ta = event.payload.get("tool_args") or event.payload.get("args") or {}
            _sig = self.eta_estimator.on_tool_call_start(
                event.session, _tn, _ta, event.enqueue_time)
            _learned = self.eta_estimator.predict(_tn, _ta)
            if _learned is not None and _learned > 0.0:
                event.payload["tool_eta_s"] = _learned
            from ._metrics import m as _m
            _m("eta_estimate", sig="/".join(_sig),
               learned=("none" if _learned is None else round(_learned, 3)),
               bootstrap=round(float(event.payload.get("tool_eta_s") or 0.0), 3))
        elif event.kind == EventKind.TOOL_CALL_END and event.session:
            self.eta_estimator.on_tool_call_end(event.session, event.enqueue_time)
        elif event.kind == EventKind.SESSION_END and event.session:
            # #238: drop the registered prefix so the warm cache stays bounded.
            getattr(router, "_session_prefix", {}).pop(event.session, None)
        try:
            state_json = await router.fetch_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kv_scheduler: /aginfer/state fetch failed for %s: %r",
                event.kind.value, exc,
            )
            from ._metrics import m as _m
            _m("state_fetch_failed", kind=event.kind.value)
            # T42 audit G2: route load-fault tally through the same
            # observability counter that already collects migrate-skip
            # reasons.  APPLY_FAILED webhook (T23+T37 / #153) will plug
            # in via the same recorder when it lands.
            if self.observability is not None:
                self.observability.record_failure("state_fetch_failed")
            return
        # Emit per-(tier, subpool) occupancy snapshot for every handled
        # event.  Bounded ~16 ms wall per cycle.  Raw data behind
        # the F3 / T14 occupancy-trajectory observability.
        from ._metrics import m as _m
        # Direct subscript: schema contract is enforced by sglang's dump;
        # a real schema break should surface as a KeyError logged with
        # full traceback (handle()'s try/except surrounds the
        # subsequent build_paper_state call which also reads these
        # fields).  Per audit: silent .get(..., default) masks the
        # schema-vs-daemon gap (cf. the G10 fix this code originally
        # tried to instrument).
        flat = _flatten_per_rank(state_json)
        pool_usage = flat["pool_usage"]
        # Authoritative HBM occupancy = MAX over subpools per DESIGN §5
        # ('admission acts when ANY subpool crosses theta_hi').
        hbm_subpools = pool_usage["HBM"]["subpools"]
        occ_hbm = max(
            (e["used_bytes"] / e["cap_bytes"]) if e["cap_bytes"] > 0 else 0.0
            for e in hbm_subpools.values()
        ) if hbm_subpools else 0.0
        dram_subpools = pool_usage["DRAM"]["subpools"]
        occ_dram = max(
            (e["used_bytes"] / e["cap_bytes"]) if e["cap_bytes"] > 0 else 0.0
            for e in dram_subpools.values()
        ) if dram_subpools else 0.0
        # Radix-resident slice (evictable bytes) as a separate signal:
        # post-T17 the daemon no longer maintains a "tree-view" stat
        # because pool_usage IS the unified allocator view.  Emit
        # evictable_bytes / cap_bytes as a proxy for "how much of HBM
        # is radix-resident" so the trajectory parse can see the
        # in-flight-vs-radix split.
        tree_occ_hbm = max(
            (e["evictable_bytes"] / e["cap_bytes"]) if e["cap_bytes"] > 0
            else 0.0
            for e in hbm_subpools.values()
        ) if hbm_subpools else 0.0
        n_units = len(flat["units"])
        _m(
            "state_fetched",
            kind=event.kind.value,
            occ_hbm=occ_hbm,              # AUTHORITATIVE pressure
            tree_occ_hbm=tree_occ_hbm,    # evictable slice for debug
            occ_dram=occ_dram,
            units=n_units,
        )
        try:
            sched_state = build_paper_state(
                state_json,
                event=event,
                tracker=self.tracker,
                lambda_acting=self.lambda_acting,
                unknown_tier_log=self._unknown_tier_log,
            )
        except Exception:  # noqa: BLE001
            logger.exception("kv_scheduler: build_paper_state raised; skip")
            _m("kv_decide", kind=event.kind.value, outcome="build_state_raised")
            return
        # #190 (bounded tracker): reclaim ENDED programs whose KV has
        # fully cleared from the snapshot.  `live_pids` = every pid that
        # still holds a unit in THIS fresh dump; an ENDED pid absent from
        # it has no residual KV and no bookkeeping left.  Runs every
        # event on the fresh state (cheap dict scan), mirroring sglang's
        # ENDED-no-units dump-GC (#186) so the daemon tracker stays
        # bounded by the live-unit set.
        live_pids = {
            sid for u in sched_state.units.values() for sid in u.holders
        }
        self.tracker.gc_ended(live_pids)
        # #240 saturation-yield: measure whether our recently-dispatched demotes
        # actually landed (the hash is no longer HBM-resident in THIS fresh dump)
        # and update the apply-rate EMA; the post-joint_decide filter below uses
        # it to yield futile explicit demotes (lock-race vs sglang's own eviction).
        self._update_demote_apply_rate(sched_state.units)
        self.last_decision_set_size = len(sched_state.decision_set)
        # DESIGN §3/§7 action-timeline plane: on TOOL_CALL_START schedule the
        # caller's idle-tail predictive promote-back for ``T_start + tool_ETA −
        # load_back`` so it lands in HBM as the resume prefill arrives.  The
        # demote half is taken by joint_decide below (the tail is this event's
        # D_t); this is the PROMOTE half, deferred onto the due-action heap and
        # belief-validated at fire (fire_due_action).  No-op without a wired
        # timeline or a payload ETA (degrades to promote-at-TOOL_CALL_END).
        if event.kind == EventKind.TOOL_CALL_START:
            self._schedule_promote_back(event, sched_state, router)
        # T40 (#184, F2): push the V_u hints for EVERY D_t unit,
        # unconditionally, BEFORE (and independent of) the joint decision.
        # The inline scorer reads the hint table at its allocation
        # callsite and cannot wait for the daemon, so the daemon refreshes
        # it every event.  No shadow cache (DESIGN §10): re-pushes are
        # absorbed by sglang's overwrite-by-stamp.  An empty D_t ⇒ empty
        # hint list ⇒ no-op push (handled in _dispatch_hints).
        await self._dispatch_hints(hints_from_state(sched_state))
        # DESIGN §9 (#194): joint_decide runs on EVERY event, even when
        # D_t is empty (LLM_PREFILL — D_t always ∅ — or an empty top-k).
        # migrate_candidates yields nothing then, but the admission
        # generators (pause/resume) still produce candidates from live
        # state, so the joint decision must NOT be short-circuited on an
        # empty decision_set.  (The greedy-era `if not decision_set:
        # return` did exactly that, leaving admission inert on
        # LLM_PREFILL — #194 audit, caught by verify/joint_decide stage F.)
        # DESIGN §9 (#194): ONE joint decision over the union action
        # space {migrate} ∪ {pause/resume}, replacing the old sequential
        # greedy-decide-then-admission decompose.  Thresholds + horizon
        # come from the router (the single source of truth, T22/§10);
        # cost calibration from the shared policy instance.
        # #251 Stage B: ONE in-engine call does joint_decide + the #223 evict-cooldown
        # filter + the #240 saturation yield — the post-decision logic that used to be
        # inline here.  Same call, same filters, no transport (the daemon still does the
        # fetch above and the dispatch below).  `evict_cooldown` is mutated in place
        # (expired-entry prune) exactly as before.
        plan = self.driver.decide(
            sched_state, event,
            costs=self.policy.costs,
            pi_u=self.policy.pi_u,
            theta_hi=router.theta_hi,
            theta_lo=router.theta_lo,
            heartbeat_s=router.heartbeat_s,
            admission_enabled=self.admission_enabled,
            evict_cooldown=getattr(router, "evict_cooldown", None),
        )
        self.decisions += 1
        self.last_plan = plan
        # #215: reconcile in-flight resumes against the fresh dump — runs EVERY
        # event (incl. the empty-plan ones below) so a clear that lands (or is
        # lost) is observed promptly.  The tracker drops a pid once the dump no
        # longer shows it PAUSED (clear landed) or after the recovery window
        # (clear lost → re-fire).
        _paused_now = {pid for pid, pu in sched_state.per_program_usage.items()
                       if pu.get("state") == "PAUSED"}
        self.tracker.reconcile_resume_acks(_paused_now, _RESUME_DEDUP_WINDOW)
        if not plan:
            # Nothing to do this event — declined (every V_u-positive
            # alternative loses) or the hysteresis dead-zone (§9).
            _m(
                "kv_decide",
                kind=event.kind.value,
                dset_size=self.last_decision_set_size,
                outcome="policy_declined",
                eta=event.payload.get("tool_eta_s"),
                cmd=str((event.payload.get("tool_args") or {}).get("command", ""))[:16],
            )
            return
        await self._dispatch_plan(plan, sched_state)

    def _update_demote_apply_rate(self, units: Dict[str, Any]) -> None:
        """#240 / #251 Stage B: delegate to the in-engine driver's apply-rate EMA
        (the logic moved verbatim into AginferDriver.update_demote_apply_rate)."""
        self.driver.update_demote_apply_rate(units)

    async def _dispatch_plan(self, plan: List[Any], sched_state) -> None:
        """Dispatch a §9 ``joint_decide`` mixed plan (#194).

        Splits the chosen candidates by lever type and routes each:
          * ``Migrate`` → the residence-set POST (same wire as before;
            ``c.id`` is the ``(uid, add_tiers, remove_tiers)`` tuple
            ``assignments_to_wire`` consumes).
          * ``Pause``   → ``tracker.pause(pid)`` (gates the proxy) AND
            ``PUT /aginfer/program_paused`` so sglang stores p's
            ``pre_pause_state`` (the §8 resume counterfactual reads it).
            **The Pause lever is DORMANT (DESIGN §9): ``joint_decide`` never
            emits a Pause today, so this branch is retained for when the
            lever is enabled but is not exercised by any current plan.**
          * ``Resume``  → ``tracker.resume(pid)`` (releases the gate) AND
            a ``PUT`` clearing the paused mark back to ``pre_pause_state``.
        """
        from baselines.knapsack import Migrate, Pause, Resume
        from ._metrics import m as _m
        migrates = [c for c in plan if isinstance(c, Migrate)]
        pauses = [c for c in plan if isinstance(c, Pause)]
        resumes = [c for c in plan if isinstance(c, Resume)]
        # S2 isolation knob: suppress the migrate/demote dispatch while keeping the
        # hint push (so the eviction scorer still gets n_holders) and pause/resume.
        # Under heavy churn the pressure-relief demote is net-counterproductive for
        # shared-prefix retention (it demotes into a full/flaky DRAM tier AND frees
        # HBM the bg flood immediately reclaims), so this isolates the pure
        # value-aware EVICTION lever from the relief lever. Off by default.
        if migrates and os.environ.get("AGINFER_DISABLE_MIGRATE"):
            migrates = []
        _m(
            "kv_decide",
            kind=sched_state.event_kind,
            dset_size=self.last_decision_set_size,
            migrates=len(migrates),
            pauses=len(pauses),
            resumes=len(resumes),
            outcome="dispatched",
        )
        if migrates:
            # #240 / #251 Stage B: remember remove-HBM hashes so the next fresh dump
            # tells us whether they landed (feeds the in-engine driver's apply-rate EMA).
            for m in migrates:
                _mid = getattr(m, "id", None)
                if isinstance(_mid, tuple) and len(_mid) >= 3 and Tier.HBM in _mid[2]:
                    self.driver.note_demote_dispatched(_mid[0])
            await self._dispatch_migrate([m.id for m in migrates])
        for c in pauses:
            await self._dispatch_pause(c.pid, sched_state)
        # #215 resume double-fire dedup: skip a resume the daemon has already
        # issued whose clear the dump hasn't reflected yet (overlay lag).  The
        # tracker owns this — it is reconciled against the fresh dump in
        # handle() and re-arms on a lost clear.  _dispatch_resume → tracker.
        # resume notes the issue; we mark it explicitly to be robust if that
        # call path changes.
        for c in resumes:
            if self.tracker.resume_in_flight(c.pid):
                continue  # already issued; dump just lags — don't re-fire
            await self._dispatch_resume(c.pid, sched_state)
            self.tracker.note_resume_issued(c.pid)

    async def _dispatch_pause(self, pid: str, sched_state) -> None:
        """Pause program ``pid``: gate the proxy + notify sglang with the
        pre-pause state (DESIGN §6 / §8)."""
        prior = sched_state.per_program_usage.get(pid, {}).get("state")
        self.tracker.pause(pid)
        self.pause_calls += 1
        if self.outbound is not None:
            self.outbound.enqueue_program_paused(
                pid=pid, state="PAUSED", pre_pause_state=prior)
        from ._metrics import m as _m
        _m("admission_pause", pid=pid, pre_pause_state=prior)

    async def _dispatch_resume(self, pid: str, sched_state) -> None:
        """Resume program ``pid``: release the gate + clear the sglang
        paused mark back to its pre-pause state (DESIGN §6 / §8)."""
        pre = sched_state.per_program_usage.get(pid, {}).get("pre_pause_state")
        self.tracker.resume(pid)
        self.resume_calls += 1
        if self.outbound is not None:
            # Restore the program to whatever it was doing pre-pause; the
            # paused mark is cleared (pre_pause_state=None on the resumed
            # record).
            self.outbound.enqueue_program_paused(
                pid=pid, state=pre or "REASONING", pre_pause_state=None)
        from ._metrics import m as _m
        _m("admission_resume", pid=pid, restored_state=pre)

    async def _dispatch_migrate(
        self, assignments: List[Tuple[str, List[Tier], List[Tier]]]
    ) -> None:
        """T36 (DESIGN §6 B4) fire-and-forget enqueue.

        Returns immediately after a ``put_nowait`` + ``uuid4()``.
        The shared ``OutboundQueue`` worker (started by main.py) pops
        + POSTs in the background; per-item failures flow back via
        the ``APPLY_FAILED`` webhook (T23+T37), not via this call's
        return path.

        ``outbound`` is REQUIRED — there is no longer a synchronous
        fallback (removed post-T36 audit).  Constructing a
        ``KvScheduler`` without ``outbound=...`` is a wiring bug that
        we surface here rather than silently dropping every migrate.
        """
        if self.outbound is None:
            raise RuntimeError(
                "KvScheduler._dispatch_migrate called without an "
                "OutboundQueue injected.  main.py wires this; tests "
                "constructing KvScheduler directly must pass "
                "outbound=OutboundQueue(...) per DESIGN §6 B4."
            )
        actions_wire = assignments_to_wire(assignments)
        batch_id = self.outbound.enqueue_migrate(actions_wire)
        self.migrate_calls += 1
        from ._metrics import m as _m
        _m(
            "migrate_enqueued",
            batch_id=batch_id,
            n_actions=len(assignments),
        )

    async def _dispatch_hints(self, hints: List[Dict[str, Any]]) -> None:
        """T40 (#184, DESIGN §6 F2) fire-and-forget hint push.

        Mirrors ``_dispatch_migrate``: ``put_nowait`` + ``uuid4`` then
        return; the shared OutboundQueue worker PUTs in the background.
        ``outbound`` is REQUIRED (same wiring contract as migrate).
        An empty hint list is a no-op (no point PUTting nothing).
        """
        if not hints:
            return
        if self.outbound is None:
            raise RuntimeError(
                "KvScheduler._dispatch_hints called without an "
                "OutboundQueue injected.  main.py wires this; tests "
                "constructing KvScheduler directly must pass "
                "outbound=OutboundQueue(...)."
            )
        from ._metrics import m as _m
        # #230: defer hint DELIVERY (computed-now, delivered-stale) for the
        # latency-budget arms.  The list is already shaped; call_later just
        # enqueues it later so sglang sees it ``_hint_delay_s`` late.
        if self._hint_delay_s > 0.0:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.call_later(self._hint_delay_s,
                            self.outbound.enqueue_hints, list(hints))
            self.hint_calls += 1
            self.hint_delayed_calls += 1
            _m("hints_enqueued", batch_id="deferred",
               n_hints=len(hints), delay_ms=int(self._hint_delay_s * 1000))
            return
        batch_id = self.outbound.enqueue_hints(hints)
        self.hint_calls += 1
        _m(
            "hints_enqueued",
            batch_id=batch_id,
            n_hints=len(hints),
        )

    # ------------------------------------------------- belief plane (§3)

    def _apply_belief_transition(self, event: Event) -> None:
        """DESIGN §3: update the program-lifecycle tracker from the event kind.

        Mirrors the proxy's observe_arrival/observe_completion semantics so the
        belief plane is driven by EVENTS (not only the proxy's request
        observation): a request-side event puts the program in REASONING; a
        TOOL_CALL_START (response done → agent acting) flips REASONING→ACTING.
        SESSION_END is owned by the SESSION_END handler (tracker.end).  Pressure
        webhooks carry no session → skipped.  All transitions are idempotent."""
        pid = event.session
        if not pid:
            return
        k = event.kind
        if k in (EventKind.SESSION_ARRIVAL, EventKind.LLM_PREFILL,
                 EventKind.TOOL_CALL_END):
            self.tracker.observe_arrival(pid)      # → REASONING
        elif k == EventKind.TOOL_CALL_START:
            self.tracker.observe_completion(pid)   # REASONING → ACTING

    # ------------------------------------------------- action-timeline (§3/§7)

    def _schedule_promote_back(self, event: Event, sched_state, router) -> None:
        """DESIGN §3/§7: place a predictive promote-back of program
        ``event.session``'s idle tail on the router's action-timeline, due at
        ``T_start + tool_ETA − load_back − margin`` so the tail is HBM-resident
        when the resume prefill arrives.  Belief-validated at fire
        (``fire_due_action``).  No-op when the timeline isn't wired, the event
        carries no usable ETA (degrade to promote-at-TOOL_CALL_END), or the
        program has no exclusive tail.

        ``T_start`` is the event's ``enqueue_time`` (the event-stream clock,
        §10 — no wall-clock call here).  ``tool_ETA`` is read from the event
        payload (``tool_eta_s``); the proxy/replay carries the typed tool's
        expected duration."""
        tl = getattr(router, "timeline", None)
        if tl is None:
            return
        session = event.session
        if not session:
            return
        eta = event.payload.get("tool_eta_s")
        if eta is None:
            return  # no reliable ETA → reactive promote-at-END (DESIGN §7)
        try:
            eta = float(eta)
        except (TypeError, ValueError):
            return
        if eta <= 0.0:
            return
        # Caller's EXCLUSIVE tail — the same units §4's table makes the
        # demote candidates for TOOL_CALL_START.
        tail = _units_for_session(sched_state.units, session)
        if not tail:
            return
        total_bytes = sum(sched_state.units[uid].n_bytes
                          for uid in tail if uid in sched_state.units)
        load_back_s = _estimate_load_back_s(sched_state, total_bytes)
        # #241: the promote runs as a prefill-only WARM (#238), whose completion
        # is prefill-class (queue+compute+load), not transfer-class.  Use
        # _WARM_LEAD_S as a FLOOR on the lead so the warm lands before the resume.
        eff_lead_cost = max(load_back_s, _WARM_LEAD_S)
        now = event.enqueue_time  # event-stream clock (perf_counter frame)
        lead = max(0.0, eta - eff_lead_cost - _PROMOTE_SAFETY_MARGIN_S)
        due = now + lead
        from_tiers = tuple({t for uid in tail if uid in sched_state.units
                            for t in sched_state.units[uid].residence})
        tl.schedule(due, PromoteAction(
            session=session, unit_hashes=tuple(tail), from_tiers=from_tiers,
            eta_s=eta, load_back_s=load_back_s, scheduled_at=now,
            reason="tool_eta"))
        self.promotes_scheduled += 1
        from ._metrics import m as _m
        _m("promote_scheduled", session=session, n_units=len(tail),
           eta_s=round(eta, 3), load_back_s=round(load_back_s, 4),
           lead_s=round(lead, 4), bytes=int(total_bytes))

    async def fire_due_action(self, payload, router) -> None:  # noqa: ANN001
        """DESIGN §3 belief-validated fire of a due action-timeline action.

        Re-reads belief (program lifecycle + a fresh ``/aginfer/state``
        residence snapshot) and turns a stale promote into an idempotent no-op
        (§10): the program already returned (no longer ACTING), ended, the tail
        was never demoted (already HBM), or the tail was dropped (nothing to
        promote).  Only genuinely-demoted units of a still-tool-bound program
        get a Migrate(→HBM).  Registered as ``router.due_action_handler`` by
        main.py and invoked by the event_worker under the dispatch lock."""
        if not isinstance(payload, PromoteAction):
            return
        from ._metrics import m as _m
        session = payload.session
        # Belief 1 — still tool-bound?  ACTING is the window between
        # TOOL_CALL_START and the program's next LLM event (program_tracker).
        # REASONING (already resumed) / PAUSED / ENDED / unknown → stale.
        st = self.tracker.state(session)
        if st != State.ACTING:
            self.promotes_skipped_stale += 1
            _m("promote_skipped", session=session,
               reason=f"state={st.value if st else None}")
            return
        # Belief 2 — fresh residence.  A promote only helps a tail currently in
        # DRAM/DISK; one already in HBM (never demoted) or dropped is a no-op.
        try:
            state_json = await router.fetch_state()
            sched_state = build_paper_state(
                state_json,
                event=Event(kind=EventKind.LLM_PREFILL, session=session),
                tracker=self.tracker,
                lambda_acting=self.lambda_acting,
                unknown_tier_log=self._unknown_tier_log,
            )
        except Exception:  # noqa: BLE001
            logger.exception("promote fire: state fetch/build failed; skip")
            self.promotes_skipped_stale += 1
            _m("promote_skipped", session=session, reason="state_fetch_failed")
            return
        assignments: List[Tuple[str, List[Tier], List[Tier]]] = []
        all_hbm = True
        for uid in payload.unit_hashes:
            u = sched_state.units.get(uid)
            if u is None:
                # Gone from the tree → DISK-evicted / dropped.  Unreachable by the
                # node-based migrate plane; needs the warm (#238).
                all_hbm = False
                continue
            res = set(u.residence)
            if Tier.HBM in res:
                continue  # already HBM (never demoted / already promoted)
            all_hbm = False
            if res & {Tier.DRAM, Tier.DISK}:
                # DESIGN §7 ``[] → {HBM}`` = load_back; KEEP the lower-tier backup.
                assignments.append((uid, [Tier.HBM], []))
        if all_hbm:
            # Whole tail already HBM-resident (never demoted) → idempotent no-op.
            self.promotes_skipped_stale += 1
            _m("promote_skipped", session=session, reason="already_hbm")
            return
        # #238: PREFER the warm (prefill-only, max_new_tokens=0) when the session's
        # prefix is registered — it uniformly stages DRAM/DISK/dropped back to HBM
        # and is the ONLY mechanism that can reach a prefix that has LEFT the radix
        # tree (DISK-evicted).  The per-unit migrate (DRAM load_back) is the
        # fallback when no prefix is registered (e.g. tests, or DRAM-only tails).
        prefix_tokens = getattr(router, "_session_prefix", {}).get(session)
        if prefix_tokens:
            ok = await router.warm_to_hbm(prefix_tokens, session)
            if ok:
                self.promotes += 1
                _m("promote_dispatched", session=session, via="warm",
                   n_tokens=len(prefix_tokens), eta_s=round(payload.eta_s, 3))
                return
            # warm dispatch failed → fall through to the migrate fallback
        if not assignments:
            self.promotes_skipped_stale += 1
            _m("promote_skipped", session=session,
               reason="no_prefix_and_no_dram_units")
            return
        await self._dispatch_migrate(assignments)
        self.promotes += len(assignments)
        _m("promote_dispatched", session=session, via="migrate",
           n=len(assignments), eta_s=round(payload.eta_s, 3))


# ----------------------------------------------------------------- attach


def attach_kv_scheduler(router, scheduler: KvScheduler) -> None:  # noqa: ANN001
    """Register the KvScheduler handler on every paper §4 event kind.

    The EventRouter's per-kind handler registry maps to a single
    method on the KvScheduler instance.  T8 (admission_controller)
    will compose ON TOP by wrapping `MEMORY_PRESSURE` and
    `PRESSURE_RESOLVED` handlers — but kv_scheduler still owns the
    migrate-half of the response.
    """
    for kind in EventKind:
        router.set_handler(kind, scheduler.handle)
