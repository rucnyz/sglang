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
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

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

from .events import Event, EventKind
from .program_tracker import ProgramTracker, State

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- calibration

# λ for a unit owned by a program in ACTING state.  Default 1/5: mean
# tool call on terminus-2's swebenchpro is ~5 s (range 1–30).  Clamped
# to [1/30, 1/1] per audit #15 — see verify/t7/README.md WORST CASE.
_LAMBDA_ACTING_FLOOR = 1.0 / 30.0
_LAMBDA_ACTING_CEIL = 1.0 / 1.0


def _env_float(key: str, default: str) -> float:
    """Parse a float env var with a CLEAR error message on malformed
    values — bare ``float(os.environ[k])`` raises a vague
    ``ValueError: could not convert string to float`` that doesn't
    mention the env var name, which would silently break the daemon
    at module import."""
    raw = os.environ.get(key, default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"env var {key}={raw!r} is not a valid float: {exc}"
        ) from exc


def _env_int(key: str, default: str) -> int:
    raw = os.environ.get(key, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"env var {key}={raw!r} is not a valid int: {exc}"
        ) from exc


_DEFAULT_LAMBDA_ACTING = _env_float("AGINFER_LAMBDA_ACTING", "0.2")

# Top-k cap on the memory_pressure decision_set.  Paper §7.1.  256 is
# enough to materially affect HBM occ on a B300 (~half a percent per
# unit at 2 KB/token × 4 k tokens/unit).
_DEFAULT_MEMORY_PRESSURE_TOPK = _env_int("AGINFER_MEMORY_PRESSURE_TOPK", "256")


def _clamp_lambda_acting(lam: float) -> float:
    return max(_LAMBDA_ACTING_FLOOR, min(_LAMBDA_ACTING_CEIL, lam))


# ----------------------------------------------------------------- adapter


_TIER_LABEL_MAP: Dict[str, Tier] = {
    "HBM": Tier.HBM,
    "DEVICE": Tier.HBM,
    "DRAM": Tier.DRAM,
    "HOST": Tier.DRAM,
    "DISK": Tier.DISK,
    "DROP": Tier.DROP,
}


def _tier_from_string(label: str) -> Optional[Tier]:
    """sglang dumps tier names as strings; map to the enum.

    Audit round-1 B1: previously fell back to ``Tier.HBM`` for any
    unrecognised label, which silently mis-classified e.g. a
    ``"ZSTD_DISK"`` unit as HBM-resident — downstream
    ``_top_k_by_regret`` would then nominate it as an HBM demote
    candidate, and ``OursGreedyPolicy.decide()`` would score it
    against HBM cost / occupancy.  The misclassification was
    invisible (no log).  Now: return ``None`` on unknown so callers
    can skip the unit and log a single warning per label.
    """
    return _TIER_LABEL_MAP.get(label.upper())


def _flatten_per_rank(state_json: Dict[str, Any]) -> Dict[str, Any]:
    """Audit round-1 B2: aggregate a multi-rank state payload.

    sglang multi-DP emits ``{"per_rank": [<per-rank dict>, ...]}``.
    The per-rank dicts each have their own ``tier_usage`` and
    ``units``; the daemon needs one global view to make a coherent
    scheduling decision (units that exist on rank R can only be
    migrated by rank R, but the policy needs to see the GLOBAL HBM
    occupancy when deciding which rank's units to demote).

    Aggregation rule:
      * tier_usage: sum used_bytes and cap_bytes across ranks (HBM is
        per-rank; total HBM is the sum).
      * units: concatenate verbatim — **no hash prefix**.  Audit
        round-2 R2-B1: previous version prefixed with ``rN/`` so the
        daemon could "route back" to the right rank, but sglang's
        ``POST /aginfer/migrate`` does an EXACT ``hash_to_node.get(h)``
        lookup; a prefix landed every action in ``skipped`` and the
        daemon never knew (200 + empty ``applied_hashes``).  sglang's
        hashes are globally unique (hex SHA256 or ``node-<id>``); if
        two ranks emit the SAME hash it's the same logical unit
        (replicated prefix) and we dedupe via ``seen_hashes`` —
        broadcasting one migrate to both ranks is the correct action.
      * time_counter: max across ranks (clocks may differ).
    Single-rank shape (no ``per_rank``) is returned unchanged.
    """
    if "per_rank" not in state_json:
        return state_json
    per_rank = state_json["per_rank"]
    agg_tu: Dict[str, Dict[str, int]] = {
        "HBM": {"used_bytes": 0, "cap_bytes": 0},
        "DRAM": {"used_bytes": 0, "cap_bytes": 0},
        "DISK": {"used_bytes": 0, "cap_bytes": 0},
    }
    # G10 fix: aggregate allocator-level pool_usage across ranks.
    # token_usage is recomputed at the end from summed used/cap; the
    # max-of-sub-pools convention (full vs swa) is preserved per rank
    # then SUMMED across ranks (a rank under pressure raises the max).
    agg_pool: Dict[str, Dict[str, int]] = {
        "HBM": {"used_bytes": 0, "cap_bytes": 0,
                "available_bytes": 0, "evictable_bytes": 0},
    }
    pool_present = False
    # Track each hash's index in agg_units so we can update on tier
    # disagreement (audit round-5 MINOR: mid-migration race where
    # rank-0 still reports HBM while rank-1 already reports DRAM —
    # prefer the COLDER tier under the assumption the migration is
    # in-progress and the unit is on its way out of hot tiers).
    agg_units: List[Dict[str, Any]] = []
    hash_to_idx: Dict[str, int] = {}
    agg_time = 0
    # Tier ordering (colder = lower value): DROP < DISK < DRAM < HBM.
    _tier_rank = {"HBM": 3, "DEVICE": 3, "DRAM": 2, "HOST": 2, "DISK": 1, "DROP": 0}
    for rank in per_rank:
        rank_tu = rank["tier_usage"]
        for label in agg_tu:
            sub = rank_tu[label]
            agg_tu[label]["used_bytes"] += int(sub["used_bytes"])
            agg_tu[label]["cap_bytes"] += int(sub["cap_bytes"])
        rank_pool = rank.get("pool_usage")
        if rank_pool and "HBM" in rank_pool:
            pool_present = True
            sub = rank_pool["HBM"]
            agg_pool["HBM"]["used_bytes"] += int(sub.get("used_bytes", 0))
            agg_pool["HBM"]["cap_bytes"] += int(sub.get("cap_bytes", 0))
            agg_pool["HBM"]["available_bytes"] += int(sub.get("available_bytes", 0))
            agg_pool["HBM"]["evictable_bytes"] += int(sub.get("evictable_bytes", 0))
        for u in rank["units"]:
            uhash = str(u["hash"])
            if uhash in hash_to_idx:
                # Replicated-prefix or in-flight migration: same hash
                # on multiple ranks.  If tiers disagree, keep the
                # COLDER one (assume migration is in progress).
                existing = agg_units[hash_to_idx[uhash]]
                new_rank = _tier_rank.get(str(u["tier"]).upper(), 99)
                old_rank = _tier_rank.get(str(existing["tier"]).upper(), 99)
                if new_rank < old_rank:
                    agg_units[hash_to_idx[uhash]] = dict(u)
                continue
            hash_to_idx[uhash] = len(agg_units)
            agg_units.append(dict(u))
        agg_time = max(agg_time, int(rank["time_counter"]))
    out: Dict[str, Any] = {
        "tier_usage": agg_tu,
        "units": agg_units,
        "time_counter": agg_time,
    }
    if pool_present:
        hbm = agg_pool["HBM"]
        cap = hbm["cap_bytes"]
        hbm["token_usage"] = (hbm["used_bytes"] / cap) if cap > 0 else 0.0
        out["pool_usage"] = agg_pool
    return out


def _log_unknown_tier_once(label: str, seen: set) -> None:
    """Log each unknown tier label exactly once to avoid log spam on
    repeated state fetches.

    Audit round-2 R2-N2: ``seen`` is per-instance (lives on the
    KvScheduler) rather than module-global, so cross-test/restart
    state doesn't leak."""
    if label in seen:
        return
    seen.add(label)
    logger.warning(
        "kv_scheduler: unknown tier label %r in /aginfer/state; "
        "unit skipped.  Add to daemon/kv_scheduler.py _TIER_LABEL_MAP.",
        label,
    )


def build_paper_state(
    state_json: Dict[str, Any],
    *,
    event: Event,
    tracker: ProgramTracker,
    unknown_tier_log: set,
    lambda_acting: float = _DEFAULT_LAMBDA_ACTING,
) -> SchedulerState:
    """Convert sglang ``/aginfer/state`` JSON → ``SchedulerState``.

    paper §3's reduced state: per-unit (tier, age, p_hat, λ), per-tier
    usage, and the event's decision_set.  ``λ`` for ACTING/PAUSED
    programs is the calibrated floor; for REASONING programs we keep
    the inline scorer's ``hits/age`` proxy so two scoring paths agree.

    Audit round-3.5: removed dead null-defense (``or {}`` / ``or []``
    / ``or 0`` patterns) on fields sglang's dump_aginfer_state always
    emits with proper types.  A missing/null field is now a contract
    violation — handle()'s try/except logs the full traceback so ops
    see exactly which field broke.  Also removed the
    ``raw.get("tier", "HBM")`` default, which was a B1-class bug
    (missing tier → silent HBM misclassification).

    sglang's emission contract:
      * tier_usage: dict with HBM / DRAM / DISK sub-dicts
        (each has used_bytes + cap_bytes, both int)
      * units: list of dicts; each has hash, tier, n_tokens, n_bytes,
        last_access_time, hit_count, session_ids
      * time_counter: int
    """
    # Audit round-5 MAJOR: scheduler emits ``unsupported_tree_cache``
    # marker when the tree cache doesn't expose ``dump_aginfer_state``
    # (e.g. a non-Unified RadixCache).  Daemon used to silently see
    # all-zero state and make no decisions — log a one-shot warning
    # per tree-cache class name so ops can grep for the misconfig.
    marker = state_json.get("unsupported_tree_cache")
    if marker:
        _log_unknown_tier_once(
            f"unsupported_tree_cache:{marker}", unknown_tier_log
        )

    # Audit round-1 B2: multi-rank sglang (TP × DP > 1) emits
    # ``{"per_rank": [...]}``; aggregate into a single global view.
    state_json = _flatten_per_rank(state_json)

    tier_usage = TierUsage(capacity_bytes={}, used_bytes={})
    raw_tu = state_json["tier_usage"]
    for label, tier in (("HBM", Tier.HBM), ("DRAM", Tier.DRAM), ("DISK", Tier.DISK)):
        sub = raw_tu[label]
        tier_usage.capacity_bytes[tier] = int(sub["cap_bytes"])
        tier_usage.used_bytes[tier] = int(sub["used_bytes"])
    # bw_free defaults: use the full per-pair BW from costs config.
    # Production T8 / measurement layer can plumb live bw_free in here.
    tier_usage.bw_free = dict(default_costs().bw)

    # G10 fix: extract allocator-truth pool_pressure for admission gating.
    # Missing key (older sglang) is acceptable — pool_pressure stays empty
    # and admission falls back to tier_usage.occupancy_ratio (the old,
    # always-~0 behavior).  Log once if seen so ops notice the rollback.
    pool_pressure: Dict[Tier, float] = {}
    raw_pool = state_json.get("pool_usage")
    if raw_pool:
        hbm_sub = raw_pool.get("HBM")
        if hbm_sub and "token_usage" in hbm_sub:
            pool_pressure[Tier.HBM] = float(hbm_sub["token_usage"])

    units_raw = state_json["units"]
    now_counter = int(state_json["time_counter"])

    units: Dict[str, ReuseUnit] = {}
    # Owner program → its ACTING-floor λ (cached per call).
    program_lambda: Dict[str, float] = {}
    for raw in units_raw:
        uhash = str(raw["hash"])
        if not uhash:
            continue
        n_tokens = int(raw["n_tokens"])
        n_bytes = int(raw["n_bytes"])
        # sglang's emit code hardcodes "tier" as a non-optional field
        # (`tier_lit` in the bytes-path / explicit dict key in the dict-
        # path).  Direct access — KeyError is sglang misbehaving.
        raw_tier_label = str(raw["tier"])
        tier = _tier_from_string(raw_tier_label)
        if tier is None:
            # Unrecognised label — skip + log once (forward-compat with
            # future tier labels we haven't taught the daemon about).
            _log_unknown_tier_once(raw_tier_label, unknown_tier_log)
            continue
        last_access = int(raw["last_access_time"])
        hits = int(raw["hit_count"])
        age = max(1, now_counter - last_access)
        # baseline λ from inline scorer (hits/age proxy).
        # `age` is `max(1, ...)` above, so it's always >= 1.
        # p_hat is computed below from the program-alive rule (T11a).
        lam = max(1e-3, hits / age)
        # Iterate holders to compute two reductions in one pass:
        #   * any_acting  → clamp λ to the ACTING floor (paper §7
        #     "expected reuse interval ~ tool duration" rule).
        #   * any_alive   → program-alive rule (T11a): if any holder's
        #     program is tracked-alive, p_hat = 1.0.  Rationale:
        #     conditional on the queryable predicate
        #     `ProgramTracker.state(sid) is not None`, monotonic-
        #     extension workloads have P(next-step reuse) = 1 within
        #     paper §7's 1-step horizon.  hits/age is unbiased only
        #     under uniform-Poisson, which multi-turn agent workloads
        #     violate; the proxy under-values young trunk units (low
        #     age, hits=1) by ~20× vs structural 1.0.  System-prompt
        #     high value emerges from T8's shared-aware aggregation
        #     across many alive holders, not from a separate rule.
        #     Known limitation: tracker has no ENDED state yet; an
        #     ended program stays state()!=None until manually
        #     forgotten.  Bounded over-estimation; v1 trade-off.
        session_ids = raw["session_ids"]
        any_acting = False
        any_alive = False
        for sid in session_ids:
            st = tracker.state(sid)
            if st is not None:
                any_alive = True
            if sid not in program_lambda:
                # Audit round-2 R2-M1: PAUSED programs are STILL
                # mid-tool-call (admission_controller pinned them).
                # Paper §7's "expected reuse interval ~ tool duration"
                # applies to them too.  Round-1 only fired the floor
                # for ACTING; PAUSED fell back to hits/age — the
                # OPPOSITE of paper intent (a high-hit prefix would
                # get high λ and be KEPT on HBM during the tool call,
                # which is exactly what admission gating is trying
                # to free up).
                program_lambda[sid] = (
                    _clamp_lambda_acting(lambda_acting)
                    if st in (State.ACTING, State.PAUSED)
                    else 0.0
                )
            if program_lambda[sid] > 0:
                any_acting = True
        if any_acting:
            lam = program_lambda[
                next(sid for sid in session_ids if program_lambda[sid] > 0)
            ]
        # Program-alive rule (see comment above).  Fall back to
        # hits/age proxy when ALL holders are unknown (orphan unit).
        if any_alive:
            p_hat = 1.0
        else:
            p_hat = min(1.0, hits / age)
        units[uhash] = ReuseUnit(
            id=uhash,
            type=UnitType.SESSION,  # platform / tool_def tags arrive later
            scope=Scope.SESSION,
            n_tokens=n_tokens,
            n_bytes=n_bytes,
            tier=tier,
            age_seconds=float(age),
            p_hat=p_hat,
            lambda_rate=lam,
            holders=list(session_ids),
        )

    decision_set = _build_decision_set(event, units, tracker)

    return SchedulerState(
        t=float(now_counter),
        units=units,
        tier_usage=tier_usage,
        event_kind=event.kind.value,
        event_session_id=event.session,
        decision_set=decision_set,
        pool_pressure=pool_pressure,
    )


# ----------------------------------------------------------------- D_t builders


def _units_for_session(
    units: Dict[str, ReuseUnit], session: Optional[str]
) -> List[str]:
    """Caller's exclusive tail — units held ONLY by this session.

    Per paper §4 table: TOOL_CALL_START / TOOL_CALL_END operate on the
    caller's tail (demote / promote), NOT the shared platform/tool_def
    prefix (which stays HBM because it's high-value to other programs).
    Concretely: a unit whose ``holders`` is exactly ``{session}``.
    """
    if session is None:
        return []
    # Audit round-2 R2-N1: previously had ``u.holders == [session] or
    # set(u.holders) == {session}``.  The set form already covered
    # everything (incl. duplicate-holder lists like ``[s, s]``); the
    # list form was redundant.  Use set semantics exclusively — it's
    # the paper meaning (a unit has a SET of holders).
    target = {session}
    return [uid for uid, u in units.items() if set(u.holders) == target]


def _shared_prefix_units(units: Dict[str, ReuseUnit]) -> List[str]:
    """Units held by >= 2 programs — the platform / tool_def candidates.

    v1 heuristic until T3's typed-unit metadata reaches the daemon.
    """
    return [uid for uid, u in units.items() if len(u.holders) >= 2]


def _top_k_by_regret(
    units: Dict[str, ReuseUnit],
    k: int,
    costs=default_costs(),
) -> List[str]:
    """Top-k units to evaluate on a memory_pressure event.

    Paper §7.1: the cheap regret proxy is "how much more it costs to
    refetch this unit from disk than it costs to hold".  We score
    each HBM-resident unit as ``p_hat * (R_drop − R_hbm) − holding``
    (the steady-state value of keeping it at HBM), then **sort
    ascending and return the first k** — i.e., the k units with the
    SMALLEST keep-value, which are the best demote candidates per
    paper §7.1.  A future maintainer should not "fix" this slice to
    ``items[-k:]``; that would invert the policy (keep the LEAST
    valuable, demote the most) — see verify/t7/regression_probe.py
    `probe_top_k_content` for the bisect demo.
    """
    if k <= 0 or not units:
        return []
    # Inline a lightweight V_u proxy here so we don't need a fully-built
    # SchedulerState yet.  Pure ordering; absolute values don't matter.
    rho_hbm = costs.rho[Tier.HBM]
    rho_disk = costs.rho[Tier.DISK]
    items: List[Tuple[float, str]] = []
    for uid, u in units.items():
        if u.tier != Tier.HBM:
            # v1: only HBM-resident units are demote candidates.  T10
            # extends this to also rank DRAM units once the daemon-
            # controlled DISK (L3 / Mooncake) tier is wired — paper §7.1
            # says regret should rank across all current-tier units.
            continue
        saved = u.p_hat * (rho_disk - rho_hbm) * u.n_tokens
        # Holding tax proxy (per unit time):
        hold = costs.h_base[Tier.HBM] * u.n_bytes
        score = saved - hold
        items.append((score, uid))
    items.sort()
    return [uid for _score, uid in items[:k]]


def _build_decision_set(
    event: Event,
    units: Dict[str, ReuseUnit],
    tracker: ProgramTracker,
) -> List[str]:
    """Paper §4 table → D_t for this event."""
    kind = event.kind
    session = event.session
    if kind == EventKind.SESSION_ARRIVAL:
        # Only shared (platform / tool_def / subagent_ctx) prefix units
        # are candidates: pull them into HBM ahead of the first prefill.
        return _shared_prefix_units(units)
    if kind in (EventKind.LLM_PREFILL,):
        # Per paper §4 LLM_PREFILL is informational (state observation
        # only); no migrate decision unless watermarks fire separately.
        return []
    if kind == EventKind.TOOL_CALL_START:
        # Caller's session tail is a demote candidate while in tool call.
        return _units_for_session(units, session)
    if kind == EventKind.TOOL_CALL_END:
        # Caller's session tail is a promote candidate (about to reuse).
        return _units_for_session(units, session)
    if kind == EventKind.SUB_DISPATCH_BLOCKING:
        # Parent tail demoted; shared platform / tool_def stays HBM.
        return _units_for_session(units, session) + _shared_prefix_units(units)
    if kind == EventKind.SUB_DISPATCH_ASYNC:
        # Only shared platform / tool_def — child's tail isn't visible
        # to the daemon's state snapshot yet.
        return _shared_prefix_units(units)
    if kind in (EventKind.MEMORY_PRESSURE, EventKind.PRESSURE_RESOLVED):
        # Top-k by regret (paper §7.1).
        return _top_k_by_regret(units, _DEFAULT_MEMORY_PRESSURE_TOPK)
    return []


# ----------------------------------------------------------------- dispatch


def _tier_to_wire(tier: Tier) -> str:
    return {
        Tier.HBM: "HBM",
        Tier.DRAM: "DRAM",
        Tier.DISK: "DISK",
        Tier.DROP: "DROP",
    }[tier]


def assignments_to_wire(
    assignments: Iterable[Tuple[str, Tier]],
) -> List[Dict[str, str]]:
    """Translate ``[(unit_hash, tier), ...]`` → migrate JSON body."""
    return [
        {"hash": uhash, "target_tier": _tier_to_wire(tier)}
        for uhash, tier in assignments
    ]


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
        http_client: Optional[httpx.AsyncClient] = None,
        policy: Optional[OursGreedyPolicy] = None,
        lambda_acting: float = _DEFAULT_LAMBDA_ACTING,
    ) -> None:
        self.tracker = tracker
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.policy = policy or OursGreedyPolicy(default_costs())
        self.lambda_acting = lambda_acting
        # Telemetry for tests.
        self.decisions: int = 0
        self.migrate_calls: int = 0
        self.last_action: Optional[Action] = None
        self.last_decision_set_size: int = 0
        # Audit round-2 R2-N2: per-instance unknown-tier log set so
        # cross-test / cross-restart state doesn't leak.
        self._unknown_tier_log: set = set()

    async def ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def handle(self, event: Event, router) -> None:  # noqa: ANN001
        """Single entry point for all paper §4 events.

        Always refetches state at entry (per design contract), builds
        D_t, runs ``decide()``, POSTs any migrate actions.  Errors
        downstream of the policy do NOT propagate — paper §9 promises
        the inline scorer is a safety net.
        """
        try:
            state_json = await router.fetch_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kv_scheduler: /aginfer/state fetch failed for %s: %r",
                event.kind.value, exc,
            )
            from ._metrics import m as _m
            _m("state_fetch_failed", kind=event.kind.value)
            return
        # Emit HBM/DRAM occupancy snapshot for every handled event.
        # (Bounded ~10k–20k lines per cycle; ~16 ms wall.)  This is the
        # raw data behind T9 G5 (HBM occ trajectory).
        from ._metrics import m as _m
        try:
            flat = _flatten_per_rank(state_json)
            tu = flat.get("tier_usage") or {}
            hbm = tu.get("HBM", {})
            dram = tu.get("DRAM", {})
            # G10: tree_occ_hbm = radix-tree view (matches tier_usage
            # field used by V_u migration scoring; near-zero under
            # in-flight decode pressure).  pool_occ_hbm = allocator
            # truth (matches sglang's `full_token_usage`, what
            # admission gates on).  Emit BOTH so the trajectory parse
            # can see the divergence in the cycle log.
            tree_occ_hbm = (
                int(hbm.get("used_bytes", 0)) / max(int(hbm.get("cap_bytes", 1)), 1)
            )
            pool_occ_hbm = tree_occ_hbm
            pool_hbm = (flat.get("pool_usage") or {}).get("HBM") or {}
            if "token_usage" in pool_hbm:
                pool_occ_hbm = float(pool_hbm["token_usage"])
            occ_dram = (
                int(dram.get("used_bytes", 0)) / max(int(dram.get("cap_bytes", 1)), 1)
            )
            n_units = len(state_json.get("units") or [])
            _m(
                "state_fetched",
                kind=event.kind.value,
                occ_hbm=pool_occ_hbm,         # AUTHORITATIVE pressure
                tree_occ_hbm=tree_occ_hbm,    # radix view for debug
                occ_dram=occ_dram,
                units=n_units,
            )
        except Exception:  # noqa: BLE001
            pass  # metric emission must not break the worker
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
        self.last_decision_set_size = len(sched_state.decision_set)
        if not sched_state.decision_set:
            # Nothing to decide on (e.g. LLM_PREFILL or empty top-k).
            _m(
                "kv_decide",
                kind=event.kind.value,
                dset_size=0,
                outcome="empty_decision_set",
            )
            return
        action = self.policy.decide(sched_state)
        self.decisions += 1
        self.last_action = action
        if not action.assignments:
            # Policy declined to migrate — paper §7 says this happens
            # whenever Vt is non-positive for every alternative tier.
            _m(
                "kv_decide",
                kind=event.kind.value,
                dset_size=self.last_decision_set_size,
                outcome="policy_declined",
            )
            return
        _m(
            "kv_decide",
            kind=event.kind.value,
            dset_size=self.last_decision_set_size,
            action_n=len(action.assignments),
            outcome="dispatched",
        )
        await self._dispatch_migrate(action.assignments)

    async def _dispatch_migrate(
        self, assignments: List[Tuple[str, Tier]]
    ) -> None:
        body = {"actions": assignments_to_wire(assignments)}
        client = await self.ensure_client()
        url = f"{self.sglang_base_url}/aginfer/migrate"
        from ._metrics import m as _m
        try:
            r = await client.post(url, json=body)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kv_scheduler: migrate POST raised: %s", exc
            )
            _m("migrate_post", status="exception", n_actions=len(assignments))
            return
        self.migrate_calls += 1
        if r.status_code >= 400:
            logger.warning(
                "kv_scheduler: migrate returned %d: %s",
                r.status_code, r.text[:200],
            )
            _m(
                "migrate_post",
                status=r.status_code,
                n_actions=len(assignments),
                applied=0,
            )
            return
        # Parse sglang's response for applied vs skipped counts.
        # Contract: /aginfer/migrate ALWAYS returns JSON with `applied: int`
        # and `skipped: list`.  Tight except so a real protocol break
        # surfaces in the log instead of being silently coerced to -1.
        try:
            resp = r.json()
            applied = int(resp["applied"])
            skipped_list = resp["skipped"]
            skipped = len(skipped_list)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "kv_scheduler: migrate response malformed (%s): body=%s",
                type(exc).__name__, r.text[:300],
            )
            applied = -1
            skipped = -1
            skipped_list = []
        _m(
            "migrate_post",
            status=r.status_code,
            n_actions=len(assignments),
            applied=applied,
            skipped=skipped,
        )
        # Per-action skip reason breakdown — high-value for finding
        # G11-class issues (race:already_on_dram / promote_not_yet_wired
        # / race:not_in_tree / etc.).  Each line ≈ 1 µs, fires at most
        # `skipped` times per POST.
        for entry in skipped_list:
            if isinstance(entry, dict):
                reason = entry.get("reason", "?")
                # Replace spaces with _ since metric format is space-sep.
                _m(
                    "migrate_skipped",
                    reason=str(reason).replace(" ", "_")[:120],
                )


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
