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

from ._fatal import fatal
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
    """Aggregate a multi-rank DESIGN §5 state payload.

    sglang multi-DP emits ``{"per_rank": [<per-rank dict>, ...]}``.
    The daemon needs one global view to make a coherent scheduling
    decision (units that exist on rank R can only be migrated by
    rank R, but the policy must see the GLOBAL HBM occupancy when
    deciding which rank's units to demote).

    Per-field aggregation rule (DESIGN §6 multi-rank table):
      * pool_usage[tier].subpools[sp].{used,cap,available,evictable}_bytes:
        SUM across ranks (each rank has its own KV pool).  page_bytes
        is identical across ranks (static deployment config); take
        rank-0.  Subpool key set must be identical across ranks
        (sglang's tree_components are TP-shared); ASSERT on mismatch.
      * link_stats[link].peak_bw_bps: SUM across ranks (DESIGN §6
        line 731: each rank has its OWN PCIe link / NVMe queue;
        aggregate peak scales with rank count).
      * link_stats[link].recent_throughput_bps: SUM across ranks
        (same reasoning — independent per-rank links).
      * link_stats[link].time_since_last_sample_s: MAX across ranks
        (the LONGEST idle window; admission's bw_free branch keys on
        the worst-case link).
      * tier_holding_cost[tier][sp].h_max_per_byte_sec: identical
        across ranks (static deployment); take rank-0.
      * throughput_ema.prefill_bps: SUM across ranks (each rank
        processes its prefill slice; aggregate is the system rate).
      * throughput_ema.decode_per_program[pid]: SUM across ranks
        (a program may have decode work on multiple ranks via TP).
      * per_program_usage[pid].hbm.committed[sp] (and dram.committed):
        SUM across ranks; state takes the most-active (REASONING >
        ACTING > PAUSED > ENDED) across ranks; unit_hashes is the
        union.  pre_pause_state mirrors state.
      * units: concatenate verbatim — **no hash prefix**.  sglang's
        hashes are globally unique (hex SHA256 or ``node-<id>``); if
        two ranks emit the SAME hash it's the same logical unit
        (replicated prefix) and we dedupe — broadcasting one migrate
        to both ranks is the correct action.  On residence disagree-
        ment between ranks we keep the COLDER union (= the unit's
        bytes are persisted somewhere even if mid-migration on one
        rank).
      * time_counter: MAX across ranks (clocks may differ).

    Single-rank shape (no ``per_rank``) is returned unchanged.
    """
    if "per_rank" not in state_json:
        return state_json
    per_rank: List[Dict[str, Any]] = state_json["per_rank"]
    if not per_rank:
        # Deployment bug: sglang reported per_rank shape with empty list.
        # A correct multi-rank deployment dumps at least rank-0.
        fatal("per_rank_empty", state=state_json)

    # ---- pool_usage: per-(tier, subpool) sum across ranks ----
    rank0 = per_rank[0]
    agg_pool: Dict[str, Dict[str, Any]] = {}
    for tier in ("HBM", "DRAM", "DISK"):
        rank0_subpools = rank0["pool_usage"][tier]["subpools"]
        sp_keys = set(rank0_subpools.keys())
        agg_subpools: Dict[str, Dict[str, int]] = {}
        for sp in sp_keys:
            agg_subpools[sp] = {
                "used_bytes": 0,
                "cap_bytes": 0,
                "available_bytes": 0,
                "evictable_bytes": 0,
                # page_bytes is static; take rank-0's value.
                "page_bytes": int(rank0_subpools[sp]["page_bytes"]),
            }
        for rank in per_rank:
            rank_subpools = rank["pool_usage"][tier]["subpools"]
            if set(rank_subpools.keys()) != sp_keys:
                # DESIGN §6 line 737: every rank runs the same architecture,
                # so the subpool key set is structurally identical; a
                # mismatch is a deployment bug, not a workload reality.
                fatal(
                    "subpool_key_mismatch_across_ranks",
                    tier=tier,
                    rank0_keys=sorted(sp_keys),
                    this_rank_keys=sorted(rank_subpools.keys()),
                    state=state_json,
                )
            for sp, fields in rank_subpools.items():
                agg_subpools[sp]["used_bytes"] += int(fields["used_bytes"])
                agg_subpools[sp]["cap_bytes"] += int(fields["cap_bytes"])
                agg_subpools[sp]["available_bytes"] += int(
                    fields["available_bytes"])
                agg_subpools[sp]["evictable_bytes"] += int(
                    fields["evictable_bytes"])
        agg_pool[tier] = {"subpools": agg_subpools}

    # ---- link_stats ----
    # DESIGN §6 line 731: peak_bw_bps AND recent_throughput_bps both
    # SUM across ranks (each rank has its own PCIe/NVMe link;
    # aggregate scales with rank count).  time_since_last_sample_s
    # takes the MAX (the longest idle window across ranks; admission's
    # bw_free branch should treat the link as idle only if EVERY rank
    # has been idle).
    rank0_links = rank0["link_stats"]
    agg_links: Dict[str, Dict[str, float]] = {}
    for link in rank0_links:
        total_peak = 0
        total_throughput = 0.0
        max_idle = 0.0
        for rank in per_rank:
            entry = rank["link_stats"][link]
            total_peak += int(entry["peak_bw_bps"])
            total_throughput += float(entry["recent_throughput_bps"])
            max_idle = max(max_idle, float(entry["time_since_last_sample_s"]))
        agg_links[link] = {
            "peak_bw_bps": total_peak,
            "recent_throughput_bps": total_throughput,
            "time_since_last_sample_s": max_idle,
        }

    # ---- tier_holding_cost: identical across ranks; take rank-0 ----
    agg_holding = rank0["tier_holding_cost"]

    # ---- throughput_ema: sum prefill, merge decode per program ----
    agg_throughput: Dict[str, Any] = {
        "prefill_bps": sum(
            float(rank["throughput_ema"]["prefill_bps"]) for rank in per_rank),
        "decode_per_program": {},
    }
    for rank in per_rank:
        for pid, bps in rank["throughput_ema"]["decode_per_program"].items():
            agg_throughput["decode_per_program"][pid] = (
                agg_throughput["decode_per_program"].get(pid, 0.0)
                + float(bps))

    # ---- per_program_usage: sum committed bytes; union unit_hashes ----
    # State preference (most-active wins): REASONING > ACTING > PAUSED > ENDED.
    _STATE_RANK = {"REASONING": 3, "ACTING": 2, "PAUSED": 1, "ENDED": 0}
    agg_programs: Dict[str, Dict[str, Any]] = {}
    for rank in per_rank:
        for pid, e in rank["per_program_usage"].items():
            agg = agg_programs.get(pid)
            if agg is None:
                agg = {
                    "hbm": {"committed": {}, "inflight": {}},
                    "dram": {"committed": {}},
                    "state": e["state"],
                    "pre_pause_state": e["pre_pause_state"],
                    "unit_hashes": [],
                }
                agg_programs[pid] = agg
            for side, side_dict in (("hbm", e["hbm"]),
                                    ("dram", e["dram"])):
                for sub_kind, sub in side_dict.items():
                    if side == "dram" and sub_kind != "committed":
                        continue
                    bucket = agg[side][sub_kind]
                    for sp, b in sub.items():
                        bucket[sp] = bucket.get(sp, 0) + int(b)
            if _STATE_RANK[e["state"]] > _STATE_RANK[agg["state"]]:
                agg["state"] = e["state"]
                agg["pre_pause_state"] = e["pre_pause_state"]
            agg["unit_hashes"].extend(e["unit_hashes"])
    # Dedup unit_hashes per program (rank-replicated units appear on
    # multiple ranks).
    for pid, agg in agg_programs.items():
        agg["unit_hashes"] = sorted(set(agg["unit_hashes"]))

    # ---- units: concat with hash-dedup (union residence on collision) ----
    agg_units: List[Dict[str, Any]] = []
    hash_to_idx: Dict[str, int] = {}
    agg_time = 0
    for rank in per_rank:
        for u in rank["units"]:
            uhash = str(u["hash"])
            if uhash in hash_to_idx:
                # Replicated-prefix: union residence (the unit is in
                # whatever tiers ANY rank reports for it; the colder
                # union is a superset of any single rank's view).
                existing = agg_units[hash_to_idx[uhash]]
                merged_residence = sorted(
                    set(existing["residence"]) | set(u["residence"]))
                # n_bytes: DESIGN §6 L736 — identical across ranks
                # (derived from architecture).  When the SAME
                # (tier, subpool) key is present on both ranks for the
                # same hash, the values MUST agree.  A "max to absorb
                # mid-migration race" fallback would silently mask a
                # deployment bug; DESIGN explicitly classifies this as
                # bug-class → fatal().  Tier-keys present on only one
                # rank are fine (one rank may be mid-migration with
                # the unit in DRAM while another rank still has HBM).
                merged_nb: Dict[str, Dict[str, int]] = {}
                for tier in set(existing["n_bytes"]) | set(u["n_bytes"]):
                    merged_nb[tier] = {}
                    e_tier = existing["n_bytes"].get(tier, {})
                    u_tier = u["n_bytes"].get(tier, {})
                    for sp in set(e_tier) | set(u_tier):
                        e_val = e_tier.get(sp)
                        u_val = u_tier.get(sp)
                        if e_val is not None and u_val is not None:
                            if int(e_val) != int(u_val):
                                fatal(
                                    "n_bytes_disagreement_across_ranks",
                                    hash=uhash,
                                    tier=tier,
                                    subpool=sp,
                                    rank_a_n_bytes=int(e_val),
                                    rank_b_n_bytes=int(u_val),
                                    state=state_json,
                                )
                            merged_nb[tier][sp] = int(e_val)
                        elif e_val is not None:
                            merged_nb[tier][sp] = int(e_val)
                        else:
                            merged_nb[tier][sp] = int(u_val)
                existing["residence"] = merged_residence
                existing["n_bytes"] = merged_nb
                continue
            hash_to_idx[uhash] = len(agg_units)
            agg_units.append(dict(u))
        agg_time = max(agg_time, int(rank["time_counter"]))

    return {
        "time_counter": agg_time,
        "throughput_ema": agg_throughput,
        "pool_usage": agg_pool,
        "per_program_usage": agg_programs,
        "units": agg_units,
        "link_stats": agg_links,
        "tier_holding_cost": agg_holding,
    }


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
    """Convert sglang ``/aginfer/state`` JSON (DESIGN §5) → ``SchedulerState``.

    Paper §3's reduced state: per-unit (residence, age, p_hat, λ),
    per-(tier, subpool) usage, and the event's decision_set.  ``λ``
    for ACTING/PAUSED programs is the calibrated floor; for REASONING
    programs we keep the inline scorer's ``hits/age`` proxy so two
    scoring paths agree.

    sglang's emission contract (post-T17):
      * pool_usage: {tier: {"subpools": {sp: {used/cap/avail/evict/page_bytes}}}}
      * units: each has hash, residence: list[Tier], n_tokens,
               n_bytes: dict[tier][sp]→int, last_access_time, hit_count,
               session_ids
      * per_program_usage: per-pid committed/inflight bytes + state
      * time_counter: int
      * link_stats / tier_holding_cost / throughput_ema: aux fields
    """
    # Halt-loudly on unsupported tree cache (DESIGN §5 invariant).
    if "unsupported_tree_cache" in state_json:
        fatal(
            "unsupported_tree_cache",
            reported_kind=state_json["unsupported_tree_cache"],
            state=state_json,
        )

    # Multi-rank aggregation: DESIGN §6 per-field table.
    state_json = _flatten_per_rank(state_json)

    # DESIGN §10 "Required positivity" + "Missing fields ... deployment
    # bugs → fatal()".  Every consumer below assumes these blocks
    # exist; failing fast with a forensic dump is strictly better than
    # a KeyError at line 420.
    for field in ("pool_usage", "link_stats", "tier_holding_cost",
                  "throughput_ema", "per_program_usage", "units",
                  "time_counter"):
        if field not in state_json:
            fatal(
                "missing_state_field",
                missing=field,
                state=state_json,
            )

    # DESIGN §10 line 2319 positivity invariant #2:
    #   h_max_per_byte_sec > 0 for every (tier, subpool).
    # Zero means operator forgot to configure h_max for a subpool —
    # the holding-tax term in V_u silently collapses to zero, the
    # policy decisions go off-paper, no log.
    #
    # Conditional (audit #161): sglang ships a pre-T12 placeholder
    # of H=0.0 for every (tier, subpool), so cold-start sees ALL
    # h_max == 0.  We treat that as "no operator config yet" and
    # let it pass (analogous to the prefill_bps "once any prefill
    # has run" qualifier).  The actual deployment bug is PARTIAL
    # configuration — SOME entries positive and SOME zero — which
    # silently zeroes the holding tax for the zero-config subpool.
    holding_values = [
        float(fields["h_max_per_byte_sec"])
        for subpools in state_json["tier_holding_cost"].values()
        for fields in subpools.values()
    ]
    any_positive = any(v > 0.0 for v in holding_values)
    any_negative = any(v < 0.0 for v in holding_values)
    if any_negative:
        # Negative is always a deployment bug; locate one and fatal.
        for tier_label, subpools in state_json["tier_holding_cost"].items():
            for sp, fields in subpools.items():
                v = float(fields["h_max_per_byte_sec"])
                if v < 0.0:
                    fatal(
                        "holding_cost_non_positive",
                        tier=tier_label, subpool=sp,
                        h_max_per_byte_sec=v,
                        state=state_json,
                    )
    if any_positive:
        # Once any positive value exists, every entry must be positive
        # (partial-config deployment bug).
        for tier_label, subpools in state_json["tier_holding_cost"].items():
            for sp, fields in subpools.items():
                v = float(fields["h_max_per_byte_sec"])
                if v <= 0.0:
                    fatal(
                        "holding_cost_non_positive",
                        tier=tier_label, subpool=sp,
                        h_max_per_byte_sec=v,
                        state=state_json,
                    )

    # DESIGN §10 line 2319 positivity invariant #3:
    #   prefill_bps > 0 ONCE ANY PREFILL HAS RUN.
    #
    # Pre-T26 reality (audit #161 follow-up): sglang's
    # _aginfer_throughput_ema ships ``prefill_bps=0.0`` as a placeholder
    # awaiting T26 measurement wiring.  The strict "0 + units > 0 →
    # fatal" check fires on every event after the first prefill,
    # treating sglang's "no measurement yet" the same as
    # "measurement broken".  Until T26 lands, we ONLY fatal on
    # NEGATIVE prefill_bps (structurally nonsense; can never be a
    # startup state).  Once T26 wires real measurement, this check
    # should re-tighten to the "zero + traffic = bug" form.
    prefill_bps = float(state_json["throughput_ema"]["prefill_bps"])
    if prefill_bps < 0.0:
        fatal(
            "prefill_bps_non_positive_with_traffic",
            prefill_bps=prefill_bps,
            units_count=len(state_json["units"]),
            time_counter=int(state_json["time_counter"]),
            state=state_json,
        )

    # ---- TierUsage: per-(tier, subpool) view from pool_usage ----
    raw_pool = state_json["pool_usage"]
    tier_usage = TierUsage()
    pool_pressure: Dict[Tier, Dict[str, float]] = {}
    for label, tier in (("HBM", Tier.HBM), ("DRAM", Tier.DRAM),
                        ("DISK", Tier.DISK)):
        subpools = raw_pool[label]["subpools"]
        tier_usage.pool_used[tier] = {}
        tier_usage.pool_cap[tier] = {}
        tier_usage.pool_available[tier] = {}
        tier_usage.pool_evictable[tier] = {}
        tier_usage.page_bytes[tier] = {}
        pool_pressure[tier] = {}
        for sp, fields in subpools.items():
            used = int(fields["used_bytes"])
            cap = int(fields["cap_bytes"])
            tier_usage.pool_used[tier][sp] = used
            tier_usage.pool_cap[tier][sp] = cap
            tier_usage.pool_available[tier][sp] = int(fields["available_bytes"])
            tier_usage.pool_evictable[tier][sp] = int(fields["evictable_bytes"])
            tier_usage.page_bytes[tier][sp] = int(fields["page_bytes"])
            pool_pressure[tier][sp] = used / cap if cap > 0 else 0.0
    # bw_free derived from link_stats: peak when link is cold-idle,
    # else (peak - recent_throughput).  Negative bw_free clamps to 0.
    raw_links = state_json["link_stats"]
    _LINK_IDLE_SECONDS = 1.0  # DESIGN §7 bw_free branch threshold
    _LINK_PAIRS = [
        ((Tier.HBM, Tier.DRAM), "HBM->DRAM"),
        ((Tier.DRAM, Tier.HBM), "DRAM->HBM"),
        ((Tier.DRAM, Tier.DISK), "DRAM->DISK"),
        ((Tier.DISK, Tier.DRAM), "DISK->DRAM"),
    ]
    for (src, dst), link_label in _LINK_PAIRS:
        entry = raw_links[link_label]
        peak = float(entry["peak_bw_bps"])
        if peak <= 0.0:
            # DESIGN §10 "Required positivity: peak_bw_bps > 0".  A
            # non-positive peak means either sglang hasn't measured the
            # link yet (mid-startup race) or the operator misconfigured
            # the deployment; either way the daemon cannot compute
            # bw_free and the policy's bw-bound bucket collapses.
            fatal(
                "peak_bw_bps_non_positive",
                link=link_label,
                peak_bw_bps=peak,
                state=state_json,
            )
        recent = float(entry["recent_throughput_bps"])
        idle = float(entry["time_since_last_sample_s"])
        # DESIGN §7: if link is cold-idle (idle > 1.0 s), assume peak
        # is fully available.  Otherwise free = peak - recent.
        bw = peak if idle > _LINK_IDLE_SECONDS else max(0.0, peak - recent)
        tier_usage.bw_free[(src, dst)] = bw

    units_raw = state_json["units"]
    now_counter = int(state_json["time_counter"])

    units: Dict[str, ReuseUnit] = {}
    # Owner program → its ACTING-floor λ (cached per call).
    program_lambda: Dict[str, float] = {}
    _RESIDENCE_TIER = {"HBM": Tier.HBM, "DRAM": Tier.DRAM,
                       "DISK": Tier.DISK}
    for raw in units_raw:
        uhash = str(raw["hash"])
        if not uhash:
            continue
        n_tokens = int(raw["n_tokens"])
        # Residence + nested n_bytes from new schema.
        residence: List[Tier] = []
        for tier_label in raw["residence"]:
            tier = _RESIDENCE_TIER.get(tier_label)
            if tier is None:
                _log_unknown_tier_once(tier_label, unknown_tier_log)
                continue
            residence.append(tier)
        if not residence:
            # Empty residence after filtering — shouldn't happen per
            # DESIGN §5 (empty-residence units don't appear in units[]);
            # skip rather than build a degenerate ReuseUnit.
            continue
        n_bytes_by_tier: Dict[Tier, Dict[str, int]] = {}
        for tier_label, sp_dict in raw["n_bytes"].items():
            tier = _RESIDENCE_TIER.get(tier_label)
            if tier is None:
                continue
            n_bytes_by_tier[tier] = {sp: int(b) for sp, b in sp_dict.items()}
        last_access = int(raw["last_access_time"])
        hits = int(raw["hit_count"])
        age = max(1, now_counter - last_access)
        lam = max(1e-3, hits / age)
        # Iterate holders to compute λ floor + p_hat (program-alive rule
        # — see prior comments in commit history for §7 justification).
        session_ids = raw["session_ids"]
        any_acting = False
        any_alive = False
        for sid in session_ids:
            st = tracker.state(sid)
            if st is not None:
                any_alive = True
            if sid not in program_lambda:
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
        if any_alive:
            p_hat = 1.0
        else:
            p_hat = min(1.0, hits / age)
        units[uhash] = ReuseUnit(
            id=uhash,
            type=UnitType.SESSION,  # platform / tool_def tags arrive later
            scope=Scope.SESSION,
            n_tokens=n_tokens,
            n_bytes_by_tier=n_bytes_by_tier,
            residence=residence,
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
        if Tier.HBM not in u.residence:
            # v1: only HBM-resident units are demote candidates.  Post-
            # T17, a unit is "in HBM" iff Tier.HBM ∈ residence (set
            # semantics — a unit can be HBM+DRAM simultaneously).
            # T10/T34 extends this to also rank DRAM units once the
            # daemon-controlled DISK (L3 / Mooncake) tier is wired —
            # paper §7.1 says regret should rank across all current-
            # tier units.
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
    assignments: Iterable[Tuple[str, List[Tier], List[Tier]]],
) -> List[Dict[str, Any]]:
    """Translate ``[(unit_hash, add_tiers, remove_tiers), ...]`` →
    DESIGN §6 ``POST /aginfer/migrate`` JSON body items.

    Each item carries ``add_tiers`` + ``remove_tiers`` (residence-set
    transitions per §7) plus an ``action_id`` opaque correlator that
    the sglang side echoes back in APPLY_FAILED webhooks (T23).
    """
    import uuid
    return [
        {
            "hash": uhash,
            "add_tiers": [_tier_to_wire(t) for t in add],
            "remove_tiers": [_tier_to_wire(t) for t in remove],
            "action_id": uuid.uuid4().hex,
        }
        for uhash, add, remove in assignments
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
        observability=None,  # daemon._observability.DaemonObservability
    ) -> None:
        self.tracker = tracker
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.policy = policy or OursGreedyPolicy(default_costs())
        self.lambda_acting = lambda_acting
        # T42: optional injection from main.py (router.observability).
        # When set, every migrate skip-reason bumps the per-reason
        # counter in addition to its per-event log line.  Left None
        # for unit tests that only care about the policy / wire path.
        self.observability = observability
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
        # Per-action skip reason breakdown.
        self._record_skips(skipped_list, _m)

    def _record_skips(self, skipped_list, _m_func=None) -> None:
        """Per-skip accounting: emit a ``migrate_skipped`` line AND
        (when observability is attached) bump the cumulative
        per-reason counter on the daemon's T42 aggregator.

        Extracted as a method so unit tests can exercise the skip-
        accounting path without spinning up the HTTP client.
        Each line ≈ 1 µs, fires at most ``len(skipped_list)`` times
        per migrate POST — high-value for finding G11-class issues
        (race:already_on_dram / promote_not_yet_wired /
        race:not_in_tree / etc.).
        """
        if _m_func is None:
            from ._metrics import m as _m_func
        for entry in skipped_list:
            # Direct subscript: sglang's /aginfer/migrate ALWAYS emits
            # `reason` per DESIGN §6 + verify/t20 enforcement.  A
            # missing field is a real protocol regression worth
            # surfacing, not a silent "?" metric value.
            # Replace spaces with _ since metric format is space-sep.
            reason = entry["reason"].replace(" ", "_")[:120]
            _m_func("migrate_skipped", reason=reason)
            if self.observability is not None:
                self.observability.record_failure(reason)


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
