"""aginfer state builder (refactor #251 increment 4): the dump→PaperState transform.

`build_paper_state` converts sglang's `/aginfer/state` JSON (DESIGN §5) into the paper-§3
`SchedulerState` the value model + joint_decide consume — plus the §4 decision-set (`D_t`).
Extracted VERBATIM (AST source segments) from dev/aginfer/daemon/kv_scheduler.py so the ENGINE
can build its own s_t in-process (no daemon, no HTTP); the daemon re-exports these names so its
own call sites + the verify suite keep working unchanged. ONE canonical copy of the transform.

Imports resolve to the in-engine package directly (the daemon reached them via baselines.*/.xxx
alias-shims). A FRESH module logger (not the daemon's) — the only move-proof straggler.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sglang.srt.mem_cache.aginfer.base import (
    ReuseUnit, Scope, SchedulerState, Tier, TierUsage, UnitType,
)
from sglang.srt.mem_cache.aginfer._fatal import fatal
from sglang.srt.mem_cache.aginfer.costs import default_costs
from sglang.srt.mem_cache.aginfer.events import Event, EventKind
from sglang.srt.mem_cache.aginfer.program_tracker import ProgramTracker, State

logger = logging.getLogger("sglang.srt.mem_cache.aginfer.state_builder")






# --- env-var helpers ---


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





# --- calibration constants (DESIGN §7) ---


_LAMBDA_ACTING_FLOOR = 1.0 / 30.0


_LAMBDA_ACTING_CEIL = 1.0 / 1.0


_DEFAULT_LAMBDA_ACTING = _env_float("AGINFER_LAMBDA_ACTING", "0.2")


_CONST_VU = bool(os.environ.get("AGINFER_CONST_VU"))


_PHAT_REUSE_ALPHA = _env_float("AGINFER_PHAT_REUSE_ALPHA", "0.5")


# T11 (DESIGN §7 "Δt" estimator priority #3, bootstrap/cold-start): the
# look-ahead window used for an ACTING holder's p_access when no sharper
# per-event ETA is available (i.e. this holder isn't the event's own
# session, or the event carries no tool_eta_s).  DESIGN §7 quotes "average
# inter-event spacing (10ms-1s range on agent workloads)" for this
# fallback; 1.0s (the busy end of that range) is the conservative choice —
# a smaller Δt would UNDERSTATE p_access for an ACTING holder we have no
# sharper signal for, biasing the holder-product toward demoting units
# that are, in fact, likely to be reused soon.
_PHAT_BOOTSTRAP_DT = _env_float("AGINFER_PHAT_BOOTSTRAP_DT", "1.0")


_DEFAULT_MEMORY_PRESSURE_TOPK = _env_int("AGINFER_MEMORY_PRESSURE_TOPK", "256")


_PROMOTE_FALLBACK_BW_BPS = _env_float("AGINFER_PROMOTE_FALLBACK_BW_BPS", "5e9")


LINK_IDLE_SECONDS = 1.0


LINK_PAIRS = [
    ((Tier.HBM, Tier.DRAM), "HBM->DRAM"),
    ((Tier.DRAM, Tier.HBM), "DRAM->HBM"),
    ((Tier.DRAM, Tier.DISK), "DRAM->DISK"),
    ((Tier.DISK, Tier.DRAM), "DISK->DRAM"),
]


_TIER_LABEL_MAP: Dict[str, Tier] = {
    "HBM": Tier.HBM,
    "DEVICE": Tier.HBM,
    "DRAM": Tier.DRAM,
    "HOST": Tier.DRAM,
    "DISK": Tier.DISK,
    "DROP": Tier.DROP,
}





# --- dump→PaperState helpers + the transform ---


def _clamp_lambda_acting(lam: float) -> float:
    return max(_LAMBDA_ACTING_FLOOR, min(_LAMBDA_ACTING_CEIL, lam))


def _p_access_holder(
    st: Optional[State],
    hits: int,
    sid: str,
    event: Event,
    program_lambda: Dict[str, float],
) -> float:
    """T11 (DESIGN §7): one holder's contribution to a unit's holder-product
    ``p_hat``, ``p_access(u, s, Δt)`` conditioned on ``s``'s OWN observable
    ``program_tracker`` state (the state-as-feature design the old single
    branch-selected p_hat — any_alive / any_ended / untracked — collapsed
    away):

      PAUSED / ENDED  -> 0.  DESIGN §7 is explicit: no access until
        admission resume (PAUSED) / the program terminated and issues no
        more requests against this unit (ENDED).

      ACTING  -> P(``sid``'s tool returns within Δt).  When ``sid`` IS the
        triggering event's own session AND that TOOL_CALL_START carries a
        real ``tool_eta_s``, Δt is BY DEFINITION that ETA (estimator
        priority #1: "the access we care about is the one that fires when
        the tool returns") -> the access is a near-certainty within its
        own window -> p_access ~= 1.  Otherwise (a co-holder we have no
        per-holder ETA for, or no payload ETA) fall back to the calibrated
        ACTING-floor rate under the bootstrap Δt (priority #3).

      REASONING / untracked (``st is None``)  -> the SAME recency-
        DECOUPLED reuse-probability proxy used pre-holder-product
        (#249/#250): one-shot (hits<=1) -> 0, demonstrated reuse -> ->1.
        We have no per-holder turn-distance signal for "in s's recent
        prefix tail" (DESIGN §7's REASONING case), so this unit-level
        hit_count stands in for it; an untracked holder (no aginfer
        TOOL_CALL protocol in play) gets the identical treatment because
        session-state is a FEATURE layered on TOP of the base estimator,
        not a fallback to a DIFFERENT one — the
        [[feedback-workload-agnostic-phat]] rule this supersedes-in-place.
    """
    if st is State.PAUSED or st is State.ENDED:
        return 0.0
    if st is State.ACTING:
        if event.kind == EventKind.TOOL_CALL_START and event.session == sid:
            eta_raw = event.payload.get("tool_eta_s")
            try:
                eta = float(eta_raw) if eta_raw is not None else 0.0
            except (TypeError, ValueError):
                eta = 0.0
            if eta > 0.0:
                return 1.0
        lam = program_lambda.get(sid, 0.0)
        return 1.0 - math.exp(-lam * _PHAT_BOOTSTRAP_DT)
    return 1.0 - math.exp(-_PHAT_REUSE_ALPHA * max(0, hits - 1))


def _estimate_load_back_s(state: SchedulerState, total_bytes: int) -> float:
    """DESIGN §7 ``load_back_latency`` estimate for the predictive promote.

    Time to bring ``total_bytes`` of a demoted tail back up to HBM, from the
    live ``bw_free`` link samples (DESIGN §5, bytes/s).  There is **no direct
    DISK→HBM link** (see ``LINK_PAIRS``) — a DISK-resident tail loads back
    two-hop (DISK→DRAM→HBM), so we cost the **DISK path** here: it is the
    slower load_back, and S1 deliberately sizes DRAM small so the idle tail
    spills onto DISK during a long tool gap.  Costing the slower path means the
    promote is scheduled a touch EARLY — win-preserving (the prefix is in HBM
    before the resume); if the tail happened to stay in DRAM the only cost is a
    slightly shorter demote window, never a missed promote.  A link with no
    sample yet (cold start) falls back to a conservative DRAM-class rate so the
    lead time never collapses to zero (which would degrade to a promote-at-
    TOOL_CALL_END)."""
    if total_bytes <= 0:
        return 0.0
    bw = state.tier_usage.bw_free

    def hop_s(pair) -> float:
        b = bw.get(pair, 0.0)
        return float(total_bytes) / (b if b > 0.0 else _PROMOTE_FALLBACK_BW_BPS)

    # DISK→DRAM then DRAM→HBM (the two-hop load_back).
    return hop_s((Tier.DISK, Tier.DRAM)) + hop_s((Tier.DRAM, Tier.HBM))


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
        "aginfer.state_builder: unknown tier label %r in /aginfer/state; "
        "unit skipped.  Add to state_builder.py _TIER_LABEL_MAP.",
        label,
    )


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
      * throughput_ema.prefill_bps / decode_per_program[pid]: SUM across
        ranks — kept consistent with the SUMmed byte fields (the daemon
        runs in a uniform N×-scaled space; see the inline rationale at the
        throughput block).  DESIGN §6 L766/L767 "mean" assumes a true-scale
        regime this impl does not use; true-scale conversion is #214.
      * per_program_usage[pid].hbm.committed[sp] (and dram.committed):
        SUM across ranks; state reconciles to the SAFE side on cross-
        rank disagreement (PAUSED > REASONING > ACTING > ENDED): PAUSED
        wins so a lagging rank can't mask a pause and starve the resume
        candidate; ENDED loses so a lagging co-holder rank isn't dropped
        prematurely.  unit_hashes is the union; pre_pause_state follows
        the winning (PAUSED) rank.
      * units: concatenate verbatim — **no hash prefix**.  sglang's
        hashes are globally unique (hex SHA256 or ``node-<id>``); if
        two ranks emit the SAME hash it's the same logical unit
        (replicated prefix) and we dedupe — broadcasting one migrate
        to both ranks is the correct action.  On residence disagree-
        ment between ranks we keep the COLDER union (= the unit's
        bytes are persisted somewhere even if mid-migration on one
        rank).  leaf flags AND-reconcile (#210), session_ids/holders
        UNION (#211), last_access_time/hit_count MAX (warmest, the
        liveness-preserving side feeding V_u), n_tokens identical→rank-0.
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
                # decode_bytes_per_token (#199) is static (architecture
                # constant); take rank-0.  Older sglang may omit it → 0.
                "decode_bytes_per_token": int(
                    rank0_subpools[sp].get("decode_bytes_per_token", 0)),
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

    # ---- throughput_ema: SUM prefill, SUM decode per program ----
    # KEEP SUM (status quo).  The daemon runs in a UNIFORM N×-scaled byte
    # space: a unit's n_bytes are IDENTICAL across ranks (the agree-or-fatal
    # check below proves KV bytes are MIRRORED, not 1/N-sharded — MLA latent
    # / replicated prefix), and pool_usage + per-program committed/inflight
    # are SUMmed, so every absolute byte AND byte-rate is N×.  Because that
    # scaling is uniform, every consumer cancels: occ is a ratio; the
    # knapsack compares N× budget vs N× weight; marginal_pause_cost =
    # (N× inflight) / (N× prefill_bps); forecast growth (N× decode_tokens ×
    # bpt) is compared against the N× cap.  MEAN-ing ONLY prefill_bps /
    # decode while committed/inflight/cap stay SUMmed would BREAK those
    # cancellations and inject a real N× error — so we do not.
    #
    # DESIGN §6 L766/L767 say "mean" for these rates: that assumes a
    # true-scale (or genuinely sharded) regime this impl does not use.
    # Converting to true-scale means de-N×-ing ALL byte fields together
    # (pool / committed / inflight / units), not just the two rates —
    # deferred to #214.  Moot today on two counts: the forecast consumer is
    # dormant until T11/T26, and this whole multi-rank merge path is dead
    # until #174 wires per-rank dumps (sglang currently returns a single
    # pre-aggregated dump → _flatten_per_rank returns it unchanged).
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
    # Cross-rank state reconciliation.  PUT /aginfer/program_paused fans
    # out to every rank's scheduler PER-RANK and is NOT cross-rank atomic
    # (tokenizer_control_mixin: "partial ok = race with another in-flight
    # PUT"); a state dump can land mid-fan-out, so the SAME pid can read
    # PAUSED on the rank that already applied and REASONING/ACTING on a
    # lagging rank.
    #
    #   PAUSED must WIN over REASONING/ACTING.  ``resume_candidates`` only
    #   considers pids whose merged ``state == "PAUSED"``; if a lagging
    #   rank's REASONING masked the pause, the program would drop out of
    #   the resume set and — since a PAUSED program emits no events of its
    #   own and can only be un-gated by a headroom Resume — STARVE to an
    #   AgentTimeout (the #211 starvation class, here via cross-rank merge
    #   instead of dropped units).  This mirrors the residence "colder
    #   superset" union and the #210 leaf-flag AND: when ranks disagree on
    #   a daemon-controlled transition, take the SAFE (liveness-preserving)
    #   side, never rank-0's permissive view.
    #
    #   ENDED LOSES to any active/paused state (kept from the original
    #   "most-active wins" rule): a rank that saw SESSION_END first must
    #   not prematurely drive the program to ENDED on a lagging co-holder
    #   rank — premature ENDED drops session_scoped KV the lagging rank may
    #   still reference (unsafe), whereas keeping it active over-retains
    #   for one event and self-heals when every rank catches up (benign).
    #
    #   REASONING > ACTING only breaks the λ tie (REASONING uses the
    #   hits/age proxy, ACTING the calibrated floor); neither gates
    #   liveness, so their relative order is cosmetic.
    _STATE_RANK = {"PAUSED": 4, "REASONING": 3, "ACTING": 2, "ENDED": 0}
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
                # #210: AND-reconcile the leaf flags.  A remove is structurally
                # safe (sglang won't reject it) only if EVERY rank agrees the
                # node is the relevant leaf — in the mid-migration window the
                # residence-union comment above calls out, one rank can have
                # the node device-resident (is_device_leaf=True) while another
                # has it evicted (False).  Keeping rank-0's permissive view
                # would let migrate_candidates propose a remove the disagreeing
                # rank rejects, re-arming the #210 apply_failed leak.  AND is
                # the stricter mirror of the (colder-superset) residence union.
                for _flag in ("is_device_leaf", "is_host_leaf", "is_tree_leaf"):
                    existing[_flag] = bool(existing.get(_flag, True)) \
                        and bool(u.get(_flag, True))
                # UNION session_ids (holders) across ranks.  Node session
                # tagging (``node.session_ids.add(pid)`` / SESSION_END
                # untagging) runs in each rank's OWN scheduler, driven by
                # the same request stream but at slightly different wall-
                # clock moments, so in a transient window the same hash can
                # carry {p0,p1} on one rank and {p0} on a rank that already
                # processed p1's SESSION_END.  Keeping rank-0's set verbatim
                # made ``holders`` ORDER-DEPENDENT (whichever rank emitted
                # the hash first won), so ``len(u.holders)`` —
                # shared_aware_prog_scores' V_u divisor — and
                # session_scoped_units' ``holders == {session}`` predicate
                # diverged by rank ordering.  Union is the colder superset
                # (a holder seen by ANY rank is a holder): it keeps a live
                # co-holder's p_hat high and never lets a stale single-rank
                # view strand a still-shared unit as session-scoped.
                existing_sids = existing.get("session_ids") or []
                u_sids = u.get("session_ids") or []
                existing["session_ids"] = sorted(
                    set(existing_sids) | set(u_sids))
                # MAX-reconcile last_access_time + hit_count (same transient-
                # divergence class as the #210 leaf flags / #211 holders
                # union).  Each rank's scheduler bumps the radix node's
                # access counters on the SAME request stream, so in a
                # propagation window one rank can read a just-accessed value
                # (recent last_access, higher hit_count) while a lagging rank
                # is stale.  These feed V_u: age = max(1, now - last_access),
                # lam = hits/age, p_hat = min(1, hits/age) for units with no
                # live holder.  A stale rank-0 last_access → larger age →
                # smaller p_hat → the unit looks COLDER → _top_k_by_regret
                # (saved = p_hat·Δρ·n_tokens) ranks it a demote candidate it
                # shouldn't be.  Take the SAFE (warmest / liveness-preserving)
                # side — MAX, mirroring the residence-union "colder superset"
                # and the holders union — so a stale single-rank view never
                # spuriously demotes a still-warm unit.  Order-independent.
                existing["last_access_time"] = max(
                    int(existing.get("last_access_time", 0)),
                    int(u.get("last_access_time", 0)))
                existing["hit_count"] = max(
                    int(existing.get("hit_count", 0)),
                    int(u.get("hit_count", 0)))
                # n_tokens: REPLICATED logical token count (every rank holds
                # the same prefix tokens; only the head-dim slice of each
                # token's KV differs), so it is identical across ranks by
                # construction — like n_bytes but architecture-derived from
                # the token count rather than the byte shape.  Take rank-0
                # (kept verbatim from the first-seen unit dict above); a
                # genuine cross-rank disagreement would be a deployment bug,
                # but unlike n_bytes there is no per-token fatal guard because
                # n_tokens is not a sharded byte quantity any consumer sums.
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
        tier_usage.decode_bytes_per_token[tier] = {}
        pool_pressure[tier] = {}
        for sp, fields in subpools.items():
            used = int(fields["used_bytes"])
            cap = int(fields["cap_bytes"])
            tier_usage.pool_used[tier][sp] = used
            tier_usage.pool_cap[tier][sp] = cap
            tier_usage.pool_available[tier][sp] = int(fields["available_bytes"])
            tier_usage.pool_evictable[tier][sp] = int(fields["evictable_bytes"])
            tier_usage.page_bytes[tier][sp] = int(fields["page_bytes"])
            # #199: optional (older sglang omits it) → default 0.
            tier_usage.decode_bytes_per_token[tier][sp] = int(
                fields.get("decode_bytes_per_token", 0))
            pool_pressure[tier][sp] = used / cap if cap > 0 else 0.0
    # bw_free derived from link_stats: peak when link is cold-idle,
    # else (peak - recent_throughput).  Negative bw_free clamps to 0.
    raw_links = state_json["link_stats"]
    for (src, dst), link_label in LINK_PAIRS:
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
        bw = peak if idle > LINK_IDLE_SECONDS else max(0.0, peak - recent)
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
        # Iterate holders to compute λ floor (unchanged — hold_time is a
        # SEPARATE quantity from p_hat's Δt, DESIGN §7 "hold_time" section).
        session_ids = raw["session_ids"]
        any_acting = False
        for sid in session_ids:
            st = tracker.state(sid)
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
        # T11 (DESIGN §7): p_hat is the holder-PRODUCT —
        #   p_hat(u, Δt) = 1 - Π_{s in u.session_ids} (1 - p_access(u, s, Δt))
        # — replacing the old single branch-selected estimate (any_alive /
        # any_ended / untracked, one formula for the WHOLE unit) with a real
        # per-holder aggregation.  This is what makes a shared prefix held by
        # N concurrent programs aggregate correctly (any one holder being
        # likely-to-access is enough to keep p_hat high) with NO ad-hoc 1/N
        # weighting, and what makes PAUSED/ENDED holders contribute EXACTLY
        # zero (not a softened prior) per the DESIGN §7 contract.
        if session_ids:
            p_not_access = 1.0
            for sid in session_ids:
                st = tracker.state(sid)
                p_not_access *= 1.0 - _p_access_holder(
                    st, hits, sid, event, program_lambda
                )
            p_hat = 1.0 - p_not_access
        else:
            # Holder-product's empty-Π convention (Π over ∅ = 1) would zero
            # p_hat for a unit with NO current holders — but a shared
            # platform/tool_def prefix genuinely sits briefly unheld between
            # sessions while remaining highly likely to be re-referenced
            # (§7.1's memory_pressure regret proxy singles these out).  With
            # no live holder to condition on, fall back to the same
            # demonstrated-reuse estimate an untracked holder would get.
            p_hat = 1.0 - math.exp(-_PHAT_REUSE_ALPHA * max(0, hits - 1))
        if _CONST_VU:
            # #208 const-V_u isolation arm: neutralise the reuse-prediction
            # signal on the daemon side too (matches the inline scorer's
            # AGINFER_CONST_VU hook) so the migrate/admission ranking is
            # reuse-blind while the machinery still runs.
            p_hat, lam = 1.0, 1.0
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
            # #210: the three structural leaf flags migrate_candidates uses
            # to mirror sglang's apply-site guards (remove_not_leaf /
            # remove_hbm_not_device_leaf / remove_dram_not_host_leaf).
            # Default True if the dump predates the field (co-shipped, so in
            # practice always present); then the policy keeps old behavior.
            is_device_leaf=bool(raw.get("is_device_leaf", True)),
            is_host_leaf=bool(raw.get("is_host_leaf", True)),
            is_tree_leaf=bool(raw.get("is_tree_leaf", True)),
        )

    # DESIGN §7 per-event override (estimator priority #1).  On TOOL_CALL_START
    # the caller is about to be idle for *this* tool's ETA, and §7 makes that
    # ETA both the reuse horizon (Δt) and the holding window (hold_time) for
    # demoting its session tail.  Plug the per-event ETA in as hold_time —
    # `_value` integrates the HBM holding tax over `hold_time = 1/lambda_rate`,
    # so set `lambda_rate = 1/ETA` on the caller's exclusive tail.  This
    # REPLACES the constant `λ_ACTING` fallback (the Poisson `1/λ` collapse §7
    # explicitly drops) for the one case where a sharper signal exists.  Without
    # it the demote value cannot tell a 0.5 s tool from a 60 s one — exactly the
    # long-predictable-gap signal S1 exploits.  hold_time is a per-DECISION
    # quantity (§7), and `units` is rebuilt every event, so this override is
    # naturally scoped to this decision and does not persist.
    if event.kind == EventKind.TOOL_CALL_START and event.session:
        _eta_raw = event.payload.get("tool_eta_s")
        try:
            _eta = float(_eta_raw) if _eta_raw is not None else 0.0
        except (TypeError, ValueError):
            _eta = 0.0
        if _eta > 0.0:
            for _uid in _units_for_session(units, event.session):
                if _uid in units:
                    units[_uid].lambda_rate = 1.0 / _eta
                    # Coupled with the predictive promote-back scheduled for
                    # this same tail (handle() → _schedule_promote_back): the
                    # reuse will be served from HBM, so the demote value must
                    # not charge it a DRAM/DISK load_back (DESIGN §7/§3 S1
                    # coupling).  value_residence reads this flag.
                    units[_uid].promote_pending = True

    decision_set = _build_decision_set(event, units, tracker)

    return SchedulerState(
        t=float(now_counter),
        units=units,
        tier_usage=tier_usage,
        event_kind=event.kind.value,
        event_session_id=event.session,
        decision_set=decision_set,
        pool_pressure=pool_pressure,
        # DESIGN §8 program/forecast inputs carried verbatim from the
        # (flattened) dump so the admission candidate generators read
        # them the same way kv_scheduler reads pool_usage.
        per_program_usage=state_json["per_program_usage"],
        throughput_ema=state_json["throughput_ema"],
    )


def _units_for_session(
    units: Dict[str, ReuseUnit], session: Optional[str]
) -> List[str]:
    """Caller's exclusive tail — units held ONLY by this session
    (``holders == {session}``).  This is DESIGN §7's ``session_tail``
    (TOOL_CALL_START/END, SUB_DISPATCH parent tail) AND
    ``session_scoped_units`` (SESSION_END) — the same exclusive
    predicate, one helper.

    EXCLUSIVE is intentional (#189, DESIGN §7 reconciled to match):
    a TOOL_CALL is a PER-PROGRAM event, so it nominates only p's
    PRIVATE units — the ones whose value changed when p went idle.  A
    SHARED platform/tool_def prefix's value did NOT change because one
    of its many holders went tool-bound (the others still need it), so
    it stays HBM and is NOT a candidate here; its residence is driven by
    SESSION_ARRIVAL (preload) + MEMORY_PRESSURE (global top-k).  Keeping
    the tail exclusive also makes the SUB_DISPATCH union
    (``_units_for_session + _shared_prefix_units``) disjoint — no
    double-scored hashes.
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
    if kind == EventKind.SESSION_END:
        # T187 (#187, DESIGN §7 decision_set table): session_scoped_
        # units(p) = units held ONLY by the ending program (holders ==
        # {p}).  After END these have no other holder, so the policy
        # demotes them to the cheapest tier with nonzero workload-prior
        # p_hat, or DROPs them.  Units p SHARED with another program
        # have holders ⊋ {p} → excluded here → untouched (they survive
        # p).  `_units_for_session` already implements the exclusive
        # (holders == {session}) form.  Pairs with the p_hat ENDED-
        # exclusion above: the ending program's SESSION_END handler
        # transitions it to ENDED BEFORE this runs, so these units
        # score with the workload-prior p_hat, not 1.0.
        return _units_for_session(units, session)
    if kind in (EventKind.MEMORY_PRESSURE, EventKind.PRESSURE_RESOLVED):
        # Top-k by regret (paper §7.1).
        return _top_k_by_regret(units, _DEFAULT_MEMORY_PRESSURE_TOPK)
    return []


def hints_from_state(sched_state) -> List[Dict[str, Any]]:  # noqa: ANN001
    """T40 (#184, DESIGN §6 ``PUT /aginfer/hints``): one hint per unit
    in ``D_t``, carrying the V_u inputs the scorer just computed.

    Wire shape per hint: ``{"hash", "p_hat", "lambda", "stamp"}``.
    ``stamp`` is sglang's own ``time_counter`` (``sched_state.t``) —
    a monotonic, daemon-restart-surviving ordering token that makes
    sglang's overwrite-by-stamp table deterministic WITHOUT any
    wall-clock call in the daemon's policy path (the §10 "no time.*
    in the transition path" invariant).

    This is the WHOLE D_t, pushed unconditionally — no shadow
    ``{hash: last_pushed}`` map, no "value changed beyond threshold"
    filter (DESIGN §10 "No daemon-side hint cache").  Redundant
    re-pushes of unchanged values are absorbed by sglang's
    overwrite-by-stamp dedupe (an equal stamp is an idempotent
    no-op).
    """
    stamp = int(sched_state.t)
    hints: List[Dict[str, Any]] = []
    for uid in sched_state.decision_set:
        u = sched_state.units.get(uid)
        if u is None:
            # D_t is derived from units, so this should not happen;
            # skip defensively rather than push a hint for a hash
            # sglang has no unit for.
            continue
        hints.append({
            "hash": uid,
            "p_hat": float(u.p_hat),
            "lambda": float(u.lambda_rate),
            # DESIGN §2 fact 1 / S2: holder-count so the inline eviction scorer can
            # value a fleet-shared prefix by N× saved-prefill (it builds units with
            # empty `holders` and can't recover the count from the node alone).
            "n_holders": len(u.holders),
            "stamp": stamp,
        })
    # S2 diagnostic: confirm the daemon actually observes shared units (n_holders>1)
    _mx = max((h["n_holders"] for h in hints), default=0)
    if _mx > 1:
        import logging as _lg
        _lg.getLogger("aginfer.kv").info(
            "[aginfer] S2 hint push: n=%d units, MAX n_holders=%d (shared prefix seen)",
            len(hints), _mx)
    return hints
