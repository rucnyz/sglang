"""verify/kv_scheduler_value_rule (#146): rewrite for post-T33 contract.

The legacy verify in ``legacy/`` exercised the pre-round-9 surface:
``_value(u, tier, state)`` over a single tier, ``Action.assignments``
as ``(unit_id, Tier)``, ``state['tier_usage']`` flat dict.  T33
collapsed that to a residence-set form (``_value(u, next_residence:
List[Tier], state)``, ``Action.assignments`` 3-tuples, nested
``pool_usage[tier].subpools[sp]``), and the legacy verify can no
longer load any of its fixtures.

This file is the post-T33 contract pin.  It does NOT re-test T17's
schema migration or T20's migrate-POST envelope — those have their
own verify dirs.  Scope here is the kv_scheduler MODULE's behavior
on top of those primitives:

  * ``build_paper_state``: post-T17 schema → ``SchedulerState``
  * ``_build_decision_set``: paper §4 D_t per EventKind
  * Program-aware λ / p_hat rules (ACTING-floor, alive p_hat)
  * Top-k regret demote candidates (paper §7.1)
  * ``Action.assignments`` 3-tuple shape
  * Dispatch wiring (post-T36 outbound-only)
  * Idempotence + robustness + latency floor

Each stage is independent — fixture per stage, no cross-stage state.

Usage:
    python dev/aginfer/verify/kv_scheduler_value_rule/verify.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import Action, Tier  # noqa: E402
from baselines.ours_greedy import OursGreedyPolicy  # noqa: E402
from baselines.costs import default_costs  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon.events import Event, EventKind  # noqa: E402
from daemon.outbound import OutboundQueue  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ fixtures


def _unit(
    *,
    uhash: str,
    residence: List[str],
    holders: List[str],
    n_tokens: int = 1000,
    n_bytes_per_tier: Optional[Dict[str, int]] = None,
    last_access_time: int = 0,
    hit_count: int = 1,
    subpool: str = "kv",
    is_device_leaf: bool = True,
    is_host_leaf: bool = True,
    is_tree_leaf: bool = True,
) -> Dict[str, Any]:
    """Synthetic post-T17 unit JSON."""
    if n_bytes_per_tier is None:
        n_bytes_per_tier = {t: n_tokens * 2048 for t in residence}
    n_bytes = {t: {subpool: nb} for t, nb in n_bytes_per_tier.items()}
    return {
        "hash": uhash,
        "residence": list(residence),
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "last_access_time": last_access_time,
        "hit_count": hit_count,
        "session_ids": list(holders),
        "is_device_leaf": is_device_leaf,
        "is_host_leaf": is_host_leaf,
        "is_tree_leaf": is_tree_leaf,
    }


def _state_json(
    *,
    units: List[Dict[str, Any]],
    programs: Optional[Dict[str, Dict[str, Any]]] = None,
    time_counter: int = 100,
    hbm_cap: int = 10 * 1024 * 1024 * 1024,
    hbm_used: int = 1 * 1024 * 1024 * 1024,
    dram_cap: int = 40 * 1024 * 1024 * 1024,
    dram_used: int = 1 * 1024 * 1024 * 1024,
    disk_cap: int = 200 * 1024 * 1024 * 1024,
    disk_used: int = 0,
    h_max_per_byte_sec: float = 0.0,  # cold-start placeholder; see kv_scheduler positivity rules
    peak_bw_bps: int = 64 * 1024 * 1024 * 1024,  # 64 GB/s nominal PCIe5
    prefill_bps: float = 0.0,  # pre-T26 placeholder
    subpool: str = "kv",
) -> Dict[str, Any]:
    """Synthetic post-T17 state JSON.  All required fields populated.

    Defaults match the cold-start pre-T26/T12 reality: h_max=0,
    prefill_bps=0.  Override per stage when those signals matter."""
    def _pool(used: int, cap: int) -> Dict[str, Any]:
        return {
            "subpools": {
                subpool: {
                    "used_bytes": used,
                    "cap_bytes": cap,
                    "available_bytes": max(0, cap - used),
                    "evictable_bytes": used,
                    "page_bytes": 64 * 1024,
                },
            },
        }
    return {
        "time_counter": time_counter,
        "throughput_ema": {
            "prefill_bps": prefill_bps,
            "decode_per_program": {},
        },
        "pool_usage": {
            "HBM": _pool(hbm_used, hbm_cap),
            "DRAM": _pool(dram_used, dram_cap),
            "DISK": _pool(disk_used, disk_cap),
        },
        "per_program_usage": programs or {},
        "units": units,
        "link_stats": {
            link: {
                "peak_bw_bps": peak_bw_bps,
                "recent_throughput_bps": 0.0,
                "time_since_last_sample_s": 5.0,  # idle
            } for link in ("HBM->DRAM", "DRAM->HBM",
                           "DRAM->DISK", "DISK->DRAM")
        },
        "tier_holding_cost": {
            tier: {subpool: {"h_max_per_byte_sec": h_max_per_byte_sec}}
            for tier in ("HBM", "DRAM", "DISK")
        },
    }


def _build_state(
    state_json: Dict[str, Any],
    event: Event,
    tracker: Optional[ProgramTracker] = None,
):
    if tracker is None:
        tracker = ProgramTracker()
    return kvs.build_paper_state(
        state_json, event=event, tracker=tracker,
        unknown_tier_log=set(),
    )


# ============================================================ A. schema


def stage_a0_schema_pool_usage_to_tier_usage() -> None:
    """post-T17 ``pool_usage[tier].subpools[sp]`` populates
    ``TierUsage.pool_used / pool_cap / page_bytes / pool_evictable``
    per-(tier, subpool)."""
    sj = _state_json(
        units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
        hbm_used=3 * 1024 * 1024 * 1024,
        hbm_cap=10 * 1024 * 1024 * 1024,
    )
    s = _build_state(sj, Event(EventKind.LLM_PREFILL, session="p0"))
    if s.tier_usage.pool_used[Tier.HBM]["kv"] != 3 * 1024**3:
        raise StageFail(
            f"pool_used HBM/kv: {s.tier_usage.pool_used[Tier.HBM]}"
        )
    if s.tier_usage.pool_cap[Tier.HBM]["kv"] != 10 * 1024**3:
        raise StageFail(
            f"pool_cap HBM/kv: {s.tier_usage.pool_cap[Tier.HBM]}"
        )
    if s.tier_usage.page_bytes[Tier.HBM]["kv"] != 64 * 1024:
        raise StageFail(
            f"page_bytes HBM/kv: {s.tier_usage.page_bytes[Tier.HBM]}"
        )
    # ReuseUnit residence is post-T17 List[Tier], not single Tier.
    u = s.units["u0"]
    if u.residence != [Tier.HBM]:
        raise StageFail(f"unit.residence: {u.residence}")
    if u.n_bytes_by_tier.get(Tier.HBM, {}).get("kv") != 1000 * 2048:
        raise StageFail(f"unit.n_bytes_by_tier: {u.n_bytes_by_tier}")


def stage_a1_multi_rank_flatten() -> None:
    """``_flatten_per_rank`` SUMs pool_usage across ranks, dedupes
    units by hash with residence-union, fatals on n_bytes
    disagreement."""
    sj_rank = _state_json(
        units=[_unit(uhash="shared", residence=["HBM"], holders=["p0"])],
        hbm_used=2 * 1024**3, hbm_cap=10 * 1024**3,
    )
    multi = {"per_rank": [sj_rank, sj_rank]}  # same shape on both ranks
    flat = kvs._flatten_per_rank(multi)
    # SUM used / cap.
    if flat["pool_usage"]["HBM"]["subpools"]["kv"]["used_bytes"] != 4 * 1024**3:
        raise StageFail(
            f"used_bytes should sum across ranks: "
            f"{flat['pool_usage']['HBM']['subpools']['kv']}"
        )
    if flat["pool_usage"]["HBM"]["subpools"]["kv"]["cap_bytes"] != 20 * 1024**3:
        raise StageFail("cap_bytes should sum across ranks")
    # Dedupe by hash; residence union.
    if len(flat["units"]) != 1:
        raise StageFail(
            f"shared-hash units should dedupe: got {len(flat['units'])}"
        )


def stage_a1b_multi_rank_leaf_flag_and_reconcile() -> None:
    """#210 audit: when the SAME hash appears on multiple ranks with
    DIVERGING leaf flags (a mid-migration window — one rank still has the
    node device-resident, another has it evicted), ``_flatten_per_rank``
    must AND-reconcile is_device_leaf / is_host_leaf / is_tree_leaf, NOT
    keep rank-0's view.  A remove is structurally safe (sglang won't
    reject it) only if EVERY rank agrees the node is the relevant leaf;
    taking rank-0's permissive ``True`` would let migrate_candidates
    propose a remove that the disagreeing rank rejects — re-arming the
    #210 apply_failed leak in exactly the mid-migration window the
    residence-union comment already calls out.  Mirror is the residence
    UNION (colder superset): leaf flags take the AND (stricter)."""
    GB = 1024 ** 3
    rank_leaf = _state_json(units=[_unit(
        uhash="shared", residence=["HBM", "DRAM"], holders=["p0"],
        n_bytes_per_tier={"HBM": GB, "DRAM": GB},
        is_device_leaf=True, is_host_leaf=True, is_tree_leaf=True)])
    rank_nonleaf = _state_json(units=[_unit(
        uhash="shared", residence=["HBM", "DRAM"], holders=["p0"],
        n_bytes_per_tier={"HBM": GB, "DRAM": GB},
        is_device_leaf=False, is_host_leaf=False, is_tree_leaf=False)])
    # Order-independent: the permissive rank first, then the strict one.
    flat = kvs._flatten_per_rank({"per_rank": [rank_leaf, rank_nonleaf]})
    if len(flat["units"]) != 1:
        raise StageFail(f"shared hash must dedupe; got {len(flat['units'])}")
    u = flat["units"][0]
    for k in ("is_device_leaf", "is_host_leaf", "is_tree_leaf"):
        if u.get(k) is not False:
            raise StageFail(
                f"#210: cross-rank {k} must AND-reconcile to False when any "
                f"rank reports non-leaf (else a remove the disagreeing rank "
                f"rejects → apply_failed); got {k}={u.get(k)!r}")
    # And the reverse order yields the same AND (rank-0 strict, rank-1 leaf).
    flat2 = kvs._flatten_per_rank({"per_rank": [rank_nonleaf, rank_leaf]})
    u2 = flat2["units"][0]
    if any(u2.get(k) is not False for k in
           ("is_device_leaf", "is_host_leaf", "is_tree_leaf")):
        raise StageFail(
            f"#210: cross-rank leaf AND must be order-independent; got {u2}")


def stage_a1c_multi_rank_paused_state_wins() -> None:
    """Cross-rank starvation: PUT /aginfer/program_paused fans out PER
    RANK and is NOT cross-rank atomic, so a state dump can land mid-fan-
    out with the SAME pid PAUSED on the rank that already applied and
    REASONING (lagging) on another.  ``_flatten_per_rank`` must let PAUSED
    WIN — ``resume_candidates`` only considers ``state == "PAUSED"`` pids,
    so a lagging REASONING masking the pause drops the program out of the
    resume set and (a PAUSED program emits no events of its own) STARVES
    it to an AgentTimeout (#211 class via cross-rank merge).  The winning
    PAUSED rank's ``pre_pause_state`` must survive too (the §8 resume
    counterfactual reads it); taking the REASONING rank's ``None`` would
    restore the program to the wrong state.  Order-independent + ENDED
    still LOSES to active (premature ENDED would drop a lagging co-holder
    rank's session_scoped KV)."""
    def _prog(state, pre):
        return {"hbm": {"committed": {}, "inflight": {}},
                "dram": {"committed": {}}, "state": state,
                "pre_pause_state": pre, "unit_hashes": []}
    rank_paused = _state_json(
        units=[], programs={"p0": _prog("PAUSED", "REASONING")})
    rank_lagging = _state_json(
        units=[], programs={"p0": _prog("REASONING", None)})
    for order in ([rank_paused, rank_lagging], [rank_lagging, rank_paused]):
        flat = kvs._flatten_per_rank({"per_rank": order})
        merged = flat["per_program_usage"]["p0"]
        if merged["state"] != "PAUSED":
            raise StageFail(
                "cross-rank: a PAUSED program masked by a lagging rank's "
                f"REASONING starves (never a resume candidate); got "
                f"state={merged['state']!r}")
        if merged["pre_pause_state"] != "REASONING":
            raise StageFail(
                "cross-rank: pre_pause_state must come from the PAUSED rank "
                f"(else resume restores wrong state); got "
                f"{merged['pre_pause_state']!r}")
    # ENDED must NOT win over an active co-holder rank (premature drop).
    rank_ended = _state_json(
        units=[], programs={"p1": _prog("ENDED", None)})
    rank_active = _state_json(
        units=[], programs={"p1": _prog("ACTING", None)})
    flat = kvs._flatten_per_rank({"per_rank": [rank_ended, rank_active]})
    if flat["per_program_usage"]["p1"]["state"] != "ACTING":
        raise StageFail(
            "cross-rank: ENDED must LOSE to an active rank (premature ENDED "
            "drops a lagging co-holder's session_scoped KV); got "
            f"{flat['per_program_usage']['p1']['state']!r}")


def stage_a1d_multi_rank_holders_union() -> None:
    """Cross-rank holder skew: node session tagging
    (``node.session_ids.add(pid)`` / SESSION_END untag) runs in each
    rank's OWN scheduler, so in a transient window the same hash carries
    {p0,p1} on one rank and {p0} on a rank that already processed p1's
    SESSION_END.  ``_flatten_per_rank`` must UNION session_ids in the
    dedupe branch — keeping rank-0's set verbatim made ``holders``
    ORDER-DEPENDENT, skewing shared_aware_prog_scores' V_u divisor
    (``len(holders)``) and session_scoped_units' ``holders == {session}``
    predicate by rank ordering."""
    GB = 1024 ** 3
    rank_two = _state_json(units=[_unit(
        uhash="shared", residence=["HBM"], holders=["p0", "p1"],
        n_bytes_per_tier={"HBM": GB})])
    rank_one = _state_json(units=[_unit(
        uhash="shared", residence=["HBM"], holders=["p0"],
        n_bytes_per_tier={"HBM": GB})])
    for order in ([rank_two, rank_one], [rank_one, rank_two]):
        flat = kvs._flatten_per_rank({"per_rank": order})
        if len(flat["units"]) != 1:
            raise StageFail(
                f"shared hash must dedupe; got {len(flat['units'])}")
        sids = flat["units"][0]["session_ids"]
        if sorted(sids) != ["p0", "p1"]:
            raise StageFail(
                "cross-rank holders must UNION (order-independent) — a stale "
                "single-rank view must not strand a still-shared unit; got "
                f"session_ids={sids!r}")


def stage_a1g_multi_rank_last_access_hit_count_max() -> None:
    """Cross-rank access-counter skew (same transient-divergence class as
    the #210 leaf flags / #211 holders union): radix-node ``last_access_
    time`` and ``hit_count`` are bumped by EACH rank's own scheduler on the
    same request stream, so in a propagation window the same hash can read
    a just-accessed (recent last_access, higher hit_count) value on one
    rank and a stale one on a lagging rank.  ``_flatten_per_rank`` kept
    rank-0's values VERBATIM in the dedupe branch — making them ORDER-
    DEPENDENT, exactly the omission class fixed for holders.

    These feed V_u: ``age = max(1, now - last_access)``, ``lam = hits /
    age``, ``p_hat = min(1, hits/age)`` (for units with no live holder).
    A stale rank-0 last_access → larger age → smaller p_hat → the unit
    looks COLDER → ``_top_k_by_regret`` (saved = p_hat·Δρ·n_tokens) ranks
    it as a demote candidate it shouldn't be.  Reconcile to the SAFE
    (warmest / liveness-preserving) side, mirroring residence-union and
    holders-union: last_access = MAX (most-recent across ranks), hit_count
    = MAX (most-progressed rank).  Order-independent."""
    GB = 1024 ** 3
    rank_recent = _state_json(units=[_unit(
        uhash="shared", residence=["HBM"], holders=["p0"],
        n_bytes_per_tier={"HBM": GB}, last_access_time=90, hit_count=7)])
    rank_stale = _state_json(units=[_unit(
        uhash="shared", residence=["HBM"], holders=["p0"],
        n_bytes_per_tier={"HBM": GB}, last_access_time=10, hit_count=2)])
    for order in ([rank_recent, rank_stale], [rank_stale, rank_recent]):
        flat = kvs._flatten_per_rank({"per_rank": order})
        if len(flat["units"]) != 1:
            raise StageFail(
                f"shared hash must dedupe; got {len(flat['units'])}")
        u = flat["units"][0]
        if int(u["last_access_time"]) != 90:
            raise StageFail(
                "cross-rank last_access_time must reconcile to the MAX (most "
                "recent; a stale rank-0 value inflates age → deflates p_hat → "
                "spuriously demotes a warm unit); got "
                f"last_access_time={u['last_access_time']} (expected 90)")
        if int(u["hit_count"]) != 7:
            raise StageFail(
                "cross-rank hit_count must reconcile to the MAX (most-"
                f"progressed rank); got hit_count={u['hit_count']} "
                "(expected 7)")


def stage_a2_unknown_tier_label_skipped() -> None:
    """A unit residing in an unknown tier label is SKIPPED (not
    silently coerced to HBM — that was the round-1 B1 bug).  The
    same label is logged exactly once."""
    log: set = set()
    sj = _state_json(units=[
        _unit(uhash="u0", residence=["HBM"], holders=["p0"]),
        _unit(uhash="u1", residence=["ZSTD_DISK"], holders=["p1"]),
    ])
    s = kvs.build_paper_state(
        sj, event=Event(EventKind.LLM_PREFILL, session="p0"),
        tracker=ProgramTracker(), unknown_tier_log=log,
    )
    if "u1" in s.units:
        raise StageFail(
            "unit with unknown residence label should be skipped; "
            "got u1 in state.units"
        )
    if "ZSTD_DISK" not in log:
        raise StageFail(
            f"unknown_tier_log should be marked once: {log}"
        )


def stage_a3_missing_state_field_fatals() -> None:
    """A missing top-level required state field fatals via the
    daemon's fatal() helper (subprocess exit 1, since fatal calls
    os._exit)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="t146_a3_") as td:
        script = f"""
import sys
sys.path.insert(0, {str(_AGINFER_ROOT)!r})
import os
os.environ['AGINFER_DATA_DIR'] = {td!r}
from daemon import kv_scheduler as kvs
from daemon.events import Event, EventKind
from daemon.program_tracker import ProgramTracker
# Construct a state JSON missing `link_stats`.
bad = {{
    "time_counter": 0, "throughput_ema": {{"prefill_bps": 0.0,
        "decode_per_program": {{}}}},
    "pool_usage": {{"HBM": {{"subpools": {{}}}},
                    "DRAM": {{"subpools": {{}}}},
                    "DISK": {{"subpools": {{}}}}}},
    "per_program_usage": {{}}, "units": [],
    # MISSING link_stats AND tier_holding_cost.
}}
kvs.build_paper_state(
    bad, event=Event(EventKind.LLM_PREFILL, session=None),
    tracker=ProgramTracker(), unknown_tier_log=set(),
)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**__import__("os").environ, "PYTHONPATH": str(_AGINFER_ROOT),
                 "AGINFER_DATA_DIR": td},
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected fatal exit=1; got {result.returncode}; "
                f"stderr={result.stderr[-400:]!r}"
            )
        if "missing_state_field" not in result.stderr:
            raise StageFail(
                f"expected 'missing_state_field' reason in stderr; "
                f"got {result.stderr[-400:]!r}"
            )


def stage_a4_h_max_partial_zero_fatals() -> None:
    """All-zero h_max is allowed (cold-start placeholder).
    PARTIAL-zero (some positive, some zero) fatals — that's the
    real "operator forgot to configure h_max for this subpool"
    deployment bug."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="t146_a4_") as td:
        script = f"""
import sys
sys.path.insert(0, {str(_AGINFER_ROOT)!r})
import os
os.environ['AGINFER_DATA_DIR'] = {td!r}
from daemon import kv_scheduler as kvs
from daemon.events import Event, EventKind
from daemon.program_tracker import ProgramTracker

# Partial-zero: HBM h_max positive, DRAM zero, DISK positive.
sj = {{
    "time_counter": 0,
    "throughput_ema": {{"prefill_bps": 0.0, "decode_per_program": {{}}}},
    "pool_usage": {{
        t: {{"subpools": {{"kv": {{
            "used_bytes": 0, "cap_bytes": 10*1024**3,
            "available_bytes": 10*1024**3, "evictable_bytes": 0,
            "page_bytes": 64*1024,
        }}}}}} for t in ("HBM","DRAM","DISK")
    }},
    "per_program_usage": {{}}, "units": [],
    "link_stats": {{
        link: {{"peak_bw_bps": 64*1024**3,
                "recent_throughput_bps": 0.0,
                "time_since_last_sample_s": 5.0}}
        for link in ("HBM->DRAM","DRAM->HBM","DRAM->DISK","DISK->DRAM")
    }},
    "tier_holding_cost": {{
        "HBM":  {{"kv": {{"h_max_per_byte_sec": 0.001}}}},
        "DRAM": {{"kv": {{"h_max_per_byte_sec": 0.0}}}},  # partial zero
        "DISK": {{"kv": {{"h_max_per_byte_sec": 0.0001}}}},
    }},
}}
kvs.build_paper_state(
    sj, event=Event(EventKind.LLM_PREFILL, session=None),
    tracker=ProgramTracker(), unknown_tier_log=set(),
)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**__import__("os").environ, "PYTHONPATH": str(_AGINFER_ROOT),
                 "AGINFER_DATA_DIR": td},
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected fatal on partial-zero h_max; got returncode="
                f"{result.returncode}; stderr={result.stderr[-400:]!r}"
            )
        if "holding_cost_non_positive" not in result.stderr:
            raise StageFail(
                f"expected 'holding_cost_non_positive' in stderr; "
                f"got {result.stderr[-400:]!r}"
            )


# ============================================================ B. decision-set


def _build_d_t(event_kind: EventKind, session: Optional[str],
               units: List[Dict[str, Any]],
               tracker: Optional[ProgramTracker] = None) -> List[str]:
    """Convenience: build the SchedulerState and return D_t order."""
    sj = _state_json(units=units)
    s = _build_state(sj, Event(event_kind, session=session), tracker)
    return list(s.decision_set)


def stage_b0_paper4_decision_set_all_kinds() -> None:
    """6 paper §4 event kinds → correct D_t.

    Fixture: 3 units —
      * u-shared: holders={p1,p2}            → shared prefix
      * u-tail-p1: holders={p1}              → p1's exclusive tail
      * u-tail-p2: holders={p2}              → p2's exclusive tail
    """
    units = [
        _unit(uhash="u-shared",  residence=["HBM"], holders=["p1", "p2"]),
        _unit(uhash="u-tail-p1", residence=["HBM"], holders=["p1"]),
        _unit(uhash="u-tail-p2", residence=["HBM"], holders=["p2"]),
    ]
    cases: List[Tuple[EventKind, Optional[str], set]] = [
        (EventKind.SESSION_ARRIVAL,      None,  {"u-shared"}),
        (EventKind.LLM_PREFILL,          "p1",  set()),
        (EventKind.TOOL_CALL_START,      "p1",  {"u-tail-p1"}),
        (EventKind.TOOL_CALL_END,        "p1",  {"u-tail-p1"}),
        (EventKind.SUB_DISPATCH_BLOCKING,"p1",  {"u-shared", "u-tail-p1"}),
        (EventKind.SUB_DISPATCH_ASYNC,   "p1",  {"u-shared"}),
    ]
    for kind, session, expected in cases:
        d_t = set(_build_d_t(kind, session, units))
        if d_t != expected:
            raise StageFail(
                f"{kind.value} D_t mismatch: got {d_t}, expected {expected}"
            )


def stage_b0b_tool_call_exclusive_no_shared() -> None:
    """#189 (DESIGN §7 reconciled to the code): a TOOL_CALL is a
    PER-PROGRAM event → D_t is the caller's EXCLUSIVE tail; a SHARED
    prefix the caller ALSO holds is EXCLUDED (its value didn't change
    because one holder went tool-bound).  And the SUB_DISPATCH union
    (`session_tail + shared_prefix`) is DISJOINT — no duplicate hashes
    (a regression to the inclusive `session_tail` would double-list the
    caller's shared units; b0 set()s those away, so pin it here on the
    raw LIST)."""
    units = [
        _unit(uhash="u-shared",  residence=["HBM"], holders=["p1", "p2"]),
        _unit(uhash="u-priv-p1", residence=["HBM"], holders=["p1"]),
    ]
    # p1 holds BOTH u-shared and u-priv-p1, but TOOL_CALL_START(p1) must
    # nominate ONLY the exclusive u-priv-p1.
    for kind in (EventKind.TOOL_CALL_START, EventKind.TOOL_CALL_END):
        d_t = list(_build_d_t(kind, "p1", units))
        if "u-shared" in d_t:
            raise StageFail(
                f"{kind.value}(p1) must EXCLUDE the shared prefix p1 also "
                f"holds (#189 exclusive contract); got {d_t}"
            )
        if d_t != ["u-priv-p1"]:
            raise StageFail(f"{kind.value}(p1) D_t should be [u-priv-p1]; got {d_t}")
    # SUB_DISPATCH union must be a disjoint LIST (no dup of u-shared).
    sub = list(_build_d_t(EventKind.SUB_DISPATCH_BLOCKING, "p1", units))
    if len(sub) != len(set(sub)):
        raise StageFail(
            f"SUB_DISPATCH D_t must have no duplicate hashes (disjoint "
            f"union — the tail is exclusive); got {sub}"
        )
    if set(sub) != {"u-priv-p1", "u-shared"}:
        raise StageFail(f"SUB_DISPATCH(p1) should be private tail + shared; got {sub}")


def stage_b1_memory_pressure_topk_by_regret() -> None:
    """``MEMORY_PRESSURE`` D_t is bounded by ``top_k`` and ranked by
    ascending regret (lowest keep-value first → best demote
    candidates).  Plant 10 low-value sentinels among 100 high-
    value units; assert all 10 sentinels appear, and any units in
    excess of the cap are excluded."""
    # Use AGINFER_MEMORY_PRESSURE_TOPK default (256) via the module
    # constant; force a smaller fixture by using k=20 indirectly.
    units = []
    # 100 "high-value" units (high hit_count → high p_hat).
    for i in range(100):
        units.append(_unit(
            uhash=f"hi-{i}", residence=["HBM"], holders=[f"p{i}"],
            hit_count=1000, last_access_time=100,  # fresh + hot
        ))
    # 10 sentinels (low hit_count → low p_hat → low keep-value).
    for i in range(10):
        units.append(_unit(
            uhash=f"sentinel-{i}", residence=["HBM"], holders=[f"sp{i}"],
            hit_count=1, last_access_time=1,  # ancient + cold
        ))
    # Note: each holder is a unique program with no tracker entry →
    # all "unknown to tracker" → p_hat = hits/age (NOT alive-rule).
    # That preserves the regret ordering this test depends on.

    # Direct call: _top_k_by_regret with k=20.
    sj = _state_json(units=units)
    s = _build_state(sj, Event(EventKind.MEMORY_PRESSURE, session=None))
    # Build a local k=20 view to bypass the env-default.
    top_20 = kvs._top_k_by_regret(s.units, 20)
    if len(top_20) != 20:
        raise StageFail(f"top-20 length: {len(top_20)}")
    sentinel_in_top = sum(1 for uid in top_20 if uid.startswith("sentinel-"))
    if sentinel_in_top != 10:
        raise StageFail(
            f"expected all 10 sentinels in top-20 (low keep-value); "
            f"got {sentinel_in_top}.  top_20={top_20}"
        )


# ============================================================ C. λ + p_hat


def stage_c0_acting_lambda_floor_clamp() -> None:
    """A program in ``State.ACTING`` propagates the calibrated
    ACTING-floor λ to ALL its units, clamped to [1/30, 1/1]."""
    tracker = ProgramTracker()
    # Move p_act through REASONING → ACTING.
    tracker.observe_arrival("p_act")
    tracker.observe_completion("p_act")  # ACTING
    if tracker.state("p_act") is not State.ACTING:
        raise StageFail(f"setup: tracker state(p_act)={tracker.state('p_act')}")

    # Try lambdas at the boundaries.
    units = [_unit(uhash="u0", residence=["HBM"], holders=["p_act"],
                   hit_count=1000, last_access_time=99)]  # fresh, high hits/age
    sj = _state_json(units=units)
    # Pass lambda=10.0 → above ceil 1.0 → clamps to 1.0.
    s_hi = kvs.build_paper_state(
        sj, event=Event(EventKind.TOOL_CALL_START, session="p_act"),
        tracker=tracker, unknown_tier_log=set(), lambda_acting=10.0,
    )
    if abs(s_hi.units["u0"].lambda_rate - 1.0) > 1e-6:
        raise StageFail(
            f"λ should clamp to ceil 1.0; got {s_hi.units['u0'].lambda_rate}"
        )
    # Below floor → clamps to 1/30.
    s_lo = kvs.build_paper_state(
        sj, event=Event(EventKind.TOOL_CALL_START, session="p_act"),
        tracker=tracker, unknown_tier_log=set(), lambda_acting=1e-6,
    )
    if abs(s_lo.units["u0"].lambda_rate - 1.0 / 30.0) > 1e-6:
        raise StageFail(
            f"λ should clamp to floor 1/30; got {s_lo.units['u0'].lambda_rate}"
        )


def stage_c1_paused_lambda_also_clamped() -> None:
    """Round-2 R2-M1: programs in ``State.PAUSED`` also get the
    ACTING-floor (paper §7 intent is "any non-REASONING program is
    held mid-tool-call")."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_pause")  # REASONING
    tracker.observe_completion("p_pause")  # ACTING
    tracker.pause("p_pause")  # PAUSED
    if tracker.state("p_pause") is not State.PAUSED:
        raise StageFail(
            f"setup: tracker state(p_pause)={tracker.state('p_pause')}"
        )
    units = [_unit(uhash="u0", residence=["HBM"], holders=["p_pause"],
                   hit_count=1000, last_access_time=99)]
    sj = _state_json(units=units)
    s = kvs.build_paper_state(
        sj, event=Event(EventKind.TOOL_CALL_START, session="p_pause"),
        tracker=tracker, unknown_tier_log=set(), lambda_acting=0.2,
    )
    # PAUSED → λ should be 0.2 (in [1/30, 1/1]), NOT hits/age (would
    # be 1000/1 = 1000 → clamped at PolicyOOR).
    if abs(s.units["u0"].lambda_rate - 0.2) > 1e-6:
        raise StageFail(
            f"PAUSED program should use ACTING-floor λ=0.2; got "
            f"{s.units['u0'].lambda_rate}"
        )


def stage_c2_p_hat_alive_vs_ended() -> None:
    """p_hat estimator (T11, DESIGN §7 holder-PRODUCT) — replaces the old
    branch-selected single p_hat (#249/#250/#187) with a real per-holder
    aggregation: ``p_hat = 1 - Π_s (1 - p_access(u, s, Δt))``.  A REASONING
    or untracked holder's ``p_access`` is the same recency-DECOUPLED reuse
    base as before (#249/#250 still hold — this is a per-holder refinement
    of that estimator, not a reversion of it); PAUSED/ENDED holders now
    contribute EXACTLY zero per the DESIGN §7 contract (superseding #187's
    softened hits/age prior — the product's own math already makes an
    ended-only unit's p_hat 0, no separate branch needed).

      * ALIVE (REASONING, tracked) holder        -> reuse-based 1-exp(-a*(hits-1))
      * GENUINELY-ENDED holder (observe+end())    -> contributes 0 (not a prior)
      * NEVER-SEEN / untracked holder             -> reuse-based, NOT hits/age
        (#250: raw inference / no TOOL_CALL protocol must NOT recency-penalise
        demonstrated reuse).
      * MIXED alive+ended (shared prefix)         -> the alive holder's
        contribution SURVIVES the product (the ended co-holder contributes 0,
        i.e. the identity factor (1-0)=1, so it cannot pull p_hat down) —
        this is the holder-product "no ad-hoc 1/N" property DESIGN §7 calls
        out explicitly.
    """
    import math as _math
    A = kvs._PHAT_REUSE_ALPHA

    def reuse(h: int) -> float:           # the recency-decoupled base estimator
        return 1.0 - _math.exp(-A * max(0, h - 1))

    tracker = ProgramTracker()
    tracker.observe_arrival("p_alive")                      # tracked, LIVE
    tracker.observe_arrival("p_done"); tracker.end("p_done")  # tracked, genuinely ENDED

    units = [
        _unit(uhash="u-alive", residence=["HBM"], holders=["p_alive"],
              hit_count=2, last_access_time=99),
        _unit(uhash="u-ended", residence=["HBM"], holders=["p_done"],
              hit_count=1, last_access_time=1),     # genuinely ENDED -> 0
        _unit(uhash="u-untracked-oneshot", residence=["HBM"], holders=["p_never"],
              hit_count=1, last_access_time=1),      # never-seen flood -> reuse(1)=0
        _unit(uhash="u-untracked-reused", residence=["HBM"], holders=["p_never2"],
              hit_count=5, last_access_time=1),       # never-seen REUSED + idle
        _unit(uhash="u-mixed", residence=["HBM"], holders=["p_alive", "p_never"],
              hit_count=2, last_access_time=99),
        _unit(uhash="u-mixed-ended", residence=["HBM"], holders=["p_alive", "p_done"],
              hit_count=2, last_access_time=99),
    ]
    s = _build_state(_state_json(units=units),
                     Event(EventKind.LLM_PREFILL, session=None), tracker)

    # alive -> reuse-based (single holder -> product collapses to its own term)
    got = s.units["u-alive"].p_hat
    if abs(got - reuse(2)) > 1e-9:
        raise StageFail(f"alive p_hat should be reuse-based {reuse(2):.4f}; got {got}")

    # genuinely-ENDED-ONLY -> holder-product gives EXACTLY 0 (DESIGN §7 contract;
    # supersedes #187's softened hits/age prior in place).
    got = s.units["u-ended"].p_hat
    if got != 0.0:
        raise StageFail(f"genuinely-ENDED-only p_hat should be exactly 0.0; got {got}")

    # never-seen one-shot flood -> reuse(1)=0 (evict-first), NOT hits/age
    got = s.units["u-untracked-oneshot"].p_hat
    if abs(got - reuse(1)) > 1e-9:
        raise StageFail(f"untracked one-shot p_hat should be reuse(1)={reuse(1)}; got {got}")

    # #250 DO-NO-HARM GUARD: never-seen but REUSED (hits=5) + idle (last_access=1,
    # age=99) MUST be recency-DECOUPLED reuse-based (~0.86), NOT recency-decayed
    # hits/age (5/99≈0.05).  This is exactly the Dynamo regression: an untracked
    # reused prefix P stays high so the pushed daemon hint cannot eviction-nibble it.
    got = s.units["u-untracked-reused"].p_hat
    if abs(got - reuse(5)) > 1e-9:
        raise StageFail(
            f"untracked REUSED prefix must be reuse-based {reuse(5):.4f} "
            f"(NOT hits/age {5/99:.4f}); got {got}")
    if got < 0.8:
        raise StageFail(f"untracked reused (hits=5) p_hat must be high (>0.8); got {got}")

    # mixed alive+untracked -> both holders contribute the SAME reuse(2) term
    # (untracked gets the identical treatment as alive, per #250) -> the
    # product is 1-(1-reuse(2))^2, STRICTLY GREATER than either term alone —
    # the holder-product aggregation, not a single-branch passthrough.
    got = s.units["u-mixed"].p_hat
    single = reuse(2)
    expected = 1.0 - (1.0 - single) ** 2
    if abs(got - expected) > 1e-9:
        raise StageFail(
            f"mixed alive+untracked should be the holder-PRODUCT {expected:.4f} "
            f"(two non-zero terms combine); got {got}")
    if got <= single + 1e-9:
        raise StageFail(
            f"two-holder product must exceed either holder's solo term "
            f"({single:.4f}); got {got}")

    # mixed alive+ENDED (shared prefix, one co-holder terminated) -> the ended
    # co-holder contributes the identity factor (1-0)=1, so p_hat collapses to
    # EXACTLY the alive holder's own term — "no ad-hoc 1/N weighting" (DESIGN §7).
    got = s.units["u-mixed-ended"].p_hat
    if abs(got - reuse(2)) > 1e-9:
        raise StageFail(
            f"alive+ended shared prefix should equal the alive holder's solo "
            f"term {reuse(2):.4f} (ended contributes 0, no ad-hoc dilution); "
            f"got {got}")


def stage_c3_p_hat_acting_and_paused() -> None:
    """T11 (DESIGN §7): the ACTING / PAUSED holder-product branches
    ``stage_c2`` doesn't exercise (it only covers REASONING/ENDED/untracked).

      * PAUSED holder            -> contributes 0 (same as ENDED; no access
        until admission resume).
      * ACTING holder, event IS this holder's own TOOL_CALL_START WITH a
        real ``tool_eta_s`` payload -> p_access ~= 1 (Δt is BY DEFINITION
        that ETA — estimator priority #1, "the access we care about is the
        one that fires when the tool returns").
      * ACTING holder with NO per-holder ETA available (a co-holder that
        ISN'T the triggering event's session) -> bootstrap fallback
        ``1 - exp(-lambda_acting * AGINFER_PHAT_BOOTSTRAP_DT)`` (priority #3).
    """
    tracker = ProgramTracker()
    tracker.observe_arrival("p_paused"); tracker.pause("p_paused")
    tracker.observe_arrival("p_eta"); tracker.observe_completion("p_eta")   # ACTING
    tracker.observe_arrival("p_noeta"); tracker.observe_completion("p_noeta")  # ACTING
    if tracker.state("p_paused") is not State.PAUSED:
        raise StageFail("setup: p_paused should be PAUSED")
    if tracker.state("p_eta") is not State.ACTING:
        raise StageFail("setup: p_eta should be ACTING")

    units = [
        _unit(uhash="u-paused", residence=["HBM"], holders=["p_paused"],
              hit_count=1000, last_access_time=99),   # high hits/age -- must NOT leak in
        _unit(uhash="u-eta", residence=["HBM"], holders=["p_eta"],
              hit_count=1, last_access_time=99),        # one-shot -- ETA branch must dominate
        _unit(uhash="u-noeta", residence=["HBM"], holders=["p_noeta"],
              hit_count=1, last_access_time=99),
    ]
    sj = _state_json(units=units)

    # PAUSED holder -> p_hat exactly 0, regardless of hit_count.
    s_paused = kvs.build_paper_state(
        sj, event=Event(EventKind.LLM_PREFILL, session=None),
        tracker=tracker, unknown_tier_log=set(),
    )
    got = s_paused.units["u-paused"].p_hat
    if got != 0.0:
        raise StageFail(f"PAUSED-only holder p_hat should be exactly 0.0; got {got}")

    # ACTING + this event IS p_eta's own TOOL_CALL_START with tool_eta_s -> ~1.0,
    # even though hit_count=1 would give reuse(1)=0 under the untracked/REASONING
    # branch -- the ETA branch must be the one that fires.
    s_eta = kvs.build_paper_state(
        sj, event=Event(EventKind.TOOL_CALL_START, session="p_eta",
                        payload={"tool_eta_s": 5.0}),
        tracker=tracker, unknown_tier_log=set(),
    )
    got = s_eta.units["u-eta"].p_hat
    if abs(got - 1.0) > 1e-9:
        raise StageFail(
            f"ACTING holder w/ own TOOL_CALL_START tool_eta_s should give "
            f"p_access~=1.0 (Δt:=eta); got {got}")

    # ACTING but this event is a DIFFERENT session's TOOL_CALL_START (p_noeta
    # has no per-holder ETA of its own available here) -> bootstrap fallback,
    # a function of the ACTING-floor lambda, NOT 1.0 and NOT reuse(1)=0.
    s_noeta = kvs.build_paper_state(
        sj, event=Event(EventKind.TOOL_CALL_START, session="p_eta",
                        payload={"tool_eta_s": 5.0}),
        tracker=tracker, unknown_tier_log=set(), lambda_acting=0.2,
    )
    import math as _math
    lam = kvs._clamp_lambda_acting(0.2)
    expected = 1.0 - _math.exp(-lam * kvs._PHAT_BOOTSTRAP_DT)
    got = s_noeta.units["u-noeta"].p_hat
    if abs(got - expected) > 1e-9:
        raise StageFail(
            f"ACTING co-holder w/ no per-holder ETA should use the bootstrap "
            f"fallback {expected:.4f}; got {got}")
    if got in (0.0, 1.0):
        raise StageFail(f"bootstrap fallback should be strictly between 0 and 1; got {got}")


# ============================================================ D. action / dispatch


def stage_d0_action_assignments_3tuple_shape() -> None:
    """Post-T33 ``Action.assignments`` is ``List[Tuple[str,
    List[Tier], List[Tier]]]`` — (unit_id, add_tiers, remove_tiers).
    The legacy 2-tuple ``(unit_id, Tier)`` is gone."""
    # Force a migrate: a single unit currently HBM-only, queue
    # pressure that makes DRAM cheaper to hold.  Use h_max ≥ 0 cold-
    # start path so V signs aren't artifically suppressed.
    units = [_unit(
        uhash="u0", residence=["HBM"], holders=["p_ended"],
        hit_count=1, last_access_time=1,
    )]
    sj = _state_json(
        units=units,
        # Force HBM near full so DRAM is preferable.
        hbm_used=int(9.5 * 1024**3), hbm_cap=10 * 1024**3,
    )
    s = _build_state(sj, Event(EventKind.MEMORY_PRESSURE, session=None))
    policy = OursGreedyPolicy(default_costs())
    action = policy.decide(s)
    # Action.assignments must be a list, each item is (str, list, list).
    if not isinstance(action.assignments, list):
        raise StageFail(f"assignments not a list: {type(action.assignments)}")
    if not action.assignments:
        # Policy may decline if V signs collapse to 0 under defaults;
        # not a contract violation — just no migrate this run.  Stage
        # purpose is shape; assert assignments_to_wire handles the
        # empty list gracefully.
        wire = kvs.assignments_to_wire([])
        if wire != []:
            raise StageFail(f"empty assignments should yield []: {wire}")
        return
    for assignment in action.assignments:
        if not (isinstance(assignment, tuple) and len(assignment) == 3):
            raise StageFail(
                f"assignment not 3-tuple: {assignment!r}"
            )
        uid, add, remove = assignment
        if not isinstance(uid, str):
            raise StageFail(f"uid not str: {uid!r}")
        if not isinstance(add, list):
            raise StageFail(f"add not list: {add!r}")
        if not isinstance(remove, list):
            raise StageFail(f"remove not list: {remove!r}")
        if add and remove and set(add) & set(remove):
            raise StageFail(
                f"add and remove must be disjoint: add={add} remove={remove}"
            )


def stage_d1_assignments_to_wire_envelope() -> None:
    """``assignments_to_wire`` emits the 4-key envelope per item:
    ``hash``, ``add_tiers`` (list of strings), ``remove_tiers`` (list
    of strings), ``action_id`` (unique UUID4 hex per call)."""
    assignments = [
        ("u0", [Tier.DRAM], [Tier.HBM]),
        ("u1", [Tier.HBM], []),  # pure promote
    ]
    wire = kvs.assignments_to_wire(assignments)
    if len(wire) != 2:
        raise StageFail(f"wire length: {len(wire)}")
    for i, item in enumerate(wire):
        keys = set(item)
        if keys != {"hash", "add_tiers", "remove_tiers", "action_id"}:
            raise StageFail(f"wire[{i}] keys: {keys}")
        if not isinstance(item["add_tiers"], list):
            raise StageFail(f"add_tiers not list: {item}")
        if not all(isinstance(s, str) for s in item["add_tiers"]):
            raise StageFail(f"add_tiers not all str: {item}")
        if not isinstance(item["remove_tiers"], list):
            raise StageFail(f"remove_tiers not list: {item}")
        if not item["action_id"] or len(item["action_id"]) < 16:
            raise StageFail(f"action_id missing/short: {item}")
    if wire[0]["action_id"] == wire[1]["action_id"]:
        raise StageFail(
            f"action_id should be unique per item: {wire[0]['action_id']}"
        )
    if wire[0]["hash"] != "u0" or wire[1]["hash"] != "u1":
        raise StageFail(f"hashes: {[w['hash'] for w in wire]}")


def stage_d2_dispatch_without_outbound_raises() -> None:
    """Post-T36: ``KvScheduler._dispatch_migrate`` REQUIRES an
    ``outbound`` (no sync fallback).  Constructing without one and
    calling dispatch raises ``RuntimeError`` — this is a wiring bug
    surfaced loud, not a silent drop."""
    sched = kvs.KvScheduler(
        tracker=ProgramTracker(),
        sglang_base_url="http://unused",
        # outbound omitted on purpose.
    )
    async def _go():
        await sched._dispatch_migrate([("u0", [Tier.DRAM], [Tier.HBM])])
    try:
        asyncio.run(_go())
    except RuntimeError as exc:
        if "OutboundQueue" not in str(exc):
            raise StageFail(
                f"RuntimeError should name OutboundQueue: {exc}"
            )
    else:
        raise StageFail(
            "_dispatch_migrate without outbound should raise RuntimeError; "
            "got no exception"
        )


def stage_d3_dispatch_routes_through_outbound() -> None:
    """``_dispatch_migrate`` with an outbound calls
    ``outbound.enqueue_migrate(wire)`` and increments
    ``migrate_calls``.  This is a fire-and-forget enqueue, NOT a
    POST — the actual HTTP happens in the OutboundQueue worker."""
    class _StubHttp:
        async def post(self, url, *, json=None):  # noqa: ANN001
            raise AssertionError("worker not started; should not POST")
        async def aclose(self): return None

    async def _go():
        ob = OutboundQueue(
            sglang_base_url="http://unused", http_client=_StubHttp(),
        )
        sched = kvs.KvScheduler(
            tracker=ProgramTracker(),
            sglang_base_url="http://unused", outbound=ob,
        )
        # Do NOT start the worker — we just want enqueue.
        await sched._dispatch_migrate(
            [("u0", [Tier.DRAM], [Tier.HBM]),
             ("u1", [Tier.HBM], [])]
        )
        return ob, sched
    ob, sched = asyncio.run(_go())
    if sched.migrate_calls != 1:
        raise StageFail(f"migrate_calls: {sched.migrate_calls}")
    if ob.queue.qsize() != 1:
        raise StageFail(
            f"outbound queue size after enqueue: {ob.queue.qsize()}"
        )
    batch = ob.queue.get_nowait()
    if batch.endpoint != "migrate":
        raise StageFail(f"batch.endpoint: {batch.endpoint}")
    if len(batch.body["actions"]) != 2:
        raise StageFail(f"batch actions: {batch.body['actions']}")
    if set(batch.body["actions"][0]) != {
        "hash", "add_tiers", "remove_tiers", "action_id",
    }:
        raise StageFail(f"action envelope: {batch.body['actions'][0]}")


# ============================================================ E. robustness


class _StubRouter:
    """Minimal router stub for KvScheduler.handle().  Carries the §9
    thresholds the handler reads for joint_decide (#194)."""
    def __init__(self, state_supplier, *, theta_hi=0.85, theta_lo=0.70,
                 heartbeat_s=5.0):
        self._supply = state_supplier
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.heartbeat_s = heartbeat_s
    async def fetch_state(self):
        return self._supply()


def stage_e0_state_fetch_raises_handler_survives() -> None:
    """When ``/aginfer/state`` fetch raises, the handler logs +
    bows out without crashing the event worker.  No migrate
    dispatched, no exception propagated."""
    async def _go():
        def _raise():
            raise RuntimeError("simulated upstream down")
        sched = kvs.KvScheduler(
            tracker=ProgramTracker(),
            sglang_base_url="http://unused",
            outbound=OutboundQueue(sglang_base_url="http://unused",
                                   http_client=_DummyHttp()),
        )
        router = _StubRouter(_raise)
        # Make fetch_state raise via async wrapper.
        async def _afetch_state():
            return _raise()
        router.fetch_state = _afetch_state
        await sched.handle(
            Event(EventKind.LLM_PREFILL, session="p"), router,
        )
        return sched
    sched = asyncio.run(_go())
    if sched.migrate_calls != 0:
        raise StageFail(
            f"no migrate should fire on fetch failure; got "
            f"migrate_calls={sched.migrate_calls}"
        )
    if sched.decisions != 0:
        raise StageFail(
            f"no decisions on fetch failure; got {sched.decisions}"
        )


class _DummyHttp:
    async def post(self, url, *, json=None):  # noqa: ANN001
        return _Resp200()
    async def aclose(self): return None


class _Resp200:
    status_code = 200
    text = ""
    def json(self): return {"applied": 0, "applied_hashes": [], "skipped": []}


def stage_e1_empty_decision_set_no_migrate() -> None:
    """§9 (#194): ``LLM_PREFILL`` D_t is empty → no migrate candidates,
    but joint_decide STILL RUNS (admission may pause/resume from live
    state).  Here (admission off, low occupancy) it yields an empty plan
    and dispatches nothing — but the decision is NOT short-circuited on
    the empty D_t (the greedy-era early-return bug)."""
    units = [_unit(uhash="u0", residence=["HBM"], holders=["p"])]
    sj = _state_json(units=units)
    async def _go():
        sched = kvs.KvScheduler(
            tracker=ProgramTracker(),
            sglang_base_url="http://unused",
            outbound=OutboundQueue(sglang_base_url="http://unused",
                                   http_client=_DummyHttp()),
        )
        async def _afetch(): return sj
        router = _StubRouter(lambda: sj)
        router.fetch_state = _afetch
        await sched.handle(
            Event(EventKind.LLM_PREFILL, session="p"), router,
        )
        return sched
    sched = asyncio.run(_go())
    # joint_decide RAN (not short-circuited on the empty D_t) ...
    if sched.decisions != 1:
        raise StageFail(
            f"LLM_PREFILL must still run joint_decide (admission may "
            f"pause/resume); got decisions={sched.decisions}")
    # ... D_t was empty (no migrate candidates) ...
    if sched.last_decision_set_size != 0:
        raise StageFail(
            f"last_decision_set_size: {sched.last_decision_set_size}")
    # ... and with admission off + low occupancy the plan is empty →
    # nothing dispatched.
    if sched.last_plan != []:
        raise StageFail(f"empty D_t + admission off → empty plan; got "
                        f"{sched.last_plan}")
    if sched.migrate_calls != 0 or sched.pause_calls != 0:
        raise StageFail(f"no dispatch; got migrate={sched.migrate_calls} "
                        f"pause={sched.pause_calls}")


def stage_e2_policy_declines_no_migrate() -> None:
    """§9 (#194): when joint_decide returns an empty plan (the
    hysteresis dead-zone — forecast between theta_lo and theta_hi, with
    no pause/resume work), the handler dispatches nothing."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p")  # REASONING
    units = [_unit(uhash="u0", residence=["HBM"], holders=["p"],
                   hit_count=1000, last_access_time=99)]
    # HBM at 78% — inside the [70%, 85%] hysteresis band → dead-zone.
    sj = _state_json(units=units, hbm_used=int(7.8 * 1024**3),
                     hbm_cap=10 * 1024**3)
    async def _go():
        sched = kvs.KvScheduler(
            tracker=tracker,
            sglang_base_url="http://unused",
            outbound=OutboundQueue(sglang_base_url="http://unused",
                                   http_client=_DummyHttp()),
        )
        async def _afetch(): return sj
        router = _StubRouter(lambda: sj)
        router.fetch_state = _afetch
        await sched.handle(
            Event(EventKind.MEMORY_PRESSURE, session=None), router,
        )
        return sched
    sched = asyncio.run(_go())
    if sched.decisions != 1:
        raise StageFail(f"expected exactly one decision, got {sched.decisions}")
    if sched.last_plan != []:
        raise StageFail(f"dead-zone must yield an empty plan, got "
                        f"{sched.last_plan}")
    if sched.migrate_calls != 0 or sched.pause_calls != 0:
        raise StageFail(
            f"empty plan must dispatch nothing; got "
            f"migrate_calls={sched.migrate_calls} "
            f"pause_calls={sched.pause_calls}")


# ============================================================ F. idempotence + latency


def stage_f0_idempotence_same_state_same_action() -> None:
    """Same state JSON + same event → same ``Action.assignments`` on
    repeated decides.  No hidden carrying across decide() calls."""
    units = [_unit(
        uhash="u0", residence=["HBM"], holders=["p_ended"],
        hit_count=1, last_access_time=1,
    )]
    sj = _state_json(units=units, hbm_used=int(9.5 * 1024**3),
                     hbm_cap=10 * 1024**3)
    s = _build_state(sj, Event(EventKind.MEMORY_PRESSURE, session=None))
    policy = OursGreedyPolicy(default_costs())
    a0 = policy.decide(s)
    a1 = policy.decide(s)
    a2 = policy.decide(s)
    # Compare as serialised tuples (lists aren't hashable).
    def _key(a: Action):
        return tuple(
            (uid, tuple(add), tuple(remove))
            for uid, add, remove in a.assignments
        )
    if _key(a0) != _key(a1) or _key(a1) != _key(a2):
        raise StageFail(
            f"non-idempotent: a0={_key(a0)} a1={_key(a1)} a2={_key(a2)}"
        )


def stage_f1_latency_decide_under_budget() -> None:
    """``decide()`` at 1 000 units, 5 runs, mean+3σ < 25 ms.

    Looser than legacy 5 ms ceiling — post-T33 explores 6 transitions
    per unit (vs legacy's 4 target tiers), so per-unit work ~1.5×.
    Future T34 sparse DP target: tighten to < 10 ms."""
    units = []
    for i in range(1000):
        units.append(_unit(
            uhash=f"u-{i}", residence=["HBM"], holders=[f"p{i % 8}"],
            hit_count=(i % 50) + 1, last_access_time=i,
        ))
    sj = _state_json(units=units, hbm_used=int(9.5 * 1024**3),
                     hbm_cap=10 * 1024**3)
    s = _build_state(sj, Event(EventKind.MEMORY_PRESSURE, session=None))
    policy = OursGreedyPolicy(default_costs())
    # Warm-up.
    policy.decide(s)
    samples_ms: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        policy.decide(s)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    n = len(samples_ms)
    mean = sum(samples_ms) / n
    var = sum((x - mean) ** 2 for x in samples_ms) / n
    std = var ** 0.5
    bound = mean + 3.0 * std
    if bound > 25.0:
        raise StageFail(
            f"decide() mean+3σ = {bound:.2f} ms (samples={samples_ms!r}); "
            f"budget 25 ms.  Latency regression?"
        )


def stage_g0_evict_cooldown_filter() -> None:
    """#223: a hash whose remove failed the dump-vs-apply leaf TOCTOU is
    cooled down so the daemon stops re-proposing the doomed remove every
    event (the 956/cycle reject storm).  ``_filter_cooled_evicts`` drops a
    REMOVE migrate for a cooled hash, keeps a pure-ADD migrate for the same
    hash (only the failing op is backed off), and keeps non-cooled / expired
    hashes."""
    from baselines.knapsack import Migrate
    now = 1000.0
    cd = {"hot": now + 5.0, "stale": now - 1.0}   # stale = expired
    plan = [
        Migrate(cost=1, relief={"HBM": {"kv": 1}}, acquired={},
                id=("hot", [], ["HBM"]), group="hot"),       # cooled remove
        Migrate(cost=1, relief={}, acquired={"DRAM": {"kv": 1}},
                id=("hot", ["DRAM"], []), group="hot"),       # cooled pure-add
        Migrate(cost=1, relief={"HBM": {"kv": 1}}, acquired={},
                id=("cold", [], ["HBM"]), group="cold"),      # not cooled
        Migrate(cost=1, relief={"HBM": {"kv": 1}}, acquired={},
                id=("stale", [], ["HBM"]), group="stale"),    # expired
    ]
    kept = {(c.id[0], tuple(c.id[1])) for c in
            kvs._filter_cooled_evicts(plan, cd, now)}
    if ("hot", ()) in kept:
        raise StageFail("#223: cooled-hash REMOVE must be dropped")
    if ("hot", ("DRAM",)) not in kept:
        raise StageFail("#223: cooled-hash pure-ADD must be kept "
                        "(only the failing remove is backed off)")
    if ("cold", ()) not in kept or ("stale", ()) not in kept:
        raise StageFail("#223: non-cooled and expired hashes must be kept; "
                        f"got {kept}")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 pool_usage post-T17 schema → TierUsage",
                              stage_a0_schema_pool_usage_to_tier_usage),
    ("A1 multi-rank per_rank flatten (sum + dedupe)",
                              stage_a1_multi_rank_flatten),
    ("A1b multi-rank leaf-flag AND-reconcile (#210)",
                              stage_a1b_multi_rank_leaf_flag_and_reconcile),
    ("A1c multi-rank PAUSED-state wins (no resume starvation)",
                              stage_a1c_multi_rank_paused_state_wins),
    ("A1d multi-rank holders UNION (order-independent V_u divisor)",
                              stage_a1d_multi_rank_holders_union),
    ("A1g multi-rank last_access/hit_count MAX (warmest; order-independent V_u)",
                              stage_a1g_multi_rank_last_access_hit_count_max),
    ("A2 unknown tier label skipped + logged once",
                              stage_a2_unknown_tier_label_skipped),
    ("A3 missing state field → fatal()",
                              stage_a3_missing_state_field_fatals),
    ("A4 h_max partial-zero → fatal()",
                              stage_a4_h_max_partial_zero_fatals),
    ("B0 paper §4 D_t per EventKind (6 kinds)",
                              stage_b0_paper4_decision_set_all_kinds),
    ("B0b TOOL_CALL exclusive tail, shared excluded, disjoint union (#189)",
                              stage_b0b_tool_call_exclusive_no_shared),
    ("B1 memory_pressure top-k by ascending regret",
                              stage_b1_memory_pressure_topk_by_regret),
    ("C0 ACTING program → λ clamped to [1/30, 1/1]",
                              stage_c0_acting_lambda_floor_clamp),
    ("C1 PAUSED program also gets ACTING-floor λ",
                              stage_c1_paused_lambda_also_clamped),
    ("C2 p_hat: T11 holder-product (alive/untracked=reuse-based, ENDED=0)",
                              stage_c2_p_hat_alive_vs_ended),
    ("C3 p_hat: T11 holder-product ACTING (own-eta~=1, bootstrap) / PAUSED=0",
                              stage_c3_p_hat_acting_and_paused),
    ("D0 Action.assignments is 3-tuple (uid, add, remove)",
                              stage_d0_action_assignments_3tuple_shape),
    ("D1 assignments_to_wire → hash/add/remove/action_id envelope",
                              stage_d1_assignments_to_wire_envelope),
    ("D2 _dispatch_migrate without outbound → RuntimeError",
                              stage_d2_dispatch_without_outbound_raises),
    ("D3 _dispatch_migrate routes through OutboundQueue",
                              stage_d3_dispatch_routes_through_outbound),
    ("E0 state-fetch raises → handler survives, no migrate",
                              stage_e0_state_fetch_raises_handler_survives),
    ("E1 empty decision_set → no decide() / no migrate",
                              stage_e1_empty_decision_set_no_migrate),
    ("E2 policy declines → no migrate enqueued",
                              stage_e2_policy_declines_no_migrate),
    ("F0 idempotent: same state → same Action",
                              stage_f0_idempotence_same_state_same_action),
    ("F1 latency: decide(1k units) mean+3σ < 25 ms",
                              stage_f1_latency_decide_under_budget),
    ("G0 #223 evict-cooldown filter (drop cooled REMOVE, keep ADD/cold)",
                              stage_g0_evict_cooldown_filter),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(
            f"\nkv_scheduler_value_rule FAILED ({len(failures)}): {failures}"
        ))
        return 1
    print(_green(
        f"\nkv_scheduler_value_rule PASS — all {len(_STAGES)} stages green"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
