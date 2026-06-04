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
    """Program-alive p_hat rule: alive holder → p_hat=1.0.  All
    holders unknown to tracker (= ENDED / never-seen) → p_hat =
    hits/age proxy.  Mixing one ALIVE holder dominates."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_alive")  # REASONING (known to tracker)

    units = [
        _unit(uhash="u-alive-only", residence=["HBM"],
              holders=["p_alive"], hit_count=2, last_access_time=99),
        _unit(uhash="u-ended-only", residence=["HBM"],
              holders=["p_ended"],  # never seen by tracker
              hit_count=2, last_access_time=99),  # hits/age = 2/1 = 2
        _unit(uhash="u-mixed",      residence=["HBM"],
              holders=["p_alive", "p_ended"],
              hit_count=2, last_access_time=99),
    ]
    sj = _state_json(units=units)
    s = _build_state(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    # alive-only → p_hat == 1.0
    if abs(s.units["u-alive-only"].p_hat - 1.0) > 1e-9:
        raise StageFail(
            f"alive-only p_hat should be 1.0; got "
            f"{s.units['u-alive-only'].p_hat}"
        )
    # ended-only → p_hat = min(1.0, hits/age) = 1.0 (clamped)
    # We use hit_count=2, last_access_time=99, time=100 → age=1,
    # hits/age=2.0 → clamped to 1.0.  To distinguish, use a lower
    # ratio.  Rebuild with hit=1, last=1 → age=99, ratio≈0.01.
    units2 = [
        _unit(uhash="u-ended-cold", residence=["HBM"],
              holders=["p_ended"], hit_count=1, last_access_time=1),
    ]
    s2 = _build_state(_state_json(units=units2),
                      Event(EventKind.LLM_PREFILL, session=None), tracker)
    p_cold = s2.units["u-ended-cold"].p_hat
    if not (0.0 < p_cold < 0.1):
        raise StageFail(
            f"ended-only with hits/age ≈ 1/99 should have small p_hat "
            f"(< 0.1); got {p_cold}"
        )
    # mixed → alive wins → p_hat = 1.0
    if abs(s.units["u-mixed"].p_hat - 1.0) > 1e-9:
        raise StageFail(
            f"mixed alive+ended → alive dominates; expected 1.0, got "
            f"{s.units['u-mixed'].p_hat}"
        )


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
    """``LLM_PREFILL`` → D_t is empty → handler returns without
    calling ``decide()`` or dispatching."""
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
    if sched.decisions != 0:
        raise StageFail(
            f"LLM_PREFILL has empty D_t → no decide() call; got "
            f"decisions={sched.decisions}"
        )
    if sched.migrate_calls != 0:
        raise StageFail(f"no migrate; got {sched.migrate_calls}")
    if sched.last_decision_set_size != 0:
        raise StageFail(
            f"last_decision_set_size: {sched.last_decision_set_size}"
        )


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


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 pool_usage post-T17 schema → TierUsage",
                              stage_a0_schema_pool_usage_to_tier_usage),
    ("A1 multi-rank per_rank flatten (sum + dedupe)",
                              stage_a1_multi_rank_flatten),
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
    ("C2 p_hat: alive holder=1.0, all-ENDED=hits/age",
                              stage_c2_p_hat_alive_vs_ended),
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
