"""admission_controller — DESIGN §8 program-level candidate generator.

Rewritten for #194 (DESIGN §9 joint_decide).  Admission is no longer an
event-driven pause/resume loop composed on top of kv_scheduler — that
"Gauss-Seidel decompose" was superseded by the single ``joint_decide``.
This module is now the **program-level candidate generator** §9
consumes; this verify pins that generator surface:

  * ``shared_aware_prog_scores`` — per-program aggregate V_u (holder-
    divided) used as Pause cost / Resume gain.
  * ``forecast`` / ``forecast_horizon`` / ``forecast_inflight_demand``
    — the §9 pressure / headroom trigger input (per-HBM-subpool).
  * ``marginal_pause_cost`` / ``pause_relief`` — Pause cost + relief
    components (DESIGN §8).
  * ``pause_candidates`` — one Pause per REASONING/ACTING program.
  * ``capacity_fits`` / ``resume_candidates`` — one Resume per PAUSED
    program that fits.

These are pure functions over a ``SchedulerState`` (built from the
post-T17 ``/aginfer/state`` schema); no server, no event loop.  The
live wiring (joint_decide selection + dispatch) is pinned by
``verify/joint_decide`` + ``verify/integration_stress``.

Run::

    cd /scratch/yuzhou/projects/sglang
    python dev/aginfer/verify/admission_controller/verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import Tier  # noqa: E402
from daemon import admission_controller as adm  # noqa: E402
from daemon.events import Event, EventKind  # noqa: E402
from daemon.kv_scheduler import build_paper_state  # noqa: E402
from daemon.program_tracker import ProgramTracker  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---------------------------------------------------------------- fixtures

GB = 1024 ** 3
MB = 1024 ** 2


def _unit(*, uhash, residence, holders, n_tokens=1000,
          n_bytes_per_tier=None, last_access_time=0, hit_count=1,
          subpool="kv") -> Dict[str, Any]:
    if n_bytes_per_tier is None:
        nb = {t: {subpool: n_tokens * 2048} for t in residence}
    else:
        nb = {t: (v if isinstance(v, dict) else {subpool: int(v)})
              for t, v in n_bytes_per_tier.items()}
    return {
        "hash": uhash, "residence": list(residence), "n_tokens": n_tokens,
        "n_bytes": nb, "last_access_time": last_access_time,
        "hit_count": hit_count, "session_ids": list(holders),
    }


def _program(state="REASONING", *, inflight=None, committed=None,
             unit_hashes=None, pre_pause_state=None,
             expected_remaining_tokens=None) -> Dict[str, Any]:
    pu = {
        "state": state, "pre_pause_state": pre_pause_state,
        "hbm": {"committed": committed or {}, "inflight": inflight or {}},
        "dram": {"committed": {}}, "unit_hashes": unit_hashes or [],
    }
    if expected_remaining_tokens is not None:
        pu["expected_remaining_tokens"] = expected_remaining_tokens
    return pu


def _sp(used, cap, page=64 * 1024, decode_bpt=None) -> Dict[str, int]:
    e = {"used_bytes": used, "cap_bytes": cap,
         "available_bytes": max(0, cap - used),
         "evictable_bytes": used, "page_bytes": page}
    if decode_bpt is not None:
        e["decode_bytes_per_token"] = decode_bpt
    return e


def _state_json(*, units, programs=None, hbm=None, dram=None, disk=None,
                prefill_bps=0.0, decode_per_program=None,
                time_counter=100) -> Dict[str, Any]:
    hbm = hbm or {"kv": _sp(1 * GB, 10 * GB)}
    dram = dram or {"kv": _sp(1 * GB, 40 * GB)}
    disk = disk or {"kv": _sp(0, 200 * GB)}
    return {
        "time_counter": time_counter,
        "throughput_ema": {"prefill_bps": prefill_bps,
                           "decode_per_program": decode_per_program or {}},
        "pool_usage": {"HBM": {"subpools": hbm},
                       "DRAM": {"subpools": dram},
                       "DISK": {"subpools": disk}},
        "per_program_usage": programs or {},
        "units": units,
        "link_stats": {link: {"peak_bw_bps": 64 * GB,
                              "recent_throughput_bps": 0.0,
                              "time_since_last_sample_s": 5.0}
                       for link in ("HBM->DRAM", "DRAM->HBM", "DRAM->DISK",
                                    "DISK->DRAM", "HBM->DISK", "DISK->HBM")},
        "tier_holding_cost": {t: {"kv": {"h_max_per_byte_sec": 0.0}}
                              for t in ("HBM", "DRAM", "DISK")},
    }


def _build(sj, tracker, event):
    return build_paper_state(sj, event=event, tracker=tracker,
                             unknown_tier_log=set())


# ---------------------------------------------------------------- stages


def stage_scoring() -> None:
    """shared_aware_prog_scores: each unit's V_u split across holders, so
    a shared prefix doesn't double-count across the programs holding it."""
    tracker = ProgramTracker()
    tracker.observe_arrival("A")
    tracker.observe_arrival("B")
    # u-shared held by A,B (each gets half); u-tail-A held by A only.
    sj = _state_json(units=[
        _unit(uhash="u-shared", residence=["HBM"], holders=["A", "B"],
              n_tokens=2000, hit_count=40),
        _unit(uhash="u-tail-A", residence=["HBM"], holders=["A"],
              n_tokens=4000, hit_count=8),
    ])
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session=None)
    st = _build(sj, tracker, ev)
    scores = adm.shared_aware_prog_scores(st)
    if set(scores) != {"A", "B"}:
        raise StageFail(f"scoring: expected programs A,B; got {set(scores)}")
    # B holds only half of u-shared; A holds half of u-shared + all of
    # its tail → A's aggregate strictly exceeds B's.
    if not (scores["A"] > scores["B"]):
        raise StageFail(f"scoring: A (shared/2 + tail) must exceed B "
                        f"(shared/2): {scores}")
    # Degenerate: a unit with one holder contributes its full V_u.
    sj1 = _state_json(units=[
        _unit(uhash="solo", residence=["HBM"], holders=["A"], hit_count=5)])
    st1 = _build(sj1, tracker, ev)
    from baselines.costs import default_costs
    from daemon.admission_controller import _value_at_current_tier
    full = _value_at_current_tier(st1.units["solo"], st1, default_costs(), 1e-4)
    if abs(adm.shared_aware_prog_scores(st1)["A"] - full) > 1e-12:
        raise StageFail("scoring: single-holder share must equal full V_u")
    print(_green("  [scoring] shared-aware V_u aggregation (holder split) OK"))


def stage_forecast() -> None:
    """forecast = per-HBM-subpool used_bytes (+ inflight term, 0 under
    the T26/T11 placeholders); horizon falls back to heartbeat_s."""
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    sj = _state_json(
        units=[_unit(uhash="u", residence=["HBM"], holders=["S"])],
        hbm={"full": _sp(3 * GB, 10 * GB), "mamba": _sp(8 * GB, 9 * GB)})
    st = _build(sj, tracker, Event(kind=EventKind.MEMORY_PRESSURE, session="S"))
    fc = adm.forecast(st, heartbeat_s=5.0)
    if fc != {"full": float(3 * GB), "mamba": float(8 * GB)}:
        raise StageFail(f"forecast must equal per-subpool used_bytes: {fc}")
    if adm.forecast_horizon(st, 5.0) != 5.0:
        raise StageFail("forecast_horizon must fall back to heartbeat_s")
    if adm.forecast_inflight_demand(st, 5.0):
        raise StageFail("inflight demand must be 0 pre-T26/T11")
    # decode_per_program populated but bytes/E[rem] still unwired → still 0.
    sj2 = _state_json(
        units=[_unit(uhash="u", residence=["HBM"], holders=["S"])],
        decode_per_program={"S": 1000.0})
    st2 = _build(sj2, tracker, Event(kind=EventKind.MEMORY_PRESSURE, session="S"))
    if adm.forecast_inflight_demand(st2, 5.0):
        raise StageFail("inflight demand must stay 0 until all inputs wired")
    print(_green("  [forecast] per-subpool used_bytes; horizon; inflight=0 OK"))


def stage_pause_cost_relief() -> None:
    """marginal_pause_cost (0 while prefill_bps=0) + pause_relief
    (inflight + committed snapshot)."""
    pu = _program("REASONING", inflight={"kv": 5 * MB, "mamba": 2 * MB},
                  committed={"kv": 3 * MB})
    if adm.marginal_pause_cost(pu, prefill_bps=0.0) != 0.0:
        raise StageFail("marginal_pause_cost must be 0 while prefill_bps=0")
    # with a measured prefill rate: inflight bytes / rate.
    mc = adm.marginal_pause_cost(pu, prefill_bps=7.0 * MB)
    if abs(mc - (7 * MB) / (7.0 * MB)) > 1e-9:
        raise StageFail(f"marginal_pause_cost = Σinflight/prefill_bps; got {mc}")
    relief = adm.pause_relief(pu)
    if relief != {"kv": 8 * MB, "mamba": 2 * MB}:
        raise StageFail(f"pause_relief = inflight+committed per sp; got {relief}")
    print(_green("  [pause-cost] marginal_pause_cost + pause_relief OK"))


def stage_pause_candidates() -> None:
    """pause_candidates: one Pause per REASONING/ACTING program; PAUSED
    + ENDED skipped; cost = V_u_program (+marginal, 0 here)."""
    tracker = ProgramTracker()
    for p in ("A", "B", "C", "D"):
        tracker.observe_arrival(p)
    tracker.observe_completion("B")  # ACTING
    tracker.pause("C")               # PAUSED
    tracker.end("D")                 # ENDED
    programs = {
        "A": _program("REASONING", inflight={"kv": 4 * MB}, unit_hashes=["uA"]),
        "B": _program("ACTING", committed={"kv": 2 * MB}, unit_hashes=["uB"]),
        "C": _program("PAUSED", inflight={"kv": 1 * MB}, unit_hashes=["uC"]),
        "D": _program("ENDED", unit_hashes=["uD"]),
    }
    sj = _state_json(
        units=[_unit(uhash=f"u{p}", residence=["HBM"], holders=[p])
               for p in ("A", "B", "C", "D")],
        programs=programs)
    st = _build(sj, tracker, Event(kind=EventKind.MEMORY_PRESSURE, session=None))
    pcs = adm.pause_candidates(st)
    if {p.pid for p in pcs} != {"A", "B"}:
        raise StageFail(f"pause_candidates must be REASONING+ACTING only "
                        f"(PAUSED/ENDED skipped); got {[p.pid for p in pcs]}")
    vprog = adm.shared_aware_prog_scores(st)
    pa = next(p for p in pcs if p.pid == "A")
    if abs(pa.cost - vprog["A"]) > 1e-12:
        raise StageFail("pause cost must equal V_u_program (marginal=0)")
    if pa.relief != {"kv": 4 * MB}:
        raise StageFail(f"A relief = inflight 4MB; got {pa.relief}")
    print(_green("  [pause-cands] REASONING/ACTING only; cost+relief OK"))


def stage_resume_candidates() -> None:
    """resume_candidates: PAUSED only; gain = V_u_program; re_use from
    expected_peak_hbm_after_resume; capacity_fits gates overflow."""
    tracker = ProgramTracker()
    tracker.pause("P")
    tracker.observe_arrival("R")
    sj = _state_json(
        units=[
            _unit(uhash="uP", residence=["DRAM"], holders=["P"],
                  n_bytes_per_tier={"DRAM": 1 * GB}),
            _unit(uhash="uR", residence=["HBM"], holders=["R"]),
        ],
        programs={"P": _program("PAUSED", unit_hashes=["uP"],
                                pre_pause_state="REASONING"),
                  "R": _program("REASONING", unit_hashes=["uR"])},
        hbm={"kv": _sp(1 * GB, 10 * GB)})
    st = _build(sj, tracker, Event(kind=EventKind.PRESSURE_RESOLVED, session="P"))
    rcs = adm.resume_candidates(st, heartbeat_s=5.0, theta_hi=0.85)
    if {r.pid for r in rcs} != {"P"}:
        raise StageFail(f"resume_candidates must be PAUSED only; "
                        f"got {[r.pid for r in rcs]}")
    if rcs[0].re_use.get("kv", 0) != 1 * GB:
        raise StageFail(f"P re_use = its DRAM-resident bytes (1GB); "
                        f"got {rcs[0].re_use}")
    # capacity_fits: HBM near cap → resume would overflow → omitted.
    sj_full = _state_json(
        units=[_unit(uhash="uP", residence=["DRAM"], holders=["P"],
                     n_bytes_per_tier={"DRAM": 2 * GB})],
        programs={"P": _program("PAUSED", unit_hashes=["uP"],
                                pre_pause_state="REASONING")},
        hbm={"kv": _sp(8 * GB, 10 * GB)})  # 8GB + 2GB re_use > 0.85*10GB
    st_full = _build(sj_full, tracker,
                     Event(kind=EventKind.PRESSURE_RESOLVED, session="P"))
    if adm.resume_candidates(st_full, heartbeat_s=5.0, theta_hi=0.85):
        raise StageFail("capacity_fits must omit a Resume that overflows "
                        "theta_hi")
    print(_green("  [resume-cands] PAUSED only; gain/re_use; capacity_fits OK"))


def stage_trajectory() -> None:
    """#199: the §8 forecast trajectory term, assembled from the now-
    exposed decode_bytes_per_token + synthetic decode_throughput +
    E[remaining].  Proves the full product activates correctly, stays 0
    under any missing input (T26/T11 degradation), and zeroes Mamba."""
    DBPT = 2048   # bytes/token for the attention subpool
    tracker = ProgramTracker()
    tracker.observe_arrival("A")

    def _st(*, decode=None, e_rem=1e9, dbpt=DBPT, inflight=None):
        sj = _state_json(
            units=[_unit(uhash="uA", residence=["HBM"], holders=["A"])],
            programs={"A": _program("REASONING",
                                    inflight=inflight if inflight is not None
                                    else {"full": 5 * MB},
                                    unit_hashes=["uA"],
                                    expected_remaining_tokens=e_rem)},
            hbm={"full": _sp(1 * GB, 10 * GB, decode_bpt=dbpt)},
            decode_per_program=decode or {})
        return _build(sj, tracker,
                      Event(kind=EventKind.MEMORY_PRESSURE, session="A"))

    # --- full activation: growth = min(E, horizon×dt) × decode_bpt ---
    # horizon = heartbeat 5s, dt = 100 tok/s → horizon×dt = 500 tok;
    # E=1e9 huge → min = 500 → demand = 500 × 2048 = 1,024,000 B.
    st = _st(decode={"A": 100.0})
    dem = adm.forecast_inflight_demand(st, horizon_s=5.0)
    if dem != {"full": 500.0 * DBPT}:
        raise StageFail(f"trajectory: demand must be 500×{DBPT}; got {dem}")
    fc = adm.forecast(st, heartbeat_s=5.0)
    if fc["full"] != float(1 * GB) + 500.0 * DBPT:
        raise StageFail(f"trajectory: forecast must add the demand; got {fc}")
    # pause_relief future_inflight_savings: inflight 5MB + growth.
    pcs = adm.pause_candidates(st, heartbeat_s=5.0)
    pa = next(p for p in pcs if p.pid == "A")
    if pa.relief.get("full") != 5 * MB + int(500.0 * DBPT):
        raise StageFail(f"trajectory: pause_relief must add future_inflight_"
                        f"savings; got {pa.relief}")

    # --- E[remaining] is the binding side of the min when small ---
    st2 = _st(decode={"A": 100.0}, e_rem=10)   # min(10, 500) = 10
    if adm.forecast_inflight_demand(st2, 5.0) != {"full": 10.0 * DBPT}:
        raise StageFail("trajectory: min must pick the smaller E[remaining]")

    # --- gating: each missing input → 0 ---
    if adm.forecast_inflight_demand(_st(decode={}), 5.0) != {}:
        raise StageFail("trajectory: no decode_throughput (T26) → 0")
    st_noe = _st(decode={"A": 100.0}, e_rem=None)
    if adm.forecast_inflight_demand(st_noe, 5.0) != {}:
        raise StageFail("trajectory: no E[remaining] (T11) → 0 (no bootstrap "
                        "over-forecast)")
    if adm.forecast_inflight_demand(_st(decode={"A": 100.0}, dbpt=0), 5.0) != {}:
        raise StageFail("trajectory: Mamba/snapshot (decode_bpt=0) → 0 growth")
    # inflight[sp]==0 → that subpool excluded
    st_idle = _st(decode={"A": 100.0}, inflight={"full": 0})
    if adm.forecast_inflight_demand(st_idle, 5.0) != {}:
        raise StageFail("trajectory: inflight[sp]==0 → no projected growth")

    # --- malformed inputs must NOT crash or poison (#199 audit) ---
    for bad in (float("nan"), float("inf"), -5.0, "oops", None):
        st_bad = _st(decode={"A": 100.0}, e_rem=bad)
        try:
            dem_bad = adm.forecast_inflight_demand(st_bad, 5.0)
        except Exception as e:  # noqa: BLE001
            raise StageFail(f"trajectory: e_rem={bad!r} must not raise, got {e!r}")
        if dem_bad != {}:
            raise StageFail(f"trajectory: malformed e_rem={bad!r} must yield 0, "
                            f"got {dem_bad}")
        try:
            adm.pause_candidates(st_bad, heartbeat_s=5.0)  # must not crash
        except Exception as e:  # noqa: BLE001
            raise StageFail(f"trajectory: pause_candidates must survive e_rem="
                            f"{bad!r}, got {e!r}")
    # malformed decode_throughput too
    for bad_dt in (float("nan"), float("inf"), -3.0, "x"):
        st_bd = _st(decode={"A": bad_dt})
        if adm.forecast_inflight_demand(st_bd, 5.0) != {}:
            raise StageFail(f"trajectory: malformed decode={bad_dt!r} must yield 0")

    # --- Σ over multiple programs + full(nonzero)+mamba(0) in ONE state ---
    tracker.observe_arrival("B")
    sj_multi = _state_json(
        units=[_unit(uhash="uA", residence=["HBM"], holders=["A"]),
               _unit(uhash="uB", residence=["HBM"], holders=["B"])],
        programs={
            "A": _program("REASONING", inflight={"full": 1 * MB, "mamba": 9 * MB},
                          committed={"full": 2 * MB}, unit_hashes=["uA"],
                          expected_remaining_tokens=1e9),
            "B": _program("REASONING", inflight={"full": 1 * MB},
                          unit_hashes=["uB"], expected_remaining_tokens=1e9),
        },
        hbm={"full": _sp(1 * GB, 10 * GB, decode_bpt=DBPT),
             "mamba": _sp(1 * GB, 9 * GB, decode_bpt=0)},  # Mamba: 0 per-token
        decode_per_program={"A": 100.0, "B": 100.0})
    st_m = _build(sj_multi, tracker,
                  Event(kind=EventKind.MEMORY_PRESSURE, session="A"))
    dem_m = adm.forecast_inflight_demand(st_m, horizon_s=5.0)
    # full: (A growth 500×DBPT) + (B growth 500×DBPT) = 1000×DBPT; mamba: 0
    # (decode_bpt=0) even though A has 9MB mamba inflight.
    if dem_m != {"full": 1000.0 * DBPT}:
        raise StageFail(f"trajectory: Σ over A+B on full, Mamba excluded; "
                        f"got {dem_m}")
    # pause_relief for A: snapshot (inflight 1MB+9MB + committed 2MB) +
    # future-savings (full 500×DBPT; mamba 0).
    pcs_m = adm.pause_candidates(st_m, heartbeat_s=5.0)
    pa_m = next(p for p in pcs_m if p.pid == "A")
    if pa_m.relief.get("full") != 1 * MB + 2 * MB + int(500.0 * DBPT):
        raise StageFail(f"trajectory: A full relief = inflight+committed+growth; "
                        f"got {pa_m.relief}")
    if pa_m.relief.get("mamba") != 9 * MB:  # snapshot only; no per-token growth
        raise StageFail(f"trajectory: A mamba relief = snapshot 9MB (no growth); "
                        f"got {pa_m.relief}")
    print(_green("  [trajectory] §8 demand/future_inflight_savings + full "
                 "T26/T11/Mamba gating + Σ-over-programs + malformed-input OK"))


_STAGES = [
    ("scoring", stage_scoring),
    ("forecast", stage_forecast),
    ("pause-cost", stage_pause_cost_relief),
    ("pause-cands", stage_pause_candidates),
    ("resume-cands", stage_resume_candidates),
    ("trajectory", stage_trajectory),
]


def main() -> int:
    print("=" * 64)
    print("admission_controller — DESIGN §8 candidate generator (#194)")
    print("=" * 64)
    failed = []
    for name, fn in _STAGES:
        try:
            fn()
        except StageFail as e:
            failed.append(name)
            print(_red(f"  [{name}] FAIL: {e}"))
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            import traceback
            print(_red(f"  [{name}] ERROR: {e}"))
            traceback.print_exc()
    print("=" * 64)
    if failed:
        print(_red(f"FAILED: {', '.join(failed)}"))
        return 1
    print(_green("admission_controller PASS — all 6 §8 stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
