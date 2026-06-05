"""verify/joint_decide (#194, DESIGN §7/§8/§9): wire the T34 multi-axis
DP into the live decision path.

#194 replaces two separate, sequential decision modules —
``OursGreedyPolicy.decide`` (per-unit greedy, single-axis capacity) +
the admission ``_on_pressure`` / ``_on_resolved`` loops (the
"Gauss-Seidel decompose" DESIGN §9 supersedes) — with ONE
``joint_decide(state, event)`` that:

  * generates unit-level Migrate candidates (§7 ``migrate_candidates``),
  * generates program-level Pause / Resume candidates (§8
    ``pause_candidates`` / ``resume_candidates``),
  * computes the per-HBM-subpool ``forecast`` and the
    ``bytes_needed`` / ``free_room`` thresholds (§8 / §9),
  * runs the pressure-phase ``knapsack_min_cost_multi`` OR the
    headroom-phase ``knapsack_max_value_multi`` (mutually exclusive),
  * returns the chosen mixed plan for the live handler to dispatch.

Stages (TDD — each builds a fixture, asserts the contract):

  A. migrate_candidates  — §7 generator: cost / relief / acquired,
                           transition enumeration, relief>0 filter
  B. forecast            — §8 per-HBM-subpool forecast (degrades to
                           used_bytes under the T26/T11 placeholders)
  C. pause/resume_cands  — §8 program generators (cost/relief, gain/re_use)
  D. joint_decide select — §9 pressure vs headroom vs dead-zone;
                           pressure-suppresses-headroom; LLM_PREFILL runs
  E. joint_decide DP      — exact vs brute-force oracle; no same-unit
                           double-count; Infeasible/BudgetExceeded→fatal
  F. live wiring         — handler dispatches the mixed plan

Usage:
    python dev/aginfer/verify/joint_decide/verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import Tier  # noqa: E402
from baselines.costs import default_costs  # noqa: E402
from baselines.ours_greedy import migrate_candidates  # noqa: E402
from baselines.knapsack import Migrate, Pause, Resume  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon import admission_controller as adm  # noqa: E402
from daemon.events import Event, EventKind  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from daemon.outbound import OutboundQueue  # noqa: E402
import asyncio  # noqa: E402


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
    n_bytes_per_tier: Optional[Dict[str, Dict[str, int]]] = None,
    last_access_time: int = 0,
    hit_count: int = 1,
    subpool: str = "kv",
) -> Dict[str, Any]:
    """Synthetic post-T17 unit JSON.  ``n_bytes_per_tier`` may be a flat
    ``{tier: bytes}`` (single subpool) or a nested ``{tier: {sp: bytes}}``."""
    if n_bytes_per_tier is None:
        n_bytes = {t: {subpool: n_tokens * 2048} for t in residence}
    else:
        n_bytes = {}
        for t, v in n_bytes_per_tier.items():
            n_bytes[t] = v if isinstance(v, dict) else {subpool: int(v)}
    return {
        "hash": uhash,
        "residence": list(residence),
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "last_access_time": last_access_time,
        "hit_count": hit_count,
        "session_ids": list(holders),
    }


def _pool(subpools: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    return {"subpools": subpools}


def _sp(used: int, cap: int, page: int = 64 * 1024) -> Dict[str, int]:
    return {
        "used_bytes": used,
        "cap_bytes": cap,
        "available_bytes": max(0, cap - used),
        "evictable_bytes": used,
        "page_bytes": page,
    }


def _state_json(
    *,
    units: List[Dict[str, Any]],
    programs: Optional[Dict[str, Dict[str, Any]]] = None,
    hbm: Optional[Dict[str, Dict[str, int]]] = None,
    dram: Optional[Dict[str, Dict[str, int]]] = None,
    disk: Optional[Dict[str, Dict[str, int]]] = None,
    time_counter: int = 100,
    prefill_bps: float = 0.0,
    decode_per_program: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    GB = 1024 ** 3
    hbm = hbm or {"kv": _sp(1 * GB, 10 * GB)}
    dram = dram or {"kv": _sp(1 * GB, 40 * GB)}
    disk = disk or {"kv": _sp(0, 200 * GB)}
    return {
        "time_counter": time_counter,
        "throughput_ema": {
            "prefill_bps": prefill_bps,
            "decode_per_program": decode_per_program or {},
        },
        "pool_usage": {
            "HBM": _pool(hbm),
            "DRAM": _pool(dram),
            "DISK": _pool(disk),
        },
        "per_program_usage": programs or {},
        "units": units,
        "link_stats": {
            link: {
                "peak_bw_bps": 64 * GB,
                "recent_throughput_bps": 0.0,
                "time_since_last_sample_s": 5.0,  # idle → full bw_free
            } for link in ("HBM->DRAM", "DRAM->HBM",
                           "DRAM->DISK", "DISK->DRAM",
                           "HBM->DISK", "DISK->HBM")
        },
        "tier_holding_cost": {
            tier: {"kv": {"h_max_per_byte_sec": 0.0}}
            for tier in ("HBM", "DRAM", "DISK")
        },
    }


def _build_state(state_json, tracker, event):
    return kvs.build_paper_state(
        state_json, event=event, tracker=tracker, unknown_tier_log=set())


# ============================================================ Stage A


def stage_a_migrate_candidates() -> None:
    """§7 migrate_candidates: per-unit transitions → Migrate(cost, relief,
    acquired), with the relief>0 filter and the (uid, add, remove) id."""
    tracker = ProgramTracker()
    tracker.observe_arrival("S")  # REASONING → p_hat=1 (alive holder)
    GB = 1024 ** 3
    nb = 2_000_000  # bytes per tier for the unit
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": nb, "DRAM": nb})],
        hbm={"kv": _sp(5 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    cands = migrate_candidates(st, ["u1"], default_costs())

    # {HBM,DRAM} → meaningful transitions with relief>0:
    #   evict HBM  (remove HBM)        relief={HBM}
    #   drop DRAM  (remove DRAM)       relief={DRAM}
    #   DROP       (remove HBM,DRAM)   relief={HBM,DRAM}
    # The write-through-style no-op edits / relief-empty ones are dropped.
    if len(cands) != 3:
        raise StageFail(
            f"A: expected 3 relief-bearing candidates for {{HBM,DRAM}}, "
            f"got {len(cands)}: {[c.id for c in cands]}")
    by_remove = {tuple(sorted(t.name for t in c.id[2])): c for c in cands}
    if set(by_remove) != {("HBM",), ("DRAM",), ("DRAM", "HBM")}:
        raise StageFail(f"A: unexpected remove-sets {set(by_remove)}")

    evict_hbm = by_remove[("HBM",)]
    if set(evict_hbm.relief) != {"HBM"} or evict_hbm.relief["HBM"]["kv"] != nb:
        raise StageFail(f"A: evict-HBM relief wrong: {evict_hbm.relief}")
    if evict_hbm.acquired:
        raise StageFail(
            f"A: evict-HBM should acquire nothing (DRAM already resident), "
            f"got {evict_hbm.acquired}")

    drop = by_remove[("DRAM", "HBM")]
    if set(drop.relief) != {"HBM", "DRAM"}:
        raise StageFail(f"A: DROP relief should free both tiers: {drop.relief}")
    drop_dram = by_remove[("DRAM",)]
    # Calibration-robust cost relationships (signs depend on h_base, but
    # these orderings hold for any positive cost calibration):
    #   * drop-DRAM keeps HBM authoritative and frees only the DRAM
    #     backup → V(current)=V({HBM}) → cost is EXACTLY 0.
    #   * DROP forgoes the saved-prefill that evict-to-DRAM retains
    #     (V({DRAM})>0 with a live holder) → DROP.cost > evictHBM.cost.
    if abs(drop_dram.cost) > 1e-12:
        raise StageFail(
            f"A: drop-DRAM (keep HBM) must cost exactly 0, got {drop_dram.cost!r}")
    if not (drop.cost > evict_hbm.cost):
        raise StageFail(
            f"A: DROP must cost more than evict→DRAM (it forgoes the "
            f"retained saved-prefill): DROP={drop.cost:.4g} "
            f"evictHBM={evict_hbm.cost:.4g}")

    # relief>0 filter: a unit on {HBM} only still yields evict→DRAM /
    # write-through(add DRAM, relief empty → dropped) / DROP.
    sj2 = _state_json(
        units=[_unit(uhash="u2", residence=["HBM"], holders=["S"])],
        hbm={"kv": _sp(5 * GB, 10 * GB)})
    st2 = _build_state(sj2, tracker, ev)
    c2 = migrate_candidates(st2, ["u2"], default_costs())
    # {HBM}: (add DRAM, keep HBM) relief={} DROPPED; (add DRAM, remove HBM)
    # relief={HBM}; (remove HBM → DROP) relief={HBM}. → 2 candidates.
    if len(c2) != 2:
        raise StageFail(
            f"A: {{HBM}} expected 2 relief-bearing candidates (write-through "
            f"is relief-empty → filtered), got {len(c2)}: {[c.id for c in c2]}")
    # the evict→DRAM candidate acquires DRAM bytes sized from HBM source.
    evict = next(c for c in c2 if c.id[1] == [Tier.DRAM])
    if set(evict.acquired) != {"DRAM"}:
        raise StageFail(f"A: evict→DRAM must acquire DRAM: {evict.acquired}")

    # empty decision set → no candidates (LLM_PREFILL D_t=∅ path).
    if migrate_candidates(st, [], default_costs()):
        raise StageFail("A: empty decision_set must yield no candidates")
    print(_green("  [A] migrate_candidates: cost/relief/acquired + filter OK"))


def _program(state="REASONING", *, inflight=None, committed=None,
             unit_hashes=None, pre_pause_state=None):
    return {
        "state": state,
        "pre_pause_state": pre_pause_state,
        "hbm": {"committed": committed or {}, "inflight": inflight or {}},
        "dram": {"committed": {}},
        "unit_hashes": unit_hashes or [],
    }


# ============================================================ Stage B


def stage_b_forecast() -> None:
    """§8 forecast: per-HBM-subpool used_bytes (+ inflight term, which is
    0 under the T26/T11 placeholders); forecast_horizon = heartbeat_s."""
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    GB = 1024 ** 3
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM"], holders=["S"])],
        hbm={"full": _sp(3 * GB, 10 * GB), "mamba": _sp(8 * GB, 9 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    fc = adm.forecast(st, heartbeat_s=5.0)
    if fc != {"full": float(3 * GB), "mamba": float(8 * GB)}:
        raise StageFail(f"B: forecast must equal HBM used_bytes per subpool, "
                        f"got {fc}")
    if adm.forecast_horizon(st, 5.0) != 5.0:
        raise StageFail("B: forecast_horizon must fall back to heartbeat_s")
    # inflight demand is 0 under the placeholder regardless of horizon.
    if adm.forecast_inflight_demand(st, 5.0):
        raise StageFail("B: forecast_inflight_demand must be 0 pre-T26/T11")
    print(_green("  [B] forecast = per-subpool used_bytes; horizon=heartbeat_s OK"))


# ============================================================ Stage C


def stage_c_program_candidates() -> None:
    """§8 pause_candidates / resume_candidates: cost/relief, gain/re_use,
    state filtering, capacity_fits gating."""
    GB = 1024 ** 3
    MB = 1024 ** 2
    # ---- pause_candidates ----
    tracker = ProgramTracker()
    tracker.observe_arrival("A")            # REASONING
    tracker.observe_arrival("B")
    tracker.observe_completion("B")         # ACTING
    tracker.pause("P")                       # PAUSED
    # NOTE: relief is the shared-aware ``committed`` snapshot (#205 — raw
    # inflight is a re-prefill COST, not relief).  Units placed off-HBM so
    # MEMORY_PRESSURE's D_t is empty and this stage stays focused on
    # candidate STRUCTURE (pid filtering, gain/re_use); the committed
    # D_t-exclusion (disjoint levers, #2) is pinned by
    # verify/admission_controller stage_disjoint.
    programs = {
        "A": _program("REASONING", committed={"kv": 7 * MB},
                      unit_hashes=["uA"]),
        "B": _program("ACTING", committed={"kv": 3 * MB},
                      unit_hashes=["uB"]),
        "P": _program("PAUSED", committed={"kv": 1 * MB},
                      unit_hashes=["uP"], pre_pause_state="REASONING"),
    }
    sj = _state_json(
        units=[
            _unit(uhash="uA", residence=["DRAM"], holders=["A"]),
            _unit(uhash="uB", residence=["DRAM"], holders=["B"]),
            _unit(uhash="uP", residence=["DRAM"], holders=["P"]),
        ],
        programs=programs,
        hbm={"kv": _sp(2 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="A")
    st = _build_state(sj, tracker, ev)

    pcs = adm.pause_candidates(st)
    pids = {p.pid for p in pcs}
    if pids != {"A", "B"}:
        raise StageFail(f"C: pause_candidates must cover REASONING+ACTING "
                        f"only (not PAUSED/ENDED), got {pids}")
    pa = next(p for p in pcs if p.pid == "A")
    # relief = committed (snapshot); future term 0, D_t empty here.
    if pa.relief != {"kv": 7 * MB}:
        raise StageFail(f"C: A pause_relief should be committed=7MB, "
                        f"got {pa.relief}")
    pb = next(p for p in pcs if p.pid == "B")
    if pb.relief != {"kv": 3 * MB}:
        raise StageFail(f"C: B pause_relief (committed 3MB) should be 3MB, "
                        f"got {pb.relief}")
    # prefill_bps=0 placeholder → marginal_pause_cost 0 → cost = V_u_program.
    vprog = adm.shared_aware_prog_scores(st)
    if abs(pa.cost - vprog["A"]) > 1e-12:
        raise StageFail(f"C: A pause cost should equal V_u_program "
                        f"(marginal=0 pre-T26): {pa.cost} vs {vprog['A']}")

    # ---- resume_candidates ----
    rcs = adm.resume_candidates(st, heartbeat_s=5.0, theta_hi=0.85)
    if {r.pid for r in rcs} != {"P"}:
        raise StageFail(f"C: resume_candidates must cover PAUSED only, "
                        f"got {[r.pid for r in rcs]}")
    rp = rcs[0]
    # uP lives on DRAM only → resume re-enters its bytes into HBM.
    if rp.re_use.get("kv", 0) <= 0:
        raise StageFail(f"C: P re_use should be >0 (uP not HBM-resident): "
                        f"{rp.re_use}")
    if abs(rp.gain - vprog["P"]) > 1e-12:
        raise StageFail(f"C: P resume gain should equal V_u_program: "
                        f"{rp.gain} vs {vprog['P']}")

    # capacity_fits gate: HBM near cap → Resume omitted (would overflow).
    sj_full = _state_json(
        units=[_unit(uhash="uP", residence=["DRAM"], holders=["P"],
                     n_bytes_per_tier={"DRAM": 2 * GB})],
        programs={"P": _program("PAUSED", unit_hashes=["uP"],
                                pre_pause_state="REASONING")},
        hbm={"kv": _sp(8 * GB, 10 * GB)},   # 80% used; +2GB re_use > 85%*10GB
    )
    st_full = _build_state(sj_full, tracker, ev)
    rcs_full = adm.resume_candidates(st_full, heartbeat_s=5.0, theta_hi=0.85)
    if rcs_full:
        raise StageFail(f"C: capacity_fits must omit Resume that overflows "
                        f"theta_hi (8GB+2GB > 8.5GB), got {[r.pid for r in rcs_full]}")
    print(_green("  [C] pause/resume candidates: cost/relief, gain/re_use, "
                 "capacity_fits OK"))


def _hbm_relief(chosen) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in chosen:
        rel = getattr(c, "relief", {})
        for sp, b in rel.get("HBM", {}).items():
            out[sp] = out.get(sp, 0) + b
    return out


# ============================================================ Stage D


def stage_d_joint_decide_select() -> None:
    """§9 phase selection: pressure / headroom / dead-zone, pressure
    suppresses headroom, LLM_PREFILL runs joint_decide."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1.0e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)

    tracker = ProgramTracker()
    tracker.observe_arrival("S")

    # --- pressure: HBM 90% (forecast=9GB > 8.5GB) → free ≥ 0.5GB ---
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
        hbm={"kv": _sp(9 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    plan = jd.joint_decide(st, ev, **kw)
    if not plan:
        raise StageFail("D: pressure phase must return a non-empty plan")
    if any(isinstance(c, Resume) for c in plan):
        raise StageFail("D: pressure plan must contain no Resume")
    relief = _hbm_relief(plan)
    if relief.get("kv", 0) < int(0.5 * GB):
        raise StageFail(f"D: pressure plan must free ≥ bytes_needed (0.5GB), "
                        f"freed {relief}")

    # --- dead-zone: HBM 78% (between theta_lo=70% and theta_hi=85%) ---
    sj_dz = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"])],
        hbm={"kv": _sp(int(7.8 * GB), 10 * GB)})
    st_dz = _build_state(sj_dz, tracker, ev)
    if jd.joint_decide(st_dz, ev, **kw) != []:
        raise StageFail("D: hysteresis dead-zone (70–85%) must return []")

    # --- headroom: HBM 5% (< theta_lo) + PAUSED program fits ---
    tracker.pause("P")
    sj_hr = _state_json(
        units=[_unit(uhash="uP", residence=["DRAM"], holders=["P"],
                     n_bytes_per_tier={"DRAM": 1 * GB})],
        programs={"P": _program("PAUSED", unit_hashes=["uP"],
                                pre_pause_state="REASONING")},
        hbm={"kv": _sp(int(0.5 * GB), 10 * GB)},
    )
    er = Event(kind=EventKind.PRESSURE_RESOLVED, session="P")
    st_hr = _build_state(sj_hr, tracker, er)
    plan_hr = jd.joint_decide(st_hr, er, **kw)
    if not (len(plan_hr) == 1 and isinstance(plan_hr[0], Resume)
            and plan_hr[0].pid == "P"):
        raise StageFail(f"D: headroom phase must resume P, got {plan_hr}")

    # --- pressure suppresses headroom: pressured subpool + a PAUSED
    #     program present → result has NO Resume (headroom doesn't run) ---
    sj_sup = _state_json(
        units=[
            _unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                  n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB}),
            _unit(uhash="uP", residence=["DRAM"], holders=["P"],
                  n_bytes_per_tier={"DRAM": 1 * GB}),
        ],
        programs={"P": _program("PAUSED", unit_hashes=["uP"],
                                pre_pause_state="REASONING")},
        hbm={"kv": _sp(9 * GB, 10 * GB)},
    )
    st_sup = _build_state(sj_sup, tracker, ev)
    plan_sup = jd.joint_decide(st_sup, ev, **kw)
    if any(isinstance(c, Resume) for c in plan_sup):
        raise StageFail("D: pressure must suppress headroom (no Resume "
                        f"while any subpool pressured), got {plan_sup}")

    # --- LLM_PREFILL: empty D_t (no migrates) but pause candidates still
    #     run; under pressure with an active program → a Pause appears ---
    sj_pf = _state_json(
        units=[_unit(uhash="uA", residence=["HBM"], holders=["A"],
                     n_bytes_per_tier={"HBM": 1 * GB})],
        programs={"A": _program("REASONING", committed={"kv": 1 * GB},
                                unit_hashes=["uA"])},
        hbm={"kv": _sp(9 * GB, 10 * GB)},
    )
    tracker.observe_arrival("A")
    epf = Event(kind=EventKind.LLM_PREFILL, session="A")
    st_pf = _build_state(sj_pf, tracker, epf)
    if st_pf.decision_set:
        raise StageFail("D: LLM_PREFILL D_t must be empty (precondition)")
    plan_pf = jd.joint_decide(st_pf, epf, **kw)
    if not any(isinstance(c, Pause) for c in plan_pf):
        raise StageFail(f"D: LLM_PREFILL must still run admission generators "
                        f"(expected a Pause under pressure), got {plan_pf}")
    print(_green("  [D] joint_decide selection: pressure/headroom/dead-zone, "
                 "suppression, LLM_PREFILL OK"))


# ============================================================ Stage E


def stage_e_dp_correctness() -> None:
    """§9 DP: no same-unit double-pick (multiple-choice constraint),
    exact vs brute-force oracle, infeasible → fatal."""
    from daemon import joint_decide as jd
    from baselines.knapsack import (knapsack_min_cost_multi,
                                     KnapsackInfeasibleError)
    import itertools

    # --- no same-unit double-pick: two transitions of unit "u" both
    #     relieve HBM; the DP must pick AT MOST ONE (else relief double-
    #     counts the same physical bytes). ---
    m_evict = Migrate(cost=1.0, relief={"HBM": {"kv": 100}},
                      acquired={"DRAM": {"kv": 100}}, id=("u", "evict"),
                      group="u")
    m_drop = Migrate(cost=5.0, relief={"HBM": {"kv": 100}}, acquired={},
                     id=("u", "drop"), group="u")
    chosen = knapsack_min_cost_multi(
        [m_evict, m_drop],
        bytes_needed={("HBM", "kv"): 100},
        cap_left={("DRAM", "kv"): 10_000},
        bucket_size={("HBM", "kv"): 1, ("DRAM", "kv"): 1},
    )
    if len(chosen) != 1:
        raise StageFail(f"E: must pick exactly one of unit u's transitions, "
                        f"got {[c.id for c in chosen]}")
    if chosen[0].id != ("u", "evict"):
        raise StageFail(f"E: should pick the cheaper transition (evict), "
                        f"got {chosen[0].id}")

    # If bytes_needed exceeds ONE transition's relief, a 0/1 knapsack
    # would (wrongly) pick both same-unit transitions to reach 200; the
    # MCKP must instead reach into OTHER groups.  With only unit u
    # available and need=150 > 100, it is genuinely infeasible (not a
    # 2× double-count) → exception.
    raised = False
    try:
        knapsack_min_cost_multi(
            [m_evict, m_drop],
            bytes_needed={("HBM", "kv"): 150},
            cap_left={("DRAM", "kv"): 10_000},
            bucket_size={("HBM", "kv"): 1, ("DRAM", "kv"): 1})
    except KnapsackInfeasibleError:
        raised = True
    if not raised:
        raise StageFail("E: need>single-transition with one unit must be "
                        "infeasible (MCKP forbids double-counting u's bytes)")

    # --- exact vs brute-force oracle over grouped candidates ---
    def brute_min_cost(items, need, groups):
        # enumerate choices: per group pick none or one member.
        best = None
        opts = []
        for g in groups:
            opts.append([None] + list(g))
        for combo in itertools.product(*opts):
            picked = [m for m in combo if m is not None]
            tot_relief = sum(m.relief.get("HBM", {}).get("kv", 0) for m in picked)
            if tot_relief < need:
                continue
            cost = sum(m.cost for m in picked)
            if best is None or cost < best[0]:
                best = (cost, picked)
        return best

    # deterministic small fixtures (no RNG per harness rules — vary by index)
    import math
    fails = 0
    for seed in range(40):
        # build 3 units, each with 2 grouped transitions, bytes vary by seed
        items = []
        groups = []
        for ui in range(3):
            base = 30 + ((seed * 7 + ui * 13) % 50)
            g = []
            a = Migrate(cost=1.0 + ((seed + ui) % 4),
                        relief={"HBM": {"kv": base}}, acquired={},
                        id=(ui, "a"), group=f"u{ui}")
            b = Migrate(cost=4.0 + ((seed * 3 + ui) % 5),
                        relief={"HBM": {"kv": base + 20}}, acquired={},
                        id=(ui, "b"), group=f"u{ui}")
            g = [a, b]
            items += g
            groups.append(g)
        need = 60 + (seed % 40)
        oracle = brute_min_cost(items, need, groups)
        try:
            dp = knapsack_min_cost_multi(
                items, {("HBM", "kv"): need}, {},
                {("HBM", "kv"): 1})
        except KnapsackInfeasibleError:
            if oracle is not None:
                fails += 1
            continue
        if oracle is None:
            fails += 1
            continue
        dp_cost = sum(c.cost for c in dp)
        # at most one per group in the DP result
        gseen = [c.group for c in dp]
        if len(gseen) != len(set(gseen)):
            raise StageFail(f"E: seed {seed} DP picked 2+ from one group: "
                            f"{[c.id for c in dp]}")
        if abs(dp_cost - oracle[0]) > 1e-9:
            fails += 1
    if fails:
        raise StageFail(f"E: DP vs brute-force mismatch on {fails}/40 fixtures")

    # --- under-relievable pressure → BEST-EFFORT, not fatal (#194).
    #     In-flight-dominated pressure (tiny radix footprint, no Pause
    #     candidate) cannot be fully relieved by migration; the daemon
    #     must free what it can and re-evaluate next event, NOT crash. ---
    GB = 1024 ** 3
    fatal_called = {"n": 0}

    def _fake_fatal(reason, **ctx):
        fatal_called["n"] += 1
        raise RuntimeError("fatal-sentinel")
    orig = jd.fatal
    jd.fatal = _fake_fatal
    try:
        tracker = ProgramTracker()
        tracker.observe_arrival("S")
        # HBM 90% (need ≈ 0.5GB) but the only D_t unit is 2 MB on {HBM}
        # with no lower tier, and no active program → no Pause.  Migration
        # can free at most 2 MB ≪ 0.5GB.
        sj = _state_json(
            units=[_unit(uhash="u1", residence=["HBM"], holders=["S"],
                         n_bytes_per_tier={"HBM": 2 * 1024 * 1024})],
            hbm={"kv": _sp(9 * GB, 10 * GB)})
        ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
        st = _build_state(sj, tracker, ev)
        plan = jd.joint_decide(st, ev, costs=default_costs(), pi_u=1e-4,
                               theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
        if fatal_called["n"] != 0:
            raise StageFail("E: under-relievable pressure must NOT fatal "
                            "(best-effort); fatal was called")
        # best-effort frees what it can: the one available unit (DROP or
        # evict→DRAM), i.e. a non-empty plan that under-relieves.
        if not plan:
            raise StageFail("E: best-effort must free what it can (non-empty "
                            "plan), got []")
        freed = sum(b for c in plan for b in c.relief.get("HBM", {}).values())
        if freed > int(0.5 * GB):
            raise StageFail(f"E: this fixture can only under-relieve; freed "
                            f"{freed} unexpectedly ≥ need")
    finally:
        jd.fatal = orig

    # --- DP blow-up STILL fatals (genuine misconfiguration) ---
    from baselines.knapsack import KnapsackBudgetExceededError
    blew = False
    try:
        knapsack_min_cost_multi(
            [Migrate(cost=1.0, relief={"HBM": {"kv": 64 * 1024}},
                     acquired={"DRAM": {"kv": i * 64 * 1024}}, id=i, group=i)
             for i in range(1, 60)],
            bytes_needed={("HBM", "kv"): 64 * 1024 * 40},
            cap_left={("DRAM", "kv"): 64 * 1024 * 100000},
            bucket_size={("HBM", "kv"): 64 * 1024, ("DRAM", "kv"): 1},
            max_dp_cells=50, best_effort=True)
    except KnapsackBudgetExceededError:
        blew = True
    if not blew:
        raise StageFail("E: DP cell ceiling must still raise "
                        "KnapsackBudgetExceededError even in best_effort")
    print(_green("  [E] DP: no same-unit double-pick, exact vs brute (40), "
                 "under-relief→best-effort (no fatal), blow-up→raise OK"))


# ============================================================ Stage F


class _DummyHttp:
    async def post(self, url, *, json=None):  # noqa: ANN001
        class _R:
            status_code = 200
            def json(self): return {}
            text = ""
        return _R()
    async def put(self, url, *, json=None):  # noqa: ANN001
        class _R:
            status_code = 200
            def json(self): return {}
            text = ""
        return _R()
    async def aclose(self): return None


class _StubRouter:
    def __init__(self, sj, *, theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0):
        self._sj = sj
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.heartbeat_s = heartbeat_s
        self.observability = None
    async def fetch_state(self):
        return self._sj


def _drain(ob):
    out = []
    while ob.queue.qsize():
        out.append(ob.queue.get_nowait())
    return out


def stage_f_live_dispatch() -> None:
    """The handler runs joint_decide and _dispatch_plan routes the mixed
    plan: Pause → tracker.pause + PUT{PAUSED, pre_pause_state}; Resume →
    tracker.resume + PUT{pre_pause_state}.  Brand-new dispatch code —
    pinned end-to-end through KvScheduler.handle (admission ON)."""
    GB = 1024 ** 3

    # --- pressure → a Pause is dispatched ---
    #   Under LLM_PREFILL D_t is empty (no migrates), so the committed
    #   radix (#205 relief source) is NOT D_t-excluded and Pause is the
    #   sole pressure lever — the honest scenario that forces a Pause.
    #   (Under MEMORY_PRESSURE the HBM unit would be in D_t → migrate's
    #   domain → committed excluded → relief 0; that disjoint behavior is
    #   pinned by verify/admission_controller stage_disjoint.)
    def _pause_case():
        tracker = ProgramTracker()
        tracker.observe_arrival("P")            # REASONING (prior state)
        ob = OutboundQueue(sglang_base_url="http://unused",
                           http_client=_DummyHttp())
        sched = kvs.KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                                outbound=ob)
        sched.admission_enabled = True
        sj = _state_json(
            units=[_unit(uhash="uP", residence=["HBM"], holders=["P"],
                         n_bytes_per_tier={"HBM": 1 * GB})],
            programs={"P": _program("REASONING", committed={"kv": 1 * GB},
                                    unit_hashes=["uP"])},
            hbm={"kv": _sp(9 * GB, 10 * GB)})
        router = _StubRouter(sj)
        asyncio.run(sched.handle(Event(EventKind.LLM_PREFILL, session="P"),
                                 router))
        return tracker, sched, _drain(ob)

    tracker, sched, batches = _pause_case()
    if tracker.state("P") is not State.PAUSED:
        raise StageFail("F: pressure → handler must tracker.pause(P)")
    if sched.pause_calls != 1:
        raise StageFail(f"F: pause_calls must be 1, got {sched.pause_calls}")
    puts = [b for b in batches if b.endpoint == "program_paused"]
    if not puts:
        raise StageFail(f"F: pause must enqueue a program_paused PUT; "
                        f"got {[b.endpoint for b in batches]}")
    body = puts[0].body
    if body.get("state") != "PAUSED" or body.get("pre_pause_state") != "REASONING":
        raise StageFail(f"F: pause PUT body must be {{PAUSED, pre=REASONING}}; "
                        f"got {body}")

    # --- headroom → a Resume is dispatched, restoring pre_pause_state ---
    def _resume_case():
        tracker = ProgramTracker()
        tracker.pause("R")                       # PAUSED
        ob = OutboundQueue(sglang_base_url="http://unused",
                           http_client=_DummyHttp())
        sched = kvs.KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                                outbound=ob)
        sched.admission_enabled = True
        sj = _state_json(
            units=[_unit(uhash="uR", residence=["DRAM"], holders=["R"],
                         n_bytes_per_tier={"DRAM": 1 * GB})],
            programs={"R": _program("PAUSED", unit_hashes=["uR"],
                                    pre_pause_state="ACTING")},
            hbm={"kv": _sp(int(0.5 * GB), 10 * GB)})  # < theta_lo → headroom
        router = _StubRouter(sj)
        asyncio.run(sched.handle(Event(EventKind.PRESSURE_RESOLVED, session="R"),
                                 router))
        return tracker, sched, _drain(ob)

    tracker, sched, batches = _resume_case()
    if sched.resume_calls != 1:
        raise StageFail(f"F: headroom → resume_calls must be 1, got "
                        f"{sched.resume_calls}")
    puts = [b for b in batches if b.endpoint == "program_paused"]
    if not puts or puts[0].body.get("state") != "ACTING":
        raise StageFail(f"F: resume PUT must restore pre_pause_state=ACTING; "
                        f"got {[b.body for b in puts]}")

    # --- admission OFF → no Pause even under pressure (kv-only arm) ---
    def _kvonly_case():
        tracker = ProgramTracker()
        tracker.observe_arrival("P")
        ob = OutboundQueue(sglang_base_url="http://unused",
                           http_client=_DummyHttp())
        sched = kvs.KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                                outbound=ob)
        # admission_enabled stays False (default)
        sj = _state_json(
            units=[_unit(uhash="uP", residence=["HBM", "DRAM"], holders=["P"],
                         n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
            programs={"P": _program("REASONING", inflight={"kv": 1 * GB},
                                    unit_hashes=["uP"])},
            hbm={"kv": _sp(9 * GB, 10 * GB)})
        router = _StubRouter(sj)
        asyncio.run(sched.handle(Event(EventKind.MEMORY_PRESSURE, session="P"),
                                 router))
        return tracker, sched
    tracker, sched = _kvonly_case()
    if sched.pause_calls != 0:
        raise StageFail("F: admission OFF must dispatch no Pause (kv-only arm)")
    if tracker.state("P") is State.PAUSED:
        raise StageFail("F: admission OFF must not pause P")
    print(_green("  [F] live dispatch: pause+PUT / resume+PUT / kv-only no-pause OK"))


# ============================================================ Stage G


def stage_g_robustness() -> None:
    """#194 audit: the cap_left≥0 clamp (over-subscribed destination must
    not reject zero-acquire DROP) + best_effort relieves ALL axes when
    caps allow."""
    from daemon import joint_decide as jd
    from baselines.knapsack import knapsack_min_cost_multi
    GB = 1024 ** 3

    # --- cap_left clamp: DRAM over-subscribed (used > cap → cap_left<0).
    #     A unit on {HBM,DRAM} can still evict-HBM (acquired={}) — the
    #     clamp keeps that DROP/evict feasible instead of spuriously
    #     infeasible. ---
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
        hbm={"kv": _sp(9 * GB, 10 * GB)},
        dram={"kv": _sp(50 * GB, 40 * GB)})   # used > cap → cap_left = -10GB
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    plan = jd.joint_decide(st, ev, costs=default_costs(), pi_u=1e-4,
                           theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    if not plan:
        raise StageFail("G: over-subscribed DRAM must NOT make pressure "
                        "infeasible — evict-HBM (acquired={}) is still valid")
    relief = _hbm_relief(plan)
    if relief.get("kv", 0) < int(0.5 * GB):
        raise StageFail(f"G: clamp must let the HBM-freeing plan through; "
                        f"freed {relief}")

    # --- best_effort relieves BOTH pressured axes when caps allow (no
    #     forced tradeoff): two DROPs, one per subpool, both chosen. ---
    a = Migrate(cost=1.0, relief={"HBM": {"full": 100}}, acquired={},
                id="a", group="ua")
    b = Migrate(cost=1.0, relief={"HBM": {"mamba": 100}}, acquired={},
                id="b", group="ub")
    chosen = knapsack_min_cost_multi(
        [a, b],
        bytes_needed={("HBM", "full"): 100, ("HBM", "mamba"): 100},
        cap_left={}, bucket_size={("HBM", "full"): 1, ("HBM", "mamba"): 1},
        best_effort=True)
    if {c.id for c in chosen} != {"a", "b"}:
        raise StageFail(f"G: best_effort must relieve BOTH axes when caps "
                        f"allow; got {[c.id for c in chosen]}")
    print(_green("  [G] cap_left≥0 clamp; best_effort relieves all axes OK"))


# ============================================================ runner

_STAGES = [
    ("A", stage_a_migrate_candidates),
    ("B", stage_b_forecast),
    ("C", stage_c_program_candidates),
    ("D", stage_d_joint_decide_select),
    ("E", stage_e_dp_correctness),
    ("F", stage_f_live_dispatch),
    ("G", stage_g_robustness),
]


def main() -> int:
    print("=" * 64)
    print("verify/joint_decide (#194) — DESIGN §7/§8/§9 joint_decide")
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
        print(_red(f"FAILED stages: {', '.join(failed)}"))
        return 1
    print(_green("ALL STAGES PASSED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
