"""verify/joint_decide (#194, DESIGN §7/§8/§9): wire the T34 multi-axis
DP into the live decision path.

#194 replaces two separate, sequential decision modules —
``OursGreedyPolicy.decide`` (per-unit greedy, single-axis capacity) +
the admission ``_on_pressure`` / ``_on_resolved`` loops (the
"Gauss-Seidel decompose" DESIGN §9 supersedes) — with ONE
``joint_decide(state, event)`` that:

  * generates unit-level Migrate candidates (§7 ``migrate_candidates``),
  * generates program-level Resume candidates (§8 ``resume_candidates``;
    the Pause lever is DORMANT — not generated, §9),
  * computes the per-HBM-subpool ``forecast`` and the destination /
    ``free_room`` budgets (§8 / §9),
  * runs ONE value-maximising knapsack (``knapsack_max_value_multi``) for
    relief (net-positive Migrates that relieve a PRESSURED subpool) and
    again for resume — both may pick the empty set (no-op),
  * returns the chosen mixed plan for the live handler to dispatch.

§9 is **value-gated, not cover**: every phase takes ONLY net-positive
actions and MAY no-op; relief and resume COEXIST (not mutually
exclusive); there is no forced relief and no infeasibility (the empty
plan is always reachable).

Stages (TDD — each builds a fixture, asserts the contract):

  A. migrate_candidates  — §7 generator: cost / relief / acquired,
                           transition enumeration, relief>0 filter
  B. forecast            — §8 per-HBM-subpool forecast (degrades to
                           used_bytes under the T26/T11 placeholders)
  C. pause/resume_cands  — §8 program generators (cost/relief, gain/re_use)
  D. joint_decide select — §9 value-gated: net-positive relief acts;
                           do-no-harm no-op on an unrelievable subpool;
                           pauses never appear; relief+resume coexist
  E. joint_decide DP      — value-max exact vs brute-force oracle; no
                           same-group double-count; blow-up→fatal
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
    is_device_leaf: bool = True,
    is_host_leaf: bool = True,
    is_tree_leaf: bool = True,
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
        "is_device_leaf": is_device_leaf,
        "is_host_leaf": is_host_leaf,
        "is_tree_leaf": is_tree_leaf,
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
    tracker.observe_arrival("S")  # REASONING (alive holder)
    GB = 1024 ** 3
    nb = 2_000_000  # bytes per tier for the unit
    # hit_count=5 → reuse-based p_hat≈0.86 (#249: alive no longer forces 1.0).
    # A REUSED unit is what gives V({DRAM})>0, so DROP forgoing the retained
    # saved-prefill is a real cost — the property this stage exercises.
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     hit_count=5, n_bytes_per_tier={"HBM": nb, "DRAM": nb})],
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


def stage_a_leaf_filter() -> None:
    """#210: migrate_candidates must mirror sglang's THREE apply-site leaf
    guards (unified_radix_cache.py 2673/2684/2687) so it never proposes a
    remove that sglang is structurally guaranteed to reject — pure waste
    that under A3 saturation produced ~86k apply_failed/cycle, zero relief,
    daemon thrash.  Pins each guard with a matched leaf/non-leaf pair:
      • remove-HBM  ⇐ is_device_leaf  (remove_hbm_not_device_leaf)
      • remove-DRAM ⇐ is_host_leaf    (remove_dram_not_host_leaf)
      • full-drop   ⇐ is_tree_leaf    (remove_not_leaf) — STRICTER than
        device-leaf: a node with disk-only children is a device leaf yet
        not a tree leaf, so device-leaf alone does not cover full-drop.
    Each non-leaf branch also asserts the OTHER candidates survive (no
    over-filtering)."""
    GB = 1024 ** 3
    nb = 2_000_000

    def _ids(unit_kwargs, residence, nbt):
        tracker = ProgramTracker()
        tracker.observe_arrival("S")
        sj = _state_json(
            units=[_unit(uhash="u1", residence=residence, holders=["S"],
                         n_bytes_per_tier=nbt, **unit_kwargs)],
            hbm={"kv": _sp(5 * GB, 10 * GB)})
        st = _build_state(
            sj, tracker, Event(kind=EventKind.MEMORY_PRESSURE, session="S"))
        cands = migrate_candidates(st, ["u1"], default_costs())
        # (sorted add-tier names, sorted remove-tier names) per candidate.
        return {(tuple(sorted(t.name for t in c.id[1])),
                 tuple(sorted(t.name for t in c.id[2]))) for c in cands}

    def _removes(ids):
        return {rem for _add, rem in ids}

    # ---- guard 1: remove-HBM ⇐ device-leaf -------------------------------
    hd = {"HBM": nb, "DRAM": nb}
    leaf = _removes(_ids({"is_device_leaf": True}, ["HBM", "DRAM"], hd))
    if ("HBM",) not in leaf:
        raise StageFail(f"device-leaf MUST allow remove-HBM; got {leaf}")
    nonleaf = _removes(_ids({"is_device_leaf": False}, ["HBM", "DRAM"], hd))
    if any("HBM" in rs for rs in nonleaf):
        raise StageFail(
            f"#210: non-device-leaf must yield NO remove-HBM migrate "
            f"(remove_hbm_not_device_leaf); got {nonleaf}")

    # ---- guard 2: remove-DRAM ⇐ host-leaf --------------------------------
    # {DRAM,DISK} (device-evicted): the ([],[DRAM]) transition drops the
    # host backup and keeps DISK (NOT a full-drop), so host-leaf is the only
    # guard in play.
    dd = {"DRAM": nb, "DISK": nb}
    hleaf = _removes(_ids({"is_host_leaf": True}, ["DRAM", "DISK"], dd))
    if ("DRAM",) not in hleaf:
        raise StageFail(f"host-leaf MUST allow remove-DRAM; got {hleaf}")
    hnon = _removes(_ids({"is_host_leaf": False}, ["DRAM", "DISK"], dd))
    if any("DRAM" in rs for rs in hnon):
        raise StageFail(
            f"#210: non-host-leaf must yield NO remove-DRAM migrate "
            f"(remove_dram_not_host_leaf); got {hnon}")
    if ("DISK",) not in hnon:
        raise StageFail(
            f"over-filtered: remove-DISK must survive when only host-leaf "
            f"is False; got {hnon}")

    # ---- guard 3: full-drop ⇐ tree-leaf ----------------------------------
    # {HBM}-only with is_device_leaf=True but is_tree_leaf=False (disk-only
    # children).  The DROP ([],[HBM]) is a full-drop → must be suppressed by
    # the stricter tree-leaf guard, which device-leaf alone does NOT cover.
    # The evict-HBM ([DRAM],[HBM]) is NOT a full-drop (lands in {DRAM}) and,
    # being a device leaf, must survive.
    h_only = {"HBM": nb}
    DROP = ((), ("HBM",))
    EVICT = (("DRAM",), ("HBM",))
    tl = _ids({"is_device_leaf": True, "is_tree_leaf": True}, ["HBM"], h_only)
    if DROP not in tl:
        raise StageFail(f"tree-leaf MUST allow full-drop; got {tl}")
    ntl = _ids({"is_device_leaf": True, "is_tree_leaf": False}, ["HBM"], h_only)
    if DROP in ntl:
        raise StageFail(
            f"#210: non-tree-leaf must yield NO full-drop migrate "
            f"(remove_not_leaf, stricter than device-leaf); got {ntl}")
    if EVICT not in ntl:
        raise StageFail(
            f"over-filtered: device-leaf evict-HBM must survive when only "
            f"tree-leaf is False; got {ntl}")

    print(_green("  [A-leaf] migrate_candidates mirrors sglang's 3 leaf "
                 "guards (remove-HBM/DRAM/full-drop) (#210) OK"))


def stage_a_inflight_holder_gate() -> None:
    """#224: migrate_candidates must NOT propose a remove-HBM for a unit any
    of whose holder programs is actively decoding — i.e. has a request in the
    running batch (T26 ``per_program_usage[pid].hbm.inflight`` > 0).

    Root cause of the TP=4 A3 ``remove_hbm_not_device_leaf`` storm (1384/cycle,
    187 hot units re-failing ~15×): such a unit is a device-leaf *at dump time*
    only in the brief gap between the active program's forward passes; by apply
    time (state-fetch p99≈80 ms, max≈840 ms later) the holder has re-locked its
    session tail, so sglang rejects.  Node-level ``lock_ref`` is the WRONG
    signal (0 at the dump instant — that is exactly why the node dumped as a
    leaf); the program-level ``inflight`` signal is STABLE across the per-pass
    lock oscillation.

    Crucially, a TOOL-PARKED program (awaiting a tool result → NOT in the
    running batch → ``inflight`` empty) keeps its idle tail EVICTABLE — that
    demote-during-the-tool-gap is the core §7/§9 value and must survive."""
    GB = 1024 ** 3
    nb = 2_000_000
    hd = {"HBM": nb, "DRAM": nb}

    def _removes(holders, programs):
        tracker = ProgramTracker()
        for h in holders:
            tracker.observe_arrival(h)
        sj = _state_json(
            units=[_unit(uhash="u1", residence=["HBM", "DRAM"],
                         holders=holders, n_bytes_per_tier=hd)],
            programs=programs, hbm={"kv": _sp(5 * GB, 10 * GB)})
        st = _build_state(
            sj, tracker,
            Event(kind=EventKind.MEMORY_PRESSURE, session=holders[0]))
        return {tuple(sorted(t.name for t in c.id[2]))
                for c in migrate_candidates(st, ["u1"], default_costs())}

    # (a) sole holder ACTIVELY DECODING (inflight>0) → NO remove-HBM at all.
    active = _removes(["p_act"],
                      {"p_act": _program("REASONING", inflight={"kv": GB})})
    if any("HBM" in rs for rs in active):
        raise StageFail(
            f"#224: active-holder (inflight>0) unit must yield NO remove-HBM "
            f"(races the device lock → remove_hbm_not_device_leaf); got {active}")
    # …but the device-retaining drop-DRAM (lock-safe) must NOT be over-filtered.
    if ("DRAM",) not in active:
        raise StageFail(
            f"#224 over-filter: remove-DRAM (keeps device, lock-safe) must "
            f"survive for an active holder; got {active}")

    # (b) sole holder TOOL-PARKED (inflight empty) → remove-HBM PRESERVED.
    parked = _removes(["p_park"], {"p_park": _program("ACTING")})
    if ("HBM",) not in parked:
        raise StageFail(
            f"#224: tool-parked holder (inflight empty) MUST keep its "
            f"remove-HBM demote — the core demote-during-tool-gap value; "
            f"got {parked}")

    # (c) SHARED unit, one active + one parked holder → ANY active holder
    #     blocks remove-HBM (the node is locked by the active one).
    shared = _removes(["p_act", "p_park"],
                      {"p_act": _program("REASONING", inflight={"kv": GB}),
                       "p_park": _program("ACTING")})
    if any("HBM" in rs for rs in shared):
        raise StageFail(
            f"#224: a shared unit with ANY actively-decoding holder must yield "
            f"NO remove-HBM (locked by the active holder); got {shared}")

    # (d) inflight signal absent (cold-start / pre-T26) → must NOT suppress
    #     (no false strand of the policy when the signal is unpopulated).
    cold = _removes(["p_x"], {})
    if ("HBM",) not in cold:
        raise StageFail(
            f"#224: absent inflight signal must NOT suppress remove-HBM "
            f"(cold-start safety); got {cold}")

    # (e) inflight present-but-ZERO {"kv": 0} → program is NOT in the running
    #     batch → must NOT suppress.  (Guards against a future refactor to a
    #     truthiness test that would gate a populated-but-idle program.)
    zero = _removes(["p_idle"],
                    {"p_idle": _program("ACTING", inflight={"kv": 0})})
    if ("HBM",) not in zero:
        raise StageFail(
            f"#224: present-but-zero inflight must NOT suppress remove-HBM "
            f"(zero bytes = not decoding); got {zero}")

    # (f) holder present in per_program_usage but with NO "hbm" key at all →
    #     treated as not-decoding (defensive .get chain); must NOT suppress.
    nohbm = _removes(["p_nh"], {"p_nh": {"state": "ACTING", "dram":
                                         {"committed": {}}}})
    if ("HBM",) not in nohbm:
        raise StageFail(
            f"#224: holder with no hbm key must NOT suppress remove-HBM; "
            f"got {nohbm}")

    # (g) multi-subpool inflight with ONE nonzero sp → suppress (any sp>0 means
    #     the program holds running-batch KV somewhere).
    multi = _removes(["p_m"],
                     {"p_m": _program("REASONING",
                                      inflight={"full": 0, "swa": GB})})
    if any("HBM" in rs for rs in multi):
        raise StageFail(
            f"#224: any nonzero inflight subpool must suppress remove-HBM; "
            f"got {multi}")

    print(_green("  [A-inflight] active blocked, parked/zero/no-hbm/cold "
                 "preserved, shared+multi-sp blocked (#224) OK"))


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
            # hit_count=5 → reuse-based p_hat≈0.86 (#249: alive no longer forces
            # 1.0).  A REUSED unit gives a non-degenerate V_u so the gain/cost ==
            # V_u_program identities this stage pins are exercised meaningfully.
            _unit(uhash="uA", residence=["DRAM"], holders=["A"], hit_count=5),
            _unit(uhash="uB", residence=["DRAM"], holders=["B"], hit_count=5),
            _unit(uhash="uP", residence=["DRAM"], holders=["P"], hit_count=5),
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
    # #260: pause cost = marginal_pause_cost (0 here, prefill_bps=0) +
    # forgone_progress — pure WORK-LOSS, NO V_u_program.  A is REASONING →
    # forgone = the horizon W; the V_u_program term was dropped (it double-counted
    # marginal_pause_cost and was what "went negative → wrongly fired").
    vprog = adm.shared_aware_prog_scores(st)
    W = adm.forecast_horizon(st, 5.0)
    if abs(pa.cost - W) > 1e-9:
        raise StageFail(f"C: REASONING A pause cost = forgone-progress horizon "
                        f"(work-loss #260): {pa.cost} vs {W}")

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
    # T11 (DESIGN §7 holder-product): PAUSED now correctly contributes
    # EXACTLY 0 to p_hat for a unit it exclusively holds (uP's only
    # holder is P) — the #126 counterfactual re-score under
    # pre_pause_state is still NOT implemented (admission_controller.py
    # resume_candidates docstring), so vprog["P"] is P's AS-PAUSED
    # (near-zero/negative) value, and #211's _RESUME_LIVENESS_FLOOR is
    # what actually keeps this Resume candidate alive.
    if vprog["P"] >= 0.0:
        raise StageFail(
            f"C: with T11 holder-product, P's exclusively-held uP should "
            f"score p_hat=0 -> a non-positive V_u_program (pure holding "
            f"cost, no save-prefill term); got vprog['P']={vprog['P']}")
    if abs(rp.gain - adm._RESUME_LIVENESS_FLOOR) > 1e-15:
        raise StageFail(
            f"C: P resume gain should be the #211 liveness floor "
            f"{adm._RESUME_LIVENESS_FLOOR} (vprog['P']={vprog['P']} < 0 "
            f"is below it); got {rp.gain}")

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
    """§9 VALUE-GATED selection: net-positive relief acts; the hysteresis
    dead-zone no-ops; an UNRELIEVABLE pegged subpool no-ops (do-no-harm);
    headroom resume runs; the Pause lever is dormant (never appears)."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1.0e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)

    tracker = ProgramTracker()
    tracker.observe_arrival("S")

    # --- pressure + a COLD migratable unit (on {HBM,DRAM} → evict-HBM is
    #     net-positive: DRAM retains the data) → relief migrates it out.
    #     No forced cover target; the action is taken because value>0. ---
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
        hbm={"kv": _sp(9 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    plan = jd.joint_decide(st, ev, **kw)
    migs = [c for c in plan if isinstance(c, Migrate)]
    if not migs:
        raise StageFail("D: pressure + net-positive relief must migrate the "
                        f"cold unit out; got {[type(c).__name__ for c in plan]}")
    # u1 emits THREE net-positive transitions (evict-HBM, drop-DRAM, DROP)
    # all sharing group=u1's hash; the multiple-choice exclusion via
    # `_ValueItem.group` must pick AT MOST ONE (else the unit's bytes
    # double-count).  Asserting exactly one pins the grouping WIRING in
    # joint_decide, not just the primitive (t34 G0 covers the primitive).
    if len(migs) != 1:
        raise StageFail(f"D: same-unit transitions must collapse to ONE "
                        f"migrate (group exclusion); got {len(migs)}: "
                        f"{[c.id for c in migs]}")
    if any(isinstance(c, Pause) for c in plan):
        raise StageFail("D: the Pause lever is DORMANT — no Pause may appear")
    if any(isinstance(c, Resume) for c in plan):
        raise StageFail("D: no PAUSED program here → no Resume expected")
    # the chosen migrate must actually relieve the pegged subpool.
    if _hbm_relief(plan).get("kv", 0) <= 0:
        raise StageFail(f"D: relief must free the pegged subpool, got "
                        f"{_hbm_relief(plan)}")

    # --- dead-zone: HBM 78% (between theta_lo=70% and theta_hi=85%) → no
    #     relief candidates generated, no paused program → empty no-op. ---
    sj_dz = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"])],
        hbm={"kv": _sp(int(7.8 * GB), 10 * GB)})
    st_dz = _build_state(sj_dz, tracker, ev)
    if jd.joint_decide(st_dz, ev, **kw) != []:
        raise StageFail("D: hysteresis dead-zone (70–85%) must return []")

    # --- DO-NO-HARM no-op: a subpool ('swa') pegged at 90% whose pressure
    #     is NOT relievable by migration (no migratable unit lives on it —
    #     the resident bytes are in-flight, modelled here as a pegged
    #     subpool with no units), while a HEALTHY 'full' subpool holds the
    #     only cold migratable unit.  Relief must NOT churn 'full' (it has
    #     room) and cannot touch 'swa' → empty plan, exactly like no daemon.
    #     This is the A3 swa regime that the value-gate must leave alone. ---
    sj_noop = _state_json(
        units=[_unit(uhash="uf", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": {"full": 1 * GB},
                                       "DRAM": {"full": 1 * GB}},
                     subpool="full")],
        hbm={"swa": _sp(9 * GB, 10 * GB),      # pegged, no migratable units
             "full": _sp(1 * GB, 10 * GB)},    # healthy, holds the cold unit
    )
    st_noop = _build_state(sj_noop, tracker, ev)
    plan_noop = jd.joint_decide(st_noop, ev, **kw)
    if plan_noop != []:
        raise StageFail("D: pegged-but-unrelievable subpool (swa) + a healthy "
                        "subpool holding the only cold unit must NO-OP "
                        f"(do-no-harm); got {[type(c).__name__ for c in plan_noop]}")

    # --- headroom: HBM 5% (< theta_lo) + PAUSED program fits → resume P,
    #     no Pause, no Migrate. ---
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
        raise StageFail(f"D: headroom must resume P (only), got {plan_hr}")

    print(_green("  [D] value-gated select: net-positive relief acts; "
                 "dead-zone + unrelievable-swa no-op; resume; pauses dormant OK"))


def stage_d_resume_starvation() -> None:
    """#211: a PAUSED program whose units were DROPPED while it was gated
    must STILL be resumed when headroom exists — else it starves to an
    AgentTimeout (a paused program emits no events and never gets its
    bytes back on its own).  ``resume_candidates`` sizes ``re_use`` from
    the program's units; once they are gone, ``re_use`` is empty, and the
    headroom phase's ``_has_reuse`` filter silently drops the Resume, so the
    program can never leave the gate.  A zero-HBM resume is the CHEAPEST
    possible action (just release the proxy gate; the program re-prefills
    and admission re-pauses it only if pressure actually returns) — it must
    never be filtered out.  Both shapes the live overlay produces are pinned:
      (a) ``unit_hashes`` still listed but the units are gone from
          ``state.units`` (DROPped post-pause), and
      (b) ``unit_hashes == []`` (the overlay's empty-residue PAUSED entry,
          unified_radix_cache._aginfer_overlay_program_states)."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1.0e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    er = Event(kind=EventKind.PRESSURE_RESOLVED, session="P")

    def _resume_pids(unit_hashes):
        tracker = ProgramTracker()
        tracker.observe_arrival("P")
        tracker.pause("P")
        sj = _state_json(
            units=[],   # P's units have been DROPped while it was gated
            programs={"P": _program("PAUSED", unit_hashes=unit_hashes,
                                    pre_pause_state="REASONING")},
            hbm={"kv": _sp(int(0.5 * GB), 10 * GB)},   # 5% occ → headroom
        )
        st = _build_state(sj, tracker, er)
        plan = jd.joint_decide(st, er, **kw)
        return [c.pid for c in plan if isinstance(c, Resume)]

    a = _resume_pids(unit_hashes=["uP"])   # dropped, hash still referenced
    if "P" not in a:
        raise StageFail(
            "#211: PAUSED program with DROPped units (hash still listed) "
            f"must still be resumed in headroom; got resumes {a}")
    b = _resume_pids(unit_hashes=[])       # overlay empty-residue entry
    if "P" not in b:
        raise StageFail(
            "#211: PAUSED program with empty unit_hashes (overlay empty-"
            f"residue) must still be resumed in headroom; got resumes {b}")

    # --- multi-subpool gate + ordering (audit finding 5): with TWO HBM
    #     subpools both below theta_lo (so `all(r>0)` headroom gate fires),
    #     a VALUABLE resume (real re_use+gain) and a floored zero-re_use
    #     resume must BOTH be granted, and the weight-0 floored one must
    #     never steal a subpool's room from the valuable one. ---
    tracker = ProgramTracker()
    for p in ("V", "Z"):
        tracker.observe_arrival(p)
        tracker.pause(p)
    sj_multi = _state_json(
        units=[_unit(uhash="uV", residence=["DRAM"], holders=["V"],
                     n_bytes_per_tier={"DRAM": {"sp_a": 1 * GB}})],
        programs={
            "V": _program("PAUSED", unit_hashes=["uV"],
                          pre_pause_state="REASONING"),
            "Z": _program("PAUSED", unit_hashes=[],   # dropped → floored
                          pre_pause_state="REASONING"),
        },
        hbm={"sp_a": _sp(int(0.5 * GB), 10 * GB),
             "sp_b": _sp(int(0.5 * GB), 10 * GB)},   # both 5% → headroom
    )
    st_multi = _build_state(sj_multi, tracker, er)
    plan_multi = jd.joint_decide(st_multi, er, **kw)
    pids_multi = {c.pid for c in plan_multi if isinstance(c, Resume)}
    if pids_multi != {"V", "Z"}:
        raise StageFail(
            "#211: multi-subpool headroom must resume BOTH the valuable (V) "
            f"and the floored zero-re_use (Z) program; got {pids_multi}")

    print(_green("  [D-starve] dropped-unit PAUSED programs still resume "
                 "in headroom; valuable+floored co-grant (#211) OK"))


def stage_d_resume_under_pressure() -> None:
    """#213: resume must run ALONGSIDE pressure — not be suppressed by it.

    The killer bug in the live A3 daemon arm: the pressure and resume phases
    were mutually exclusive (pressure ran whenever ANY subpool crossed
    theta_hi; resume only in the else).  Under a permanently-pegged subpool
    (A3 swa ~0.99) the pressure phase is ALWAYS active → resume never fires →
    the daemon pauses monotonically and agents starve to AgentTimeout.

    New contract: a paused program that FITS (zero re_use on the pegged
    subpool — e.g. its units were DROPped) resumes even while that subpool is
    pressured.  A paused program that does NOT fit (re_use on the pegged
    subpool) stays suppressed (capacity_fits, exercised by stage D).  So the
    plan under pressure carries BOTH the pressure response AND the un-starve
    resume."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1.0e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    tracker.observe_arrival("P")
    tracker.pause("P")
    # Single HBM subpool pegged at 90% (> theta_hi) → pressure phase active.
    # An ACTIVE program S holds the HBM-resident pressure; a PAUSED program P
    # has DROPped units (empty re_use → zero weight on the pegged subpool).
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
        programs={
            "S": _program("REASONING", committed={"kv": 9 * GB},
                          unit_hashes=["u1"]),
            "P": _program("PAUSED", unit_hashes=[],   # dropped → empty re_use
                          pre_pause_state="REASONING"),
        },
        hbm={"kv": _sp(9 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    plan = jd.joint_decide(st, ev, **kw)
    # The net-positive relief migrate (cold {HBM,DRAM} unit) must be taken...
    if not any(isinstance(c, Migrate) for c in plan):
        raise StageFail(
            "D-press-resume: the net-positive relief migrate must be taken "
            f"under pressure; got {[type(c).__name__ for c in plan]}")
    # #260: Pause is LIVE, but the only active program here (S) is REASONING —
    # hold_frac=0 (its relief does not persist; it resumes at once) so its pause
    # gain ≤ 0 and it is NEVER paused.  do-no-harm: an actively-decoding agent is
    # never stalled (the A3 regression is closed by the cost model, robustly).
    if any(isinstance(c, Pause) for c in plan):
        raise StageFail("D-press-resume: a REASONING agent must NEVER be paused "
                        "(do-no-harm #260); got a Pause in the plan")
    # ...AND the un-starve resume of P COEXISTS with the relief in one plan
    #    (relief and resume are NOT mutually exclusive — the #213 fix).
    resumed = {c.pid for c in plan if isinstance(c, Resume)}
    if "P" not in resumed:
        raise StageFail(
            "#213: a dropped-units PAUSED program must resume even under "
            "pressure (relief + resume coexist); got "
            f"resumes={resumed}, plan={[type(c).__name__ for c in plan]}")
    print(_green("  [D-press-resume] relief migrate + un-starve resume coexist "
                 "in one plan under pressure; no pause (#213) OK"))


# ============================================================ Stage E


def stage_e_dp_correctness() -> None:
    """§9 value-max DP (`knapsack_max_value_multi`): no same-group double-
    pick (multiple-choice), exact vs brute-force value oracle, empty set
    when nothing pays, multi-axis budget respected, blow-up → raise."""
    from baselines.knapsack import (knapsack_max_value_multi,
                                     KnapsackBudgetExceededError, Resume)
    import itertools

    # --- no same-group double-pick: two transitions of unit "u" both
    #     consume DRAM budget; the DP must pick AT MOST ONE (the higher
    #     value), never both (that would double-count u's physical bytes). ---
    a = Resume(gain=10.0, re_use={"DRAM": {"kv": 100}}, pid=("u", "a"), group="u")
    b = Resume(gain=4.0, re_use={"DRAM": {"kv": 100}}, pid=("u", "b"), group="u")
    chosen = knapsack_max_value_multi(
        [a, b], budget={("DRAM", "kv"): 10_000},
        bucket_size={("DRAM", "kv"): 1})
    if [c.pid for c in chosen] != [("u", "a")]:
        raise StageFail(f"E: must pick the single higher-value group member, "
                        f"got {[c.pid for c in chosen]}")

    # --- empty set when no item pays: all-negative-value items → []. ---
    neg = knapsack_max_value_multi(
        [Resume(gain=-1.0, re_use={"DRAM": {"kv": 10}}, pid="n1", group="g1"),
         Resume(gain=-5.0, re_use={"DRAM": {"kv": 10}}, pid="n2", group="g2")],
        budget={("DRAM", "kv"): 10_000}, bucket_size={("DRAM", "kv"): 1})
    if neg != []:
        raise StageFail(f"E: all-negative-value items must yield the empty "
                        f"set (value-gated no-op), got {[c.pid for c in neg]}")

    # --- exact vs brute-force value oracle over grouped, MULTI-AXIS items.
    #     Deterministic fixtures (no RNG per harness rules — vary by index). ---
    def brute_max_value(groups, budget):
        best_val, best_pick = 0.0, []          # empty set is always allowed
        opts = [[None] + list(g) for g in groups]
        for combo in itertools.product(*opts):
            picked = [m for m in combo if m is not None]
            use = {}
            for m in picked:
                for (t, sp), bd in (((t, sp), v)
                                    for t, d in m.re_use.items()
                                    for sp, v in d.items()):
                    use[(t, sp)] = use.get((t, sp), 0) + bd
            if any(use.get(ax, 0) > cap for ax, cap in budget.items()):
                continue
            val = sum(m.gain for m in picked)
            if val > best_val:
                best_val, best_pick = val, picked
        return best_val

    fails = 0
    for seed in range(40):
        groups = []
        for ui in range(3):
            wa = 30 + ((seed * 7 + ui * 13) % 50)
            wb = 20 + ((seed * 5 + ui * 11) % 40)
            # gains can be +/-; two axes (DRAM kv, DISK kv) so the budget
            # bind is multi-dimensional.
            ga = ((seed + ui) % 7) - 2.0
            gb = ((seed * 3 + ui) % 6) - 1.0
            g = [Resume(gain=ga, re_use={"DRAM": {"kv": wa}}, pid=(ui, "a"),
                        group=f"u{ui}"),
                 Resume(gain=gb, re_use={"DISK": {"kv": wb}}, pid=(ui, "b"),
                        group=f"u{ui}")]
            groups.append(g)
        items = [m for g in groups for m in g]
        budget = {("DRAM", "kv"): 60 + (seed % 40),
                  ("DISK", "kv"): 50 + (seed % 30)}
        oracle = brute_max_value(groups, budget)
        dp = knapsack_max_value_multi(
            items, budget,
            bucket_size={("DRAM", "kv"): 1, ("DISK", "kv"): 1})
        # at most one per group
        gseen = [c.group for c in dp]
        if len(gseen) != len(set(gseen)):
            raise StageFail(f"E: seed {seed} DP picked 2+ from one group: "
                            f"{[c.pid for c in dp]}")
        # budget respected
        use = {}
        for m in dp:
            for t, d in m.re_use.items():
                for sp, v in d.items():
                    use[(t, sp)] = use.get((t, sp), 0) + v
        if any(use.get(ax, 0) > cap for ax, cap in budget.items()):
            raise StageFail(f"E: seed {seed} DP exceeded budget: {use} vs {budget}")
        dp_val = sum(c.gain for c in dp)
        if abs(dp_val - oracle) > 1e-9:
            fails += 1
    if fails:
        raise StageFail(f"E: value DP vs brute-force mismatch on {fails}/40")

    # --- DP blow-up STILL raises (genuine misconfiguration: many distinct
    #     bucket-deltas across an axis exceed the reachable-cell ceiling). ---
    blew = False
    try:
        knapsack_max_value_multi(
            [Resume(gain=1.0, re_use={"DRAM": {"kv": i * 64 * 1024}},
                    pid=i, group=i) for i in range(1, 60)],
            budget={("DRAM", "kv"): 64 * 1024 * 100000},
            bucket_size={("DRAM", "kv"): 64 * 1024},
            max_dp_cells=50)
    except KnapsackBudgetExceededError:
        blew = True
    if not blew:
        raise StageFail("E: DP cell ceiling must raise KnapsackBudgetExceededError")
    print(_green("  [E] value DP: no same-group double-pick, empty-on-no-pay, "
                 "exact vs brute (40, multi-axis), budget held, blow-up→raise OK"))


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
    plan: Migrate → POST /aginfer/migrate; Resume → tracker.resume +
    PUT{pre_pause_state}.  The Pause lever is dormant, so no pause is ever
    dispatched.  Pinned end-to-end through KvScheduler.handle (admission ON)."""
    GB = 1024 ** 3

    # --- pressure → a net-positive relief Migrate is dispatched (POST
    #     /aginfer/migrate), and NO pause is dispatched (dormant lever). ---
    def _migrate_case():
        tracker = ProgramTracker()
        tracker.observe_arrival("P")            # REASONING
        ob = OutboundQueue(sglang_base_url="http://unused",
                           http_client=_DummyHttp())
        sched = kvs.KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                                outbound=ob)
        sched.admission_enabled = True
        sj = _state_json(
            units=[_unit(uhash="uP", residence=["HBM", "DRAM"], holders=["P"],
                         n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
            programs={"P": _program("REASONING", committed={"kv": 1 * GB},
                                    unit_hashes=["uP"])},
            hbm={"kv": _sp(9 * GB, 10 * GB)})
        router = _StubRouter(sj)
        asyncio.run(sched.handle(Event(EventKind.MEMORY_PRESSURE, session="P"),
                                 router))
        return tracker, sched, _drain(ob)

    tracker, sched, batches = _migrate_case()
    if sched.migrate_calls != 1:
        raise StageFail(f"F: pressure → relief migrate must dispatch once, "
                        f"got migrate_calls={sched.migrate_calls}")
    if sched.pause_calls != 0:
        raise StageFail("F: Pause lever is dormant — pause_calls must be 0")
    if tracker.state("P") is State.PAUSED:
        raise StageFail("F: no program may be paused (dormant pause lever)")
    migs = [b for b in batches if b.endpoint == "migrate"]
    if not migs:
        raise StageFail(f"F: relief must enqueue a /aginfer/migrate POST; "
                        f"got {[b.endpoint for b in batches]}")
    if any(b.endpoint == "program_paused" for b in batches):
        raise StageFail("F: no program_paused PUT may be enqueued (no pause)")

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
    if sched.resume_calls != 0:
        raise StageFail("F: admission OFF must dispatch no Resume (kv-only arm)")
    if tracker.state("P") is State.PAUSED:
        raise StageFail("F: admission OFF must not pause P")
    print(_green("  [F] live dispatch: migrate+POST / resume+PUT / "
                 "kv-only relief-only OK"))


# ============================================================ Stage G


def stage_g_robustness() -> None:
    """#194 audit + value-gate: the destination-budget ≥0 clamp (an over-
    subscribed destination must not reject a zero-acquire evict) + the
    value-gate excludes a HOT (cost ≥ 0) relief candidate end-to-end."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3

    # --- budget clamp: DRAM over-subscribed (used > cap → free room < 0).
    #     A unit on {HBM,DRAM} can still evict-HBM (acquired={}) — the
    #     clamp keeps that evict in play instead of rejecting it on a
    #     spurious negative budget. ---
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": 1 * GB, "DRAM": 1 * GB})],
        hbm={"kv": _sp(9 * GB, 10 * GB)},
        dram={"kv": _sp(50 * GB, 40 * GB)})   # used > cap → free room = -10GB
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    plan = jd.joint_decide(st, ev, costs=default_costs(), pi_u=1e-4,
                           theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    if not plan:
        raise StageFail("G: over-subscribed DRAM must NOT block relief — "
                        "evict-HBM (acquired={}) is still net-positive & valid")
    if _hbm_relief(plan).get("kv", 0) <= 0:
        raise StageFail(f"G: clamp must let the HBM-freeing plan through; "
                        f"freed {_hbm_relief(plan)}")

    # --- value-gate excludes a HOT candidate, end-to-end.  Inject (via
    #     migrate_candidates) a HOT HBM-relieving migrate (cost > 0 → moving
    #     it LOSES value) alongside a COLD one (cost < 0).  The value-gate
    #     keeps only cost < 0, so the plan must carry the cold migrate and
    #     NOT the hot one — pins the `cost < 0` filter wiring, not the
    #     real-cost coincidence. ---
    inject = [
        Migrate(cost=-5.0, relief={"HBM": {"kv": 1 * GB}}, acquired={},
                id=("cold", [], ["HBM"]), group="cold"),
        Migrate(cost=+5.0, relief={"HBM": {"kv": 1 * GB}}, acquired={},
                id=("hot", [], ["HBM"]), group="hot"),
    ]
    orig_mc = jd.migrate_candidates
    jd.migrate_candidates = lambda *a, **k: list(inject)
    try:
        plan2 = jd.joint_decide(st, ev, costs=default_costs(), pi_u=1e-4,
                                theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    finally:
        jd.migrate_candidates = orig_mc
    tags = [c.id[0] for c in plan2 if isinstance(c, Migrate)]
    if "cold" not in tags:
        raise StageFail(f"G: the COLD (cost<0) migrate must be taken; got {tags}")
    if "hot" in tags:
        raise StageFail(f"G: the HOT (cost≥0) migrate must be value-gated OUT; "
                        f"got {tags}")
    print(_green("  [G] destination-budget ≥0 clamp; value-gate excludes the "
                 "hot (cost≥0) relief candidate end-to-end OK"))


# ============================================================ Stage H


def stage_h_relief_targets_pressured_subpool() -> None:
    """SF-3 (value-gated rewrite): relief must target the PRESSURED subpool
    and never a candidate that grows it.

    Two ways a non-targeted candidate must be excluded by the pressured-
    subpool filter (`any(sp in pressured_sps for sp in c.relief['HBM'])`):

      1. A candidate that relieves a HEALTHY HBM subpool (not the pegged
         one) is dropped — relief never churns a subpool with room.
      2. A ``promote+drop_disk`` candidate ({DRAM,DISK} → {HBM,DRAM}) that
         relieves DISK while ACQUIRING bytes back into the pegged HBM
         subpool has empty ``relief['HBM']`` → it cannot pass the filter,
         so it can never grow the very subpool under pressure — even with a
         strongly negative cost that a pure value-max DP would want.

    Pinned end-to-end through joint_decide by injecting candidates (so it
    tests the WIRING of the filter, not a real-cost coincidence)."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)

    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    # 'swa' pegged at 90%, 'full' healthy at 10%.  Inject three candidates:
    #   • cold migrate that relieves the PEGGED 'swa'  → MUST be taken
    #   • cold migrate that relieves the HEALTHY 'full' → MUST be dropped
    #   • negative-cost DISK-relieving HBM(swa)-ACQUIRER → MUST be dropped
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM", "DRAM"], holders=["S"],
                     n_bytes_per_tier={"HBM": {"swa": 1 * GB},
                                       "DRAM": {"swa": 1 * GB}}, subpool="swa")],
        hbm={"swa": _sp(9 * GB, 10 * GB), "full": _sp(1 * GB, 10 * GB)},
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    inject = [
        Migrate(cost=-5.0, relief={"HBM": {"swa": 1 * GB}}, acquired={},
                id=("relieve_swa", [], ["HBM"]), group="g_swa"),
        Migrate(cost=-5.0, relief={"HBM": {"full": 1 * GB}}, acquired={},
                id=("relieve_full", [], ["HBM"]), group="g_full"),
        Migrate(cost=-10.0, relief={"DISK": {"kv": 1 * GB}},
                acquired={"HBM": {"swa": 1 * GB}},
                id=("acquire_swa", ["HBM"], ["DISK"]), group="g_acq"),
    ]
    orig_mc = jd.migrate_candidates
    jd.migrate_candidates = lambda *a, **k: list(inject)
    try:
        plan = jd.joint_decide(st, ev, **kw)
    finally:
        jd.migrate_candidates = orig_mc
    tags = [c.id[0] for c in plan if isinstance(c, Migrate)]
    if "relieve_swa" not in tags:
        raise StageFail(f"H: the migrate that relieves the PEGGED swa subpool "
                        f"must be taken; got {tags}")
    if "relieve_full" in tags:
        raise StageFail(f"H: a migrate relieving the HEALTHY 'full' subpool "
                        f"must be dropped (relief targets the bottleneck); {tags}")
    for c in plan:
        if (getattr(c, "acquired", {}) or {}).get("HBM"):
            raise StageFail(
                "H: no relief candidate may ACQUIRE into the pegged HBM "
                f"subpool (it grows the bottleneck); got {c.id} "
                f"acquired={c.acquired}")
    print(_green("  [H] relief targets the pressured subpool only; never "
                 "churns a healthy subpool nor grows the pegged one (SF-3) OK"))


def stage_i_off_budget_consumption_rejected() -> None:
    """ROUND-2 audit: the value knapsack must NOT treat consumption on an
    axis ABSENT from its budget as free.  A migrate that relieves the pegged
    subpool but ACQUIRES into a destination subpool that is not configured
    (or has no room) must be rejected (0 room), never silently over-
    subscribed.  Inject two relief candidates for the pegged 'swa':
      • an over-subscriber: relieves swa, acquires DRAM['xx'] 2GB where DRAM
        only has subpool 'kv' configured → off-budget → MUST be rejected
      • a clean DROP: relieves swa, acquires nothing → MUST be taken
    Without the union-budget fix the over-subscriber's 2GB lands on an axis
    the DP never budgeted → free → wrongly taken."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)
    tracker = ProgramTracker()
    tracker.observe_arrival("S")
    sj = _state_json(
        units=[_unit(uhash="u1", residence=["HBM"], holders=["S"],
                     n_bytes_per_tier={"HBM": {"swa": 1 * GB}}, subpool="swa")],
        hbm={"swa": _sp(9 * GB, 10 * GB)},
        dram={"kv": _sp(0, 40 * GB)},          # NO 'xx' subpool configured
    )
    ev = Event(kind=EventKind.MEMORY_PRESSURE, session="S")
    st = _build_state(sj, tracker, ev)
    inject = [
        Migrate(cost=-10.0, relief={"HBM": {"swa": 1 * GB}},
                acquired={"DRAM": {"xx": 2 * GB}},   # off-budget destination
                id=("oversub", ["DRAM"], ["HBM"]), group="g_over"),
        Migrate(cost=-5.0, relief={"HBM": {"swa": 1 * GB}}, acquired={},
                id=("clean_drop", [], ["HBM"]), group="g_drop"),
    ]
    orig_mc = jd.migrate_candidates
    jd.migrate_candidates = lambda *a, **k: list(inject)
    try:
        plan = jd.joint_decide(st, ev, **kw)
    finally:
        jd.migrate_candidates = orig_mc
    tags = [c.id[0] for c in plan if isinstance(c, Migrate)]
    if "oversub" in tags:
        raise StageFail(
            "I: a migrate acquiring into an UNCONFIGURED destination subpool "
            "(DRAM['xx'], 0 room) must be rejected, not taken for free; "
            f"got {tags}")
    if "clean_drop" not in tags:
        raise StageFail(f"I: the clean no-acquire DROP relieving swa must be "
                        f"taken; got {tags}")
    print(_green("  [I] off-budget consumption is 0-room (rejected), never "
                 "silently free (round-2 audit) OK"))


def stage_j_resume_dedup() -> None:
    """#215 (tracker-authority rework): a resume the daemon has already issued
    must NOT be re-dispatched every event while the dump's overlay lag still
    shows PAUSED (pure waste) — the program_tracker, reconciled against the
    fresh dump, owns this (no parallel dedup cache).  But a clear that never
    lands (lost PUT — the outbound queue does not retry) MUST still recover
    (re-fire) after the bounded window, else the program re-starves.  Drives
    KvScheduler.handle with a FIXED dump (overlay never advances) to model the
    lag/loss, then flips the dump to model the clear landing."""
    GB = 1024 ** 3
    from daemon import kv_scheduler as _kvs
    win = _kvs._RESUME_DEDUP_WINDOW
    tracker = ProgramTracker()
    tracker.pause("R")
    ob = OutboundQueue(sglang_base_url="http://unused", http_client=_DummyHttp())
    sched = kvs.KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                            outbound=ob)
    sched.admission_enabled = True
    sj = _state_json(
        units=[_unit(uhash="uR", residence=["DRAM"], holders=["R"],
                     n_bytes_per_tier={"DRAM": 1 * GB})],
        programs={"R": _program("PAUSED", unit_hashes=["uR"],
                                pre_pause_state="ACTING")},
        hbm={"kv": _sp(int(0.5 * GB), 10 * GB)})   # 5% → headroom, R fits
    router = _StubRouter(sj)
    ev = Event(EventKind.PRESSURE_RESOLVED, session="R")

    # event 1: R PAUSED in the dump and fits → resume fires once; the tracker
    # now reports the resume in flight.
    asyncio.run(sched.handle(ev, router))
    if sched.resume_calls != 1:
        raise StageFail(f"J: first event must resume R once, got "
                        f"{sched.resume_calls}")
    if not tracker.resume_in_flight("R"):
        raise StageFail("J: tracker must report R's resume in flight after dispatch")
    # event 2: SAME dump (overlay lag) → within the dedup window → no re-fire.
    asyncio.run(sched.handle(ev, router))
    if sched.resume_calls != 1:
        raise StageFail(f"#215: a resume re-proposed within the window "
                        f"(overlay lag) must NOT re-fire; got {sched.resume_calls}")
    # keep the SAME dump (clear never lands): the tracker must re-arm after the
    # window and the resume must recover (re-fire) — bounded by win+1 events.
    fired_again_at = None
    for k in range(win + 2):
        asyncio.run(sched.handle(ev, router))
        if sched.resume_calls == 2:
            fired_again_at = k
            break
    if fired_again_at is None:
        raise StageFail(f"#215: a lost clear must recover (re-fire) within the "
                        f"window ({win}); resume_calls stuck at {sched.resume_calls}")
    # the dump finally reflects the clear (R no longer PAUSED): no further
    # resume, and the tracker's in-flight record is pruned.
    calls_before = sched.resume_calls
    router._sj["per_program_usage"]["R"]["state"] = "ACTING"
    asyncio.run(sched.handle(ev, router))
    if sched.resume_calls != calls_before:
        raise StageFail(f"J: once the dump clears PAUSED, no further resume; "
                        f"{sched.resume_calls} vs {calls_before}")
    if tracker.resume_in_flight("R"):
        raise StageFail("J: tracker in-flight record must be pruned once the "
                        "dump confirms the clear (no longer PAUSED)")
    print(_green("  [J] resume dedup via program_tracker: suppresses re-fire in "
                 "the overlay-lag window, recovers a lost clear, prunes on clear "
                 "(#215) OK"))


def stage_k_no_evict_reuse_imminent_tail() -> None:
    """#223: at TOOL_CALL_END the decision set is the caller's session TAIL,
    which is reuse-imminent (the session resumes and extends it next turn —
    DESIGN §7 marks it a *promote* candidate).  Evicting it is a futile
    dump→apply TOCTOU (the frontier leaf gains a device child by apply time →
    sglang rejects remove_hbm_not_device_leaf).  Under value-gating the evict
    is even net-POSITIVE (HBM holding at 0.9 occ is dear, DRAM preserves the
    data), so the value-gate alone does NOT protect it — joint_decide must
    suppress remove-HBM relief specifically at TOOL_CALL_END.  The same hot
    unit under MEMORY_PRESSURE (the cold-unit top-k path) is still evictable,
    so relief is not crippled."""
    from daemon import joint_decide as jd
    GB = 1024 ** 3
    kw = dict(costs=default_costs(), pi_u=1.0e-4,
              theta_hi=0.85, theta_lo=0.70, heartbeat_s=5.0)

    def _plan(kind):
        tracker = ProgramTracker()
        tracker.observe_arrival("S")
        tracker.observe_completion("S")          # ACTING (alive holder)
        sj = _state_json(
            units=[_unit(uhash="tail", residence=["HBM"], holders=["S"],
                         n_bytes_per_tier={"HBM": 1 * GB}, last_access_time=999)],
            programs={"S": _program("ACTING", committed={"kv": 1 * GB},
                                    unit_hashes=["tail"])},
            hbm={"kv": _sp(9 * GB, 10 * GB)},     # 90% → pressured
            time_counter=1000)
        ev = Event(kind=kind, session="S")
        st = _build_state(sj, tracker, ev)
        plan = jd.joint_decide(st, ev, **kw)
        # remove-HBM migrates in the plan (the #223 futile evict)
        return [c for c in plan if isinstance(c, Migrate)
                and Tier.HBM in c.id[2]]

    # net-positive sanity: the evict IS net-positive, so suppression is real
    # work, not vacuous (the value-gate would otherwise take it — proven by
    # the MEMORY_PRESSURE branch below taking it).
    end_evicts = _plan(EventKind.TOOL_CALL_END)
    if end_evicts:
        raise StageFail(
            "#223: TOOL_CALL_END must NOT evict the reuse-imminent session "
            f"tail (futile TOCTOU); got remove-HBM {[c.id[0] for c in end_evicts]}")
    mp_evicts = _plan(EventKind.MEMORY_PRESSURE)
    if not mp_evicts:
        raise StageFail(
            "#223: MEMORY_PRESSURE relief must still evict (the cold-unit "
            "top-k path is not crippled by the TOOL_CALL_END guard); got none")
    print(_green("  [K] reuse-imminent TOOL_CALL_END tail not evicted (futile "
                 "TOCTOU), MEMORY_PRESSURE relief intact (#223) OK"))


# ============================================================ runner

_STAGES = [
    ("A", stage_a_migrate_candidates),
    ("A-leaf", stage_a_leaf_filter),
    ("A-inflight", stage_a_inflight_holder_gate),
    ("B", stage_b_forecast),
    ("C", stage_c_program_candidates),
    ("D", stage_d_joint_decide_select),
    ("D-starve", stage_d_resume_starvation),
    ("D-press-resume", stage_d_resume_under_pressure),
    ("E", stage_e_dp_correctness),
    ("F", stage_f_live_dispatch),
    ("G", stage_g_robustness),
    ("H", stage_h_relief_targets_pressured_subpool),
    ("I", stage_i_off_budget_consumption_rejected),
    ("J", stage_j_resume_dedup),
    ("K", stage_k_no_evict_reuse_imminent_tail),
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
