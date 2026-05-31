"""G10 regression probe — daemon HBM occupancy = allocator truth.

Self-contained unit-test of the G10 fix (commit landing 2026-05-31):

  sglang `dump_aginfer_state` now emits a `pool_usage` field whose
  HBM.token_usage matches sglang's own `full_token_usage` metric
  (= the value that fires the memory_pressure webhook).

  daemon `parse_state` / `SchedulerState.pool_pressure` pipes that
  value through, and `admission_controller._hbm_occ` prefers it
  over `tier_usage` (which is radix-tree-keyed and ~0 under
  in-flight decode pressure → G10 root cause).

Three sub-probes, each demonstrates a pre-fix FAIL / post-fix PASS.
Run::

    cd /scratch/yuzhou/projects/sglang/dev/aginfer
    python verify/g10/probe.py

Expected: each probe prints PASS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AGINFER = _HERE.parent.parent
sys.path.insert(0, str(_AGINFER))

from baselines.base import SchedulerState, Tier, TierUsage  # noqa: E402
from daemon.admission_controller import AdmissionController  # noqa: E402
from daemon.kv_scheduler import _flatten_per_rank  # noqa: E402


_FAIL = 0
_PASS = 0


def _ok(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        print(f"PASS  {name}")
        _PASS += 1
    else:
        print(f"FAIL  {name}  — {detail}")
        _FAIL += 1


# ----------------------------------------------------------------- probe 1

def probe_1_admission_uses_pool_pressure() -> None:
    """admission._hbm_occ MUST prefer pool_pressure over tier_usage.

    Construct a SchedulerState where the two disagree (radix view says
    HBM is empty, pool_pressure says 0.91).  _hbm_occ must return 0.91.
    """
    state = SchedulerState(
        t=0.0, units={}, tier_usage=TierUsage(
            capacity_bytes={Tier.HBM: 1000, Tier.DRAM: 0, Tier.DISK: 0},
            used_bytes={Tier.HBM: 0, Tier.DRAM: 0, Tier.DISK: 0},
        ),
        event_kind="state_fetched",
        pool_pressure={Tier.HBM: 0.91},
    )
    occ = AdmissionController._hbm_occ(state)
    _ok(
        "probe_1_admission_uses_pool_pressure_when_present",
        abs(occ - 0.91) < 1e-9,
        f"got {occ}, expected 0.91",
    )


# ----------------------------------------------------------------- probe 2

def probe_2_falls_back_when_pool_pressure_empty() -> None:
    """When pool_pressure is empty (older sglang, no pool_usage field),
    _hbm_occ must fall back to tier_usage.used_bytes / cap_bytes.
    """
    state = SchedulerState(
        t=0.0, units={}, tier_usage=TierUsage(
            capacity_bytes={Tier.HBM: 1000, Tier.DRAM: 0, Tier.DISK: 0},
            used_bytes={Tier.HBM: 250, Tier.DRAM: 0, Tier.DISK: 0},
        ),
        event_kind="state_fetched",
        pool_pressure={},
    )
    occ = AdmissionController._hbm_occ(state)
    _ok(
        "probe_2_falls_back_to_tier_usage_when_pool_pressure_empty",
        abs(occ - 0.25) < 1e-9,
        f"got {occ}, expected 0.25",
    )


# ----------------------------------------------------------------- probe 3

def probe_3_multi_rank_aggregates_pool_usage() -> None:
    """_flatten_per_rank must aggregate pool_usage across ranks and
    recompute token_usage = sum(used) / sum(cap)."""
    state = {
        "per_rank": [
            {
                "tier_usage": {
                    "HBM":  {"used_bytes": 0, "cap_bytes": 1000},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 0},
                    "DISK": {"used_bytes": 0, "cap_bytes": 0},
                },
                "pool_usage": {"HBM": {
                    "used_bytes": 900, "cap_bytes": 1000,
                    "available_bytes": 100, "evictable_bytes": 0,
                    "token_usage": 0.9,
                }},
                "units": [],
                "time_counter": 1,
            },
            {
                "tier_usage": {
                    "HBM":  {"used_bytes": 0, "cap_bytes": 1000},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 0},
                    "DISK": {"used_bytes": 0, "cap_bytes": 0},
                },
                "pool_usage": {"HBM": {
                    "used_bytes": 100, "cap_bytes": 1000,
                    "available_bytes": 900, "evictable_bytes": 0,
                    "token_usage": 0.1,
                }},
                "units": [],
                "time_counter": 2,
            },
        ],
    }
    flat = _flatten_per_rank(state)
    hbm = flat.get("pool_usage", {}).get("HBM", {})
    # Aggregate: used 900+100=1000, cap 1000+1000=2000, token_usage=0.5
    _ok(
        "probe_3_multi_rank_pool_usage_aggregates",
        hbm.get("used_bytes") == 1000
        and hbm.get("cap_bytes") == 2000
        and abs(hbm.get("token_usage", 0) - 0.5) < 1e-9,
        f"got {hbm}",
    )


# ----------------------------------------------------------------- probe 4

def probe_4_single_rank_passthrough_no_pool_usage() -> None:
    """Older sglang (no pool_usage field) — _flatten_per_rank must
    pass the snapshot through unchanged when per_rank absent."""
    state = {
        "tier_usage": {
            "HBM":  {"used_bytes": 50, "cap_bytes": 1000},
            "DRAM": {"used_bytes": 0, "cap_bytes": 0},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": [],
        "time_counter": 1,
    }
    flat = _flatten_per_rank(state)
    _ok(
        "probe_4_no_per_rank_passthrough",
        flat is state and "pool_usage" not in flat,
        f"got {flat}",
    )


# ----------------------------------------------------------------- probe 5

def probe_5_parse_state_populates_pool_pressure() -> None:
    """build_paper_state must extract pool_usage.HBM.token_usage into
    SchedulerState.pool_pressure[Tier.HBM]."""
    from daemon.events import Event, EventKind
    from daemon.kv_scheduler import build_paper_state
    from daemon.program_tracker import ProgramTracker
    state_json = {
        "tier_usage": {
            "HBM":  {"used_bytes": 0,    "cap_bytes": 1000},
            "DRAM": {"used_bytes": 0,    "cap_bytes": 0},
            "DISK": {"used_bytes": 0,    "cap_bytes": 0},
        },
        "pool_usage": {"HBM": {
            "used_bytes": 870, "cap_bytes": 1000,
            "available_bytes": 80, "evictable_bytes": 50,
            "token_usage": 0.87,
        }},
        "units": [],
        "page_size": 1,
        "bytes_per_token": 128,
        "time_counter": 0,
    }
    event = Event(kind=EventKind.SESSION_ARRIVAL, session="sess-probe", payload={})
    tracker = ProgramTracker()
    sched = build_paper_state(
        state_json,
        event=event,
        tracker=tracker,
        unknown_tier_log=set(),
    )
    _ok(
        "probe_5_parse_state_populates_pool_pressure_HBM",
        sched.pool_pressure.get(Tier.HBM) == 0.87,
        f"got {sched.pool_pressure}",
    )


# ----------------------------------------------------------------- driver

def main() -> int:
    probe_1_admission_uses_pool_pressure()
    probe_2_falls_back_when_pool_pressure_empty()
    probe_3_multi_rank_aggregates_pool_usage()
    probe_4_single_rank_passthrough_no_pool_usage()
    probe_5_parse_state_populates_pool_pressure()
    print(f"\n{_PASS} pass / {_FAIL} fail")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
