#!/usr/bin/env python3
"""verify/action_timeline — the DESIGN §3/§7 action-timeline plane (#235).

Proves the predictive-promote-back machinery that S1 depends on:

  A. ActionTimeline heap — schedule / pop_due ordering, due filtering, counters
     (the event-stream is the clock: pop_due(now) returns exactly the actions
     whose due_time <= now, in due order).
  B. _estimate_load_back_s — DISK two-hop load_back from live bw_free.
  C. _schedule_promote_back — TOOL_CALL_START with a payload ETA schedules ONE
     PromoteAction for the caller's exclusive tail at T_start+ETA−load_back−margin;
     no-ETA / no-tail / shared-tail → no schedule (degrade to promote-at-END).
  D. fire_due_action belief-validation — ACTING + demoted tail → Migrate(→HBM);
     not-ACTING / already-HBM / dropped → idempotent no-op (counts as stale).
  E. EventRouter._fire_due_actions — the drain wired into the event worker fires
     the registered callback for every due payload using the event clock.

Pure + in-process (no GPU, no sglang).  Run: python verify/action_timeline/verify.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.base import Tier  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon.action_timeline import ActionTimeline, PromoteAction  # noqa: E402
from daemon.event_router import EventRouter  # noqa: E402
from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402


class StageFail(Exception):
    pass


# ----------------------------------------------------------------- fixtures


def _sp(used: int, cap: int, page: int = 64 * 1024) -> Dict[str, int]:
    return {"used_bytes": used, "cap_bytes": cap,
            "available_bytes": max(0, cap - used),
            "evictable_bytes": used, "page_bytes": page}


def _unit(*, uhash: str, residence: List[str], holders: List[str],
          n_tokens: int = 1000, subpool: str = "kv") -> Dict[str, Any]:
    return {
        "hash": uhash, "residence": list(residence), "n_tokens": n_tokens,
        "n_bytes": {t: {subpool: n_tokens * 2048} for t in residence},
        "last_access_time": 0, "hit_count": 1, "session_ids": list(holders),
        "is_device_leaf": True, "is_host_leaf": True, "is_tree_leaf": True,
    }


def _state_json(*, units: List[Dict[str, Any]],
                programs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    GB = 1024 ** 3
    return {
        "time_counter": 100,
        "throughput_ema": {"prefill_bps": 0.0, "decode_per_program": {}},
        "pool_usage": {
            "HBM": {"subpools": {"kv": _sp(1 * GB, 10 * GB)}},
            "DRAM": {"subpools": {"kv": _sp(1 * GB, 40 * GB)}},
            "DISK": {"subpools": {"kv": _sp(0, 200 * GB)}},
        },
        "per_program_usage": programs or {},
        "units": units,
        "link_stats": {
            link: {"peak_bw_bps": 64 * GB, "recent_throughput_bps": 0.0,
                   "time_since_last_sample_s": 5.0}  # idle → bw_free=peak
            for link in ("HBM->DRAM", "DRAM->HBM", "DRAM->DISK",
                         "DISK->DRAM", "HBM->DISK", "DISK->HBM")
        },
        "tier_holding_cost": {
            t: {"kv": {"h_max_per_byte_sec": 0.0}}
            for t in ("HBM", "DRAM", "DISK")},
    }


def _build(state_json, tracker, event):
    return kvs.build_paper_state(state_json, event=event, tracker=tracker,
                                 unknown_tier_log=set())


class _FakeOutbound:
    """Captures enqueue_migrate batches (the only method fire/dispatch uses)."""
    def __init__(self) -> None:
        self.batches: List[List[Dict[str, Any]]] = []

    def enqueue_migrate(self, actions: List[Dict[str, Any]]) -> str:
        self.batches.append(list(actions))
        return "batch-%d" % len(self.batches)


class _FakeRouter:
    """Minimal router surface the scheduler touches: a timeline, an async
    fetch_state returning a fixed dump, and the threshold attrs build paths
    read indirectly (unused on the fire path)."""
    def __init__(self, state_json=None) -> None:
        self.timeline = ActionTimeline()
        self.due_action_handler = None
        self._state_json = state_json

    async def fetch_state(self) -> Dict[str, Any]:
        return self._state_json


def _mk_sched(tracker, outbound):
    return kvs.KvScheduler(tracker=tracker,
                           sglang_base_url="http://127.0.0.1:30000",
                           outbound=outbound)


# ----------------------------------------------------------------- stages


def stage_a_heap() -> None:
    tl = ActionTimeline()
    assert tl.pending() == 0 and tl.next_due_time() == float("inf")
    tl.schedule(30.0, "c")
    tl.schedule(10.0, "a")
    tl.schedule(20.0, "b")
    if tl.scheduled != 3 or tl.pending() != 3:
        raise StageFail(f"A: scheduled/pending wrong: {tl.scheduled}/{tl.pending()}")
    if tl.next_due_time() != 10.0:
        raise StageFail(f"A: next_due_time {tl.next_due_time()} != 10.0")
    # nothing due before 10
    if tl.pop_due(9.99) != []:
        raise StageFail("A: popped before earliest due")
    # due at 25 → a, b in order (not c)
    got = tl.pop_due(25.0)
    if got != ["a", "b"]:
        raise StageFail(f"A: due-order wrong: {got}")
    if tl.pending() != 1 or tl.fired != 2:
        raise StageFail(f"A: post-pop pending/fired {tl.pending()}/{tl.fired}")
    if tl.pop_due(100.0) != ["c"]:
        raise StageFail("A: tail not drained")
    if tl.pending() != 0:
        raise StageFail("A: heap not empty after full drain")
    print("  A heap: schedule/pop_due ordering + due-filter + counters OK")


def stage_b_loadback() -> None:
    tracker = ProgramTracker(); tracker.observe_arrival("S")
    nb_tokens = 1000
    sj = _state_json(units=[_unit(uhash="u1", residence=["HBM"], holders=["S"],
                                  n_tokens=nb_tokens)])
    st = _build(sj, tracker, Event(kind=EventKind.LLM_PREFILL, session="S"))
    total_bytes = nb_tokens * 2048
    GB = 1024 ** 3
    # idle links → bw_free=peak=64GB/s on each hop; DISK two-hop:
    expect = total_bytes / (64 * GB) + total_bytes / (64 * GB)
    got = kvs._estimate_load_back_s(st, total_bytes)
    if abs(got - expect) > 1e-9:
        raise StageFail(f"B: load_back {got} != two-hop {expect}")
    if kvs._estimate_load_back_s(st, 0) != 0.0:
        raise StageFail("B: zero bytes should be 0 load_back")
    print(f"  B load_back: DISK two-hop = {got*1e3:.4f} ms for {total_bytes}B OK")


def stage_c_schedule() -> None:
    tracker = ProgramTracker(); tracker.observe_arrival("S")
    out = _FakeOutbound(); sched = _mk_sched(tracker, out)
    router = _FakeRouter()

    # exclusive tail of S, currently HBM-resident
    sj = _state_json(units=[_unit(uhash="tail", residence=["HBM"], holders=["S"])])
    ev = Event(kind=EventKind.TOOL_CALL_START, session="S",
               payload={"tool_eta_s": 5.0})
    object.__setattr__(ev, "enqueue_time", 1000.0)  # frozen dataclass
    st = _build(sj, tracker, ev)
    sched._schedule_promote_back(ev, st, router)
    if router.timeline.pending() != 1 or sched.promotes_scheduled != 1:
        raise StageFail(f"C: expected 1 scheduled, got {router.timeline.pending()}")
    pa = router.timeline.pop_due(1e18)[0]
    if not isinstance(pa, PromoteAction) or pa.unit_hashes != ("tail",):
        raise StageFail(f"C: wrong payload {pa}")
    if abs(pa.eta_s - 5.0) > 1e-9:
        raise StageFail(f"C: eta {pa.eta_s}")
    lb = kvs._estimate_load_back_s(st, st.units["tail"].n_bytes)
    expect_due = 1000.0 + max(0.0, 5.0 - lb - kvs._PROMOTE_SAFETY_MARGIN_S)
    # re-derive via a fresh schedule to read the due_time off the heap
    router2 = _FakeRouter()
    sched._schedule_promote_back(ev, st, router2)
    item = router2.timeline._heap[0]
    if abs(item.due_time - expect_due) > 1e-6:
        raise StageFail(f"C: due {item.due_time} != {expect_due}")

    # no ETA → no schedule (degrade to promote-at-END)
    r3 = _FakeRouter()
    ev2 = Event(kind=EventKind.TOOL_CALL_START, session="S", payload={})
    object.__setattr__(ev2, "enqueue_time", 1000.0)
    sched._schedule_promote_back(ev2, st, r3)
    if r3.timeline.pending() != 0:
        raise StageFail("C: no-ETA must not schedule")

    # shared tail (2 holders) → not an exclusive tail → no schedule
    r4 = _FakeRouter()
    sj2 = _state_json(units=[_unit(uhash="shared", residence=["HBM"],
                                   holders=["S", "T"])])
    st2 = _build(sj2, tracker, ev)
    sched._schedule_promote_back(ev, st2, r4)
    if r4.timeline.pending() != 0:
        raise StageFail("C: shared (non-exclusive) tail must not schedule")
    print("  C schedule: ETA→1 promote@T+ETA−loadback; no-ETA/shared→none OK")


async def _fire(sched, router, payload):
    await sched.fire_due_action(payload, router)


def stage_d_fire() -> None:
    # ACTING + demoted (DRAM) tail → Migrate(→HBM)
    tracker = ProgramTracker()
    tracker.observe_arrival("S"); tracker.observe_completion("S")  # → ACTING
    if tracker.state("S") != State.ACTING:
        raise StageFail("D: setup — S not ACTING")
    out = _FakeOutbound(); sched = _mk_sched(tracker, out)
    sj_dram = _state_json(units=[_unit(uhash="tail", residence=["DRAM"],
                                       holders=["S"])])
    router = _FakeRouter(sj_dram)
    pa = PromoteAction(session="S", unit_hashes=("tail",), eta_s=5.0)
    asyncio.run(_fire(sched, router, pa))
    if sched.promotes != 1 or len(out.batches) != 1:
        raise StageFail(f"D: ACTING+DRAM should promote; promotes={sched.promotes} "
                        f"batches={len(out.batches)}")
    act = out.batches[0][0]
    # DESIGN §7 [] -> {HBM}: add HBM, KEEP the DRAM backup (remove nothing).
    if act["hash"] != "tail" or act["add_tiers"] != ["HBM"] or \
            act["remove_tiers"] != []:
        raise StageFail(f"D: wrong promote wire {act}")

    # not ACTING (already REASONING) → stale no-op
    tr2 = ProgramTracker(); tr2.observe_arrival("S")  # REASONING
    out2 = _FakeOutbound(); s2 = _mk_sched(tr2, out2)
    asyncio.run(_fire(s2, _FakeRouter(sj_dram), pa))
    if out2.batches or s2.promotes != 0 or s2.promotes_skipped_stale != 1:
        raise StageFail("D: non-ACTING must be a stale no-op")

    # ACTING but tail already HBM (never demoted) → no-op
    tr3 = ProgramTracker(); tr3.observe_arrival("S"); tr3.observe_completion("S")
    out3 = _FakeOutbound(); s3 = _mk_sched(tr3, out3)
    sj_hbm = _state_json(units=[_unit(uhash="tail", residence=["HBM"],
                                      holders=["S"])])
    asyncio.run(_fire(s3, _FakeRouter(sj_hbm), pa))
    if out3.batches or s3.promotes != 0 or s3.promotes_skipped_stale != 1:
        raise StageFail("D: already-HBM must be a no-op")

    # ACTING but tail dropped (absent from units) → no-op
    tr4 = ProgramTracker(); tr4.observe_arrival("S"); tr4.observe_completion("S")
    out4 = _FakeOutbound(); s4 = _mk_sched(tr4, out4)
    sj_drop = _state_json(units=[_unit(uhash="other", residence=["HBM"],
                                       holders=["S"])])
    asyncio.run(_fire(s4, _FakeRouter(sj_drop), pa))
    if out4.batches or s4.promotes != 0 or s4.promotes_skipped_stale != 1:
        raise StageFail("D: dropped tail must be a no-op")
    print("  D fire: ACTING+DRAM→promote(→HBM); !ACTING/HBM/dropped→stale no-op OK")


def stage_e_router_drain() -> None:
    bus = EventBus()
    router = EventRouter(bus=bus, sglang_base_url="http://127.0.0.1:30000")
    fired: List[Any] = []

    async def _cb(payload, r):
        fired.append(payload)

    router.timeline = ActionTimeline()
    router.due_action_handler = _cb
    router.timeline.schedule(500.0, "due_a")
    router.timeline.schedule(2000.0, "not_yet")

    async def _run():
        # event clock = 1000.0 → only due_a (<=1000) fires
        await router._fire_due_actions(1000.0)
    asyncio.run(_run())
    if fired != ["due_a"]:
        raise StageFail(f"E: drain fired {fired}, expected ['due_a']")
    if router.timeline.pending() != 1:
        raise StageFail("E: not-yet action should remain on heap")
    # unwired timeline is inert (no crash)
    r2 = EventRouter(bus=EventBus(), sglang_base_url="http://x")
    asyncio.run(r2._fire_due_actions(1e18))
    print("  E router drain: event-clock fires due payload, leaves future, "
          "unwired inert OK")


STAGES = [
    ("A heap", stage_a_heap),
    ("B load_back", stage_b_loadback),
    ("C schedule", stage_c_schedule),
    ("D fire", stage_d_fire),
    ("E router drain", stage_e_router_drain),
]


def main() -> int:
    print("=== verify/action_timeline (DESIGN §3/§7 predictive promote, #235) ===")
    failed = 0
    for name, fn in STAGES:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  [{name}] FAIL: {exc}")
            traceback.print_exc()
    print("=" * 60)
    print("RESULT:", "PASS" if failed == 0 else f"FAIL ({failed} stage(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
