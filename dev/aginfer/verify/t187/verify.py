"""T187 — SESSION_END migrate D_t (#187, DESIGN §4 / §7 SESSION_END
normal path).

Sibling of F5 (#185).  F5 owns the SESSION_END state-transition +
gate-release + PUT.  This task wires the MIGRATE half: on SESSION_END
for program p, the daemon re-scores D_t = session_scoped_units(p)
(units held ONLY by p) and demotes/drops them — they have no other
holder once p ends, and p contributes 0 to their future p_hat
(DESIGN §4 table, §7 decision_set, "SESSION_END normal path").

Two coupled changes:

  1. `build_paper_state` p_hat rule: an ENDED holder no longer counts
     as "alive" — a unit held only by ended programs falls back to
     the workload-prior hits/age (was: stuck at 1.0 forever post-#185
     because State.ENDED != None made any_alive True).
  2. `_build_decision_set` SESSION_END branch → session_scoped_units
     (exclusive holders == {sid}); shared units excluded (survive p).
  3. The SESSION_END handler composes: tracker.end(p) FIRST (so the
     scorer sees ENDED), THEN kv_scheduler.handle (migrate D_t), THEN
     the PUT {ENDED}.

Stages:

  A. decision_set
    A0 SESSION_END → session_scoped (exclusive) units; SHARED units
       (held by ≥2 programs) excluded
    A1 SESSION_END with only-shared units → empty D_t
  B. p_hat ENDED-exclusion (the scoring anchor)
    B0 unit held ONLY by an explicitly end()-ed program → p_hat is
       the workload-prior (< 1.0), NOT 1.0
    B1 unit held by an ENDED + a LIVE program → p_hat == 1.0
       (live co-holder dominates; the unit survives p)
    B2 (regression) never-seen holder still → workload-prior
  C. composed handler (stub demote-all policy)
    C0 SESSION_END handler → tracker ENDED + migrate batch (for the
       session_scoped units) + PUT {ENDED}, migrate ENQUEUED BEFORE
       the PUT
    C1 handler with kv_scheduler=None → pure F5 (ENDED + PUT, no
       migrate) — back-compat
  D. real composed router (main.py attach order)
    D0 attach_kv_scheduler THEN attach_session_end_handler(…, sched):
       a real SESSION_END routes to the composite → program ENDED +
       sched ran the migrate (migrate_calls advanced) + PUT enqueued
  E. real policy: scoring drives the decision
    E0 an ENDED session-scoped cold HBM unit is DEMOTED/DROPPED by the
       real OursGreedyPolicy; an identical LIVE program's HBM unit is
       NOT (p_hat=1.0 keeps it) — proves the p_hat change, not a stub,
       drives demotion
  F. shared unit survives end-to-end
    F0 p+q hold a unit; SESSION_END(p) via the real policy → that
       unit is NOT in the migrate plan (excluded from D_t)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import Action, Tier  # noqa: E402
from baselines.costs import default_costs  # noqa: E402
from baselines.ours_greedy import OursGreedyPolicy  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon.events import Event, EventKind, EventBus  # noqa: E402
from daemon.event_router import (  # noqa: E402
    EventRouter,
    make_session_end_handler,
    attach_session_end_handler,
)
from daemon.kv_scheduler import KvScheduler, attach_kv_scheduler  # noqa: E402
from daemon.outbound import OutboundBatch, OutboundQueue  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


def _real_policy() -> OursGreedyPolicy:
    return OursGreedyPolicy(default_costs())


# ============================================================ stubs / fixtures


class _Resp:
    status_code = 200
    text = ""
    def json(self): return {}


class _DummyHttp:
    async def post(self, *a, **k): return _Resp()
    async def request(self, *a, **k): return _Resp()
    async def aclose(self): return None


def _new_outbound() -> OutboundQueue:
    return OutboundQueue(sglang_base_url="http://unused", http_client=_DummyHttp())


def _unit(
    *,
    uhash: str,
    residence: List[str],
    holders: List[str],
    n_tokens: int = 1000,
    last_access_time: int = 0,
    hit_count: int = 1,
    subpool: str = "kv",
) -> Dict[str, Any]:
    n_bytes = {t: {subpool: n_tokens * 2048} for t in residence}
    return {
        "hash": uhash,
        "residence": list(residence),
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "last_access_time": last_access_time,
        "hit_count": hit_count,
        "session_ids": list(holders),
    }


def _state_json(*, units: List[Dict[str, Any]], time_counter: int = 100,
                subpool: str = "kv") -> Dict[str, Any]:
    GB = 1024 * 1024 * 1024

    def _pool(used: int, cap: int) -> Dict[str, Any]:
        return {"subpools": {subpool: {
            "used_bytes": used, "cap_bytes": cap,
            "available_bytes": max(0, cap - used),
            "evictable_bytes": used, "page_bytes": 64 * 1024,
        }}}
    return {
        "time_counter": time_counter,
        "throughput_ema": {"prefill_bps": 0.0, "decode_per_program": {}},
        "pool_usage": {
            "HBM": _pool(1 * GB, 10 * GB),
            "DRAM": _pool(1 * GB, 40 * GB),
            "DISK": _pool(0, 200 * GB),
        },
        "per_program_usage": {},
        "units": units,
        "link_stats": {link: {
            "peak_bw_bps": 64 * GB, "recent_throughput_bps": 0.0,
            "time_since_last_sample_s": 5.0,
        } for link in ("HBM->DRAM", "DRAM->HBM", "DRAM->DISK", "DISK->DRAM")},
        "tier_holding_cost": {tier: {subpool: {"h_max_per_byte_sec": 0.0}}
                              for tier in ("HBM", "DRAM", "DISK")},
    }


def _build(sj, event, tracker):
    return kvs.build_paper_state(sj, event=event, tracker=tracker,
                                 unknown_tier_log=set())


class _FakeRouter:
    def __init__(self, state_json):
        self._sj = state_json
        self.observability = None

    async def fetch_state(self):
        return self._sj


class _DemoteAllPolicy:
    """Drops every D_t unit from HBM (add=[], remove=[HBM])."""
    def decide(self, state) -> Action:
        plan = []
        for uid in state.decision_set:
            u = state.units.get(uid)
            if u is not None and Tier.HBM in u.residence:
                plan.append((uid, [], [Tier.HBM]))
        return Action(assignments=plan)


def _sched(tracker, ob, policy):
    return KvScheduler(tracker=tracker, sglang_base_url="http://unused",
                       policy=policy, outbound=ob)


def _drain(ob) -> List[OutboundBatch]:
    out = []
    while ob.queue.qsize():
        out.append(ob.queue.get_nowait())
    return out


# ============================================================ A. decision_set


def stage_a0_session_end_is_session_scoped() -> None:
    tracker = ProgramTracker()
    units = {
        u["hash"]: u for u in [
            _unit(uhash="excl-p", residence=["HBM"], holders=["p"]),
            _unit(uhash="shared", residence=["HBM"], holders=["p", "q"]),
            _unit(uhash="excl-q", residence=["HBM"], holders=["q"]),
        ]
    }
    # _build_decision_set takes the ReuseUnit dict — build it via
    # build_paper_state to get the right types, then re-derive D_t.
    sj = _state_json(units=list(units.values()))
    s = _build(sj, Event(EventKind.SESSION_END, session="p"), tracker)
    dset = set(s.decision_set)
    if dset != {"excl-p"}:
        raise StageFail(
            f"SESSION_END(p) D_t must be p's exclusive units only "
            f"{{excl-p}}; got {dset}"
        )


def stage_a1_only_shared_empty_dt() -> None:
    tracker = ProgramTracker()
    sj = _state_json(units=[
        _unit(uhash="shared", residence=["HBM"], holders=["p", "q"]),
    ])
    s = _build(sj, Event(EventKind.SESSION_END, session="p"), tracker)
    if s.decision_set:
        raise StageFail(
            f"SESSION_END(p) with only shared units → empty D_t; got "
            f"{s.decision_set}"
        )


# ============================================================ B. p_hat ENDED


def stage_b0_ended_holder_low_p_hat() -> None:
    """The anchor: a unit held ONLY by an explicitly end()-ed program
    must fall back to the workload-prior p_hat (< 1.0).  Pre-#187 this
    was 1.0 (State.ENDED != None → any_alive True), which would keep
    the unit in HBM forever after its program ended."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p")
    tracker.end("p")  # State.ENDED
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["p"],
              hit_count=1, last_access_time=1),  # age=99 → prior ≈ 0.01
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    if not (0.0 < p < 0.1):
        raise StageFail(
            f"unit held only by an ENDED program must use the workload-"
            f"prior p_hat (< 0.1), NOT 1.0; got {p}"
        )


def stage_b1_ended_plus_live_survives() -> None:
    tracker = ProgramTracker()
    tracker.observe_arrival("p_live")     # alive
    tracker.observe_arrival("p_end")
    tracker.end("p_end")                  # ENDED
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["p_end", "p_live"],
              hit_count=1, last_access_time=1),
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    if abs(p - 1.0) > 1e-9:
        raise StageFail(
            f"a live co-holder must keep p_hat=1.0 (unit survives the "
            f"ended program); got {p}"
        )


def stage_b2_never_seen_still_prior() -> None:
    """Regression: a never-seen holder (None) is unchanged by the
    ENDED carve-out — still workload-prior."""
    tracker = ProgramTracker()
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["ghost"],
              hit_count=1, last_access_time=1),
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    if not (0.0 < p < 0.1):
        raise StageFail(f"never-seen holder should be workload-prior; got {p}")


# ============================================================ C. composed handler


def stage_c0_handler_ends_migrates_puts() -> None:
    """The handler: end() → migrate (session_scoped) → PUT, with the
    migrate batch enqueued BEFORE the PUT."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DemoteAllPolicy())
        sj = _state_json(units=[
            _unit(uhash="excl-p", residence=["HBM"], holders=["p"]),
            _unit(uhash="shared", residence=["HBM"], holders=["p", "q"]),
        ])
        handler = make_session_end_handler(tracker, ob, sched)
        await handler(Event(EventKind.SESSION_END, session="p"),
                      _FakeRouter(sj))
        return tracker, _drain(ob)
    tracker, batches = asyncio.run(_go())
    if tracker.state("p") is not State.ENDED:
        raise StageFail("handler must transition p to ENDED")
    eps = [b.endpoint for b in batches]
    if "migrate" not in eps:
        raise StageFail(f"handler must enqueue a migrate; got {eps}")
    if "program_paused" not in eps:
        raise StageFail(f"handler must enqueue the PUT; got {eps}")
    # migrate before PUT (DESIGN: migrate then PUT)
    if eps.index("migrate") > eps.index("program_paused"):
        raise StageFail(f"migrate must be enqueued before the PUT; got {eps}")
    # the migrate must target the session-scoped unit only
    mig = next(b for b in batches if b.endpoint == "migrate")
    hashes = {a["hash"] for a in mig.body["actions"]}
    if hashes != {"excl-p"}:
        raise StageFail(
            f"migrate must target p's exclusive units only; got {hashes}"
        )
    put = next(b for b in batches if b.endpoint == "program_paused")
    if put.body.get("state") != "ENDED":
        raise StageFail(f"PUT must be ENDED; got {put.body!r}")


def stage_c1_no_scheduler_pure_f5() -> None:
    """Back-compat: kv_scheduler=None → pure F5 (ENDED + PUT, no
    migrate)."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        ob = _new_outbound()
        handler = make_session_end_handler(tracker, ob, None)
        await handler(Event(EventKind.SESSION_END, session="p"), _FakeRouter({}))
        return tracker, _drain(ob)
    tracker, batches = asyncio.run(_go())
    if tracker.state("p") is not State.ENDED:
        raise StageFail("pure-F5 handler must still end the program")
    eps = [b.endpoint for b in batches]
    if eps != ["program_paused"]:
        raise StageFail(f"kv_scheduler=None → only the PUT; got {eps}")


# ============================================================ D. composed router


def stage_d0_composed_router_runs_migrate_and_f5() -> None:
    """main.py attach order: kv_scheduler blanket-attaches every kind,
    then attach_session_end_handler(…, sched) OVERRIDES SESSION_END
    with the composite.  A real SESSION_END must end the program AND
    run the migrate (sched advanced) AND enqueue the PUT."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        ob = _new_outbound()
        router = EventRouter(bus=EventBus(), sglang_base_url="http://unused")
        sched = _sched(tracker, ob, _DemoteAllPolicy())
        attach_kv_scheduler(router, sched)
        attach_session_end_handler(router, tracker, ob, sched)
        # Stub the router's state fetch so handle() builds a real D_t.
        sj = _state_json(units=[
            _unit(uhash="excl-p", residence=["HBM"], holders=["p"]),
        ])
        async def _fake_fetch():
            return sj
        router.fetch_state = _fake_fetch
        handler = router._handlers.get(EventKind.SESSION_END.value)
        if handler is None:
            raise StageFail("no SESSION_END handler registered")
        before = sched.migrate_calls
        await handler(Event(EventKind.SESSION_END, session="p"), router)
        return tracker, ob, sched, before
    tracker, ob, sched, before = asyncio.run(_go())
    if tracker.state("p") is not State.ENDED:
        raise StageFail("composite must end the program (F5 ran)")
    if sched.migrate_calls <= before:
        raise StageFail(
            "composite must run kv_scheduler.handle (migrate_calls did "
            "not advance — the migrate D_t was skipped)"
        )
    eps = [b.endpoint for b in _drain(ob)]
    if "migrate" not in eps or "program_paused" not in eps:
        raise StageFail(f"composite must enqueue migrate + PUT; got {eps}")


# ============================================================ E. real policy


def stage_e0_real_policy_keep_value_lower_for_ended() -> None:
    """The scoring change (not a stub) drives the decision: with two
    otherwise-IDENTICAL cold HBM units, the real policy assigns a
    strictly LOWER keep-in-HBM value V_u to the one whose only holder
    has ENDED than to the one whose holder is still live (p_hat=1.0).

    This is the direct, cost-model-robust consequence of #187's p_hat
    carve-out: V_u(keep) = p_hat·[R(DROP) − R(HBM)] − holding, and
    R(DROP) > R(HBM), so a lower p_hat ⇒ lower keep-value ⇒ the ending
    program's units are demoted/dropped FIRST under pressure.  (The
    absolute keep-vs-demote threshold is a value-rule property tested
    in kv_scheduler_value_rule; here we pin the COMPARATIVE effect,
    which is exactly what SESSION_END relies on.)  We also confirm the
    policy actually puts the ENDED unit in its demote plan."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_live")
    tracker.observe_arrival("p_end")
    tracker.end("p_end")
    sj = _state_json(units=[
        _unit(uhash="u-ended", residence=["HBM"], holders=["p_end"],
              hit_count=1, last_access_time=1),   # ENDED → low p_hat
        _unit(uhash="u-live", residence=["HBM"], holders=["p_live"],
              hit_count=1, last_access_time=1),    # live  → p_hat 1.0
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    policy = _real_policy()
    v_ended = policy._value(s.units["u-ended"], [Tier.HBM], s)
    v_live = policy._value(s.units["u-live"], [Tier.HBM], s)
    if not (v_ended < v_live):
        raise StageFail(
            f"ENDED holder must lower the keep-in-HBM value: "
            f"V(ended)={v_ended} should be < V(live)={v_live}"
        )
    # And the ENDED unit must appear in the policy's demote plan.
    s.decision_set = ["u-ended"]
    action = policy.decide(s)
    moved = {uid for uid, _add, _rm in action.assignments}
    if "u-ended" not in moved:
        raise StageFail(
            "real policy must demote/drop the ENDED session-scoped unit; "
            f"assignments={action.assignments}"
        )


# ============================================================ F. shared survives


def stage_f0_shared_unit_survives() -> None:
    """End-to-end with the real policy: a unit shared by p and a live
    q is NOT in SESSION_END(p)'s migrate plan (excluded from D_t)."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        tracker.observe_arrival("q")  # q stays live
        ob = _new_outbound()
        sched = _sched(tracker, ob, _real_policy())
        sj = _state_json(units=[
            _unit(uhash="excl-p", residence=["HBM"], holders=["p"],
                  hit_count=1, last_access_time=1),
            _unit(uhash="shared", residence=["HBM"], holders=["p", "q"],
                  hit_count=1, last_access_time=1),
        ])
        handler = make_session_end_handler(tracker, ob, sched)
        await handler(Event(EventKind.SESSION_END, session="p"),
                      _FakeRouter(sj))
        return _drain(ob)
    batches = asyncio.run(_go())
    mig = [b for b in batches if b.endpoint == "migrate"]
    moved = set()
    for b in mig:
        for a in b.body["actions"]:
            moved.add(a["hash"])
    if "shared" in moved:
        raise StageFail(
            f"shared unit (held by live q) must survive SESSION_END(p); "
            f"got migrated {moved}"
        )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 SESSION_END D_t = session_scoped (exclusive) units", stage_a0_session_end_is_session_scoped),
    ("A1 SESSION_END with only-shared units → empty D_t", stage_a1_only_shared_empty_dt),
    ("B0 ENDED-only holder → workload-prior p_hat (< 1.0)", stage_b0_ended_holder_low_p_hat),
    ("B1 ENDED + live holder → p_hat 1.0 (survives)", stage_b1_ended_plus_live_survives),
    ("B2 never-seen holder → workload-prior (regression)", stage_b2_never_seen_still_prior),
    ("C0 handler: end → migrate(session_scoped) → PUT", stage_c0_handler_ends_migrates_puts),
    ("C1 handler kv_scheduler=None → pure F5", stage_c1_no_scheduler_pure_f5),
    ("D0 composed router runs migrate AND F5", stage_d0_composed_router_runs_migrate_and_f5),
    ("E0 real policy: ENDED lowers keep-value + demotes", stage_e0_real_policy_keep_value_lower_for_ended),
    ("F0 shared unit survives SESSION_END(p)", stage_f0_shared_unit_survives),
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
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT187 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT187 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
