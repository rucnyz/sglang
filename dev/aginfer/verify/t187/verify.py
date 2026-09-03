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
                subpool: str = "kv", hbm_used: Optional[int] = None,
                hbm_cap: int = 10 * 1024 ** 3) -> Dict[str, Any]:
    GB = 1024 * 1024 * 1024
    if hbm_used is None:
        hbm_used = 1 * GB

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
            "HBM": _pool(hbm_used, hbm_cap),
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
    def __init__(self, state_json, *, theta_hi=0.85, theta_lo=0.70,
                 heartbeat_s=5.0):
        self._sj = state_json
        self.observability = None
        # §9 thresholds the joint_decide handler reads (#194).
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.heartbeat_s = heartbeat_s

    async def fetch_state(self):
        return self._sj


# (#194) The old _DemoteAllPolicy / _RaisingPolicy stubs are gone — the
# post-joint handler runs joint_decide (not policy.decide), so the
# migrate path is driven by the real default OursGreedyPolicy under
# pressure, and the C2 "dispatch error" is injected at the outbound
# enqueue (see stage_c2).


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
    must NOT be p_hat=1.0.  Pre-#187 this was 1.0 (State.ENDED != None
    → any_alive True), which would keep the unit in HBM forever after
    its program ended.

    Post-T11 (DESIGN §7 holder-product): an ENDED holder's own
    ``p_access`` contributes EXACTLY 0 (not a softened hits/age prior
    — #187's compromise, superseded in place).  A unit held ONLY by
    ENDED holders therefore has p_hat EXACTLY 0.0 — the product's own
    math, no separate branch needed."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p")
    tracker.end("p")  # State.ENDED
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["p"],
              hit_count=1, last_access_time=1),
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    if p != 0.0:
        raise StageFail(
            f"unit held only by an ENDED program must have p_hat EXACTLY "
            f"0.0 (T11 holder-product), NOT 1.0; got {p}"
        )


def stage_b1_ended_plus_live_survives() -> None:
    """T11 holder-product: the ENDED co-holder contributes the identity
    factor (1-0)=1, so p_hat collapses to EXACTLY the live holder's own
    reuse-based term — "no ad-hoc dilution" (DESIGN §7).  hit_count=2 so
    the live holder's term is non-trivially positive (hit_count=1 would
    give reuse(1)=0 for EITHER state, which cannot distinguish "survives"
    from "diluted to 0")."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_live")     # alive
    tracker.observe_arrival("p_end")
    tracker.end("p_end")                  # ENDED
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["p_end", "p_live"],
              hit_count=2, last_access_time=1),
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    import math as _math
    expected = 1.0 - _math.exp(-kvs._PHAT_REUSE_ALPHA * 1)  # reuse(2)
    if abs(p - expected) > 1e-9:
        raise StageFail(
            f"a live co-holder must keep p_hat at its OWN reuse-based term "
            f"{expected:.4f} (unit survives the ended program, undiluted); "
            f"got {p}"
        )


def stage_b2_never_seen_still_prior() -> None:
    """Regression: a never-seen holder (None) is unchanged by the
    ENDED carve-out — still the reuse-based estimate (#250), not
    forced to 0/1 by liveness alone.  hit_count=2 so the estimate is
    non-trivially positive (see stage_b1 rationale)."""
    tracker = ProgramTracker()
    sj = _state_json(units=[
        _unit(uhash="u", residence=["HBM"], holders=["ghost"],
              hit_count=2, last_access_time=1),
    ])
    s = _build(sj, Event(EventKind.LLM_PREFILL, session=None), tracker)
    p = s.units["u"].p_hat
    import math as _math
    expected = 1.0 - _math.exp(-kvs._PHAT_REUSE_ALPHA * 1)  # reuse(2)
    if abs(p - expected) > 1e-9:
        raise StageFail(
            f"never-seen holder should be the reuse-based estimate "
            f"{expected:.4f}; got {p}")


def stage_b3_carve_out_is_event_agnostic() -> None:
    """audit G2 (blast radius): the ENDED p_hat carve-out fires on
    EVERY event, not just SESSION_END.  On a MEMORY_PRESSURE event
    triggered by an unrelated live program, a leftover unit held only
    by an ENDED program scores p_hat EXACTLY 0 (T11 holder-product; so
    it's a demote candidate) — this is the intended latent-bug fix
    (pre-#187 it was pinned at 1.0 forever).  Guards against a future
    refactor re-pinning ENDED only outside the SESSION_END path."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_end")
    tracker.end("p_end")
    sj = _state_json(units=[
        _unit(uhash="u-ended", residence=["HBM"], holders=["p_end"],
              hit_count=1, last_access_time=1),
    ])
    # NON-SESSION_END event (no session) — the carve-out must still
    # apply during pressure-driven scoring.
    s = _build(sj, Event(EventKind.MEMORY_PRESSURE, session=None), tracker)
    p = s.units["u-ended"].p_hat
    if p != 0.0:
        raise StageFail(
            f"ENDED carve-out must apply on MEMORY_PRESSURE too (event-"
            f"agnostic); got p_hat={p} (want exactly 0.0)"
        )
    # and the unit is a top-k regret candidate (in D_t for pressure)
    if "u-ended" not in s.decision_set:
        raise StageFail(
            f"the leftover ENDED unit should be a pressure demote "
            f"candidate; D_t={s.decision_set}"
        )


# ============================================================ C. composed handler


def stage_c0_handler_ends_migrates_puts() -> None:
    """The handler: end() → migrate (session_scoped) → PUT, with the
    migrate batch enqueued BEFORE the PUT.

    #194: SESSION_END migration is now pressure-gated (§9 joint_decide).
    The fixture sits just over theta_hi (HBM 85%+1MB → bytes_needed≈1MB)
    so p's exclusive unit (2MB HBM) is the chosen demote candidate; the
    shared unit is excluded from D_t (held by q too) so it survives."""
    GB = 1024 ** 3
    MB = 1024 ** 2
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        ob = _new_outbound()
        sched = _sched(tracker, ob, None)   # real default OursGreedyPolicy
        sj = _state_json(
            units=[
                _unit(uhash="excl-p", residence=["HBM"], holders=["p"]),
                _unit(uhash="shared", residence=["HBM"], holders=["p", "q"]),
            ],
            hbm_used=int(8.5 * GB) + 1 * MB, hbm_cap=10 * GB)
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


def stage_c2_migrate_error_still_puts_ended() -> None:
    """audit B1: if the migrate step (kv_scheduler.handle) raises, the
    F5 PUT {ENDED} MUST still be enqueued — sglang has to learn the
    program ended even when the data-plane migrate decision blew up.
    The state transition (already done) and the PUT are the F5
    contract; the migrate is best-effort on top.

    #194 audit: simulate a REAL migrate-DISPATCH error (the post-joint
    handler no longer calls ``policy.decide``, so the old _RaisingPolicy
    was never reached).  Pressured state → joint_decide yields a migrate
    → ``_dispatch_migrate`` calls ``enqueue_migrate``, which we make
    raise; the F5 PUT (a separate ``enqueue_program_paused``) must
    survive."""
    GB = 1024 ** 3
    MB = 1024 ** 2
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p")
        ob = _new_outbound()
        # Real dispatch failure: the migrate enqueue throws (e.g. queue
        # full / serialisation error), but program_paused still works.
        def _boom(*a, **k):
            raise RuntimeError("simulated migrate dispatch failure")
        ob.enqueue_migrate = _boom  # type: ignore[assignment]
        sched = _sched(tracker, ob, None)   # real default OursGreedyPolicy
        sj = _state_json(
            units=[_unit(uhash="excl-p", residence=["HBM"], holders=["p"])],
            hbm_used=int(8.5 * GB) + 1 * MB, hbm_cap=10 * GB)  # pressured
        handler = make_session_end_handler(tracker, ob, sched)
        # The handler must NOT propagate the migrate error.
        await handler(Event(EventKind.SESSION_END, session="p"),
                      _FakeRouter(sj))
        return tracker, _drain(ob)
    tracker, batches = asyncio.run(_go())
    if tracker.state("p") is not State.ENDED:
        raise StageFail("program must still be ENDED despite migrate error")
    eps = [b.endpoint for b in batches]
    if "program_paused" not in eps:
        raise StageFail(
            f"F5 PUT {{ENDED}} must still be enqueued when the migrate "
            f"step raises; got {eps}"
        )


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
        sched = _sched(tracker, ob, None)   # real default OursGreedyPolicy
        attach_kv_scheduler(router, sched)
        attach_session_end_handler(router, tracker, ob, sched)
        # Stub the router's state fetch so handle() builds a real D_t.
        # Pressured just over the router's default theta_hi (0.7) so the
        # joint_decide pressure phase demotes p's exclusive unit (#194).
        GB = 1024 ** 3
        MB = 1024 ** 2
        sj = _state_json(
            units=[_unit(uhash="excl-p", residence=["HBM"], holders=["p"])],
            hbm_used=7 * GB + 1 * MB, hbm_cap=10 * GB)
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
    carve-out (T11 holder-product: ENDED contributes exactly 0):
    V_u(keep) = p_hat·[R(DROP) − R(HBM)] − holding, and R(DROP) >
    R(HBM), so a lower p_hat ⇒ lower keep-value ⇒ the ending program's
    units are demoted/dropped FIRST under pressure.  (The absolute
    keep-vs-demote threshold is a value-rule property tested in
    kv_scheduler_value_rule; here we pin the COMPARATIVE effect, which
    is exactly what SESSION_END relies on.)  We also confirm the
    policy actually puts the ENDED unit in its demote plan.

    hit_count=2 (not 1): hit_count=1 collapses the REASONING/untracked
    reuse-based term to reuse(1)=0 regardless of state, which cannot
    distinguish "ENDED (p_hat=0, holder-product's own zero)" from "live
    (p_hat=reuse(hits), incidentally also 0 at hits=1)" — exactly the
    tie this stage used to hit pre-T11."""
    tracker = ProgramTracker()
    tracker.observe_arrival("p_live")
    tracker.observe_arrival("p_end")
    tracker.end("p_end")
    sj = _state_json(units=[
        _unit(uhash="u-ended", residence=["HBM"], holders=["p_end"],
              hit_count=2, last_access_time=1),   # ENDED → p_hat=0 exactly
        _unit(uhash="u-live", residence=["HBM"], holders=["p_live"],
              hit_count=2, last_access_time=1),    # live  → reuse(2) > 0
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
    """The survival mechanism for live/shared units under SESSION_END
    is D_t EXCLUSION, not a scorer keep-decision: a unit shared by p
    and a live q has holders ⊋ {p}, so it is never in SESSION_END(p)'s
    D_t and the policy never even scores it for demotion.  (This is
    why #187 cannot over-evict a surviving program's KV — every unit
    in p's D_t belongs exclusively to the ending p.)  Driven through
    the real policy + real handler end-to-end."""
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


# ============================================================ G. tracker GC


def stage_g0_handle_gcs_ended_no_units() -> None:
    """#190: kv_scheduler.handle reclaims an ENDED program once its KV
    has fully cleared from the snapshot.  Drives the REAL handle →
    build_paper_state → gc_ended path: p-dead is ENDED and holds no
    unit in the dump → reclaimed; p-live still holds a unit → kept."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p-live")
        tracker.observe_arrival("p-dead")
        tracker.end("p-dead")            # ENDED, but no unit cites it
        ob = _new_outbound()
        sched = _sched(tracker, ob, None)   # real default OursGreedyPolicy
        sj = _state_json(units=[
            _unit(uhash="u", residence=["HBM"], holders=["p-live"]),
        ])
        await sched.handle(Event(EventKind.LLM_PREFILL, session="p-live"),
                           _FakeRouter(sj))
        return tracker
    tracker = asyncio.run(_go())
    if tracker.state("p-dead") is not None:
        raise StageFail(
            "handle() must GC an ENDED program with no live units "
            f"(#190); got {tracker.state('p-dead')}"
        )
    if tracker.state("p-live") is not State.REASONING:
        raise StageFail("a live unit-holding program must NOT be GC'd")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 SESSION_END D_t = session_scoped (exclusive) units", stage_a0_session_end_is_session_scoped),
    ("A1 SESSION_END with only-shared units → empty D_t", stage_a1_only_shared_empty_dt),
    ("B0 ENDED-only holder → p_hat EXACTLY 0 (T11 holder-product)", stage_b0_ended_holder_low_p_hat),
    ("B1 ENDED + live holder → p_hat = live's own term (undiluted)", stage_b1_ended_plus_live_survives),
    ("B2 never-seen holder → reuse-based estimate (regression)", stage_b2_never_seen_still_prior),
    ("B3 ENDED carve-out is event-agnostic (MEMORY_PRESSURE)", stage_b3_carve_out_is_event_agnostic),
    ("C0 handler: end → migrate(session_scoped) → PUT", stage_c0_handler_ends_migrates_puts),
    ("C1 handler kv_scheduler=None → pure F5", stage_c1_no_scheduler_pure_f5),
    ("C2 migrate error still enqueues F5 PUT {ENDED}", stage_c2_migrate_error_still_puts_ended),
    ("D0 composed router runs migrate AND F5", stage_d0_composed_router_runs_migrate_and_f5),
    ("E0 real policy: ENDED lowers keep-value + demotes", stage_e0_real_policy_keep_value_lower_for_ended),
    ("F0 shared unit survives via D_t exclusion", stage_f0_shared_unit_survives),
    ("G0 handle() GCs ENDED-no-units program (#190 bounded tracker)", stage_g0_handle_gcs_ended_no_units),
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
