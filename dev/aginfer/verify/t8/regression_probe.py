"""T8 regression probe — bisect demos for audit findings.

Same protocol as verify/t7/regression_probe.py:
1. Re-inject the regression we want to catch (via monkey-patch).
2. Run the corresponding tightened assertion.
3. Confirm it FAILS (regression catchable).
4. Restore production code.
5. Confirm assertion PASSES.

Each probe prints ``PASS  <finding>: pre-fix FAIL → post-fix PASS``.
"""
from __future__ import annotations

import asyncio
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_AGINFER_ROOT))

import daemon.admission_controller as adm_mod  # noqa: E402
from baselines.base import Tier  # noqa: E402
from daemon.admission_controller import (  # noqa: E402
    AdmissionController,
    attach_admission_controller,
    shared_aware_prog_scores,
)
from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import EventRouter  # noqa: E402
from daemon.kv_scheduler import (  # noqa: E402
    KvScheduler,
    attach_kv_scheduler,
    build_paper_state,
)
from daemon.program_tracker import ProgramTracker, State  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@asynccontextmanager
async def run_server(app: FastAPI, host: str, port: int):
    cfg = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    if not server.started:
        raise RuntimeError(f"uvicorn never started on :{port}")
    try:
        yield
    finally:
        server.should_exit = True
        await task


def _bisect_outcome(name: str, pre: bool, post: bool) -> None:
    if not pre and post:
        print(f"PASS  {name}: pre-fix FAIL → post-fix PASS")
    else:
        print(f"FAIL  {name}: pre_fix_passed={pre} post_fix_passed={post}")
        raise AssertionError(f"regression probe FAILED for {name}")


# ---------------------------------------------------------------- B1


def probe_b1_holding_tax_restored() -> None:
    """B1: `_value_at_current_tier` must include the holding-tax term
    (paper §7), not just saved-prefill.  Pre-fix: `cap=0` short-
    circuits `holding_unit_cost` to 0, so only the prefill term
    counts.  Post-fix: use real tier_usage from the state.

    Build two units:
      * u-A: small (1 KB), high p_hat (hits=100)
      * u-B: huge (100 MB), low p_hat (hits=1)

    With holding tax INCLUDED, u-B's score is dragged DOWN by the
    holding term (big bytes × long hold time).  With holding tax
    DROPPED (bug), u-B scores via saved_prefill only — and since
    `saved_prefill ∝ p_hat × n_tokens`, the small high-hit unit
    actually wins.

    Score formula sanity:
      * Pre-fix: V = p_hat * (R_DROP - R_HBM) — same SIGN both units.
      * Post-fix: V_B has a huge negative holding term → V_B << V_A.

    Probe: assert program holding u-B scores LOWER than program
    holding u-A.  Pre-fix this fails (no holding tax), post-fix
    passes.
    """
    name = "B1 (holding tax in shared_aware_prog_scores)"

    state = {
        "tier_usage": {
            "HBM": {"used_bytes": 100 * 1024 * 1024, "cap_bytes": 100 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": [
            {  # small + hot
                "hash": "u-A",
                "tier": "HBM",
                "n_tokens": 1,
                "n_bytes": 1024,
                "last_access_time": 100,
                "hit_count": 100,
                "session_ids": ["prog-A"],
            },
            {  # huge + cold
                "hash": "u-B",
                "tier": "HBM",
                "n_tokens": 1,
                "n_bytes": 100 * 1024 * 1024,
                "last_access_time": 100,
                "hit_count": 1,
                "session_ids": ["prog-B"],
            },
        ],
        "time_counter": 101,
    }

    def _check() -> bool:
        from daemon.kv_scheduler import build_paper_state as _bps
        s = _bps(
            state, event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=ProgramTracker(), unknown_tier_log=set(),
        )
        scores = shared_aware_prog_scores(s)
        # FIX: huge cold unit drags prog-B BELOW prog-A.
        return scores["prog-B"] < scores["prog-A"]

    post_fix_passed = _check()

    # PRE-FIX simulation: patch _value_at_current_tier to drop the
    # holding tax (the original bug behavior).
    saved = adm_mod._value_at_current_tier
    from baselines.ours_greedy import reload_cost
    from baselines.costs import default_costs as _default_costs

    def _bug_value(u, state, costs, pi_u):
        # PRE-FIX bug: only saved_prefill, no holding tax.
        return u.p_hat * (
            reload_cost(u, Tier.DROP, costs, pi_u)
            - reload_cost(u, u.tier, costs, pi_u)
        )

    adm_mod._value_at_current_tier = _bug_value
    try:
        pre_fix_passed = _check()
    finally:
        adm_mod._value_at_current_tier = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- B2


def probe_b2_composition_order_safety() -> None:
    """B2: `attach_admission_controller` must FAIL LOUDLY if no T7
    handler is registered for MEMORY_PRESSURE / PRESSURE_RESOLVED.

    Pre-fix: attach_admission silently captures `prior=None`; if
    kv_scheduler attaches later, it overwrites admission's wrap.
    Post-fix: attach_admission raises AttachError (or similar) if
    the kind has no prior handler.

    Probe: bare router with no T7 handler; call attach_admission.
    Pre-fix passes (silently); post-fix raises.
    """
    name = "B2 (attach_admission requires prior kv_scheduler handler)"

    def _check_post() -> bool:
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        tracker = ProgramTracker()
        admission = AdmissionController(tracker=tracker, theta_hi=0.8, theta_lo=0.6)
        # Bare router, no kv_scheduler attach.
        try:
            attach_admission_controller(router, admission)
        except Exception:
            return True  # raised loudly — fix is in place
        return False

    def _check_pre() -> bool:
        # Simulate pre-fix by reproducing the OLD attach inline
        # (silently captures prior=None instead of raising).  The
        # SAME test assertion as POST applies: "attach should raise
        # if no prior".  Under the bug, attach does NOT raise → the
        # assertion fails → return False.
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        tracker = ProgramTracker()
        admission = AdmissionController(tracker=tracker, theta_hi=0.8, theta_lo=0.6)
        raised = False
        try:
            for kind in (EventKind.MEMORY_PRESSURE, EventKind.PRESSURE_RESOLVED):
                prior = router._handlers.get(kind.value)

                async def _composite(evt, r, _prior=prior, _adm=admission):
                    if _prior is not None:
                        await _prior(evt, r)
                    await _adm.handle(evt, r)

                router.set_handler(kind, _composite)
        except Exception:
            raised = True
        return raised  # PRE-fix: no raise → False; assertion fails

    post_fix_passed = _check_post()
    pre_fix_passed = _check_pre()
    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- N4


async def probe_n4_drain_resumes_per_event() -> None:
    """N4: `pressure_resolved` is edge-triggered on sglang's side.
    If admission only resumes ONE program per event, paused programs
    strand forever once the state stabilises at OK.

    Pre-fix: 5 paused programs + one `pressure_resolved` → 1 resumed,
    4 stranded indefinitely.
    Post-fix: drain FIFO until next resume would cross theta_hi
    (paper §9 / T8 README §3 spec).

    Probe: pause 5 programs, fire ONE pressure_resolved with low occ
    state, assert all 5 resumed (state is well below theta_lo so
    full drain is safe).
    """
    name = "N4 (pressure_resolved drains FIFO until theta_hi headroom)"

    n_programs = 5
    pressure_state = {
        "tier_usage": {
            "HBM": {"used_bytes": 1024, "cap_bytes": 1 << 30},  # ~0 occ
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": [
            {
                "hash": f"u-tail-{i}",
                "tier": "HBM",
                "n_tokens": 1,
                "n_bytes": 1024,
                "last_access_time": 0,
                "hit_count": 1,
                "session_ids": [f"prog-{i}"],
            }
            for i in range(n_programs)
        ],
        "time_counter": 1,
    }
    state_holder = {"state": pressure_state}
    stub_app = FastAPI()

    @stub_app.get("/aginfer/state")
    async def _s() -> Any:
        return state_holder["state"]

    @stub_app.post("/aginfer/migrate")
    async def _m(raw: Request) -> Any:
        await raw.body()
        return {"applied": 0, "applied_hashes": [], "skipped": []}

    async def _run() -> int:
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
        for i in range(n_programs):
            tracker.observe_arrival(f"prog-{i}")
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url=url)
        sched = KvScheduler(tracker=tracker, sglang_base_url=url)
        attach_kv_scheduler(router, sched)
        admission = AdmissionController(
            tracker=tracker, theta_hi=0.8, theta_lo=0.5
        )
        attach_admission_controller(router, admission)
        # Pre-pause 5 programs into the FIFO (simulate prior pressure events).
        for i in range(n_programs):
            tracker.pause(f"prog-{i}")
            admission._paused_fifo.append(f"prog-{i}")

        await router.start()
        async with run_server(stub_app, "127.0.0.1", port):
            # One pressure_resolved event.
            await router.bus.emit(
                Event(kind=EventKind.PRESSURE_RESOLVED,
                      payload={"state": "OK", "occ": 0.0})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            await router.stop()
            await sched.aclose()
        # T6 contract: tracker.resume() doesn't transition state out
        # of PAUSED until the next observe_arrival.  So we check the
        # admission's FIFO length (popped on each resume).
        return len(admission._paused_fifo)

    still_paused_post = await _run()
    post_fix_passed = (still_paused_post == 0)

    # PRE-FIX: monkey-patch _on_resolved to ONLY resume the oldest one.
    saved = AdmissionController._on_resolved

    async def _bug_on_resolved(self, event, router_):
        self._paused_fifo = [
            pid for pid in self._paused_fifo
            if self.tracker.state(pid) == State.PAUSED
        ]
        if not self._paused_fifo:
            return
        state_json = await router_.fetch_state()
        sched_state = build_paper_state(
            state_json, event=event, tracker=self.tracker,
            unknown_tier_log=self._unknown_tier_log,
        )
        occ = self._hbm_occ(sched_state)
        if occ >= self.theta_lo:
            return
        victim = self._paused_fifo.pop(0)
        self.tracker.resume(victim)
        self.resume_decisions += 1

    AdmissionController._on_resolved = _bug_on_resolved
    try:
        still_paused_pre = await _run()
    finally:
        AdmissionController._on_resolved = saved
    pre_fix_passed = (still_paused_pre == 0)

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- R2-M1


def probe_r2_m1_step1_catches_holding_tax_revert() -> None:
    """R2-M1: round-1 verify step [1]'s numerical assertion was a
    tautology (computed expected via the same function the
    production code calls).  The round-2 rewrite hand-derives V_u
    from paper §7 primitives (`reload_cost` + `holding_unit_cost`
    directly), so a regression in `_value_at_current_tier` is
    catchable.

    This probe verifies the NEW step [1] catches a B1-style revert
    (drop holding tax).  Pre-fix uses the round-1 tautology (re-
    importing _value_at_current_tier); post-fix uses the round-2
    hand-derived assertion.
    """
    name = "R2-M1 (verify step [1] catches holding-tax revert, not tautological)"

    from baselines.costs import default_costs
    from baselines.ours_greedy import reload_cost, holding_unit_cost
    from daemon.kv_scheduler import build_paper_state as _bps

    # The state from `make_pressure_state(n_programs=4)` equivalent:
    state = {
        "tier_usage": {
            "HBM": {"used_bytes": 35_651_584, "cap_bytes": 33_554_432},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": [
            {
                "hash": "u-shared",
                "tier": "HBM",
                "n_tokens": 1024,
                "n_bytes": 2097152,
                "last_access_time": 100,
                "hit_count": 200,
                "session_ids": [f"prog-{i}" for i in range(4)],
            },
            *[
                {
                    "hash": f"u-tail-{i}",
                    "tier": "HBM",
                    "n_tokens": 4096,
                    "n_bytes": 8388608,
                    "last_access_time": 100 - i,
                    "hit_count": (4 - i) * 8,
                    "session_ids": [f"prog-{i}"],
                }
                for i in range(4)
            ],
        ],
        "time_counter": 200,
    }
    s_built = _bps(
        state,
        event=Event(kind=EventKind.MEMORY_PRESSURE),
        tracker=ProgramTracker(),
        unknown_tier_log=set(),
    )

    def _paper_v_u(u) -> float:
        save = u.p_hat * (
            reload_cost(u, Tier.DROP, default_costs(), 1.0e-4)
            - reload_cost(u, u.tier, default_costs(), 1.0e-4)
        )
        used = s_built.tier_usage.used_bytes.get(u.tier, 0)
        cap = s_built.tier_usage.capacity_bytes.get(u.tier, 0)
        h = holding_unit_cost(u.tier, used, cap, default_costs())
        hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
        return save - h * u.n_bytes * hold_time

    def _check_with_step1_logic(use_paper_primitives: bool) -> bool:
        """Replay verify step [1]'s assertion.

        ``use_paper_primitives=True``: round-2 hand-derived (the FIX).
        ``use_paper_primitives=False``: round-1 tautology (used the
        function under test to compute expected).
        """
        scores = shared_aware_prog_scores(s_built)
        v_shared = (
            _paper_v_u(s_built.units["u-shared"])
            if use_paper_primitives
            else adm_mod._value_at_current_tier(
                s_built.units["u-shared"], s_built, default_costs(), 1.0e-4
            )
        )
        per_holder = v_shared / 4
        for i in range(4):
            tail = s_built.units[f"u-tail-{i}"]
            v_tail = (
                _paper_v_u(tail)
                if use_paper_primitives
                else adm_mod._value_at_current_tier(
                    tail, s_built, default_costs(), 1.0e-4
                )
            )
            expected = v_tail + per_holder
            actual = scores[f"prog-{i}"]
            if abs(actual - expected) >= 1e-9:
                return False  # assertion would fail
        return True  # assertion passes

    # Simulate a B1-revert: `_value_at_current_tier` drops the holding
    # term (only saved_prefill).
    saved = adm_mod._value_at_current_tier

    def _bug_value(u, state, costs, pi_u):
        return u.p_hat * (
            reload_cost(u, Tier.DROP, costs, pi_u)
            - reload_cost(u, u.tier, costs, pi_u)
        )

    adm_mod._value_at_current_tier = _bug_value
    try:
        # The contract: "does the verify step CATCH the B1 revert?"
        # = "does the assertion FAIL under the bug?"
        # Pre-fix (tautology): assertion passes under bug → does NOT
        # catch → pre_fix_passed = False.
        pre_fix_passed = not _check_with_step1_logic(use_paper_primitives=False)
        # Post-fix (paper primitives): assertion fails under bug →
        # DOES catch → post_fix_passed = True.
        post_fix_passed = not _check_with_step1_logic(use_paper_primitives=True)
    finally:
        adm_mod._value_at_current_tier = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- R2-M2


def probe_r2_m2_composite_overwrite_refused() -> None:
    """R2-M2: after `attach_admission_controller`, any future
    `router.set_handler(MEMORY_PRESSURE/PRESSURE_RESOLVED, X)` (e.g.,
    by a future T9/T10 subsystem registering its own handler) would
    silently REPLACE the admission composite — admission stops
    firing, no log.  Symmetric to round-1 B2 (which only guarded
    the "before" direction).

    POST-FIX: `set_handler` raises `RuntimeError` on overwrite of a
    wrapped composite unless `force=True`.
    PRE-FIX: silent overwrite.

    Probe: attach kv_scheduler + admission, then attempt to overwrite
    MEMORY_PRESSURE with a fresh stub handler.  Post-fix raises;
    pre-fix succeeds and admission's composite is gone.
    """
    name = "R2-M2 (set_handler refuses overwrite of wrapped composite)"

    def _check_post() -> bool:
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        tracker = ProgramTracker()
        sched = KvScheduler(tracker=tracker, sglang_base_url="http://x")
        attach_kv_scheduler(router, sched)
        admission = AdmissionController(
            tracker=tracker, theta_hi=0.8, theta_lo=0.6
        )
        attach_admission_controller(router, admission)

        async def _new_handler(evt, r):
            pass

        try:
            router.set_handler(EventKind.MEMORY_PRESSURE, _new_handler)
        except RuntimeError:
            return True  # raised loudly — fix is in place
        return False  # silently overwrote — bug

    def _check_pre() -> bool:
        # Simulate pre-fix by reproducing the OLD set_handler inline
        # (no overwrite-guard).  The assertion is the SAME as POST.
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        tracker = ProgramTracker()
        sched = KvScheduler(tracker=tracker, sglang_base_url="http://x")
        attach_kv_scheduler(router, sched)
        admission = AdmissionController(
            tracker=tracker, theta_hi=0.8, theta_lo=0.6
        )
        attach_admission_controller(router, admission)

        async def _new_handler(evt, r):
            pass

        # Bypass the guard: simulate the OLD behavior by direct dict
        # mutation, demonstrating that without the guard the composite
        # disappears silently.
        raised = False
        try:
            router._handlers[EventKind.MEMORY_PRESSURE.value] = _new_handler
        except Exception:
            raised = True
        return raised  # pre-fix: no raise → False; assertion fails

    post_fix_passed = _check_post()
    pre_fix_passed = _check_pre()
    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- main


def main() -> None:
    print("=== T8 regression_probe: audit round-1 + round-2 bisect demos ===")
    print()
    print("--- round 1 ---")
    probe_b1_holding_tax_restored()
    probe_b2_composition_order_safety()
    asyncio.run(probe_n4_drain_resumes_per_event())
    print()
    print("--- round 2 ---")
    probe_r2_m1_step1_catches_holding_tax_revert()
    probe_r2_m2_composite_overwrite_refused()
    print()
    print("=== T8 regression_probe PASSED ===")


if __name__ == "__main__":
    main()
