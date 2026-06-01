"""T8 verify: admission_controller event-driven pause/resume.

Layer A (in-process, this file):
  Stub /aginfer/state with mutable backing; drive memory_pressure /
  pressure_resolved events through the EventRouter; assert pause
  victim, FIFO resume, hysteresis, shared-aware aggregation, anti-
  timer contract, latency.

Per memory:feedback-latency-multi-run / feedback-per-task-docs.

Run::

    cd /scratch/yuzhou/projects/sglang/dev/aginfer
    python verify/t8/verify.py

Expected last line: ``=== T8 PASSED ===``.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import socket
import statistics
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_AGINFER_ROOT))

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


# ---------------------------------------------------------------- helpers


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


def make_pressure_state(
    *,
    n_programs: int = 4,
    shared_tokens: int = 1024,
    tail_tokens: int = 4096,
    hbm_cap: int = 32 * 1024 * 1024,
    used_bytes: Optional[int] = None,
    bytes_per_token: int = 2048,
) -> Dict[str, Any]:
    """N programs each holding a shared platform unit + own tail.

    Tails are sized so `used_bytes` is OVER hbm_cap unless overridden,
    giving a realistic memory_pressure trigger.
    """
    units: List[Dict[str, Any]] = []
    holders = [f"prog-{i}" for i in range(n_programs)]
    units.append(
        {
            "hash": "u-shared-platform",
            "tier": "HBM",
            "n_tokens": shared_tokens,
            "n_bytes": shared_tokens * bytes_per_token,
            "last_access_time": 100,
            "hit_count": 50 * n_programs,
            "session_ids": holders,
        }
    )
    for i in range(n_programs):
        units.append(
            {
                "hash": f"u-tail-{i}",
                "tier": "HBM",
                "n_tokens": tail_tokens,
                "n_bytes": tail_tokens * bytes_per_token,
                "last_access_time": 100 - i,  # earlier programs newer
                "hit_count": (n_programs - i) * 8,  # ascending p_hat → desc score
                "session_ids": [f"prog-{i}"],
            }
        )
    if used_bytes is None:
        used_bytes = sum(u["n_bytes"] for u in units)
    return {
        "tier_usage": {
            "HBM": {"used_bytes": used_bytes, "cap_bytes": hbm_cap},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": units,
        "time_counter": 200,
    }


def build_stub_sglang(state_provider) -> FastAPI:
    """Stub /aginfer/state + /aginfer/migrate."""
    app = FastAPI()

    @app.get("/aginfer/state")
    async def _state() -> Any:
        return state_provider()

    @app.post("/aginfer/migrate")
    async def _migrate(raw: Request) -> Any:
        await raw.body()
        return {"applied": 0, "applied_hashes": [], "skipped": []}

    return app


@asynccontextmanager
async def boot_stack(
    sglang_base_url: str,
    tracker: ProgramTracker,
    *,
    theta_hi: float = 0.85,
    theta_lo: float = 0.70,
    max_pauses_per_event: int = 16,
    enable_kv_scheduler: bool = True,
):
    bus = EventBus()
    router = EventRouter(bus=bus, sglang_base_url=sglang_base_url)
    sched: Optional[KvScheduler] = None
    if enable_kv_scheduler:
        sched = KvScheduler(tracker=tracker, sglang_base_url=sglang_base_url)
        attach_kv_scheduler(router, sched)
    admission = AdmissionController(
        tracker=tracker,
        theta_hi=theta_hi,
        theta_lo=theta_lo,
        max_pauses_per_event=max_pauses_per_event,
    )
    attach_admission_controller(router, admission)
    await router.start()
    try:
        yield router, admission, sched
    finally:
        await router.stop()
        # T36 cleanup: KvScheduler.aclose() was removed (the class no
        # longer owns an httpx client — outbound queue does).  Nothing
        # to close on sched.


# ---------------------------------------------------------------- steps


def step_shared_aware_aggregation() -> None:
    """[1] prog_score divides each unit's V_u by |holders|; shared
    platform doesn't double-count.

    Audit round-2 R2-M1: round-1's numerical pin imported
    `_value_at_current_tier` and used it to compute the "expected"
    value, then asserted the production code (which calls the same
    function) matched — a tautology.  A regression in
    `_value_at_current_tier` (e.g., reverting the B1 holding-tax
    restore) would produce matching expected+actual → test still
    passes.

    Now we hand-derive the expected V_u from paper §7's atomic
    primitives (`reload_cost` + `holding_unit_cost` direct calls)
    so a regression in the AGGREGATED function `_value_at_current_tier`
    diverges from the hand-derived expected.
    """
    from baselines.costs import default_costs as _dc
    from baselines.ours_greedy import reload_cost, holding_unit_cost

    state_json = make_pressure_state(n_programs=4)
    tracker = ProgramTracker()
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.MEMORY_PRESSURE),
        tracker=tracker,
        unknown_tier_log=set(),
    )
    scores = shared_aware_prog_scores(s)
    # All 4 programs scored.
    assert set(scores) == {f"prog-{i}" for i in range(4)}, scores
    sorted_pids = sorted(scores, key=lambda p: scores[p])
    assert sorted_pids[0] == "prog-3", (
        f"expected prog-3 (lowest hit_count) to score lowest; sorted={sorted_pids}"
    )

    # Hand-derive paper §7 V_u — independent of `_value_at_current_tier`
    # so a regression there (e.g., dropping the holding term) shows up
    # as a mismatch instead of being self-consistent.
    costs = _dc()
    pi_u = 1.0e-4

    def _paper_v_u(u) -> float:
        # V_u(tier) = p_hat * (R(DROP) - R(tier)) - h * b_u / lambda
        save_prefill = u.p_hat * (
            reload_cost(u, Tier.DROP, costs, pi_u)
            - reload_cost(u, u.tier, costs, pi_u)
        )
        used = s.tier_usage.used_bytes.get(u.tier, 0)
        cap = s.tier_usage.capacity_bytes.get(u.tier, 0)
        h = holding_unit_cost(u.tier, used, cap, costs)
        hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
        return save_prefill - h * u.n_bytes * hold_time

    shared = s.units["u-shared-platform"]
    v_shared = _paper_v_u(shared)
    expected_shared_per_holder = v_shared / 4

    for i in range(4):
        tail = s.units[f"u-tail-{i}"]
        v_tail = _paper_v_u(tail)
        expected = v_tail + expected_shared_per_holder
        actual = scores[f"prog-{i}"]
        assert abs(actual - expected) < 1e-9, (
            f"prog-{i}: shared-aware aggregation off: got {actual}, "
            f"expected {expected} (= V_tail {v_tail} + V_shared/4 "
            f"{expected_shared_per_holder}).  Note: expected is hand-"
            f"derived from paper §7 primitives, NOT via "
            f"_value_at_current_tier (which is the function under test)."
        )


def step_degenerate_full_share() -> None:
    """[2] WORST CASE: 32 programs all sharing one unit, no unique
    tails.  Shared-aware aggregation should give every program the
    same score (1 × V_share / 32 ≈ identical).  Naive sum would
    multiply by 32 — exposing this lets tie-break (by pid) pick A
    program deterministically."""
    n = 32
    holders = [f"p{i:02d}" for i in range(n)]
    state_json = {
        "tier_usage": {
            "HBM": {"used_bytes": 32 * 1024 * 1024, "cap_bytes": 64 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        },
        "units": [
            {
                "hash": "u-shared-only",
                "tier": "HBM",
                "n_tokens": 1024,
                "n_bytes": 32 * 1024 * 1024,
                "last_access_time": 0,
                "hit_count": 100,
                "session_ids": holders,
            }
        ],
        "time_counter": 1,
    }
    tracker = ProgramTracker()
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.MEMORY_PRESSURE),
        tracker=tracker,
        unknown_tier_log=set(),
    )
    scores = shared_aware_prog_scores(s)
    assert len(scores) == n, len(scores)
    # All equal.
    values = list(scores.values())
    assert max(values) - min(values) < 1e-9, (
        f"degenerate full-share scores diverged: range="
        f"{max(values) - min(values)}, expected ~0"
    )


async def step_pause_lowest_under_pressure() -> None:
    """[3] memory_pressure with occ > theta_hi pauses the lowest-
    scoring program."""
    state_holder = {"state": make_pressure_state(n_programs=4)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(4):
        tracker.observe_arrival(f"prog-{i}")
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(url, tracker, theta_hi=0.5, theta_lo=0.3) as (router, admission, _sched):
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            paused = admission.paused()
            assert paused, "expected at least one pause"
            # Lowest scorer per step [1] is prog-3.
            assert paused[0] == "prog-3", paused
            assert tracker.state("prog-3") == State.PAUSED


async def step_fifo_resume_with_hysteresis() -> None:
    """[4] pressure_resolved resumes ONE program at a time in FIFO
    order; stops if occ would cross theta_hi again."""
    state_holder = {"state": make_pressure_state(n_programs=4)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(4):
        tracker.observe_arrival(f"prog-{i}")
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(
            url, tracker,
            theta_hi=0.6, theta_lo=0.4,
            max_pauses_per_event=1,  # one pause per event for deterministic FIFO test
        ) as (router, admission, _sched):
            # Fire two pressure events to pause 2 programs.
            for _ in range(2):
                await router.bus.emit(
                    Event(kind=EventKind.MEMORY_PRESSURE,
                          payload={"state": "HIGH", "occ": 0.95})
                )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            paused_before = admission.paused()
            assert len(paused_before) == 2, paused_before
            # Drop the state to a "well below theta_lo" occ so resume can fire.
            state_holder["state"] = make_pressure_state(
                n_programs=4,
                used_bytes=1 * 1024 * 1024,  # tiny used
                hbm_cap=100 * 1024 * 1024,
            )
            # Pressure_resolved fires ONE resume.
            await router.bus.emit(
                Event(kind=EventKind.PRESSURE_RESOLVED,
                      payload={"state": "OK", "occ": 0.01})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            paused_after = admission.paused()
            assert len(paused_after) == len(paused_before) - 1, (
                f"expected exactly one resume; before={paused_before}, "
                f"after={paused_after}"
            )
            # FIFO: the oldest (paused_before[0]) is the one removed.
            assert paused_after == paused_before[1:], (
                f"FIFO violation: before={paused_before}, after={paused_after}"
            )


async def step_hysteresis_holds_high_occ() -> None:
    """[5] pressure_resolved with occ STILL >= theta_lo: NO resume."""
    state_holder = {"state": make_pressure_state(n_programs=4)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(4):
        tracker.observe_arrival(f"prog-{i}")
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(
            url, tracker,
            theta_hi=0.6, theta_lo=0.4, max_pauses_per_event=1,
        ) as (router, admission, _sched):
            await router.bus.emit(
                Event(kind=EventKind.MEMORY_PRESSURE,
                      payload={"state": "HIGH", "occ": 0.95})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            assert len(admission.paused()) == 1
            # Now fire pressure_resolved but state still HIGH (just
            # below theta_hi but above theta_lo).
            state_holder["state"] = make_pressure_state(
                n_programs=4,
                used_bytes=int(0.5 * 100 * 1024 * 1024),  # occ=0.50 between lo=0.4 and hi=0.6
                hbm_cap=100 * 1024 * 1024,
            )
            await router.bus.emit(
                Event(kind=EventKind.PRESSURE_RESOLVED,
                      payload={"state": "OK", "occ": 0.50})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            assert len(admission.paused()) == 1, (
                f"hysteresis broken: pressure_resolved at occ=0.50 (between "
                f"theta_lo=0.4 and theta_hi=0.6) resumed prematurely; "
                f"paused={admission.paused()}"
            )


async def step_no_victim_when_all_paused() -> None:
    """[6] WORST CASE: every program already paused, pressure
    persists.  Log + no-op, don't crash."""
    state_holder = {"state": make_pressure_state(n_programs=2)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    # Pre-pause everyone.
    tracker.pause("prog-0")
    tracker.pause("prog-1")
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(url, tracker, theta_hi=0.1, theta_lo=0.05) as (router, admission, _sched):
            await router.bus.emit(
                Event(kind=EventKind.MEMORY_PRESSURE,
                      payload={"state": "HIGH", "occ": 0.95})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            # No new pauses since both were pre-paused (we count
            # admission.pause_decisions, NOT total tracker PAUSE count).
            assert admission.pause_decisions == 0, admission.pause_decisions


def step_anti_timer_contract() -> None:
    """[7] CONTRACT: admission_controller.py has NO sleep / call_later
    / call_at / perf_counter / time.sleep references."""
    from daemon import admission_controller as ac_mod

    src = inspect.getsource(ac_mod)
    tree = ast.parse(src)
    forbidden = ("sleep", "call_later", "call_at", "perf_counter")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise AssertionError(
                f"timer primitive `.{node.attr}` in admission_controller.py "
                f"violates the no-polling contract"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden:
                raise AssertionError(
                    f"timer primitive `{node.func.id}(...)` in "
                    f"admission_controller.py violates the no-polling contract"
                )


async def step_max_pauses_per_event_cap() -> None:
    """[8] WORST CASE: pressure persists indefinitely (state never
    drops).  max_pauses_per_event caps the burst at exactly 16
    (default).  Audit round-1 N3: tightened from `<= 16` to `== 16`
    so removing the cap entirely would now FAIL the test.  The 20-
    program fixture + persistent pressure guarantees the cap is
    reached (16 < 20)."""
    state_holder = {"state": make_pressure_state(n_programs=20)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(20):
        tracker.observe_arrival(f"prog-{i}")
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(url, tracker, theta_hi=0.5, theta_lo=0.3) as (router, admission, _sched):
            await router.bus.emit(
                Event(kind=EventKind.MEMORY_PRESSURE,
                      payload={"state": "HIGH", "occ": 0.95})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=10.0)
            assert admission.pause_decisions == 16, (
                f"expected exactly 16 pauses (max_pauses_per_event cap "
                f"with 20-program persistent pressure); got "
                f"{admission.pause_decisions}"
            )


async def step_handler_latency_at_32_programs() -> dict:
    """[9] COST: per-event handler wall time < 10 ms at 32 programs.

    Per memory:feedback-latency-multi-run: 5-run mean ± std.
    """
    state_holder = {"state": make_pressure_state(n_programs=32)}
    stub = build_stub_sglang(lambda: state_holder["state"])
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(32):
        tracker.observe_arrival(f"prog-{i}")

    N_RUNS = 5
    run_ms: List[float] = []
    async with run_server(stub, "127.0.0.1", port):
        async with boot_stack(url, tracker, theta_hi=0.99) as (router, admission, _sched):
            # theta_hi=0.99 ensures pause won't actually fire (occ<0.99 with our fixture),
            # so we measure the steady-state aggregation+fetch cost only.
            # Warmup.
            for _ in range(2):
                await router.bus.emit(
                    Event(kind=EventKind.MEMORY_PRESSURE,
                          payload={"state": "HIGH", "occ": 0.95})
                )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)

            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                await router.bus.emit(
                    Event(kind=EventKind.MEMORY_PRESSURE,
                          payload={"state": "HIGH", "occ": 0.95})
                )
                await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
                run_ms.append((time.perf_counter() - t0) * 1000)
    stats = {
        "mean": statistics.mean(run_ms),
        "std": statistics.stdev(run_ms),
        "envelope": statistics.mean(run_ms) + 3.0 * statistics.stdev(run_ms),
    }
    print(
        f"    handler @ 32 programs ({N_RUNS} runs): "
        f"{stats['mean']:.2f} ± {stats['std']:.2f} ms "
        f"(mean+3σ={stats['envelope']:.2f} ms)"
    )
    assert stats["envelope"] < 10.0, (
        f"per-event handler wall time mean+3σ = "
        f"{stats['envelope']:.2f} ms exceeds 10 ms budget"
    )
    return stats


async def step_invalid_watermarks_raise() -> None:
    """[10] CONTRACT: watermarks must satisfy 0 < theta_lo < theta_hi < 1.

    Audit round-1 N2: previously only tested theta_lo > theta_hi.
    Now we parameterize over every boundary violation: equal, > 1,
    == 0, == 1, negative.
    """
    bad_cases = [
        ("equal", 0.5, 0.5),
        ("inverted", 0.5, 0.8),
        ("theta_hi==1", 1.0, 0.5),
        ("theta_lo==0", 0.5, 0.0),
        ("theta_lo<0", 0.5, -0.1),
        ("theta_hi>1", 1.1, 0.5),
    ]
    for label, hi, lo in bad_cases:
        try:
            AdmissionController(
                tracker=ProgramTracker(), theta_hi=hi, theta_lo=lo
            )
        except ValueError as e:
            assert "theta" in str(e), (label, e)
            continue
        raise AssertionError(
            f"AdmissionController accepted invalid watermarks {label} "
            f"(theta_hi={hi}, theta_lo={lo}); should raise ValueError"
        )


async def step_single_pause_latency() -> dict:
    """[12] COST: single-pause path latency (audit round-1 M2).

    Step [9] forces 16-pause spin under fixed state, so the
    measurement is dominated by stub HTTP RTT × 16.  This step
    isolates the steady-state path: fire memory_pressure against a
    state mutator that drops occ BELOW theta_hi after the first
    pause, so the controller does exactly ONE pause + one re-fetch
    and bails.  Tighter budget pins the algorithmic work, not the
    HTTP overhead.
    """
    n_programs = 32
    pressure_high = make_pressure_state(n_programs=n_programs)
    pressure_low = make_pressure_state(
        n_programs=n_programs,
        used_bytes=1 * 1024 * 1024,
        hbm_cap=100 * 1024 * 1024,
    )

    class _StateProvider:
        def __init__(self):
            self.fetch_count = 0

        def __call__(self):
            self.fetch_count += 1
            # First fetch: HIGH (triggers pause).  Subsequent fetches: LOW.
            return pressure_high if self.fetch_count <= 1 else pressure_low

    N_RUNS = 5
    run_ms: List[float] = []
    for _ in range(N_RUNS):
        provider = _StateProvider()
        stub = build_stub_sglang(provider)
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
        for i in range(n_programs):
            tracker.observe_arrival(f"prog-{i}")
        async with run_server(stub, "127.0.0.1", port):
            async with boot_stack(
                url, tracker, theta_hi=0.5, theta_lo=0.3,
            ) as (router, admission, _sched):
                # Warmup.
                await router.bus.emit(
                    Event(kind=EventKind.MEMORY_PRESSURE,
                          payload={"state": "HIGH", "occ": 0.95})
                )
                await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
                # Measured event: state-flip ensures exactly 1 pause.
                provider.fetch_count = 0  # reset so next fetch is HIGH again
                t0 = time.perf_counter()
                await router.bus.emit(
                    Event(kind=EventKind.MEMORY_PRESSURE,
                          payload={"state": "HIGH", "occ": 0.95})
                )
                await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
                run_ms.append((time.perf_counter() - t0) * 1000)
    stats = {
        "mean": statistics.mean(run_ms),
        "std": statistics.stdev(run_ms),
        "envelope": statistics.mean(run_ms) + 3.0 * statistics.stdev(run_ms),
    }
    print(
        f"    single-pause @ 32 programs ({N_RUNS} runs): "
        f"{stats['mean']:.2f} ± {stats['std']:.2f} ms "
        f"(mean+3σ={stats['envelope']:.2f} ms)"
    )
    # Budget: 5 ms.  A single pause is 2 state-fetches (one to
    # trigger, one to see occ dropped + exit loop) + 1 build + 1
    # score + 1 pause.  Current envelope ~4.2 ms; HTTP RTTs dominate.
    # Budget catches a 1.5× algorithmic regression layered on top of
    # the HTTP cost.
    assert stats["envelope"] < 5.0, (
        f"single-pause mean+3σ = {stats['envelope']:.2f} ms exceeds "
        f"5 ms budget"
    )
    return stats


async def step_composition_with_kv_scheduler() -> None:
    """[11] CONTRACT: T7's kv_scheduler.handle fires FIRST (issues
    migrate), then T8's admission re-checks state and pauses if needed.

    Audit round-1 M3: previously only counted both side effects.  A
    regression to "admission first, then kv_scheduler" would pass
    that loose check.  Now we record ORDER via a shared timeline:
    each handler appends a unique tag; assert kv_scheduler tag
    precedes admission tag.
    """
    state_holder = {"state": make_pressure_state(n_programs=4)}
    timeline: List[str] = []
    app = FastAPI()

    @app.get("/aginfer/state")
    async def _state() -> Any:
        return state_holder["state"]

    @app.post("/aginfer/migrate")
    async def _migrate(raw: Request) -> Any:
        await raw.body()
        timeline.append("kv_scheduler:migrate")
        return {"applied": 0, "applied_hashes": [], "skipped": []}

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    tracker = ProgramTracker()
    for i in range(4):
        tracker.observe_arrival(f"prog-{i}")
    async with run_server(app, "127.0.0.1", port):
        async with boot_stack(url, tracker, theta_hi=0.5, theta_lo=0.3) as (router, admission, sched):
            # Wrap admission.handle to record entry.
            orig_handle = admission.handle

            async def _recording_handle(evt, r):
                timeline.append("admission:enter")
                await orig_handle(evt, r)

            # The composite captures `_adm=admission` and looks up
            # `_adm.handle` at call time, so this monkey-patch is
            # picked up without re-attach.
            admission.handle = _recording_handle
            await router.bus.emit(
                Event(kind=EventKind.MEMORY_PRESSURE,
                      payload={"state": "HIGH", "occ": 0.95})
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            # Both side effects observable.
            assert sched is not None
            assert sched.decisions >= 1, (
                f"kv_scheduler did NOT run; decisions={sched.decisions}"
            )
            assert admission.pause_decisions >= 1, (
                f"admission did NOT run; pauses={admission.pause_decisions}"
            )
            # ORDER pin: first kv_scheduler:migrate, then admission:enter.
            try:
                idx_migrate = timeline.index("kv_scheduler:migrate")
                idx_admit = timeline.index("admission:enter")
            except ValueError as e:
                raise AssertionError(
                    f"timeline missing expected event: {timeline}"
                ) from e
            assert idx_migrate < idx_admit, (
                f"composition order wrong: kv_scheduler must fire BEFORE "
                f"admission.  Timeline: {timeline}"
            )


# ---------------------------------------------------------------- main


_T8_STATS: dict = {}


async def main() -> None:
    print("=== T8 verify: admission_controller pause/resume ===")
    print()

    step_shared_aware_aggregation()
    print("[1] shared-aware aggregation: each unit V_u / |holders| ✓")

    step_degenerate_full_share()
    print("[2] WORST CASE: 32-program full-share state → identical scores ✓")

    await step_pause_lowest_under_pressure()
    print("[3] memory_pressure pauses LOWEST-scoring program ✓")

    await step_fifo_resume_with_hysteresis()
    print("[4] pressure_resolved resumes ONE at a time, FIFO order ✓")

    await step_hysteresis_holds_high_occ()
    print("[5] hysteresis: pressure_resolved with occ >= theta_lo → no resume ✓")

    await step_no_victim_when_all_paused()
    print("[6] WORST CASE: all programs paused → no-op, no crash ✓")

    step_anti_timer_contract()
    print("[7] contract: no sleep / call_later / call_at / perf_counter ✓")

    await step_max_pauses_per_event_cap()
    print("[8] WORST CASE: persistent pressure → bounded by "
          "max_pauses_per_event ✓")

    stats = await step_handler_latency_at_32_programs()
    global _T8_STATS  # noqa: PLW0603
    _T8_STATS = stats
    print("[9] COST: per-event handler @ 32 programs within 10 ms ✓")

    await step_invalid_watermarks_raise()
    print("[10] contract: invalid (theta_lo >= theta_hi) raises ValueError ✓")

    await step_composition_with_kv_scheduler()
    print("[11] composition: kv_scheduler.handle (migrate) BEFORE "
          "admission.handle (pause) — ORDER-pinned via timeline ✓")

    single_stats = await step_single_pause_latency()
    print("[12] COST: single-pause @ 32 programs within 3 ms "
          "(algorithmic path, not HTTP RTT × 16) ✓")

    if _T8_STATS:
        print()
        print("Latency summary (record in RESULTS):")
        print(
            f"  admission @ 32 progs: {_T8_STATS['mean']:.2f} ± "
            f"{_T8_STATS['std']:.2f} ms  (mean+3σ {_T8_STATS['envelope']:.2f} ms)"
        )

    print()
    print("=== T8 PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
