"""T7 verify: kv_scheduler event handlers + dispatch.

Layer A (in-process, this file):
  Stub /aginfer/state + stub /aginfer/migrate; drive paper §4 events
  through the EventRouter; assert decision_set per event-kind, action
  direction, migrate body, top-k bound, latency budget.

Per memory:feedback-latency-multi-run: timing claims use mean ± std
over N >= 3 runs.  Per memory:feedback-per-task-docs: noteworthy
findings land in this folder's README RESULTS section.

Run::

    cd /scratch/yuzhou/projects/sglang/dev/aginfer
    python verify/t7/verify.py

Expected last line: ``=== T7 PASSED ===``.
"""
from __future__ import annotations

import asyncio
import json
import socket
import statistics
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request

# Make the sibling daemon / baselines packages importable.
_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import EventRouter  # noqa: E402
from daemon.kv_scheduler import (  # noqa: E402
    KvScheduler,
    assignments_to_wire,
    attach_kv_scheduler,
    build_paper_state,
)
from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from baselines.base import Tier  # noqa: E402


# ---------------------------------------------------------------- helpers


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@asynccontextmanager
async def run_server(app: FastAPI, host: str, port: int):
    cfg = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    # Wait for startup.
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


# ---------------------------------------------------------------- fixtures


def make_synthetic_state(
    *,
    n_programs: int = 4,
    platform_tokens: int = 1024,
    session_tail_tokens: int = 4096,
    hbm_cap_bytes: int = 8 * 1024 * 1024,  # 8 MiB to force pressure
    dram_cap_bytes: int = 1024 * 1024 * 1024,
    disk_cap_bytes: int = 64 * 1024 * 1024 * 1024,
    bytes_per_token: int = 2048,
    age_offset: int = 1,
) -> Dict[str, Any]:
    """Build a /aginfer/state JSON that mirrors the T7 README fixture:

      * 1 shared "platform" prefix held by all N programs;
      * N per-program "session" tails (varying age).
    """
    units: List[Dict[str, Any]] = []
    holders = [f"prog-{i}" for i in range(n_programs)]
    units.append(
        {
            "hash": "u-shared-platform",
            "tier": "HBM",
            "n_tokens": platform_tokens,
            "n_bytes": platform_tokens * bytes_per_token,
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
                "n_tokens": session_tail_tokens,
                "n_bytes": session_tail_tokens * bytes_per_token,
                # Older programs less recently used.
                "last_access_time": 100 - i,
                "hit_count": 4,
                "session_ids": [f"prog-{i}"],
            }
        )
    used_hbm = sum(u["n_bytes"] for u in units)
    return {
        "tier_usage": {
            "HBM": {"used_bytes": used_hbm, "cap_bytes": hbm_cap_bytes},
            "DRAM": {"used_bytes": 0, "cap_bytes": dram_cap_bytes},
            "DISK": {"used_bytes": 0, "cap_bytes": disk_cap_bytes},
        },
        "units": units,
        "time_counter": 100 + age_offset,
        "bytes_per_token": bytes_per_token,
        "page_size": 16,
    }


def build_stub_sglang(state_provider) -> tuple[FastAPI, List[Dict[str, Any]]]:
    """A stub of sglang's /aginfer/state + /aginfer/migrate endpoints.

    ``state_provider`` is a zero-arg callable that returns the current
    state JSON (so a test can mutate it between events).  Migrate POSTs
    are captured into the returned list.
    """
    app = FastAPI()
    migrate_calls: List[Dict[str, Any]] = []

    @app.get("/aginfer/state")
    async def _state() -> Any:
        return state_provider()

    @app.post("/aginfer/migrate")
    async def _migrate(raw: Request) -> Any:
        body = await raw.json()
        migrate_calls.append(body)
        actions = body.get("actions", []) if isinstance(body, dict) else []
        return {
            "applied": len(actions),
            "applied_hashes": [a.get("hash") for a in actions if isinstance(a, dict)],
            "skipped": [],
        }

    return app, migrate_calls


@asynccontextmanager
async def boot_router(
    sglang_base_url: str, tracker: ProgramTracker
) -> tuple[EventRouter, KvScheduler]:
    bus = EventBus()
    router = EventRouter(bus=bus, sglang_base_url=sglang_base_url)
    scheduler = KvScheduler(
        tracker=tracker, sglang_base_url=sglang_base_url
    )
    attach_kv_scheduler(router, scheduler)
    await router.start()
    try:
        yield router, scheduler
    finally:
        await router.stop()
        await scheduler.aclose()


# ---------------------------------------------------------------- steps


def step_build_paper_state_smoke() -> None:
    """[1] Pure-Python: build_paper_state on a synthetic state JSON
    returns a SchedulerState whose decision_set matches the per-event
    table (paper §4)."""
    tracker = ProgramTracker()
    # prog-2 is in tool call → its tail unit should be a demote candidate
    # under λ_ACTING.
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-2")
    tracker.observe_completion("prog-2")  # ACTING

    state_json = make_synthetic_state(n_programs=4)

    # session_arrival: only shared-prefix units in D_t.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.SESSION_ARRIVAL, session="prog-NEW"),
        tracker=tracker,
    )
    assert s.decision_set == ["u-shared-platform"], s.decision_set

    # tool_call_start(prog-0): its own session tail.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.TOOL_CALL_START, session="prog-0"),
        tracker=tracker,
    )
    assert s.decision_set == ["u-tail-0"], s.decision_set

    # tool_call_end(prog-0): same tail (promote angle).
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.TOOL_CALL_END, session="prog-0"),
        tracker=tracker,
    )
    assert s.decision_set == ["u-tail-0"], s.decision_set

    # sub_dispatch_blocking(prog-1): own tail + shared.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.SUB_DISPATCH_BLOCKING, session="prog-1"),
        tracker=tracker,
    )
    assert set(s.decision_set) == {"u-tail-1", "u-shared-platform"}, s.decision_set

    # sub_dispatch_async: only shared prefix.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.SUB_DISPATCH_ASYNC, session="prog-1"),
        tracker=tracker,
    )
    assert s.decision_set == ["u-shared-platform"], s.decision_set

    # llm_prefill: informational only, no decisions.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.LLM_PREFILL, session="prog-0"),
        tracker=tracker,
    )
    assert s.decision_set == [], s.decision_set

    # ACTING-floor λ applied to prog-2's tail unit.
    s = build_paper_state(
        state_json,
        event=Event(kind=EventKind.TOOL_CALL_START, session="prog-2"),
        tracker=tracker,
        lambda_acting=0.2,
    )
    tail2 = s.units["u-tail-2"]
    # λ for prog-2's tail must come from ACTING floor (0.2), not from
    # hits/age (which would be 4/age ≈ different).
    assert tail2.lambda_rate == 0.2, tail2.lambda_rate
    # Shared prefix is held by prog-2 too → also clamped to ACTING.
    plat = s.units["u-shared-platform"]
    assert plat.lambda_rate == 0.2, plat.lambda_rate


def step_top_k_bounded() -> None:
    """[2] memory_pressure event's D_t is bounded by top-k regardless
    of tree size, AND the SELECTED units are the lowest-V (best demote
    candidates per paper §7.1) — not just any 256 units.

    Audit round-1 M1+N7: previously only asserted length.  A regression
    that returned ``items[-k:]`` (the HIGHEST-V units, i.e. the
    BEST-to-KEEP) would have passed.  We now plant 10 "obvious
    low-V" sentinels among 10 000 noisy fillers and assert ALL 10
    sentinels appear in the returned decision_set.

    Sentinel design: huge n_bytes (=> massive holding tax) + zero
    hit_count (=> p_hat = 0) → V_u is most negative.  Fillers have
    high hit_count + tiny n_bytes → V_u positive or only mildly
    negative.  Any correct sort/slice puts the 10 sentinels first.
    """
    from daemon import kv_scheduler as _kvs

    tracker = ProgramTracker()
    units: List[Dict[str, Any]] = []
    # Fillers: high hit_count, tiny size → strong KEEP signal.
    for i in range(9_990):
        units.append(
            {
                "hash": f"u-filler-{i}",
                "tier": "HBM",
                "n_tokens": 16,
                "n_bytes": 32_768,
                "last_access_time": i,
                "hit_count": 1000,  # high p_hat → V_u positive
                "session_ids": [],
            }
        )
    # Sentinels: zero hit_count, huge size → strong DEMOTE signal.
    sentinel_hashes = [f"u-sentinel-{j}" for j in range(10)]
    for j, h in enumerate(sentinel_hashes):
        units.append(
            {
                "hash": h,
                "tier": "HBM",
                "n_tokens": 4096,
                "n_bytes": 4096 * 2048,  # 8 MiB each — biggest in the pool
                "last_access_time": 10_000 - j,  # young (high age denom)
                "hit_count": 0,                  # zero p_hat
                "session_ids": [],
            }
        )
    state_json = {
        "tier_usage": {
            "HBM": {
                "used_bytes": sum(u["n_bytes"] for u in units),
                "cap_bytes": 1024 * 1024 * 1024,
            },
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": units,
        "time_counter": 20_000,
    }
    s = build_paper_state(
        state_json,
        event=Event(
            kind=EventKind.MEMORY_PRESSURE,
            payload={"state": "HIGH", "occ": 0.95},
        ),
        tracker=tracker,
    )
    assert len(s.decision_set) == _kvs._DEFAULT_MEMORY_PRESSURE_TOPK, (
        f"top-k bound violated: {len(s.decision_set)} != "
        f"{_kvs._DEFAULT_MEMORY_PRESSURE_TOPK}"
    )
    # Audit M1+N7: assert all 10 sentinels are in the selected 256.
    # A regression that flipped the sort direction (or returned
    # `items[-k:]`) would put the 10 high-keep fillers in instead and
    # this would fail.
    missing = [h for h in sentinel_hashes if h not in s.decision_set]
    assert not missing, (
        f"top-k selection failed to include obvious low-V sentinels: "
        f"{missing} (current top-k must be returning high-keep units "
        f"instead — check _top_k_by_regret sort direction / slice)"
    )


async def step_event_to_migrate_e2e() -> None:
    """[3] End-to-end: event arrives → kv_scheduler fetches state →
    decide() runs → migrate POST hits the stub.

    A regression that (a) drops the migrate POST, (b) sends an empty
    body, or (c) sends an action whose hash doesn't match a real
    unit, would trip the assertions below.
    """
    state_holder = {"state": make_synthetic_state(n_programs=4)}
    stub_app, migrate_calls = build_stub_sglang(lambda: state_holder["state"])
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"

    tracker = ProgramTracker()
    # prog-2 in tool call so tail can be demoted; rest REASONING.
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    tracker.observe_arrival("prog-2")
    tracker.observe_arrival("prog-3")
    tracker.observe_completion("prog-2")

    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            # Fire a tool_call_start(prog-2): prog-2's tail must be a
            # demote candidate (its λ_ACTING floor + small p_hat keeps
            # HBM expensive vs DRAM).
            await router.bus.emit(
                Event(kind=EventKind.TOOL_CALL_START, session="prog-2")
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)

    assert scheduler.decisions >= 1, scheduler.decisions
    # The decision_set was just prog-2's tail.
    assert scheduler.last_decision_set_size == 1, (
        scheduler.last_decision_set_size
    )
    if not migrate_calls:
        # Action was empty (decide() declined).  That's a legitimate
        # outcome — assert state-derived expectation matches.
        action = scheduler.last_action
        assert action is not None
        assert not action.assignments, action.assignments
    else:
        # We DID send a migrate.  Body must reference prog-2's tail.
        body = migrate_calls[-1]
        actions = body.get("actions", [])
        assert any(
            a.get("hash") == "u-tail-2" for a in actions
        ), f"migrate body missing u-tail-2: {actions!r}"
        # Audit round-1 M2: previously asserted only ``!= "HBM"`` which
        # allowed DROP.  Paper §4 / README line 50 says the tail moves
        # to DRAM during a tool call (catastrophic-demote DROP would
        # save no bytes and re-prefill on return).  Tighten to DRAM
        # OR DISK — both are non-catastrophic; DROP is the regression
        # we want to catch.
        for a in actions:
            if a["hash"] == "u-tail-2":
                assert a["target_tier"] in ("DRAM", "DISK"), (
                    f"tool_call_start demoted to wrong tier "
                    f"{a['target_tier']!r}; expected DRAM/DISK, NOT DROP "
                    f"or HBM"
                )


async def step_no_migrate_when_action_empty() -> None:
    """[4] WORST CASE: every unit high-p_hat + low age → decide()
    returns Action(assignments=[]) → ZERO migrate POSTs sent.

    Pins paper §7's "no churn if nothing's worth moving" claim.  A
    regression that always POSTs an empty body (or worse, all-DROP)
    would trip the migrate-call-count assertion.
    """
    # Custom state: tail units have absurdly high hit_count + young age.
    state = make_synthetic_state(n_programs=2)
    for u in state["units"]:
        u["hit_count"] = 1_000_000
        u["last_access_time"] = 100  # equal to time_counter -1 (young)
    state["time_counter"] = 101

    state_holder = {"state": state}
    stub_app, migrate_calls = build_stub_sglang(lambda: state_holder["state"])
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"

    tracker = ProgramTracker()
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.92},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
    assert scheduler.migrate_calls == 0, (
        f"expected 0 migrate POSTs (nothing worth moving), got "
        f"{scheduler.migrate_calls}"
    )
    assert scheduler.last_action is not None
    assert not scheduler.last_action.assignments, (
        scheduler.last_action.assignments
    )


async def step_state_fetch_failure_recovers() -> None:
    """[5] WORST CASE: /aginfer/state returns 500 → kv_scheduler logs
    + bows out; tracker untouched; event_worker keeps draining.

    A regression that lets the fetch exception propagate would crash
    the event_worker (which would leave the daemon unresponsive to
    further webhooks).
    """
    state_holder = {"fail": True}

    stub_app = FastAPI()

    @stub_app.get("/aginfer/state")
    async def _bad():  # noqa: ANN001
        if state_holder["fail"]:
            from fastapi.responses import JSONResponse

            return JSONResponse({"error": "boom"}, status_code=500)
        return make_synthetic_state(n_programs=1)

    @stub_app.post("/aginfer/migrate")
    async def _migrate(raw: Request) -> Any:
        return {"applied": 0, "applied_hashes": [], "skipped": []}

    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"
    tracker = ProgramTracker()
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            # Bad fetch event.
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            # No decisions were made (fetch failed before build_paper_state).
            assert scheduler.decisions == 0, scheduler.decisions
            assert scheduler.migrate_calls == 0, scheduler.migrate_calls

            # Worker is still alive: flip the stub, fire again, succeed.
            state_holder["fail"] = False
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
    assert router.events_handled >= 2, (
        f"second event was not drained: events_handled={router.events_handled}"
    )


async def step_decide_latency_at_1k_units() -> dict:
    """[6] COST: decide() < 50 ms at 1 000 units on a memory_pressure
    event.  Per memory:feedback-latency-multi-run, multi-run mean ±
    std.  Catches a regression that would let decide() scale linearly
    past the README's 50 ms budget.
    """
    from daemon import kv_scheduler as _kvs

    tracker = ProgramTracker()
    # Build a 1k-unit synthetic state without rebuilding for each run.
    units: List[Dict[str, Any]] = []
    for i in range(1000):
        units.append(
            {
                "hash": f"u-{i}",
                "tier": "HBM",
                "n_tokens": 128,
                "n_bytes": 128 * 2048,
                "last_access_time": i,
                "hit_count": (i * 11) % 17,
                "session_ids": [],
            }
        )
    state_json = {
        "tier_usage": {
            "HBM": {"used_bytes": sum(u["n_bytes"] for u in units),
                    "cap_bytes": 256 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": units,
        "time_counter": 2000,
    }
    from baselines.ours_greedy import OursGreedyPolicy
    from baselines.costs import default_costs

    policy = OursGreedyPolicy(default_costs())

    N_RUNS = 5
    run_decide_ms: List[float] = []
    run_build_ms: List[float] = []
    for _ in range(N_RUNS):
        # Warmup
        _ = build_paper_state(
            state_json,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=tracker,
        )
        # Build cost.
        t0 = time.perf_counter()
        s = build_paper_state(
            state_json,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=tracker,
        )
        build_ms = (time.perf_counter() - t0) * 1000
        # Decide cost.
        t0 = time.perf_counter()
        _ = policy.decide(s)
        decide_ms = (time.perf_counter() - t0) * 1000
        run_build_ms.append(build_ms)
        run_decide_ms.append(decide_ms)
    stats = {
        "build_mean": statistics.mean(run_build_ms),
        "build_std": statistics.stdev(run_build_ms),
        "decide_mean": statistics.mean(run_decide_ms),
        "decide_std": statistics.stdev(run_decide_ms),
    }
    print(
        f"    decide() @ 1k units, {N_RUNS} runs: "
        f"build {stats['build_mean']:.2f} ± {stats['build_std']:.2f} ms; "
        f"decide {stats['decide_mean']:.2f} ± {stats['decide_std']:.2f} ms"
    )
    # Audit round-1 N5: README budget is decide < 50 ms but actual is
    # ~1.5 ms — 33× headroom hides a noticeable regression.  Tighten to
    # mean+3σ < 5 ms (still ~3× headroom over current; catches an
    # O(N²) reintroduction).  Build budget stays 5 ms (already tight).
    decide_env = stats["decide_mean"] + 3.0 * stats["decide_std"]
    build_env = stats["build_mean"] + 3.0 * stats["build_std"]
    assert decide_env < 5.0, (
        f"decide() mean+3σ = {decide_env:.2f} ms exceeds 5 ms budget "
        f"(was 50 ms before audit round-1 N5 tightened)"
    )
    assert build_env < 5.0, (
        f"build_paper_state mean+3σ = {build_env:.2f} ms exceeds 5 ms budget"
    )
    return stats


async def step_lambda_acting_sweep() -> None:
    """[7] Sensitivity: λ_ACTING ∈ {1/30, 1/5, 1/1, 2/1} — the
    demote/promote SIGN must be stable across the in-range values and
    must SATURATE (not invert) at the floor / ceiling.

    Forces the calibration justification: tells the user "any λ in
    the [1/30, 1/1] envelope produces qualitatively the same
    decision".  Catches a regression that flipped the sign somewhere
    in the value-function plumbing.
    """
    from daemon import kv_scheduler as _kvs

    state_json = make_synthetic_state(n_programs=2)
    tracker = ProgramTracker()
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    tracker.observe_completion("prog-1")  # ACTING

    from baselines.ours_greedy import OursGreedyPolicy
    from baselines.costs import default_costs

    policy = OursGreedyPolicy(default_costs())
    # Audit round-1 N1: previously {1/30, 1/5, 1/1, 2/1} tested only
    # ceiling clamp.  Add 1/100 below the floor to pin floor clamping
    # too — a regression removing `max(_FLOOR, ...)` from
    # _clamp_lambda_acting would now diverge λ=1/100 from λ=1/30.
    lams = [1 / 100, 1 / 30, 1 / 5, 1 / 1, 2 / 1]
    actions_per_lam: Dict[float, List[Tuple[str, Tier]]] = {}
    for lam in lams:
        s = build_paper_state(
            state_json,
            event=Event(
                kind=EventKind.MEMORY_PRESSURE,
                payload={"state": "HIGH", "occ": 0.95},
            ),
            tracker=tracker,
            lambda_acting=lam,
        )
        a = policy.decide(s)
        actions_per_lam[lam] = list(a.assignments)
    # Print for debug.
    for lam, acts in actions_per_lam.items():
        print(f"    λ_ACTING={lam:.3f}: {len(acts)} migration(s)")
    # Saturation contract: the ceiling-clamped value (2/1 → clamped to
    # 1/1) must produce the SAME action set as λ=1/1 (no oscillation
    # caused by an out-of-range input).
    assert actions_per_lam[1 / 1] == actions_per_lam[2 / 1], (
        f"clamping was bypassed: λ=1/1 → {actions_per_lam[1/1]} vs "
        f"λ=2/1 → {actions_per_lam[2/1]}"
    )
    # Audit round-1 N1: floor clamp.  λ=1/100 (below floor 1/30) must
    # produce the SAME action set as λ=1/30.  A regression removing
    # the floor would let λ=1/100 propagate → larger hold_time →
    # different demote decisions.
    assert actions_per_lam[1 / 100] == actions_per_lam[1 / 30], (
        f"floor clamp bypassed: λ=1/100 → {actions_per_lam[1/100]} vs "
        f"λ=1/30 → {actions_per_lam[1/30]}"
    )
    # Floor-clamped value (1/30) is the FLOOR of the envelope.  No
    # action should have target=DROP (we don't want catastrophic
    # demotion at the floor; the test passes if decisions stay tier-
    # local).
    for lam, acts in actions_per_lam.items():
        for _h, tier in acts:
            assert tier != Tier.DROP, (
                f"λ={lam} produced a DROP migration; floor calibration is "
                f"over-trusting the ACTING signal"
            )


def step_assignments_to_wire_contract() -> None:
    """[8] Pure unit test: ``assignments_to_wire`` produces the
    schema sglang's ``POST /aginfer/migrate`` expects.

    Pinned because a refactor that swaps "target_tier" for "tier"
    (or moves to ints) would silently break dispatch — sglang would
    400, kv_scheduler logs the rejection but keeps running.
    """
    out = assignments_to_wire(
        [
            ("u-1", Tier.DRAM),
            ("u-2", Tier.DROP),
            ("u-3", Tier.HBM),
        ]
    )
    assert out == [
        {"hash": "u-1", "target_tier": "DRAM"},
        {"hash": "u-2", "target_tier": "DROP"},
        {"hash": "u-3", "target_tier": "HBM"},
    ], out


async def step_idempotent_repeat_event() -> None:
    """[9] Replay the SAME event 3× and assert the migrate body of the
    last call equals the first (modulo state drift, which is zero
    here because the stub state is frozen).

    Audit round-1 N4: previous version used ``if migrate_calls:`` —
    if the policy declined to migrate (legit outcome) the step
    silently passed without checking anything.  Now we construct a
    fixture that GUARANTEES a migrate: pressure event on a state
    with sentinel low-V units (same trick as step [2]).
    """
    # Build a state that's certain to trigger a migrate.
    units: List[Dict[str, Any]] = [
        {
            "hash": f"u-keeper-{i}",
            "tier": "HBM",
            "n_tokens": 16,
            "n_bytes": 32_768,
            "last_access_time": 1000 + i,
            "hit_count": 1000,
            "session_ids": ["prog-keeper"],
        }
        for i in range(50)
    ]
    # Sentinels — must be demoted under top-k regret + decide().
    for j in range(5):
        units.append(
            {
                "hash": f"u-sentinel-{j}",
                "tier": "HBM",
                "n_tokens": 4096,
                "n_bytes": 4096 * 2048,
                "last_access_time": 2000,
                "hit_count": 0,
                "session_ids": [],
            }
        )
    state = {
        "tier_usage": {
            "HBM": {
                "used_bytes": sum(u["n_bytes"] for u in units),
                "cap_bytes": 32 * 1024 * 1024,  # tight; forces pressure
            },
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": units,
        "time_counter": 3000,
    }
    state_holder = {"state": state}
    stub_app, migrate_calls = build_stub_sglang(lambda: state_holder["state"])
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"
    tracker = ProgramTracker()
    tracker.observe_arrival("prog-keeper")
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            for _ in range(3):
                await router.bus.emit(
                    Event(
                        kind=EventKind.MEMORY_PRESSURE,
                        payload={"state": "HIGH", "occ": 0.95},
                    )
                )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
    # N4 fix: REQUIRE that all 3 migrates fired.
    assert len(migrate_calls) == 3, (
        f"expected exactly 3 migrate POSTs (one per replayed event); "
        f"got {len(migrate_calls)}.  Fixture must force a migrate "
        f"otherwise step [9] silently asserts nothing."
    )
    # And all 3 bodies must match (paper §9 idempotence).
    first = migrate_calls[0]
    for i, body in enumerate(migrate_calls[1:], start=1):
        assert body == first, (
            f"non-idempotent migrate body across repeats: replay #{i} "
            f"differs from first.\n  first={first}\n  later={body}"
        )


async def step_all_event_kinds_registered() -> None:
    """[11] Audit round-1 M3: ``attach_kv_scheduler`` must register
    the kv_scheduler handler for ALL 8 paper §4 EventKind values.
    The previous coverage tested 7 of 8 (PRESSURE_RESOLVED was never
    fired).  A regression like ``for kind in EventKind: if kind !=
    PRESSURE_RESOLVED: router.set_handler(...)`` would silently fall
    back to ``_noop_handler`` for that kind.

    Two-part assertion:
      (a) For every EventKind value, ``router._handlers[kind.value]``
          is the kv_scheduler handler (not the noop fallback).
      (b) Fire a real PRESSURE_RESOLVED event end-to-end; assert
          ``scheduler.last_decision_set_size`` is non-zero
          (handler was actually invoked).
    """
    # Audit round-3 VACUOUS-1: previous version compared bound-method
    # identity (``handler == scheduler.handle``), which is fragile —
    # a future ``functools.partial`` wrapper would fail equality but
    # routing would be functionally correct (and vice versa, a
    # wrapper that intercepts but still calls handle would pass
    # equality while breaking behavior).  Replace with FUNCTIONAL
    # pin: fire one event of EACH kind in turn and assert
    # ``last_decision_set_size`` was actually touched by
    # ``scheduler.handle`` (the sentinel-overwrite check).
    state_holder = {"state": make_synthetic_state(n_programs=2)}
    stub_app, _migrate_calls = build_stub_sglang(lambda: state_holder["state"])
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"
    tracker = ProgramTracker()
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    SENTINEL = -42
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            for kind in EventKind:
                scheduler.last_decision_set_size = SENTINEL
                await router.bus.emit(
                    Event(
                        kind=kind,
                        session="prog-0",
                        payload={"state": "HIGH", "occ": 0.95},
                    )
                )
                await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
                # If routing landed on _noop_handler instead of
                # scheduler.handle, the sentinel would still be -42
                # (noop doesn't touch this field).
                assert scheduler.last_decision_set_size != SENTINEL, (
                    f"EventKind.{kind.name} was NOT routed to "
                    f"scheduler.handle; sentinel stayed {SENTINEL} "
                    f"(handler probably fell back to _noop_handler)."
                )


async def step_migrate_5xx_does_not_crash() -> None:
    """[12] Audit round-1 N2: a 5xx from /aginfer/migrate must NOT
    crash the event_worker.  ``_dispatch_migrate`` logs + bows out;
    counters reflect the attempt.

    A regression that added ``r.raise_for_status()`` would propagate
    the exception → event_worker would log + drop the event but
    ``handler_failures`` would tick.  Pin the no-failure contract.
    """
    state_holder = {
        "state": make_synthetic_state(n_programs=2),
        "fail_migrate": True,
    }
    stub_app = FastAPI()

    @stub_app.get("/aginfer/state")
    async def _state() -> Any:
        return state_holder["state"]

    @stub_app.post("/aginfer/migrate")
    async def _migrate(raw: Request) -> Any:
        await raw.body()
        if state_holder["fail_migrate"]:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"error": "simulated 500"}, status_code=500
            )
        return {"applied": 0, "applied_hashes": [], "skipped": []}

    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"
    tracker = ProgramTracker()
    tracker.observe_arrival("prog-0")
    tracker.observe_arrival("prog-1")
    tracker.observe_completion("prog-1")  # ACTING
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            # Build a state that triggers a migrate.
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            handler_failures_after_500 = router.handler_failures
            # Worker still alive: flip the stub, fire again.
            state_holder["fail_migrate"] = False
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
    # N2 contract: handler must NOT have raised on 5xx.
    assert handler_failures_after_500 == 0, (
        f"handler raised on migrate 5xx (expected log+continue); "
        f"handler_failures={handler_failures_after_500}"
    )
    # Both events were handled (worker alive after the failure).
    assert router.events_handled >= 2, router.events_handled
    # The actual POST WAS attempted (counter increments before status
    # check in _dispatch_migrate).
    assert scheduler.migrate_calls >= 1, scheduler.migrate_calls


def step_env_var_binding() -> None:
    """[13] Audit round-1 N3: ENV_VAR → module-level constant binding.

    A subprocess sets ``AGINFER_MEMORY_PRESSURE_TOPK=7``, imports
    ``daemon.kv_scheduler``, reads the module constant.  A regression
    that renamed the env var key would strand operators on the
    default.
    """
    import subprocess

    probe = (
        "import sys, json; "
        f"sys.path.insert(0, {str(_AGINFER_ROOT)!r}); "
        "from daemon import kv_scheduler as k; "
        "print(json.dumps({"
        "'topk': k._DEFAULT_MEMORY_PRESSURE_TOPK, "
        "'lam':  k._DEFAULT_LAMBDA_ACTING}))"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", probe],
        env={
            **{k: v for k, v in __import__("os").environ.items()
               if k.startswith(("PATH", "PYTHON", "LD_", "CONDA"))},
            "AGINFER_MEMORY_PRESSURE_TOPK": "7",
            "AGINFER_LAMBDA_ACTING": "0.42",
        },
        timeout=15,
    ).decode().strip().splitlines()[-1]
    parsed = json.loads(out)
    assert parsed["topk"] == 7, (
        f"AGINFER_MEMORY_PRESSURE_TOPK -> _DEFAULT_MEMORY_PRESSURE_TOPK "
        f"binding broken: got {parsed['topk']}"
    )
    assert abs(parsed["lam"] - 0.42) < 1e-9, (
        f"AGINFER_LAMBDA_ACTING -> _DEFAULT_LAMBDA_ACTING binding broken: "
        f"got {parsed['lam']}"
    )


async def step_unknown_event_kind_safe() -> None:
    """[10] An unmapped event kind (defensive — paper §4 only has 8 but
    a future kind might land before its handler) MUST not crash the
    worker.  We send an event whose kind is in the enum but for which
    decision_set is empty: assert no decisions, no crash, worker keeps
    going."""
    state_holder = {"state": make_synthetic_state(n_programs=1)}
    stub_app, migrate_calls = build_stub_sglang(lambda: state_holder["state"])
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"
    tracker = ProgramTracker()
    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with boot_router(stub_url, tracker) as (router, scheduler):
            # LLM_PREFILL produces empty D_t (informational).  Worker
            # must drain it without calling decide().
            await router.bus.emit(
                Event(kind=EventKind.LLM_PREFILL, session="prog-0")
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
    assert scheduler.decisions == 0, scheduler.decisions
    assert scheduler.migrate_calls == 0, scheduler.migrate_calls
    # But the event WAS handled by the worker.
    assert router.events_handled == 1, router.events_handled


# ---------------------------------------------------------------- main


_T7_STATS: dict = {}


async def main() -> None:
    print("=== T7 verify: kv_scheduler event handlers + dispatch ===")
    print()

    step_build_paper_state_smoke()
    print("[1] build_paper_state: D_t per paper §4 table (6 event kinds) ✓")

    step_top_k_bounded()
    print("[2] memory_pressure D_t bounded at top-k = 256 (paper §7.1) ✓")

    await step_event_to_migrate_e2e()
    print("[3] event arrives → state fetch → decide() → migrate POST ✓")

    await step_no_migrate_when_action_empty()
    print("[4] WORST CASE: nothing worth moving → 0 migrate POSTs ✓")

    await step_state_fetch_failure_recovers()
    print("[5] WORST CASE: /aginfer/state 500 → log + continue; worker "
          "drains next event ✓")

    stats = await step_decide_latency_at_1k_units()
    global _T7_STATS  # noqa: PLW0603
    _T7_STATS = stats
    print("[6] COST: decide() @ 1k units, build + decide within budget "
          "(5-run mean ± std) ✓")

    await step_lambda_acting_sweep()
    print("[7] λ_ACTING sweep {1/30, 1/5, 1/1, 2/1}: clamp saturates; no "
          "DROP migrations at floor ✓")

    step_assignments_to_wire_contract()
    print("[8] assignments_to_wire schema matches sglang's "
          "POST /aginfer/migrate contract ✓")

    await step_idempotent_repeat_event()
    print("[9] same event replayed 3× → identical migrate body "
          "(paper §9 idempotence) ✓")

    await step_unknown_event_kind_safe()
    print("[10] LLM_PREFILL (empty D_t) drains without crashing worker ✓")

    await step_all_event_kinds_registered()
    print("[11] audit M3 fix: all 8 EventKinds routed to kv_scheduler "
          "(incl. PRESSURE_RESOLVED) + end-to-end fire ✓")

    await step_migrate_5xx_does_not_crash()
    print("[12] audit N2 fix: /aginfer/migrate 500 → log+continue; "
          "handler_failures stays 0; worker drains next event ✓")

    step_env_var_binding()
    print("[13] audit N3 fix: AGINFER_* env vars actually bind to "
          "module-level constants (subprocess probe) ✓")

    if _T7_STATS:
        print()
        print("Latency summary (record in RESULTS):")
        print(
            f"  build_paper_state @ 1k units: "
            f"{_T7_STATS['build_mean']:.2f} ± {_T7_STATS['build_std']:.2f} ms"
        )
        print(
            f"  decide() @ 1k units:          "
            f"{_T7_STATS['decide_mean']:.2f} ± {_T7_STATS['decide_std']:.2f} ms"
        )

    print()
    print("=== T7 PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
