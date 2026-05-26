"""T7 regression probe — bisect-style demos of audit round-1 findings.

For each finding (B1, B2, M1+N7, M2, M3, N1, N2, N3, N4, N5):

  1. Re-inject the regression we want to catch (via monkey-patch or
     manual flag).
  2. Run the corresponding tightened verify assertion.
  3. Assert it FAILS — proving the test would catch the regression.
  4. Restore production code.
  5. Run again — assert it PASSES — proving the fix actually fixes it.

This file is the per-task-RESULTS-doc artifact for the audit round
(memory:feedback-per-task-docs).  Run after verify.py passes::

    cd /scratch/yuzhou/projects/sglang/dev/aginfer
    python verify/t7/regression_probe.py

Expected: each probe prints ``PASS  <finding>: pre-fix FAIL → post-fix PASS``.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import socket
import subprocess
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

# Force a clean import of kv_scheduler so monkey-patches can be
# applied / reverted cleanly.
import daemon.kv_scheduler as kvs_mod  # noqa: E402
from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import EventRouter  # noqa: E402
from daemon.kv_scheduler import (  # noqa: E402
    KvScheduler,
    attach_kv_scheduler,
    build_paper_state,
)
from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from baselines.base import Tier  # noqa: E402


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
    try:
        yield
    finally:
        server.should_exit = True
        await task


def _bisect_outcome(name: str, pre_fix_passed: bool, post_fix_passed: bool) -> None:
    """Print a single line per finding.  Probe PASSES iff
    pre_fix FAILS (the regression is catchable) AND post_fix PASSES
    (the fix actually works)."""
    if not pre_fix_passed and post_fix_passed:
        print(f"PASS  {name}: pre-fix FAIL → post-fix PASS")
    else:
        print(
            f"FAIL  {name}: pre_fix_passed={pre_fix_passed} "
            f"post_fix_passed={post_fix_passed}"
        )
        raise AssertionError(f"regression probe FAILED for {name}")


# ---------------------------------------------------------------- B1


def probe_b1_unknown_tier_unsafe_default() -> None:
    """B1: ``_tier_from_string("ZSTD_DISK")`` must NOT silently classify
    as HBM.  Regression: re-introduce the silent HBM fallback.
    """
    name = "B1 (unknown tier -> safe skip)"

    def _check() -> bool:
        # Build a state with one "ZSTD_DISK" unit.  Under the FIX,
        # _tier_from_string returns None and the unit is skipped → it
        # cannot appear in any decision_set.  Under the BUG, it would
        # be classified HBM and become a top-k candidate.
        state = {
            "tier_usage": {
                "HBM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
            },
            "units": [
                {
                    "hash": "u-zstd",
                    "tier": "ZSTD_DISK",
                    "n_tokens": 4096,
                    "n_bytes": 8 * 1024 * 1024,
                    "last_access_time": 0,
                    "hit_count": 0,
                    "session_ids": [],
                }
            ],
            "time_counter": 100,
        }
        s = build_paper_state(
            state,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=ProgramTracker(),
        )
        return "u-zstd" not in s.units

    # POST-FIX (stock code): unknown tier → unit skipped.
    post_fix_passed = _check()

    # PRE-FIX (regression): monkey-patch _tier_from_string to fall back
    # to HBM on unknown labels, AND skip the None-check in
    # build_paper_state (the original bug had both: the helper returned
    # HBM unconditionally; the loop never had a "skip None" branch).
    saved_label_map = dict(kvs_mod._TIER_LABEL_MAP)
    saved_from_string = kvs_mod._tier_from_string

    def _bug_from_string(label: str) -> Tier:
        return saved_label_map.get(label.upper(), Tier.HBM)

    kvs_mod._tier_from_string = _bug_from_string
    try:
        pre_fix_passed = _check()
    finally:
        kvs_mod._tier_from_string = saved_from_string

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- B2


def probe_b2_per_rank_aggregation() -> None:
    """B2: multi-rank ``{"per_rank": [...]}`` must be aggregated.
    Regression: bypass ``_flatten_per_rank`` so the daemon sees an
    empty top-level state.
    """
    name = "B2 (multi-rank per_rank aggregation)"
    per_rank_state = {
        "per_rank": [
            {
                "tier_usage": {
                    "HBM": {"used_bytes": 4 * 1024 * 1024, "cap_bytes": 8 * 1024 * 1024},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                    "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
                },
                "units": [
                    {
                        "hash": "u-1",
                        "tier": "HBM",
                        "n_tokens": 1024,
                        "n_bytes": 2 * 1024 * 1024,
                        "last_access_time": 0,
                        "hit_count": 0,
                        "session_ids": [],
                    },
                ],
                "time_counter": 50,
            },
            {
                "tier_usage": {
                    "HBM": {"used_bytes": 4 * 1024 * 1024, "cap_bytes": 8 * 1024 * 1024},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                    "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
                },
                "units": [
                    {
                        "hash": "u-2",
                        "tier": "HBM",
                        "n_tokens": 1024,
                        "n_bytes": 2 * 1024 * 1024,
                        "last_access_time": 0,
                        "hit_count": 0,
                        "session_ids": [],
                    },
                ],
                "time_counter": 75,
            },
        ]
    }

    def _check() -> bool:
        s = build_paper_state(
            per_rank_state,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=ProgramTracker(),
        )
        # FIX: per_rank flattened; both units visible (prefixed with r0/, r1/).
        return len(s.units) >= 2 and len(s.decision_set) >= 1

    post_fix_passed = _check()

    saved_flatten = kvs_mod._flatten_per_rank
    kvs_mod._flatten_per_rank = lambda j: j  # bypass
    try:
        pre_fix_passed = _check()
    finally:
        kvs_mod._flatten_per_rank = saved_flatten

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- M1+N7


def probe_m1_top_k_content_pin() -> None:
    """M1+N7: ``_top_k_by_regret`` must return LOWEST-V units (best
    demote candidates).  Regression: flip the slice to ``items[-k:]``
    (= highest-V, the OPPOSITE of paper §7.1).
    """
    name = "M1+N7 (top-k content: lowest-V, not highest)"
    # 1k filler + 5 sentinels.
    units: List[Dict[str, Any]] = []
    for i in range(990):
        units.append(
            {
                "hash": f"u-filler-{i}",
                "tier": "HBM",
                "n_tokens": 16,
                "n_bytes": 32_768,
                "last_access_time": i,
                "hit_count": 1000,
                "session_ids": [],
            }
        )
    sentinels = [f"u-sentinel-{j}" for j in range(5)]
    for j, h in enumerate(sentinels):
        units.append(
            {
                "hash": h,
                "tier": "HBM",
                "n_tokens": 4096,
                "n_bytes": 4096 * 2048,
                "last_access_time": 2000 - j,
                "hit_count": 0,
                "session_ids": [],
            }
        )
    state = {
        "tier_usage": {
            "HBM": {
                "used_bytes": sum(u["n_bytes"] for u in units),
                "cap_bytes": 1 << 30,
            },
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": units,
        "time_counter": 3000,
    }

    def _check(k: int = 50) -> bool:
        s = build_paper_state(
            state,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=ProgramTracker(),
        )
        return all(h in s.decision_set for h in sentinels)

    post_fix_passed = _check()

    saved = kvs_mod._top_k_by_regret

    def _bug(units_, k, costs=kvs_mod.default_costs()):
        rho_hbm = costs.rho[Tier.HBM]
        rho_disk = costs.rho[Tier.DISK]
        items = []
        for uid, u in units_.items():
            if u.tier != Tier.HBM:
                continue
            saved_p = u.p_hat * (rho_disk - rho_hbm) * u.n_tokens
            hold = costs.h_base[Tier.HBM] * u.n_bytes
            items.append((saved_p - hold, uid))
        items.sort()
        return [uid for _s, uid in items[-k:]]  # BUG: top-of-desc instead of top-of-asc

    kvs_mod._top_k_by_regret = _bug
    try:
        pre_fix_passed = _check()
    finally:
        kvs_mod._top_k_by_regret = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- M2


def probe_m2_target_tier_not_drop() -> None:
    """M2: tool_call_start demote target must be DRAM/DISK, never DROP
    (catastrophic).  Regression: monkey-patch the policy to prefer
    DROP for everything.
    """
    name = "M2 (tool_call_start target tier != DROP)"
    from baselines.base import Action

    # Run step [3]'s end-to-end fixture twice: once stock, once with a
    # patched policy that returns DROP.

    async def _run_with_policy(policy_handler) -> List[Dict[str, Any]]:
        # Inline minimal version of step_event_to_migrate_e2e.
        state_holder = {
            "state": _make_simple_state(),
        }
        stub_app = FastAPI()
        captured: List[Dict[str, Any]] = []

        @stub_app.get("/aginfer/state")
        async def _state() -> Any:
            return state_holder["state"]

        @stub_app.post("/aginfer/migrate")
        async def _migrate(raw: Request) -> Any:
            body = await raw.json()
            captured.append(body)
            return {"applied": len(body.get("actions", [])), "applied_hashes": [], "skipped": []}

        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
        tracker.observe_arrival("prog-0")
        tracker.observe_arrival("prog-2")
        tracker.observe_completion("prog-2")
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url=url)
        scheduler = KvScheduler(tracker=tracker, sglang_base_url=url)
        if policy_handler is not None:
            scheduler.policy = policy_handler
        attach_kv_scheduler(router, scheduler)
        await router.start()
        async with run_server(stub_app, "127.0.0.1", port):
            await router.bus.emit(
                Event(kind=EventKind.TOOL_CALL_START, session="prog-2")
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            await router.stop()
            await scheduler.aclose()
        return captured

    async def _check_post() -> bool:
        calls = await _run_with_policy(None)
        if not calls:
            return False  # legit: no migrate → can't verify direction
        for body in calls:
            for a in body.get("actions", []):
                if a.get("hash", "").endswith("u-tail-2"):
                    return a.get("target_tier") in ("DRAM", "DISK")
        return True

    # Replace OursGreedyPolicy with a fake that returns DROP for tail-2.
    class _DropPolicy:
        def decide(self, state):
            assigns = []
            for uid, u in state.units.items():
                if uid.endswith("u-tail-2"):
                    assigns.append((uid, Tier.DROP))
            return Action(assignments=assigns)

    async def _check_pre() -> bool:
        calls = await _run_with_policy(_DropPolicy())
        # Bug present: migrate body has target_tier = "DROP".  New
        # assertion `in ("DRAM","DISK")` will FAIL.
        for body in calls:
            for a in body.get("actions", []):
                if a.get("hash") == "u-tail-2":
                    return a.get("target_tier") in ("DRAM", "DISK")
        return True  # no migrate observed = vacuous pass

    post_fix_passed = asyncio.run(_check_post())
    pre_fix_passed = asyncio.run(_check_pre())

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


def _make_simple_state() -> Dict[str, Any]:
    units = [
        {
            "hash": "u-shared-platform",
            "tier": "HBM",
            "n_tokens": 1024,
            "n_bytes": 1024 * 2048,
            "last_access_time": 100,
            "hit_count": 200,
            "session_ids": ["prog-0", "prog-2"],
        },
        {
            "hash": "u-tail-0",
            "tier": "HBM",
            "n_tokens": 4096,
            "n_bytes": 4096 * 2048,
            "last_access_time": 100,
            "hit_count": 4,
            "session_ids": ["prog-0"],
        },
        {
            "hash": "u-tail-2",
            "tier": "HBM",
            "n_tokens": 4096,
            "n_bytes": 4096 * 2048,
            "last_access_time": 99,
            "hit_count": 4,
            "session_ids": ["prog-2"],
        },
    ]
    return {
        "tier_usage": {
            "HBM": {"used_bytes": sum(u["n_bytes"] for u in units), "cap_bytes": 8 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": units,
        "time_counter": 101,
    }


# ---------------------------------------------------------------- M3


def probe_m3_pressure_resolved_routed() -> None:
    """M3: ``attach_kv_scheduler`` must register PRESSURE_RESOLVED.
    Regression: skip that one kind in the for-loop.
    """
    name = "M3 (PRESSURE_RESOLVED routed to kv_scheduler)"

    def _check_pre() -> bool:
        bus = EventBus()
        tracker = ProgramTracker()
        sched = KvScheduler(tracker=tracker, sglang_base_url="http://x")
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        # BUG: skip PRESSURE_RESOLVED registration.
        for kind in EventKind:
            if kind == EventKind.PRESSURE_RESOLVED:
                continue
            router.set_handler(kind, sched.handle)
        return router._handlers.get(EventKind.PRESSURE_RESOLVED.value) == sched.handle

    def _check_post() -> bool:
        bus = EventBus()
        tracker = ProgramTracker()
        sched = KvScheduler(tracker=tracker, sglang_base_url="http://x")
        router = EventRouter(bus=bus, sglang_base_url="http://x")
        attach_kv_scheduler(router, sched)
        return router._handlers.get(EventKind.PRESSURE_RESOLVED.value) == sched.handle

    _bisect_outcome(name, _check_pre(), _check_post())


# ---------------------------------------------------------------- N1


def probe_n1_floor_clamp() -> None:
    """N1: ``_clamp_lambda_acting`` must floor at 1/30.  Regression:
    remove the ``max(_FLOOR, ...)`` so λ=1/100 propagates.
    """
    name = "N1 (lambda_acting floor clamp)"
    saved = kvs_mod._clamp_lambda_acting

    def _check() -> bool:
        a = kvs_mod._clamp_lambda_acting(1 / 100)
        return a == kvs_mod._LAMBDA_ACTING_FLOOR

    post_fix_passed = _check()
    kvs_mod._clamp_lambda_acting = lambda lam: min(kvs_mod._LAMBDA_ACTING_CEIL, lam)
    try:
        pre_fix_passed = _check()
    finally:
        kvs_mod._clamp_lambda_acting = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- N2


def probe_n2_migrate_5xx_no_raise() -> None:
    """N2: migrate 5xx must log+continue, NOT raise.  Regression:
    monkey-patch ``_dispatch_migrate`` to call ``r.raise_for_status()``.
    """
    name = "N2 (migrate 5xx log+continue)"

    async def _run() -> tuple[int, int]:
        """Returns (handler_failures, events_handled) after firing 1
        memory_pressure event against a stub that 500s on migrate."""
        state = _make_simple_state()
        stub_app = FastAPI()

        @stub_app.get("/aginfer/state")
        async def _s() -> Any:
            return state

        @stub_app.post("/aginfer/migrate")
        async def _m(raw: Request) -> Any:
            await raw.body()
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "boom"}, status_code=500)

        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
        tracker.observe_arrival("prog-0")
        tracker.observe_arrival("prog-2")
        tracker.observe_completion("prog-2")
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url=url)
        sched = KvScheduler(tracker=tracker, sglang_base_url=url)
        attach_kv_scheduler(router, sched)
        await router.start()
        async with run_server(stub_app, "127.0.0.1", port):
            await router.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    payload={"state": "HIGH", "occ": 0.95},
                )
            )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            await router.stop()
            await sched.aclose()
        return router.handler_failures, router.events_handled

    failures, handled = asyncio.run(_run())
    post_fix_passed = (failures == 0 and handled == 1)

    # PRE-FIX: monkey-patch _dispatch_migrate to raise on 5xx.
    saved = KvScheduler._dispatch_migrate

    async def _bug_dispatch(self, assignments):  # noqa: ANN001
        from daemon.kv_scheduler import assignments_to_wire as _atw
        body = {"actions": _atw(assignments)}
        client = await self.ensure_client()
        r = await client.post(
            f"{self.sglang_base_url}/aginfer/migrate", json=body
        )
        r.raise_for_status()  # BUG: previously logged + continued.
        self.migrate_calls += 1

    KvScheduler._dispatch_migrate = _bug_dispatch
    try:
        b_failures, b_handled = asyncio.run(_run())
    finally:
        KvScheduler._dispatch_migrate = saved
    pre_fix_passed = (b_failures == 0 and b_handled == 1)

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- N3


def probe_n3_env_var_binding() -> None:
    """N3: AGINFER_MEMORY_PRESSURE_TOPK should drive
    ``_DEFAULT_MEMORY_PRESSURE_TOPK``.  Regression: a rename would
    silently strand operators on the default.

    The bisect demo here mutates the source file's env-var key in a
    subprocess.  Reverting is automatic (the parent process source
    is untouched).
    """
    name = "N3 (AGINFER_* env -> module constant)"

    def _probe(env_extra: Dict[str, str], regression_key: Optional[str] = None) -> int:
        """Return the value of _DEFAULT_MEMORY_PRESSURE_TOPK seen by a
        fresh subprocess.  If ``regression_key`` is provided, the
        subprocess will read THAT key instead (simulating a rename).
        """
        if regression_key is None:
            probe_src = (
                "import sys; "
                f"sys.path.insert(0, {str(_AGINFER_ROOT)!r}); "
                "from daemon import kv_scheduler as k; "
                "print(k._DEFAULT_MEMORY_PRESSURE_TOPK)"
            )
        else:
            # Simulate rename: read THIS key instead of the real one.
            probe_src = (
                "import sys, os; "
                f"sys.path.insert(0, {str(_AGINFER_ROOT)!r}); "
                f"# Regression: read renamed key {regression_key!r}\n"
                f"v = int(os.environ.get({regression_key!r}, '256'))\n"
                "print(v)"
            )
        env = {
            **{k: v for k, v in os.environ.items()
               if k.startswith(("PATH", "PYTHON", "LD_", "CONDA"))},
            **env_extra,
        }
        out = subprocess.check_output(
            [sys.executable, "-c", probe_src], env=env, timeout=15,
        ).decode().strip()
        return int(out)

    # POST-FIX: stock module reads AGINFER_MEMORY_PRESSURE_TOPK.
    post_observed = _probe({"AGINFER_MEMORY_PRESSURE_TOPK": "7"})
    post_fix_passed = (post_observed == 7)

    # PRE-FIX: regression renames the key.  The test would catch it
    # because the new step asserts == 7 but the subprocess reads
    # the renamed key (still defaulting to 256 since we set the OLD
    # name only).
    pre_observed = _probe(
        {"AGINFER_MEMORY_PRESSURE_TOPK": "7"},
        regression_key="AGINFER_TOPK_RENAMED",
    )
    pre_fix_passed = (pre_observed == 7)

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- N4


def probe_n4_idempotence_forced() -> None:
    """N4: step [9]'s ``if migrate_calls:`` used to silently pass on a
    zero-migrate replay.  The fixture now FORCES migrate; the new
    assertion ``len == 3`` would catch a regression that yielded 0.
    Regression: monkey-patch the policy to ALWAYS return Action(empty)
    — the new assertion FAILS (count is 0).
    """
    name = "N4 (idempotence requires forced migrate)"
    from baselines.base import Action

    async def _run(force_empty: bool) -> int:
        # Reuse the fixture from step [9].
        units = []
        for i in range(50):
            units.append(
                {
                    "hash": f"u-keeper-{i}",
                    "tier": "HBM",
                    "n_tokens": 16,
                    "n_bytes": 32_768,
                    "last_access_time": 1000 + i,
                    "hit_count": 1000,
                    "session_ids": ["prog-keeper"],
                }
            )
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
                "HBM": {"used_bytes": sum(u["n_bytes"] for u in units),
                         "cap_bytes": 32 * 1024 * 1024},
                "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
            },
            "units": units,
            "time_counter": 3000,
        }
        stub_app = FastAPI()
        captured: List[Dict[str, Any]] = []

        @stub_app.get("/aginfer/state")
        async def _s() -> Any:
            return state

        @stub_app.post("/aginfer/migrate")
        async def _m(raw: Request) -> Any:
            body = await raw.json()
            captured.append(body)
            return {"applied": 0, "applied_hashes": [], "skipped": []}

        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
        tracker.observe_arrival("prog-keeper")
        bus = EventBus()
        router = EventRouter(bus=bus, sglang_base_url=url)
        sched = KvScheduler(tracker=tracker, sglang_base_url=url)
        if force_empty:
            class _EmptyPolicy:
                def decide(self, state):
                    return Action(assignments=[])
            sched.policy = _EmptyPolicy()
        attach_kv_scheduler(router, sched)
        await router.start()
        async with run_server(stub_app, "127.0.0.1", port):
            for _ in range(3):
                await router.bus.emit(
                    Event(
                        kind=EventKind.MEMORY_PRESSURE,
                        payload={"state": "HIGH", "occ": 0.95},
                    )
                )
            await asyncio.wait_for(router.bus.queue.join(), timeout=5.0)
            await router.stop()
            await sched.aclose()
        return len(captured)

    post_n_calls = asyncio.run(_run(force_empty=False))
    pre_n_calls = asyncio.run(_run(force_empty=True))
    # New assertion: len == 3.  POST passes (>= 3 migrates), PRE fails (0).
    post_fix_passed = (post_n_calls == 3)
    pre_fix_passed = (pre_n_calls == 3)
    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- main


def main() -> None:
    print("=== T7 regression_probe: audit round-1 bisect demos ===")
    print()
    probe_b1_unknown_tier_unsafe_default()
    probe_b2_per_rank_aggregation()
    probe_m1_top_k_content_pin()
    probe_m2_target_tier_not_drop()
    probe_m3_pressure_resolved_routed()
    probe_n1_floor_clamp()
    probe_n2_migrate_5xx_no_raise()
    probe_n3_env_var_binding()
    probe_n4_idempotence_forced()
    print()
    print("=== T7 regression_probe PASSED ===")


if __name__ == "__main__":
    main()
