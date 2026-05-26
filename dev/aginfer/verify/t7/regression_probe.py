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
    # Audit round-2 R2-N3: previously yielded even if uvicorn never
    # bound; downstream probes would silently exercise a non-listening
    # port and produce vacuous PASS/FAIL.  Match verify.py's guard.
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
        # Audit round-2 R2-X2: align with `_check_pre`'s exact match
        # (no rN/ prefix any more after R2-B1 fix; single-rank fixture
        # has bare hashes).
        for body in calls:
            for a in body.get("actions", []):
                if a.get("hash") == "u-tail-2":
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

    Audit round-2 R2-M2: the previous probe was a tautology — it
    pointed the subprocess at a literally renamed key (without
    importing kv_scheduler at all), so it demonstrated "subprocess
    reads a different key", not "if source renames the key, verify
    catches it".  This rewrite actually modifies the SOURCE PATH:
    writes a shadow copy of ``daemon/kv_scheduler.py`` with the env
    var key sed-replaced, points the subprocess at THAT module, and
    observes whether ``AGINFER_MEMORY_PRESSURE_TOPK=7`` still drives
    the constant.

    POST-FIX (stock source): subprocess imports stock module, env var
    binds → constant == 7.
    PRE-FIX (sed'd source): subprocess imports shadow with renamed
    env var; setting the OLD key has no effect → constant == 256
    (default).  Test would catch this.
    """
    name = "N3 (AGINFER_* env -> module constant, real bisect)"
    import tempfile, shutil

    src_root = _AGINFER_ROOT
    real_src = (src_root / "daemon" / "kv_scheduler.py").read_text()

    def _probe_with_source(custom_src: str, env_extra: Dict[str, str]) -> int:
        """Lay down a shadow copy of the daemon package, run a
        subprocess that imports it from the shadow path, returns
        ``_DEFAULT_MEMORY_PRESSURE_TOPK``."""
        shadow = tempfile.mkdtemp(prefix="aginfer_probe_n3_")
        try:
            # Copy minimal package: daemon/__init__.py + the shadowed
            # kv_scheduler.py.  Other daemon modules aren't imported
            # at module-level constants, but kv_scheduler does
            # ``from .events import``, so we need events.py and
            # program_tracker.py too.  Easiest: shutil.copytree.
            shutil.copytree(src_root / "daemon", Path(shadow) / "daemon")
            # Also copy baselines (kv_scheduler imports from it).
            shutil.copytree(
                src_root / "baselines", Path(shadow) / "baselines",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            # Overwrite the shadow kv_scheduler with the custom source.
            (Path(shadow) / "daemon" / "kv_scheduler.py").write_text(custom_src)
            probe_src = (
                f"import sys; sys.path.insert(0, {shadow!r}); "
                "from daemon import kv_scheduler as k; "
                "print(k._DEFAULT_MEMORY_PRESSURE_TOPK)"
            )
            env = {
                **{k: v for k, v in os.environ.items()
                   if k.startswith(("PATH", "PYTHON", "LD_", "CONDA"))},
                **env_extra,
            }
            out = subprocess.check_output(
                [sys.executable, "-c", probe_src], env=env, timeout=20,
            ).decode().strip().splitlines()[-1]
            return int(out)
        finally:
            shutil.rmtree(shadow, ignore_errors=True)

    # POST-FIX: stock source.
    post_observed = _probe_with_source(real_src, {"AGINFER_MEMORY_PRESSURE_TOPK": "7"})
    post_fix_passed = (post_observed == 7)

    # PRE-FIX: source with the env var renamed.  Setting the OLD env
    # var should NOT bind; constant should be 256 (default).
    renamed_src = real_src.replace(
        "AGINFER_MEMORY_PRESSURE_TOPK", "AGINFER_TOPK_RENAMED"
    )
    assert renamed_src != real_src, "source-replace did not match the env var key"
    pre_observed = _probe_with_source(
        renamed_src, {"AGINFER_MEMORY_PRESSURE_TOPK": "7"}
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

    Audit round-2 R2-X1: this probe builds its own fixture (50
    keepers + 5 sentinels + tight HBM cap + memory_pressure replay)
    that mirrors verify [9]'s ``step_idempotent_repeat_event``
    fixture line-by-line.  If verify [9] is later refactored to use
    a different fixture, this probe will drift; the alternative
    (import step_idempotent_repeat_event and monkey-patch its
    KvScheduler.policy) requires invasive surgery on the verify
    module.  Keeping a STAND-IN fixture here is the explicit
    trade-off.
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


# ================================================================ round 2


# ---------------------------------------------------------------- R2-B1


async def probe_r2_b1_multi_rank_hash_round_trip() -> None:
    """R2-B1: round-1's per_rank fix prefixed hashes with ``rN/`` so
    the daemon could "route back" to the right rank.  But sglang's
    ``POST /aginfer/migrate`` does an EXACT ``hash_to_node.get(h)``
    lookup — `r0/u-1` is not in the tree, lookup misses, sglang
    returns 200 with empty ``applied_hashes`` (everything in
    ``skipped``).  The daemon sees a "successful" POST and the bug
    is invisible.

    This probe simulates the real lookup: stub records the tree of
    hashes it knows about (the unprefixed ones), and ``applied_hashes``
    is the intersection.  Probe assertion: ``applied_hashes`` is
    non-empty (i.e., dispatch is actually reaching sglang).

    PRE-FIX (rN/ prefix kept): dispatched body has ``r0/u-1``,
    intersection is empty, FAIL.
    POST-FIX (no prefix): dispatched body has ``u-1``, intersection
    non-empty, PASS.
    """
    name = "R2-B1 (multi-rank hash round-trip)"

    per_rank_state = {
        "per_rank": [
            {
                "tier_usage": {
                    "HBM": {"used_bytes": 8 * 1024 * 1024, "cap_bytes": 16 * 1024 * 1024},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                    "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
                },
                "units": [
                    {
                        "hash": "u-rank0-sentinel",
                        "tier": "HBM",
                        "n_tokens": 4096,
                        "n_bytes": 8 * 1024 * 1024,
                        "last_access_time": 0,
                        "hit_count": 0,
                        "session_ids": [],
                    },
                ],
                "time_counter": 50,
            },
            {
                "tier_usage": {
                    "HBM": {"used_bytes": 8 * 1024 * 1024, "cap_bytes": 16 * 1024 * 1024},
                    "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                    "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
                },
                "units": [
                    {
                        "hash": "u-rank1-sentinel",
                        "tier": "HBM",
                        "n_tokens": 4096,
                        "n_bytes": 8 * 1024 * 1024,
                        "last_access_time": 0,
                        "hit_count": 0,
                        "session_ids": [],
                    },
                ],
                "time_counter": 75,
            },
        ]
    }
    # sglang's tree has the UNPREFIXED hashes (this is the contract).
    sglang_tree_hashes = {"u-rank0-sentinel", "u-rank1-sentinel"}

    async def _check() -> bool:
        """Returns True iff at least one dispatched hash actually
        landed in ``applied_hashes`` (i.e., the migrate actually did
        something on sglang's side)."""
        stub_app = FastAPI()
        applied_observed: List[List[str]] = []

        @stub_app.get("/aginfer/state")
        async def _s() -> Any:
            return per_rank_state

        @stub_app.post("/aginfer/migrate")
        async def _m(raw: Request) -> Any:
            body = await raw.json()
            actions = body.get("actions", []) or []
            dispatched = [a.get("hash") for a in actions if isinstance(a, dict)]
            # Mimic sglang's exact-hash lookup.
            applied = [h for h in dispatched if h in sglang_tree_hashes]
            applied_observed.append(applied)
            return {
                "applied": len(applied),
                "applied_hashes": applied,
                "skipped": [
                    {"hash": h, "reason": "not_in_tree"}
                    for h in dispatched if h not in sglang_tree_hashes
                ],
            }

        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        tracker = ProgramTracker()
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
        # PASS iff some POST landed at least one hash in the tree.
        return any(applied for applied in applied_observed)

    # POST-FIX (no prefix) — stock code under test.  Run first; if the
    # prod fix isn't applied yet, this returns False (bug present).
    post_fix_passed = await _check()

    # PRE-FIX simulation: monkey-patch _flatten_per_rank to re-introduce
    # the rN/ prefix.
    saved_flatten = kvs_mod._flatten_per_rank

    def _bug_flatten(state_json: Dict[str, Any]) -> Dict[str, Any]:
        per_rank = state_json.get("per_rank")
        if not isinstance(per_rank, list) or not per_rank:
            return state_json
        agg_tu: Dict[str, Dict[str, int]] = {
            "HBM": {"used_bytes": 0, "cap_bytes": 0},
            "DRAM": {"used_bytes": 0, "cap_bytes": 0},
            "DISK": {"used_bytes": 0, "cap_bytes": 0},
        }
        agg_units: List[Dict[str, Any]] = []
        agg_time = 0
        for r_idx, rank in enumerate(per_rank):
            if not isinstance(rank, dict):
                continue
            rank_tu = rank.get("tier_usage", {}) or {}
            for label in agg_tu:
                sub = rank_tu.get(label, {}) or {}
                agg_tu[label]["used_bytes"] += int(sub.get("used_bytes", 0) or 0)
                agg_tu[label]["cap_bytes"] += int(sub.get("cap_bytes", 0) or 0)
            for u in rank.get("units", []) or []:
                if not isinstance(u, dict):
                    continue
                u2 = dict(u)
                uhash = str(u2.get("hash", ""))
                if uhash and not uhash.startswith(f"r{r_idx}/"):
                    u2["hash"] = f"r{r_idx}/{uhash}"  # BUG
                agg_units.append(u2)
            agg_time = max(agg_time, int(rank.get("time_counter", 0) or 0))
        return {
            "tier_usage": agg_tu, "units": agg_units, "time_counter": agg_time,
        }

    kvs_mod._flatten_per_rank = _bug_flatten
    try:
        pre_fix_passed = await _check()
    finally:
        kvs_mod._flatten_per_rank = saved_flatten

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- R2-M1


def probe_r2_m1_paused_gets_acting_floor() -> None:
    """R2-M1: a program that is PAUSED (admission_controller pinned
    mid-tool-call) is STILL mid-tool-call; paper §7 "expected reuse
    interval ~ tool duration" applies.  Round-1 only triggered the
    ACTING-floor for ``State.ACTING``, so PAUSED programs fell back
    to the ``hits/age`` proxy — the OPPOSITE of paper intent (a
    high-hit prefix would get a high λ and be KEPT on HBM during the
    tool call, exactly what we're trying to avoid).

    PRE-FIX: ``build_paper_state`` with a PAUSED holder → λ derives
    from ``hits/age`` (e.g. for a frequently-accessed unit, that's
    much higher than ACTING-floor 1/5 = 0.2).
    POST-FIX: PAUSED → ACTING-floor 0.2, same as ACTING.
    """
    name = "R2-M1 (PAUSED gets ACTING-floor lambda)"

    state_json = {
        "tier_usage": {
            "HBM": {"used_bytes": 8 * 1024 * 1024, "cap_bytes": 64 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": [
            {
                "hash": "u-paused-tail",
                "tier": "HBM",
                "n_tokens": 4096,
                "n_bytes": 4096 * 2048,
                "last_access_time": 100,
                "hit_count": 1000,  # high hits → high hits/age → high λ
                "session_ids": ["prog-paused"],
            },
        ],
        "time_counter": 101,
    }

    def _check() -> bool:
        tracker = ProgramTracker()
        tracker.observe_arrival("prog-paused")
        # Drive into PAUSED via the public pause() helper.  (We don't
        # assert tracker.state == PAUSED here because the pre-fix
        # monkey-patch below intentionally LIES about that state to
        # simulate the round-1 "PAUSED leaks through" bug.)
        tracker.pause("prog-paused")
        s = build_paper_state(
            state_json,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=tracker,
            lambda_acting=0.2,
        )
        u = s.units["u-paused-tail"]
        # FIX: PAUSED → ACTING-floor.  λ exactly 0.2.
        return abs(u.lambda_rate - 0.2) < 1e-9

    post_fix_passed = _check()

    # PRE-FIX: monkey-patch the ACTING-floor logic in build_paper_state.
    # The cleanest way is to monkey-patch ``State`` so that PAUSED is
    # treated as REASONING by the equality check.  We can't change the
    # enum at runtime safely; instead we monkey-patch
    # ``ProgramTracker.state`` to lie (claim PAUSED programs are
    # REASONING).  Same effective regression: the inner branch falls
    # into the non-ACTING bucket.
    saved_state = ProgramTracker.state

    def _lying_state(self, pid):  # type: ignore[no-redef]
        s = saved_state(self, pid)
        if s == State.PAUSED:
            return State.REASONING  # BUG simulation
        return s

    ProgramTracker.state = _lying_state  # type: ignore[assignment]
    try:
        pre_fix_passed = _check()
    finally:
        ProgramTracker.state = saved_state  # type: ignore[assignment]

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- R2-N1


def probe_r2_n1_units_for_session_set_semantics() -> None:
    """R2-N1: `_units_for_session` previously had a redundant
    ``u.holders == [session] or set(u.holders) == {session}`` check.
    For a sane state (holders is a unique list) the two are equivalent,
    but the set version silently accepts ``["s", "s"]`` while the list
    version rejects it.  Round-2 fix: use set semantics exclusively
    (the paper meaning — a unit has a SET of holders, not a list).

    PRE-FIX (the OR was redundant in the round-1 code so the set
    semantics already "won"; the regression we want to catch is a
    REVERT to list-only): swap `_units_for_session` for a list-only
    version; assert a duplicate-holder unit is missed.
    POST-FIX: the dedup via `set(u.holders)` handles ``["s", "s"]``.
    """
    name = "R2-N1 (_units_for_session set semantics for dup holders)"
    # The paper-true intent: a unit held by ["prog-X", "prog-X"] IS
    # exclusively held by prog-X.  A list-only equality check would
    # miss it because ``["prog-X", "prog-X"] != ["prog-X"]``.
    state_json = {
        "tier_usage": {
            "HBM": {"used_bytes": 4 * 1024 * 1024, "cap_bytes": 16 * 1024 * 1024},
            "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
            "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
        },
        "units": [
            {
                "hash": "u-dup-holder",
                "tier": "HBM",
                "n_tokens": 2048,
                "n_bytes": 4 * 1024 * 1024,
                "last_access_time": 0,
                "hit_count": 1,
                # T3's session_id passthrough is a set-on-the-wire too,
                # but a future bug (idempotent re-add) could yield a
                # duplicated list.  Pin the set semantics.
                "session_ids": ["prog-X", "prog-X"],
            },
        ],
        "time_counter": 10,
    }

    def _check() -> bool:
        tracker = ProgramTracker()
        s = build_paper_state(
            state_json,
            event=Event(kind=EventKind.TOOL_CALL_START, session="prog-X"),
            tracker=tracker,
        )
        return "u-dup-holder" in s.decision_set

    post_fix_passed = _check()

    # PRE-FIX: monkey-patch _units_for_session to use list-only check.
    saved = kvs_mod._units_for_session

    def _bug_units_for_session(units, session):
        if session is None:
            return []
        return [
            uid for uid, u in units.items() if u.holders == [session]
        ]

    kvs_mod._units_for_session = _bug_units_for_session
    try:
        pre_fix_passed = _check()
    finally:
        kvs_mod._units_for_session = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


# ---------------------------------------------------------------- R2-N2


def probe_r2_n2_unknown_tier_log_scope() -> None:
    """R2-N2: the ``_logged_unknown_tiers`` set was a module-global
    in round-1, which meant test #1 in a process triggering the warn
    log would suppress test #2 from re-logging the same label.
    Surprised ops during debugging.

    Bisect: invoke ``_log_unknown_tier_once("ZSTD_TEST")`` twice via
    two separate KvScheduler instances.  Under the FIX (instance set):
    each instance has its own set → both calls log → ``seen[0]`` and
    ``seen[1]`` both contain the label (i.e., logged at least once
    per instance).  Under the BUG (module global): first call logs,
    second call finds the label already in the global set → does NOT
    re-log; the second instance has no record.

    The observable: did the SECOND instance's set grow when its
    handler was invoked with an unknown tier?
    """
    name = "R2-N2 (unknown-tier log is instance-scoped, not module)"

    def _check() -> bool:
        s1 = KvScheduler(
            tracker=ProgramTracker(), sglang_base_url="http://x"
        )
        s2 = KvScheduler(
            tracker=ProgramTracker(), sglang_base_url="http://x"
        )
        state_with_unknown = {
            "tier_usage": {
                "HBM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
            },
            "units": [
                {
                    "hash": "u-zstd",
                    "tier": "ZSTD_TEST",
                    "n_tokens": 16,
                    "n_bytes": 32_768,
                    "last_access_time": 0,
                    "hit_count": 0,
                    "session_ids": [],
                }
            ],
            "time_counter": 1,
        }
        # Drive both schedulers through build_paper_state with the
        # instance's log set wired in.
        build_paper_state(
            state_with_unknown,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=s1.tracker,
            unknown_tier_log=s1._unknown_tier_log,
        )
        build_paper_state(
            state_with_unknown,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=s2.tracker,
            unknown_tier_log=s2._unknown_tier_log,
        )
        return (
            "ZSTD_TEST" in s1._unknown_tier_log
            and "ZSTD_TEST" in s2._unknown_tier_log
        )

    post_fix_passed = _check()

    # PRE-FIX: revert to module-global set.  Both instances point at
    # the same set; second instance won't re-log.
    # Simulate: monkey-patch the build_paper_state signature so the
    # ``unknown_tier_log`` parameter is ignored and a shared global
    # set is used instead.
    shared_global: set = set()
    saved = kvs_mod.build_paper_state

    def _bug_build(
        state_json, *, event, tracker, lambda_acting=0.2, now_counter=None,
        unknown_tier_log=None,
    ):
        # Force the module-global behavior: ignore the per-instance
        # log set, point all calls at one shared set.
        return saved(
            state_json,
            event=event,
            tracker=tracker,
            lambda_acting=lambda_acting,
            now_counter=now_counter,
            unknown_tier_log=shared_global,
        )

    kvs_mod.build_paper_state = _bug_build
    try:
        # Run the same _check, but with bug-build replacing the
        # signature.  Because build_paper_state is the module-level
        # function we bind it back here for the in-process call.
        import importlib
        # Re-import the probe's reference too:
        import daemon.kv_scheduler  # noqa: F401
        # Simulate: only ONE instance's set ever sees the label.
        s1 = KvScheduler(
            tracker=ProgramTracker(), sglang_base_url="http://x"
        )
        s2 = KvScheduler(
            tracker=ProgramTracker(), sglang_base_url="http://x"
        )
        state_with_unknown = {
            "tier_usage": {
                "HBM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DRAM": {"used_bytes": 0, "cap_bytes": 1 << 30},
                "DISK": {"used_bytes": 0, "cap_bytes": 1 << 40},
            },
            "units": [
                {
                    "hash": "u-zstd",
                    "tier": "ZSTD_TEST",
                    "n_tokens": 16,
                    "n_bytes": 32_768,
                    "last_access_time": 0,
                    "hit_count": 0,
                    "session_ids": [],
                }
            ],
            "time_counter": 1,
        }
        kvs_mod.build_paper_state(
            state_with_unknown,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=s1.tracker,
            unknown_tier_log=s1._unknown_tier_log,
        )
        kvs_mod.build_paper_state(
            state_with_unknown,
            event=Event(kind=EventKind.MEMORY_PRESSURE),
            tracker=s2.tracker,
            unknown_tier_log=s2._unknown_tier_log,
        )
        # Under the bug, both call paths share `shared_global`; the
        # per-instance sets stay empty.
        pre_fix_passed = (
            "ZSTD_TEST" in s1._unknown_tier_log
            and "ZSTD_TEST" in s2._unknown_tier_log
        )
    finally:
        kvs_mod.build_paper_state = saved

    _bisect_outcome(name, pre_fix_passed, post_fix_passed)


def main() -> None:
    print("=== T7 regression_probe: audit round-1 + round-2 bisect demos ===")
    print()
    print("--- round 1 ---")
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
    print("--- round 2 ---")
    asyncio.run(probe_r2_b1_multi_rank_hash_round_trip())
    probe_r2_m1_paused_gets_acting_floor()
    probe_r2_n1_units_for_session_set_semantics()
    probe_r2_n2_unknown_tier_log_scope()
    print()
    print("=== T7 regression_probe PASSED ===")


if __name__ == "__main__":
    main()
