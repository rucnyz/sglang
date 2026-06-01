"""T24 — HASH_COLLISION webhook + detection (#182, DESIGN §4 + §10).

Detection in `apply_aginfer_migrations` (sglang side) emits
`{"hash_collisions": [{"key", "node_a_summary", "node_b_summary"},
...]}` in the result dict.  The scheduler's `_fire_hash_collisions`
hooks each into `AginferWebhookFirer.fire_hash_collision()`.  The
daemon's `_hash_collision_handler` calls `fatal('hash_collision',
...)` on receipt — deployment-bug class.

Verify (no GPU; everything stubbed):

  A. Sglang side (webhook firer)
    A0 fire_hash_collision builds correct payload, posts to
       notify_url (`kind=hash_collision`, key, both summaries)
    A1 _send_hash_collision retries on 5xx, gives up after 3
    A2 fire_hash_collision is non-blocking (returns immediately)

  B. Scheduler wiring
    B0 `_fire_hash_collisions` skips entries without a key
    B1 `_fire_hash_collisions` no-ops when aginfer_webhook is None

  C. Daemon side
    C0 EventKind.HASH_COLLISION is defined
    C1 _hash_collision_handler calls fatal() with full context

  D. Subprocess integration
    D0 Real daemon (subprocess) + fake sglang stub firing the
       webhook → daemon exits 1 + forensic file lands with
       `reason=hash_collision` + all context fields
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import httpx


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- A. webhook firer (in-process, with a captured-POST stub) ----


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _run_capture_server(captured: List[Dict[str, Any]],
                        status_seq: List[int]) -> tuple:
    """Spin a tiny stdlib HTTPServer in a thread; capture POST
    bodies + return configurable status codes from `status_seq`.

    Stdlib instead of uvicorn to avoid asyncio-loop collision with
    the AginferWebhookFirer's background loop (also asyncio-based).

    Returns (port, stop_callable)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"i": 0, "lock": threading.Lock()}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # silence default access log

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {"_raw": raw.decode("utf-8", "replace")}
            with state["lock"]:
                captured.append(body)
                if state["i"] < len(status_seq):
                    code = status_seq[state["i"]]
                    state["i"] += 1
                else:
                    code = 200
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    def stop():
        server.shutdown()
        server.server_close()
        t.join(timeout=3.0)

    return port, stop


def _wait_firer_loop(firer, timeout_s: float = 2.0) -> None:
    """AginferWebhookFirer starts its background loop in a thread;
    wait until the loop is actually running before firing."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        loop = firer._loop
        if loop is not None and loop.is_running():
            return
        time.sleep(0.01)
    raise RuntimeError("firer background loop never came up")


def stage_a0_payload_shape() -> None:
    from sglang.srt.managers.aginfer_webhook import AginferWebhookFirer

    captured: List[Dict[str, Any]] = []
    port, stop = _run_capture_server(captured, status_seq=[200])
    try:
        firer = AginferWebhookFirer(
            notify_url=f"http://127.0.0.1:{port}/aginfer/event",
            theta_hi=0.9, theta_lo=0.7, theta_crit=0.95,
            heartbeat_s=5.0,
        )
        _wait_firer_loop(firer)
        firer.fire_hash_collision(
            key="deadbeef",
            node_a_summary={"node_id": 1, "n_tokens": 64,
                            "residence": ["HBM"],
                            "hash_value": "deadbeef"},
            node_b_summary={"node_id": 2, "n_tokens": 128,
                            "residence": ["HBM", "DRAM"],
                            "hash_value": "deadbeef"},
        )
        # Wait for delivery.
        deadline = time.time() + 5.0
        while time.time() < deadline and not captured:
            time.sleep(0.05)
        if not captured:
            raise StageFail("webhook never delivered")
        body = captured[0]
        if body.get("kind") != "hash_collision":
            raise StageFail(f"kind wrong: {body!r}")
        if body.get("key") != "deadbeef":
            raise StageFail(f"key wrong: {body!r}")
        for side in ("node_a_summary", "node_b_summary"):
            s = body.get(side)
            if not isinstance(s, dict):
                raise StageFail(f"{side} not dict: {body!r}")
            for k in ("node_id", "n_tokens", "residence",
                      "hash_value"):
                if k not in s:
                    raise StageFail(f"{side} missing {k!r}: {s!r}")
    finally:
        stop()


def stage_a1_retry_on_5xx() -> None:
    """First two attempts 503, third 200 → exactly 3 POSTs captured."""
    from sglang.srt.managers.aginfer_webhook import AginferWebhookFirer

    captured: List[Dict[str, Any]] = []
    port, stop = _run_capture_server(
        captured, status_seq=[503, 503, 200],
    )
    try:
        firer = AginferWebhookFirer(
            notify_url=f"http://127.0.0.1:{port}/aginfer/event",
            theta_hi=0.9, theta_lo=0.7, theta_crit=0.95,
            heartbeat_s=5.0,
        )
        _wait_firer_loop(firer)
        firer.fire_hash_collision(
            key="cafebabe",
            node_a_summary={"node_id": 10},
            node_b_summary={"node_id": 20},
        )
        # Backoff: 0.1 + 0.4 = 0.5 s minimum; allow generous slack.
        deadline = time.time() + 8.0
        while time.time() < deadline and len(captured) < 3:
            time.sleep(0.05)
        if len(captured) != 3:
            raise StageFail(
                f"expected exactly 3 POSTs (2 retries + final); "
                f"got {len(captured)}"
            )
    finally:
        stop()


def stage_a2_non_blocking() -> None:
    """fire_hash_collision returns quickly even if the target is
    unreachable.  Important: it's called from the scheduler's main
    loop and must NOT block."""
    from sglang.srt.managers.aginfer_webhook import AginferWebhookFirer

    # Notify URL points at an unused port; this stresses the
    # transport-error path.
    bad_port = _free_port()
    firer = AginferWebhookFirer(
        notify_url=f"http://127.0.0.1:{bad_port}/aginfer/event",
        theta_hi=0.9, theta_lo=0.7, theta_crit=0.95,
        heartbeat_s=5.0,
    )
    _wait_firer_loop(firer)
    t0 = time.perf_counter()
    firer.fire_hash_collision(
        key="bad", node_a_summary={"id": 1}, node_b_summary={"id": 2},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > 50.0:
        raise StageFail(
            f"fire_hash_collision blocked the caller for "
            f"{elapsed_ms:.1f} ms (must be < 50 ms — runs from "
            f"the scheduler hot path)"
        )


# ---- B. scheduler wiring (unit-level) ----


class _StubFirer:
    """Captures fire_hash_collision calls for stage B."""
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
    def fire_hash_collision(self, *, key, node_a_summary, node_b_summary):
        self.calls.append({
            "key": key,
            "node_a_summary": node_a_summary,
            "node_b_summary": node_b_summary,
        })


def stage_b0_skips_missing_key() -> None:
    """An entry without a `key` field is silently dropped (defensive
    — schedule may produce malformed entries during refactors)."""
    from sglang.srt.managers.scheduler import Scheduler
    firer = _StubFirer()
    # _fire_hash_collisions is a bound method; we can mimic with an
    # ad-hoc shim that mirrors what the production method does.
    # Cheaper: call the method on a synthetic instance with just the
    # webhook attribute set.
    class _Shim:
        aginfer_webhook = firer
        _fire_hash_collisions = Scheduler._fire_hash_collisions
    s = _Shim()
    _Shim._fire_hash_collisions(s, [
        {"key": "good", "node_a_summary": {}, "node_b_summary": {}},
        {"node_a_summary": {}, "node_b_summary": {}},   # missing key
        {"key": "", "node_a_summary": {}, "node_b_summary": {}},  # empty
    ])
    if len(firer.calls) != 1:
        raise StageFail(
            f"only the entry with a real key should fire; got "
            f"{len(firer.calls)} calls"
        )


def stage_b1_no_webhook_is_noop() -> None:
    """When aginfer_webhook is None (sglang launched without
    --aginfer-notify-url), _fire_hash_collisions returns silently."""
    from sglang.srt.managers.scheduler import Scheduler
    class _Shim:
        aginfer_webhook = None
        _fire_hash_collisions = Scheduler._fire_hash_collisions
    s = _Shim()
    # Should not raise.
    _Shim._fire_hash_collisions(s, [
        {"key": "k1", "node_a_summary": {}, "node_b_summary": {}},
    ])


# ---- C. daemon side ----


def stage_c0_event_kind_defined() -> None:
    from daemon.events import EventKind
    if not hasattr(EventKind, "HASH_COLLISION"):
        raise StageFail("EventKind.HASH_COLLISION not defined")
    if EventKind.HASH_COLLISION.value != "hash_collision":
        raise StageFail(
            f"EventKind.HASH_COLLISION value wrong: "
            f"{EventKind.HASH_COLLISION.value!r}"
        )


def stage_c1_handler_invokes_fatal_with_context() -> None:
    """Mock fatal() and call _hash_collision_handler — assert fatal
    receives all the context keys from the event payload."""
    from daemon.event_router import _hash_collision_handler
    from daemon.events import Event, EventKind
    import daemon._fatal as _fatal_mod

    captured: Dict[str, Any] = {}

    def fake_fatal(reason, **ctx):
        captured["reason"] = reason
        captured["ctx"] = ctx
        raise RuntimeError("fatal-stub")

    orig = _fatal_mod.fatal
    _fatal_mod.fatal = fake_fatal
    try:
        evt = Event(
            kind=EventKind.HASH_COLLISION,
            session=None,
            payload={
                "key": "deadbeef",
                "node_a_summary": {"node_id": 1, "n_tokens": 64},
                "node_b_summary": {"node_id": 2, "n_tokens": 128},
                "ts": 1234.5,
                "ts_monotonic": 6789.0,
            },
        )
        try:
            asyncio.run(_hash_collision_handler(evt, router=None))
        except RuntimeError as e:
            if "fatal-stub" not in str(e):
                raise StageFail(f"unexpected exc: {e}")
    finally:
        _fatal_mod.fatal = orig

    if captured.get("reason") != "hash_collision":
        raise StageFail(f"reason wrong: {captured!r}")
    ctx = captured.get("ctx", {})
    for k in ("key", "node_a_summary", "node_b_summary",
              "ts", "ts_monotonic"):
        if k not in ctx:
            raise StageFail(f"fatal() context missing {k!r}: {ctx!r}")
    if ctx["key"] != "deadbeef":
        raise StageFail(f"key not propagated: {ctx!r}")


# ---- D. subprocess integration ----

_SUBPROCESS_SCRIPT = r"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, {aginfer_root!r})
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
os.environ["AGINFER_DATA_DIR"] = {data_dir!r}

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response


# ---- fake sglang side: an empty /aginfer/thresholds endpoint so the
# daemon's bootstrap doesn't halt (irrelevant for the test).
# ---- but we also need the daemon to come up.  Daemon needs sglang_
# base_url to be reachable for the cold_start_probe (which logs +
# continues on failure, so we can leave it dead).

# Launch daemon as subprocess.
DAEMON_PORT = {daemon_port}
SGLANG_DEAD_PORT = 30099  # nothing listens

daemon_env = os.environ.copy()
daemon_env["PYTHONPATH"] = {aginfer_root!r}
daemon_env["AGINFER_DATA_DIR"] = {data_dir!r}
daemon_log = Path({data_dir!r}) / "daemon.log"
daemon = subprocess.Popen(
    [
        sys.executable, "-m", "daemon.main",
        "--sglang-base-url", f"http://127.0.0.1:{{SGLANG_DEAD_PORT}}",
        "--host", "127.0.0.1", "--port", str(DAEMON_PORT),
        "--kv-scheduler", "enabled",
        "--admission-controller", "enabled",
    ],
    env=daemon_env, cwd={aginfer_root!r},
    stdout=open(daemon_log, "w"), stderr=subprocess.STDOUT,
)


async def main():
    # Wait for daemon /health.
    deadline = time.time() + 30.0
    async with httpx.AsyncClient(timeout=2.0) as cli:
        while time.time() < deadline:
            try:
                r = await cli.get(f"http://127.0.0.1:{{DAEMON_PORT}}/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            print("daemon never came up", file=sys.stderr)
            sys.exit(2)
        # Fire HASH_COLLISION webhook (simulating what
        # aginfer_webhook._send_hash_collision would POST).
        body = {{
            "kind": "hash_collision",
            "key": "deadbeefcafe",
            "node_a_summary": {{"node_id": 100, "n_tokens": 64,
                                  "residence": ["HBM"],
                                  "hash_value": "deadbeefcafe",
                                  "session_ids": ["pA"], "hit_count": 5}},
            "node_b_summary": {{"node_id": 200, "n_tokens": 128,
                                  "residence": ["HBM", "DRAM"],
                                  "hash_value": "deadbeefcafe",
                                  "session_ids": ["pB"], "hit_count": 9}},
            "ts": time.time(),
            "ts_monotonic": time.monotonic(),
        }}
        try:
            await cli.post(
                f"http://127.0.0.1:{{DAEMON_PORT}}/aginfer/event",
                json=body, timeout=5.0,
            )
        except Exception as e:
            print(f"post raised: {{e!r}}", file=sys.stderr)

    # Wait for daemon to fatal (it should exit 1 within seconds).
    wait_deadline = time.time() + 30.0
    while time.time() < wait_deadline and daemon.poll() is None:
        await asyncio.sleep(0.2)

asyncio.run(main())

if daemon.poll() is None:
    daemon.send_signal(__import__("signal").SIGTERM)
    try:
        daemon.wait(timeout=5)
    except subprocess.TimeoutExpired:
        daemon.kill()
        daemon.wait(timeout=3)

sys.exit(daemon.returncode if daemon.returncode is not None else 99)
"""


def stage_d0_subprocess_daemon_fatals() -> None:
    with tempfile.TemporaryDirectory(prefix="t24_d0_") as td:
        data_dir = Path(td)
        port = _free_port()
        body = _SUBPROCESS_SCRIPT.format(
            aginfer_root=str(_AGINFER_ROOT),
            data_dir=str(data_dir),
            daemon_port=port,
        )
        result = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True, timeout=90,
        )
        if result.returncode != 1:
            raise StageFail(
                f"daemon should fatal(rc=1) on HASH_COLLISION; got "
                f"rc={result.returncode}; stderr_tail="
                f"{result.stderr[-600:]!r}"
            )
        # Forensic file should be in data_dir/forensic/.
        forensic_dir = data_dir / "forensic"
        matches = (sorted(forensic_dir.glob("hash_collision_*.json"))
                   if forensic_dir.exists() else [])
        if not matches:
            raise StageFail(
                f"no forensic file in {forensic_dir}; "
                f"existing={list(data_dir.iterdir()) if data_dir.exists() else 'none'}"
            )
        payload = json.loads(matches[0].read_text())
        if payload.get("reason") != "hash_collision":
            raise StageFail(f"reason wrong: {payload.get('reason')!r}")
        ctx = payload.get("context", {})
        for k in ("key", "node_a_summary", "node_b_summary"):
            if k not in ctx:
                raise StageFail(f"context missing {k!r}: keys={list(ctx)}")
        if ctx["key"] != "deadbeefcafe":
            raise StageFail(f"key not propagated: {ctx['key']!r}")
        # Each summary should retain the structural fields.
        for side in ("node_a_summary", "node_b_summary"):
            s = ctx[side]
            for sk in ("node_id", "n_tokens", "residence",
                       "hash_value", "hit_count"):
                if sk not in s:
                    raise StageFail(f"{side} missing {sk!r}: {s!r}")


# ---- run ----

_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 fire_hash_collision payload shape",        stage_a0_payload_shape),
    ("A1 _send_hash_collision retries on 5xx (3 attempts total)",
                                                    stage_a1_retry_on_5xx),
    ("A2 fire_hash_collision is non-blocking",      stage_a2_non_blocking),
    ("B0 _fire_hash_collisions skips missing key",  stage_b0_skips_missing_key),
    ("B1 _fire_hash_collisions no-ops when webhook=None",
                                                    stage_b1_no_webhook_is_noop),
    ("C0 EventKind.HASH_COLLISION defined",         stage_c0_event_kind_defined),
    ("C1 daemon handler → fatal('hash_collision', …)",
                                                    stage_c1_handler_invokes_fatal_with_context),
    ("D0 subprocess: daemon receives + fatals + forensic",
                                                    stage_d0_subprocess_daemon_fatals),
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
        print(_red(f"\nT24 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT24 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
