"""integration_stress shared harness: sglang + daemon launchers.

Lifecycle managers + helpers for the six A–F stress flavors.  Each
flavor's verify file imports from here so the heavy launch logic
(model load, hicache init, CUDA graph capture, daemon startup) is
implemented once.

Conventions
-----------
* sglang on port 30030 (avoids clashing with smoke 30001, v4flash
  30002, t15 30015).
* daemon on port 9130.
* GPUs from env ``T_INT_GPUS`` (default "5,6"); single-GPU sglang
  TP=1 unless ``--tp 2`` requested explicitly.
* ``SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`` (required for /aginfer/state
  to emit the post-T17 schema).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import httpx


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 30030
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 9130

GPUS = os.environ.get("T_INT_GPUS", "5,6")
MODEL = os.environ.get("T_INT_MODEL", "Qwen/Qwen3-0.6B")


@dataclass
class StackHandles:
    sglang_proc: subprocess.Popen
    daemon_proc: subprocess.Popen
    sglang_log: Path
    daemon_log: Path


def _launch_sglang(log_path: Path, *, tp: int = 1) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", SGLANG_HOST, "--port", str(SGLANG_PORT),
        "--tp", str(tp),
        "--mem-fraction-static", "0.10",
        "--enable-hierarchical-cache",
        "--hicache-ratio", "1.2",
        "--hicache-write-policy", "write_through_selective",
        "--max-running-requests", "32",
        "--trust-remote-code",
        # T22 (#155): sglang's AginferWebhookFirer is only constructed
        # when this flag is set; without it the scheduler's
        # update_aginfer_thresholds returns ok=False with reason
        # "sglang launched without --aginfer-notify-url".  Flavor E
        # needs the firer to exist.  Point at the daemon's /aginfer/
        # event endpoint so the integration is real on this leg too.
        "--aginfer-notify-url", f"http://{DAEMON_HOST}:{DAEMON_PORT}/aginfer/event",
    ]
    f = open(log_path, "w")
    return subprocess.Popen(
        cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
    )


def _launch_daemon(
    log_path: Path,
    *,
    extra_args: Optional[list] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_AGINFER_ROOT)
    args = [
        sys.executable, "-m", "daemon.main",
        "--sglang-base-url", f"http://{SGLANG_HOST}:{SGLANG_PORT}",
        "--host", DAEMON_HOST, "--port", str(DAEMON_PORT),
        "--kv-scheduler", "enabled",
        "--admission-controller", "enabled",
        "--observability-summary-every-n", "50",
    ]
    if extra_args:
        args.extend(extra_args)
    f = open(log_path, "w")
    return subprocess.Popen(
        args, env=env, stdout=f, stderr=subprocess.STDOUT,
        cwd=str(_AGINFER_ROOT),
    )


async def _wait_http(
    url: str, *, timeout_s: float, label: str,
) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as cli:
        while time.time() < deadline:
            try:
                r = await cli.get(url)
                if r.status_code == 200:
                    return
            except (httpx.RequestError, httpx.HTTPError):
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(
        f"{label} on {url} not ready after {timeout_s:.0f}s"
    )


async def wait_sglang(timeout_s: float = 480.0) -> None:
    await _wait_http(
        f"http://{SGLANG_HOST}:{SGLANG_PORT}/health_generate",
        timeout_s=timeout_s, label="sglang",
    )


async def wait_daemon(timeout_s: float = 30.0) -> None:
    await _wait_http(
        f"http://{DAEMON_HOST}:{DAEMON_PORT}/health",
        timeout_s=timeout_s, label="daemon",
    )


@contextlib.contextmanager
def stack(
    results_dir: Path,
    *,
    tp: int = 1,
    daemon_extra_args: Optional[list] = None,
    skip_daemon: bool = False,
) -> Iterator[StackHandles]:
    """Bring up sglang + daemon as subprocesses, yield handles, tear
    down on exit.  Even on exception, both processes are terminated
    via SIGTERM with a 15s grace then SIGKILL."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    sglang_log = results_dir / f"{ts}_sglang.log"
    daemon_log = results_dir / f"{ts}_daemon.log"
    # ORDER: daemon FIRST.  sglang's bootstrap_thresholds_into_server_
    # args (T22 #165) calls GET daemon:/aginfer/thresholds at launch
    # and halts on failure ("deployment-ordering bug" per
    # daemon/aginfer_webhook.py).  The daemon, in contrast, tolerates
    # a dead sglang at startup (cold_start_probe logs and continues).
    daemon_proc: Optional[subprocess.Popen] = None
    sglang_proc: Optional[subprocess.Popen] = None
    try:
        if not skip_daemon:
            daemon_proc = _launch_daemon(daemon_log, extra_args=daemon_extra_args)
            try:
                asyncio.run(wait_daemon())
            except Exception:
                print(f"[harness] daemon startup failed; log={daemon_log}")
                raise
        sglang_proc = _launch_sglang(sglang_log, tp=tp)
        try:
            asyncio.run(wait_sglang())
        except Exception:
            print(f"[harness] sglang startup failed; log={sglang_log}")
            raise
        handles = StackHandles(
            sglang_proc=sglang_proc,
            daemon_proc=daemon_proc,  # type: ignore[arg-type]
            sglang_log=sglang_log,
            daemon_log=daemon_log,
        )
        yield handles
    finally:
        # Tear down sglang first (so it doesn't try to fire pending
        # webhooks against a dying daemon).
        for label, proc in (("sglang", sglang_proc), ("daemon", daemon_proc)):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(f"[harness] {label} did not exit on TERM; killing")
                    proc.kill()
                    proc.wait(timeout=5)
