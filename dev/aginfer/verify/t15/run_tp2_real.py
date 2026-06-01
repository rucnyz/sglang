"""T15 real-run driver: DP=2 sglang under sustained eviction churn.

PLAN §1 T15 asks for "TP > 1 deployment under high hint churn".
Note: ``/aginfer/state`` AGGREGATES across TP ranks
(``http_server.py`` exposes ``per_rank`` only when the
communicator returns multiple responses, which happens for DP > 1,
not TP > 1 — TP ranks within one DP group share a logical
scheduler).  The pre-aggregation per-TP-rank view is not exposed
as an endpoint today.

So this driver runs DP=2 — two independent KV pools serving the
same model.  Under DP, divergence between replicas is
EXPECTED-BY-DESIGN: each replica serves a different program
subset, so eviction sets differ across windows.  The point of this
real-run is to:

  (1) Prove the detector + parser work end-to-end against real
      sglang ``per_rank`` JSON (not just synthetic dicts).
  (2) Confirm /aginfer/state's per_rank wire format under DP > 1.

For the TP-rank divergence half of T15 (the actual §6 invariant
the probe is meant to catch), see task #174 — needs a sglang patch
exposing per-TP-rank state pre-aggregation.

Usage:
    python dev/aginfer/verify/t15/run_tp2_real.py [--duration 60]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from detector import detect_divergence, summarise  # noqa: E402


HOST = "127.0.0.1"
PORT = 30015  # avoid clashing with smoke (30001) / v4flash (30002) ports
MODEL = os.environ.get("T15_MODEL", "Qwen/Qwen3-0.6B")
GPUS = os.environ.get("T15_GPUS", "5,6")


def _launch_sglang(log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    # UnifiedRadixCache is the only tree cache that exposes the
    # aginfer state schema; without this flag sglang falls back to
    # HiRadixCache and /aginfer/state returns `unsupported_tree_cache`.
    env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", HOST, "--port", str(PORT),
        "--dp", "2",  # two DP replicas → /aginfer/state exposes per_rank
        # Cap HBM to ~20 GB per rank so eviction triggers at modest
        # request volume — we want churn, not endurance.
        "--mem-fraction-static", "0.10",
        "--enable-hierarchical-cache",
        # Small DRAM mirror for fast startup; HBM eviction still works.
        "--hicache-ratio", "1.2",
        "--hicache-write-policy", "write_through_selective",
        "--max-running-requests", "32",
        "--trust-remote-code",
    ]
    f = open(log_path, "w")
    return subprocess.Popen(
        cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
    )


async def _wait_ready(timeout_s: float = 480.0) -> None:
    """Poll /health until ready or timeout."""
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as cli:
        while time.time() < deadline:
            try:
                r = await cli.get(f"http://{HOST}:{PORT}/health_generate")
                if r.status_code == 200:
                    return
            except (httpx.RequestError, httpx.HTTPError):
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(
        f"sglang on {HOST}:{PORT} not ready after {timeout_s}s"
    )


async def _churn_worker(idx: int, deadline_ts: float) -> int:
    """Fire varied prompts to force radix-cache turnover.  Each
    prompt is unique (uuid in the system message) so there is no
    cross-request prefix reuse; the radix tree fills with new
    leaves until eviction kicks in.

    Returns the count of requests this worker completed."""
    n = 0
    # Padding text inflates the prompt so each request commits enough
    # pages into the radix tree to register as a unit (UnifiedRadix's
    # unit boundary is page-aligned; 8-token prompts often stay below
    # the threshold).
    PAD = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
        "do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco "
        "laboris nisi ut aliquip ex ea commodo consequat. "
    ) * 8  # ~ 1.2 k tokens of constant filler — shared prefix candidate
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
    ) as cli:
        while time.time() < deadline_ts:
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "system",
                     "content": f"{PAD} (worker-tag-{idx}-{uuid.uuid4().hex[:8]})"},
                    {"role": "user",
                     "content": (
                         f"reply with the literal token 'ack' (request "
                         f"{n} from worker {idx} with nonce "
                         f"{uuid.uuid4().hex[:12]})"
                     )},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            }
            try:
                r = await cli.post(
                    f"http://{HOST}:{PORT}/v1/chat/completions",
                    json=body,
                )
                if r.status_code == 200:
                    n += 1
            except (httpx.RequestError, asyncio.TimeoutError):
                pass
    return n


async def _poll_state(period_s: float, deadline_ts: float) -> List[Dict[str, Any]]:
    """Snapshot /aginfer/state every ``period_s`` until deadline.
    Returns the time-ordered list of state dumps."""
    snapshots: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as cli:
        while time.time() < deadline_ts:
            try:
                r = await cli.get(f"http://{HOST}:{PORT}/aginfer/state")
                if r.status_code == 200:
                    snapshots.append(r.json())
            except (httpx.RequestError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(period_s)
    return snapshots


async def _amain(duration_s: float, n_workers: int) -> int:
    deadline = time.time() + duration_s
    workers = [
        asyncio.create_task(_churn_worker(i, deadline))
        for i in range(n_workers)
    ]
    snapshots_task = asyncio.create_task(_poll_state(0.5, deadline))

    counts = await asyncio.gather(*workers)
    snapshots = await snapshots_task

    print(f"[T15] traffic generated: {sum(counts)} requests across "
          f"{n_workers} workers")
    print(f"[T15] snapshots captured: {len(snapshots)}")

    # Filter to multi-rank snapshots only (single-rank would be a
    # sglang misconfiguration; bail loudly).
    if not snapshots:
        print("[T15] FAIL: zero snapshots — sglang state endpoint dead")
        return 2
    if "per_rank" not in snapshots[0]:
        print("[T15] FAIL: snapshot lacks per_rank — sglang launched DP=1?")
        return 2
    n_ranks = len(snapshots[0]['per_rank'])
    print(f"[T15] ranks per snapshot: {n_ranks}")

    # Total unit-population per snapshot (eviction triggers iff this
    # cycles up and down).
    pops = [
        sum(len(r.get("units", [])) for r in s["per_rank"])
        for s in snapshots
    ]
    print(f"[T15] unit population range: min={min(pops)} max={max(pops)} "
          f"final={pops[-1]}")

    reports = detect_divergence(snapshots)
    n_windows = len(snapshots) - 1
    print()
    print(summarise(reports))
    print(f"\n[T15] {len(reports)} divergent windows / {n_windows} total")

    # Under DP > 1, EXPECTED-BY-DESIGN: each replica serves a
    # different program subset, so eviction sets MUST differ.  The
    # contract this real-run validates is *parser correctness*, not
    # the §6 cross-TP-rank-eviction invariant (that needs a sglang
    # patch to expose per-TP-rank state pre-aggregation — task #174).
    if not reports and n_windows > 5:
        print("[T15] WARN: zero divergence across DP replicas under "
              "churn — either traffic was load-balanced 100% to one "
              "replica, or replicas evicted in lockstep (very "
              "unlikely under unique-prompt churn).  Inspect snapshot "
              "trajectory before treating this as a pass.")
    print(f"[T15] parser handled {len(snapshots)} real per_rank JSON "
          f"snapshots without raising — detector contract green")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds of churn traffic (default 60)")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent prompt workers (default 16)")
    ap.add_argument("--keep-server", action="store_true",
                    help="leave sglang running after exit (for debug)")
    args = ap.parse_args()

    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    server_log = results_dir / f"{ts}_t15_tp2_sglang.log"

    print(f"[T15] launching sglang TP=2 on GPUs {GPUS} port {PORT}")
    print(f"[T15] model: {MODEL}")
    print(f"[T15] server log: {server_log}")
    proc = _launch_sglang(server_log)
    try:
        asyncio.run(_wait_ready())
    except Exception as exc:
        print(f"[T15] sglang startup failed: {exc}")
        print(f"[T15] check {server_log}")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        return 2
    print("[T15] sglang ready; starting churn")

    try:
        rc = asyncio.run(_amain(args.duration, args.workers))
    finally:
        if not args.keep_server:
            print("[T15] shutting sglang down")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
