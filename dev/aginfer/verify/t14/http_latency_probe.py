"""Measure HTTP-observed latency on /aginfer/state under load.

Companion to stress_probe.py, which reports the SCHEDULER-INTERNAL
state_dump_metrics latency (compute time including GIL preemption).
This probe measures the latency CLIENT-SIDE — the time the daemon
actually blocks waiting for a state-fetch response.

#160 fix added an HTTP-layer cache + bg refresh task; this probe
validates that user-observed latency is consistently low while
the scheduler-internal metric (which the cache doesn't fix) can
stay spiky.

Usage:
    AGINFER_VERIFY_BASE=http://127.0.0.1:30040 \\
    python dev/aginfer/verify/t14/http_latency_probe.py \\
        --duration 60 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import List

import httpx


def _q(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


async def _poll_loop(
    base_url: str, duration_s: float, samples: List[float],
) -> None:
    deadline = time.time() + duration_s
    async with httpx.AsyncClient(timeout=30.0) as cli:
        while time.time() < deadline:
            t0 = time.perf_counter()
            try:
                r = await cli.get(f"{base_url}/aginfer/state")
                if r.status_code == 200:
                    samples.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass


async def main_async(args: argparse.Namespace) -> int:
    all_samples: List[float] = []
    pollers = [
        _poll_loop(args.base, args.duration, all_samples)
        for _ in range(args.concurrency)
    ]
    await asyncio.gather(*pollers)
    if not all_samples:
        print(f"[http_latency] no successful polls", file=sys.stderr)
        return 2
    p50 = _q(all_samples, 0.50)
    p95 = _q(all_samples, 0.95)
    p99 = _q(all_samples, 0.99)
    mx = max(all_samples)
    n = len(all_samples)
    print(f"[http_latency] N={n}  p50={p50:.2f}ms  p95={p95:.2f}ms  "
          f"p99={p99:.2f}ms  max={mx:.2f}ms")
    return 0 if p99 < args.threshold_ms else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AGINFER_VERIFY_BASE", ""))
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--threshold-ms", type=float, default=50.0)
    args = ap.parse_args()
    if not args.base:
        print("set --base or AGINFER_VERIFY_BASE")
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
