"""#160 re-verification driver: re-run T14 stress probe with the
original #160 trigger setup to honestly check whether the
state-dump path still trips the PLAN F3-revisit p99 > 50 ms
threshold.

Original (2026-05-31) reading: peak p99 = 321.94 ms (6.4× over
threshold) — diagnosis was contention/GC-driven spike on the
state-dump call competing with prefill/decode in the scheduler.

Setup mirrors verify/t14/README.md "Stress measurement":
  * Qwen3-0.6B, TP=1, GPU 6
  * --max-total-tokens 65536 (tight cap → eviction churn under
    light prompt volume)
  * HiCache + write_through (NOT write_through_selective)
  * 32 concurrent unique-prefix chats × 90 s
  * stress_probe polls /aginfer/state every 150 ms

Pass = peak p99 < 50 ms (i.e. trigger no longer fires) → #160 closable.
Fail = peak p99 >= 50 ms → trigger still fires; #160 stays open.

Usage:
    python dev/aginfer/verify/t14/run_stress_real.py
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


HOST = "127.0.0.1"
PORT = 30040  # avoid clash with smoke / v4flash / t15 / integration_stress
MODEL = os.environ.get("T160_MODEL", "Qwen/Qwen3-0.6B")
GPU = os.environ.get("T160_GPU", "6")


_HERE = Path(__file__).resolve().parent


def _launch_sglang(log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", HOST, "--port", str(PORT),
        "--tp", "1",
        # Tight HBM cap = the original #160 trigger setup.
        "--max-total-tokens", "65536",
        "--enable-hierarchical-cache",
        "--hicache-ratio", "1.5",
        # ORIGINAL #160 used full write_through (not selective) —
        # mirror that for an apples-to-apples comparison.
        "--hicache-write-policy", "write_through",
        "--max-running-requests", "32",
        "--trust-remote-code",
    ]
    f = open(log_path, "w")
    return subprocess.Popen(
        cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
    )


async def _wait_ready(timeout_s: float = 480.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as cli:
        while time.time() < deadline:
            try:
                r = await cli.get(
                    f"http://{HOST}:{PORT}/health_generate"
                )
                if r.status_code == 200:
                    return
            except (httpx.RequestError, httpx.HTTPError):
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(
        f"sglang on {HOST}:{PORT} not ready after {timeout_s:.0f}s"
    )


def main() -> int:
    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    sglang_log = results_dir / f"{ts}_t160_sglang.log"
    samples_tsv = results_dir / f"{ts}_t160_revisit_stress_samples.tsv"

    print(f"[#160] launching sglang TP=1 on GPU {GPU} port {PORT}")
    print(f"[#160] sglang log: {sglang_log}")
    proc = _launch_sglang(sglang_log)
    try:
        try:
            asyncio.run(_wait_ready())
        except Exception as exc:
            print(f"[#160] sglang startup failed: {exc}")
            return 2
        print("[#160] sglang ready; starting stress probe (32 conc × 90 s)")
        # Reuse the existing stress_probe.py — same parameters as
        # the original 2026-05-31 #160-opening run.
        env = os.environ.copy()
        env["AGINFER_VERIFY_BASE"] = f"http://{HOST}:{PORT}"
        rc = subprocess.call(
            [
                sys.executable,
                str(_HERE / "stress_probe.py"),
                "--concurrency", "32",
                "--duration", "90",
                "--max-tokens", "200",
                "--prefix-min-tokens", "256",
                "--prefix-max-tokens", "512",
                "--out", str(samples_tsv),
                # Use default --threshold-ms 50.0
            ],
            env=env,
        )
        # stress_probe.py exits 1 when peak_p99_ms >= --threshold-ms.
        print(f"[#160] stress_probe returncode = {rc}")
        print(f"[#160] samples TSV: {samples_tsv}")
        if rc == 0:
            print(
                "[#160] peak p99 stayed UNDER 50 ms threshold — "
                "#160 trigger no longer fires under this fixture; "
                "candidate for closure."
            )
        elif rc == 1:
            print(
                "[#160] peak p99 CROSSED 50 ms threshold — F3-revisit "
                "still warranted.  #160 stays open."
            )
        else:
            print(f"[#160] stress_probe failed with rc={rc}")
        return rc
    finally:
        if proc.poll() is None:
            print("[#160] tearing down sglang")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
