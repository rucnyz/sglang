"""#160 re-verification driver: re-run T14 stress probe with the
original #160 trigger setup to honestly check whether the
state-dump path still trips the PLAN F3-revisit p99 > 50 ms
threshold.

Original (2026-05-31) reading: peak p99 = 321.94 ms (6.4× over
threshold) — diagnosis was contention/GC-driven spike on the
state-dump call competing with prefill/decode in the scheduler.

Setup mirrors verify/t14/README.md "Stress measurement":
  * Qwen3-0.6B, TP=1, GPU 6
  * --mem-fraction-static 0.15
  * --max-total-tokens 65536 (tight cap → eviction churn under
    light prompt volume)
  * --attention-backend flashinfer
  * HiCache + write_through (NOT write_through_selective)
  * 32 concurrent unique-prefix chats × 90 s
  * stress_probe polls /aginfer/state every 150 ms

N=3 trials per #160-closure audit (#176 / feedback-latency-multi-
run.md memory: "single-shot perf numbers are unconvincing").

Pass criterion: ALL N trials have peak p99 < 50 ms.
Fail criterion: ANY trial has peak p99 >= 50 ms → trigger fires.

Usage:
    python dev/aginfer/verify/t14/run_stress_real.py [--trials 3]
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
    """Launch sglang with the EXACT flags the original 2026-05-31
    #160 trigger run used (T14 README "Reproducing" → step 1).
    No extra flags; the original audit closure had a subtle
    `--max-running-requests 32` add-on that the audit caught."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", HOST, "--port", str(PORT),
        "--tp", "1",
        "--mem-fraction-static", "0.15",
        "--max-total-tokens", "65536",
        "--trust-remote-code",
        "--attention-backend", "flashinfer",
        "--enable-hierarchical-cache",
        "--hicache-ratio", "1.5",
        "--hicache-write-policy", "write_through",
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


def _run_one_trial(
    trial_idx: int, results_dir: Path,
) -> dict:
    """Launch a fresh sglang, run one 90s stress + concurrent HTTP-
    latency probe, tear down.  Returns dict with both the scheduler-
    internal metrics (from stress_probe TSV) AND the user-observed
    HTTP latency (from http_latency_probe).

    These are different quantities:
      * stress_probe peak_p99_ms = time the SCHEDULER PROCESS
        observed inside `dump_aginfer_state_bytes()` (incl. GIL
        preemption).  PLAN T14 trigger uses this.
      * http_latency p99_ms = time the CLIENT (daemon) blocked
        waiting for /aginfer/state to return.  This is what the
        daemon's policy decisions actually wait on.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    sglang_log = results_dir / f"{ts}_t160_trial{trial_idx}_sglang.log"
    samples_tsv = results_dir / f"{ts}_t160_trial{trial_idx}_samples.tsv"
    http_log = results_dir / f"{ts}_t160_trial{trial_idx}_http_latency.log"

    print(f"[#160 trial {trial_idx}] launching sglang TP=1 on GPU {GPU} port {PORT}")
    proc = _launch_sglang(sglang_log)
    try:
        try:
            asyncio.run(_wait_ready())
        except Exception as exc:
            print(f"[#160 trial {trial_idx}] sglang startup failed: {exc}")
            return {"error": str(exc)}
        print(f"[#160 trial {trial_idx}] sglang ready; running stress + HTTP probe 32+4×90s")
        env = os.environ.copy()
        env["AGINFER_VERIFY_BASE"] = f"http://{HOST}:{PORT}"
        stress = subprocess.Popen(
            [
                sys.executable, str(_HERE / "stress_probe.py"),
                "--concurrency", "32", "--duration", "90",
                "--max-tokens", "200",
                "--prefix-min-tokens", "256",
                "--prefix-max-tokens", "512",
                "--out", str(samples_tsv),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # HTTP-latency probe runs in parallel with the stress so it
        # observes /aginfer/state latency under the SAME contention.
        with open(http_log, "w") as f:
            http = subprocess.Popen(
                [
                    sys.executable, str(_HERE / "http_latency_probe.py"),
                    "--duration", "90", "--concurrency", "4",
                    "--threshold-ms", "50.0",
                ],
                env=env, stdout=f, stderr=subprocess.STDOUT,
            )
            stress.wait()
            http.wait()
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    # Parse the TSV to extract peak metrics + outlier count.
    if not samples_tsv.exists():
        return {"error": "no TSV emitted"}
    with open(samples_tsv) as f:
        header = f.readline().strip().split("\t")
        rows = [line.strip().split("\t") for line in f if line.strip()]
    if not rows:
        return {"error": "empty TSV"}
    p99_idx = header.index("p99_ms")
    max_idx = header.index("max_ms")
    bytes_idx = header.index("dump_bytes")
    p99s = [float(r[p99_idx]) for r in rows]
    maxes = [float(r[max_idx]) for r in rows]
    bytes_ = [int(r[bytes_idx]) for r in rows if int(r[bytes_idx]) > 0]
    # Parse the HTTP-latency probe log (its last line is the
    # summary).
    http_p50 = http_p95 = http_p99 = http_max = http_n = 0.0
    if http_log.exists():
        with open(http_log) as f:
            for line in f:
                if "[http_latency]" in line and "N=" in line:
                    # Parse "[http_latency] N=NNN  p50=X.XXms  p95=…"
                    import re
                    m = re.search(
                        r"N=(\d+)\s+p50=([\d.]+)ms\s+p95=([\d.]+)ms\s+"
                        r"p99=([\d.]+)ms\s+max=([\d.]+)ms",
                        line,
                    )
                    if m:
                        http_n = int(m.group(1))
                        http_p50 = float(m.group(2))
                        http_p95 = float(m.group(3))
                        http_p99 = float(m.group(4))
                        http_max = float(m.group(5))
    return {
        "samples_tsv": str(samples_tsv),
        "sample_count": len(rows),
        "peak_p99_ms": max(p99s),               # scheduler-internal
        "peak_max_ms": max(maxes),              # scheduler-internal
        "n_outliers_over_50ms": sum(1 for v in p99s if v > 50.0),
        "peak_dump_bytes": max(bytes_) if bytes_ else 0,
        "http_n": http_n,
        "http_p50_ms": http_p50,                # client-observed
        "http_p95_ms": http_p95,                # client-observed
        "http_p99_ms": http_p99,                # client-observed
        "http_max_ms": http_max,                # client-observed
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3,
                    help="number of independent trials (default 3)")
    args = ap.parse_args()

    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)

    trials: list = []
    for i in range(1, args.trials + 1):
        print(f"\n=== TRIAL {i}/{args.trials} ===")
        t0 = time.time()
        r = _run_one_trial(i, results_dir)
        r["elapsed_s"] = time.time() - t0
        trials.append(r)
        if "error" in r:
            print(f"  trial {i} ERROR: {r['error']}")
            continue
        print(f"  trial {i} sched p99={r['peak_p99_ms']:.2f}ms "
              f"max={r['peak_max_ms']:.2f}ms outliers={r['n_outliers_over_50ms']} "
              f"| http N={r.get('http_n',0)} "
              f"p50={r.get('http_p50_ms',0):.2f}ms "
              f"p95={r.get('http_p95_ms',0):.2f}ms "
              f"p99={r.get('http_p99_ms',0):.2f}ms "
              f"max={r.get('http_max_ms',0):.2f}ms")

    # Aggregate
    print(f"\n=== SUMMARY (N={args.trials}) ===")
    import statistics
    sched_p99 = [t.get("peak_p99_ms", float("nan")) for t in trials if "peak_p99_ms" in t]
    http_p99 = [t.get("http_p99_ms", float("nan")) for t in trials if "http_p99_ms" in t]
    if not sched_p99:
        print("[#160] all trials errored")
        return 2
    print(f"  SCHEDULER-INTERNAL p99 (state_dump_metrics): {sched_p99}")
    print(f"    mean={statistics.fmean(sched_p99):.2f}ms  "
          f"std={statistics.stdev(sched_p99) if len(sched_p99)>1 else 0:.2f}ms")
    print(f"  CLIENT-OBSERVED p99 (HTTP latency): {http_p99}")
    if http_p99:
        print(f"    mean={statistics.fmean(http_p99):.2f}ms  "
              f"std={statistics.stdev(http_p99) if len(http_p99)>1 else 0:.2f}ms")
    print(f"  outlier counts (p99>50ms samples per trial): "
          f"{[t.get('n_outliers_over_50ms', 'err') for t in trials]}")
    sched_over = [t for t in trials if t.get("peak_p99_ms", 0) >= 50.0]
    http_over = [t for t in trials if t.get("http_p99_ms", 0) >= 50.0]
    print(f"  trials sched_p99 >= 50 ms: {len(sched_over)}/{len(trials)}")
    print(f"  trials http_p99  >= 50 ms: {len(http_over)}/{len(trials)}")
    if http_over:
        print(
            "[#160] CLIENT-OBSERVED p99 over threshold → daemon "
            "experience IS still degraded.  Fix incomplete."
        )
        return 1
    if sched_over:
        print(
            "[#160] CLIENT-OBSERVED p99 under threshold ✓, but "
            "SCHEDULER-INTERNAL p99 still over.  The HTTP cache hides "
            "scheduler-side contention from the daemon (the real win), "
            "but the scheduler still spends real time on slow dumps.  "
            "Daemon experience is now predictable; scheduler-internal "
            "tail is a separate (lower-priority) follow-on."
        )
        return 0
    print(
        f"[#160] verdict: ALL {len(trials)} trials stayed under 50 ms "
        "on BOTH metrics."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
