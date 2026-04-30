#!/usr/bin/env python3
"""Layer 2 regression+benefit test suite driver.

Runs a fixed set of jobs (workload × arm) under a GPU-pool scheduler
that occupies as many free H200s as possible. Each job's output is a
single per-job JSON with the metrics the suite cares about. The driver
aggregates them into a PASS/FAIL table with per-row gates.

Usage:
    python dev/eval/regression_suite/run.py [--out OUT_DIR] [--gpus N] [--baseline-gpu G]

Conventions:
    - GPU 0 is reserved by an external process (per the user's setup).
    - Free GPUs are detected by `nvidia-smi --query-gpu=memory.used`.
    - Each job runs with CUDA_VISIBLE_DEVICES set to a single integer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SGLANG_DIR = SCRIPT_DIR.parent.parent.parent
PYTHON = SGLANG_DIR / ".venv" / "bin" / "python"


@dataclass
class Job:
    name: str           # unique id, e.g. "R1_steady_baseline"
    workload: str       # workload key
    arm: str            # "baseline" or "prelude"
    runner: str         # path to the bash runner under workloads/
    port: int           # listen port unique to job
    extra_env: dict = field(default_factory=dict)
    # Acceptance gates evaluated against the produced metrics.json.
    # The driver evaluates these AFTER all jobs of a workload finish so it
    # can compare baseline vs prelude.
    pass_metric: Optional[str] = None        # e.g. "input_tps"
    pass_min_relative: Optional[float] = None  # 0.97 = require ≥ 97% baseline
    pass_max_relative: Optional[float] = None  # 1.03 = ≤ 103% (regression check)


@dataclass
class JobResult:
    job: Job
    ok: bool
    metrics: dict
    log_tail: str
    duration_s: float
    gpu: int


def discover_free_gpus(reserved: set[int]) -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()
    free = []
    for line in out:
        idx, mem_mib = [s.strip() for s in line.split(",")]
        idx = int(idx)
        mem_mib = int(mem_mib)
        # A "free" GPU has < 1 GiB used (some baseline drivers reserve a
        # few hundred MB, and our prior runs may have left state).
        if idx in reserved:
            continue
        if mem_mib < 1024:
            free.append(idx)
    return free


def run_job(job: Job, gpu: int, out_dir: Path) -> JobResult:
    job_dir = out_dir / job.name
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"
    metrics_path = job_dir / "metrics.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PORT"] = str(job.port)
    env["OUT_DIR"] = str(job_dir)
    env["METRICS_PATH"] = str(metrics_path)
    for k, v in job.extra_env.items():
        env[k] = str(v)
    t0 = time.time()
    with open(log_path, "wb") as logf:
        proc = subprocess.run(
            ["bash", job.runner],
            cwd=SGLANG_DIR,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=60 * 30,
        )
    duration = time.time() - t0
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {"error": "metrics.json could not be parsed"}
    log_tail = log_path.read_text(errors="replace").splitlines()[-30:]
    return JobResult(
        job=job, ok=(proc.returncode == 0 and bool(metrics)),
        metrics=metrics, log_tail="\n".join(log_tail),
        duration_s=duration, gpu=gpu,
    )


def schedule(jobs: list[Job], gpus: list[int], out_dir: Path) -> list[JobResult]:
    """Run jobs over a GPU pool. One job per GPU; new job starts when a GPU frees."""
    results: list[JobResult] = []
    pending = list(jobs)
    in_flight: dict[int, tuple[Job, int]] = {}  # GPU -> (job, future_id)

    print(f"[suite] scheduling {len(pending)} jobs over {len(gpus)} GPUs: {gpus}")

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        future_to_gpu: dict = {}
        free = list(gpus)

        def submit(job: Job, gpu: int):
            print(f"[suite] launch  job={job.name:<28} gpu={gpu} port={job.port}")
            f = pool.submit(run_job, job, gpu, out_dir)
            future_to_gpu[f] = gpu

        # Prime
        while free and pending:
            submit(pending.pop(0), free.pop(0))

        while future_to_gpu:
            for f in as_completed(list(future_to_gpu.keys())):
                gpu = future_to_gpu.pop(f)
                try:
                    r = f.result()
                except Exception as e:
                    print(f"[suite] job on GPU {gpu} crashed: {e}")
                    continue
                results.append(r)
                status = "OK" if r.ok else "FAIL"
                print(f"[suite] finish job={r.job.name:<28} gpu={gpu} dur={r.duration_s:.1f}s {status}")
                # Refill
                if pending:
                    submit(pending.pop(0), gpu)
                break  # re-loop to as_completed with the (possibly extended) future set
    return results


def evaluate(results: list[JobResult]) -> tuple[list[dict], bool]:
    """Group baseline+prelude pairs by workload, evaluate gates."""
    by_workload: dict = {}
    for r in results:
        by_workload.setdefault(r.job.workload, {})[r.job.arm] = r
    rows = []
    all_pass = True
    for wl, arms in sorted(by_workload.items()):
        bl = arms.get("baseline")
        pr = arms.get("prelude")
        if bl is None or pr is None or not bl.ok or not pr.ok:
            rows.append({
                "workload": wl,
                "status": "INCOMPLETE",
                "reason": "missing arm or job failed",
            })
            all_pass = False
            continue
        # Pull metrics. Each workload's runner writes its own keys.
        bl_m, pr_m = bl.metrics, pr.metrics
        tps_bl = bl_m.get("input_tps") or bl_m.get("input_throughput") or 0
        tps_pr = pr_m.get("input_tps") or pr_m.get("input_throughput") or 0
        ttft_bl = bl_m.get("mean_ttft_ms") or 0
        ttft_pr = pr_m.get("mean_ttft_ms") or 0
        e2e_bl = bl_m.get("median_e2e_ms") or bl_m.get("median_e2e_latency_ms") or 0
        e2e_pr = pr_m.get("median_e2e_ms") or pr_m.get("median_e2e_latency_ms") or 0
        xfers = pr_m.get("xpool_transfers", 0)
        # Job's pass criteria
        job = pr.job
        delta_pct = 100 * (tps_pr - tps_bl) / tps_bl if tps_bl > 0 else 0
        if job.pass_metric == "input_tps":
            ok = (job.pass_min_relative is None or tps_pr >= job.pass_min_relative * tps_bl) \
                 and (job.pass_max_relative is None or tps_pr <= job.pass_max_relative * tps_bl)
        elif job.pass_metric == "median_e2e_ms":
            ok = (job.pass_min_relative is None or e2e_pr >= job.pass_min_relative * e2e_bl) \
                 and (job.pass_max_relative is None or e2e_pr <= job.pass_max_relative * e2e_bl)
        else:
            ok = True  # no gate set
        rows.append({
            "workload": wl,
            "baseline_tps": tps_bl,
            "prelude_tps": tps_pr,
            "delta_tps_pct": delta_pct,
            "baseline_ttft_ms": ttft_bl,
            "prelude_ttft_ms": ttft_pr,
            "baseline_e2e_ms": e2e_bl,
            "prelude_e2e_ms": e2e_pr,
            "transfers": xfers,
            "status": "PASS" if ok else "FAIL",
        })
        if not ok:
            all_pass = False
    return rows, all_pass


def render(rows: list[dict]) -> str:
    out = []
    out.append(f"{'workload':<20} {'baseline TPS':>13} {'prelude TPS':>13} "
               f"{'Δ %':>7} {'BL TTFT':>9} {'PR TTFT':>9} "
               f"{'BL E2E':>9} {'PR E2E':>9} {'xfers':>6}  status")
    out.append("-" * 110)
    for r in rows:
        if r.get("status") == "INCOMPLETE":
            out.append(f"{r['workload']:<20}  -- INCOMPLETE: {r.get('reason','')}")
            continue
        out.append(
            f"{r['workload']:<20} "
            f"{r['baseline_tps']:>13.1f} "
            f"{r['prelude_tps']:>13.1f} "
            f"{r['delta_tps_pct']:>+6.1f}% "
            f"{r['baseline_ttft_ms']:>8.1f}ms "
            f"{r['prelude_ttft_ms']:>8.1f}ms "
            f"{r['baseline_e2e_ms']:>8.1f}ms "
            f"{r['prelude_e2e_ms']:>8.1f}ms "
            f"{r['transfers']:>6}  {r['status']}"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reserved-gpus", default="0",
                    help="comma-separated GPU IDs to skip (e.g. '0' for shared GPU 0)")
    args = ap.parse_args()

    reserved = {int(x) for x in args.reserved_gpus.split(",") if x.strip()}
    free = discover_free_gpus(reserved)
    if not free:
        print("ERROR: no free GPUs detected"); sys.exit(2)

    out_dir = Path(args.out) if args.out else Path(f"/tmp/regsuite_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[suite] out_dir = {out_dir}")

    # Import the manifest (kept in sibling jobs.py)
    sys.path.insert(0, str(SCRIPT_DIR))
    from jobs import build_manifest  # noqa
    jobs = build_manifest()
    results = schedule(jobs, free, out_dir)
    rows, all_pass = evaluate(results)
    table = render(rows)
    print()
    print(table)

    summary_path = out_dir / "regression_check.md"
    with open(summary_path, "w") as f:
        f.write("# Layer 2 regression+benefit suite\n\n")
        f.write("```\n")
        f.write(table)
        f.write("\n```\n\n")
        f.write(f"Overall: {'PASS' if all_pass else 'FAIL'}\n")
    print(f"\n[suite] summary written to {summary_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
