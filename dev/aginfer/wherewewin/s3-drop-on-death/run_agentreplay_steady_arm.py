#!/usr/bin/env python3
"""Run one steady-state AgentReplay Dead-KV arm on direct SGLang.

This runner imports AgentReplay's scheduler and HTTP request implementation.  A
single trace continuously introduces root sessions across warmup, measurement,
and cooldown windows.  Long-lived programs revisit their prefixes while short
programs finish and become dead KV.  In ``ours`` mode, SESSION_END runs at
program completion on an independent control semaphore; baseline sends no END
until post-measurement cleanup.

Artifacts contain aggregate metrics, safe per-request timing (no token IDs), and
summarized state samples.  Use only on a dedicated deployment: the runner flushes
the cache before and after the arm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_agentreplay_with_telemetry as telemetry  # noqa: E402


SCHEMA_VERSION = 1
TIERS = ("HBM", "DRAM", "DISK")


def utc_now() -> str:
    return telemetry.utc_now()


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"n": 0}

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (upper - position) + ordered[upper] * (
            position - lower
        )

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_trace(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"trace line {line_number} is not an object")
            if row.get("steady_role") not in {"live", "churn"}:
                raise ValueError(
                    f"trace line {line_number} has no valid steady_role"
                )
            records.append(dict(row))
    if not records:
        raise ValueError("steady trace is empty")
    return records


def source_programs(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for record in records:
        program_id = record.get("program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError("steady trace contains an invalid program_id")
        result.setdefault(program_id, []).append(dict(record))
    for program_id, rows in result.items():
        rows.sort(key=lambda row: int(row.get("step") or 0))
        steps = [int(row.get("step") or 0) for row in rows]
        if steps != list(range(1, len(rows) + 1)):
            raise ValueError(f"program {program_id!r} has non-consecutive steps")
        if len({row["steady_role"] for row in rows}) != 1:
            raise ValueError(f"program {program_id!r} changes steady_role")
        parents = {row.get("parent_program_id") for row in rows}
        roots = {row.get("steady_root_id") for row in rows}
        if len(parents) != 1 or len(roots) != 1:
            raise ValueError(f"program {program_id!r} changes bundle metadata")
        parent = rows[0].get("parent_program_id")
        if parent is not None:
            if parent not in result:
                raise ValueError(f"program {program_id!r} has a missing parent")
            spawn = rows[0].get("spawned_at_step")
            parent_steps = {int(row["step"]) for row in result[parent]}
            if not isinstance(spawn, int) or spawn not in parent_steps:
                raise ValueError(f"program {program_id!r} has an invalid spawn step")
        root = rows[0].get("steady_root_id")
        if not isinstance(root, str) or root not in result:
            raise ValueError(f"program {program_id!r} has an invalid steady root")
        if bool(rows[0].get("steady_is_root")) != (program_id == root):
            raise ValueError(f"program {program_id!r} has inconsistent root metadata")
    return result


def blocking_children(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, str | None], dict[str, set[str]]]:
    parents: dict[str, str | None] = {}
    blocking: dict[str, set[str]] = {}
    for program_id, rows in programs.items():
        parent = rows[0].get("parent_program_id")
        parents[program_id] = str(parent) if parent is not None else None
        if parent is not None and not bool(rows[0].get("background_spawn")):
            blocking.setdefault(str(parent), set()).add(program_id)
    return parents, blocking


def load_agentreplay(root: pathlib.Path):
    sys.path.insert(0, str(root))
    try:
        driver = importlib.import_module("agentreplay.driver")
    finally:
        sys.path.pop(0)
    required = (
        "_one_request",
        "_end_program_http",
        "_runtime_program_id",
        "replay",
    )
    missing = [name for name in required if not hasattr(driver, name)]
    if missing:
        raise RuntimeError(f"AgentReplay driver is missing: {', '.join(missing)}")
    parameters = inspect.signature(driver.replay).parameters
    if "end_max_conc" not in parameters:
        raise RuntimeError(
            "AgentReplay driver lacks independent SESSION_END concurrency"
        )
    if getattr(driver, "httpx", None) is None:
        raise RuntimeError("AgentReplay requires httpx")
    return driver


def flush_cache(base_url: str, timeout: float) -> None:
    url = base_url.rstrip("/") + f"/flush_cache?timeout={timeout:g}"
    request = urllib.request.Request(url, data=b"{}", method="POST")
    with telemetry.no_proxy_opener().open(request, timeout=timeout + 5) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError(f"cache flush returned HTTP {response.status}")
        response.read()


def wait_empty(state_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while True:
        payload, error, status = telemetry.fetch_json(state_url, min(10.0, timeout))
        if error is not None:
            raise RuntimeError(f"state request failed: status={status}, error={error}")
        last = telemetry.analyze_state(payload, set(), set())
        if last["unit_count"] == 0 and last["program_usage_count"] == 0:
            return last
        if time.monotonic() >= deadline:
            raise RuntimeError(f"cache did not become empty: {last}")
        time.sleep(0.2)


def safe_request_row(
    record: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = (
        "ok",
        "status",
        "error",
        "ttft_ms",
        "e2e_ms",
        "tpot_ms",
        "n_out",
        "want_out",
        "cached",
        "prompt",
        "force_exact",
        "started_elapsed_seconds",
        "completed_elapsed_seconds",
    )
    result = {key: row.get(key) for key in allowed if key in row}
    result.update(
        {
            "program_id_sha256": hashlib.sha256(
                str(record["program_id"]).encode()
            ).hexdigest(),
            "step": int(record.get("step") or 0),
            "role": record["steady_role"],
            "traffic_class": (
                "live_revisit"
                if record["steady_role"] == "live"
                and int(record.get("step") or 0) > 1
                else (
                    "live_initial"
                    if record["steady_role"] == "live"
                    else "churn"
                )
            ),
            "is_root": bool(record.get("steady_is_root")),
            "root_id_sha256": hashlib.sha256(
                str(record.get("steady_root_id")).encode()
            ).hexdigest(),
            "scheduled_session_arrival_s": float(
                record.get("scheduled_session_arrival_s") or 0.0
            ),
        }
    )
    return result


def request_window_metrics(
    rows: Sequence[Mapping[str, Any]],
    start: float,
    end: float,
    *,
    include_breakdown: bool = True,
) -> dict[str, Any]:
    duration = end - start
    completed = [
        row
        for row in rows
        if start <= float(row.get("completed_elapsed_seconds") or -1) < end
    ]
    started = [
        row
        for row in rows
        if start <= float(row.get("started_elapsed_seconds") or -1) < end
    ]
    ok_completed = [row for row in completed if row.get("ok")]
    ok_started = [row for row in started if row.get("ok")]
    output_tokens = sum(int(row.get("n_out") or 0) for row in ok_completed)
    prompt_tokens = sum(int(row.get("prompt") or 0) for row in ok_started)
    cached_tokens = sum(int(row.get("cached") or 0) for row in ok_started)
    metrics: dict[str, Any] = {
        "window_seconds": duration,
        "requests_started": len(started),
        "requests_completed": len(completed),
        "successful_completions": len(ok_completed),
        "completion_request_rate_per_second": len(ok_completed) / duration,
        # Tokens are charged at request completion.  With a measurement window
        # much longer than request latency, boundary error is negligible.
        "completion_accounted_output_tokens": output_tokens,
        "completion_accounted_goodput_tok_s": output_tokens / duration,
        "start_cohort_prompt_tokens": prompt_tokens,
        "start_cohort_cached_tokens": cached_tokens,
        "start_cohort_cache_hit": (
            cached_tokens / prompt_tokens if prompt_tokens else 0.0
        ),
        "start_cohort_ttft_ms": summarize(
            [
                float(row["ttft_ms"])
                for row in ok_started
                if row.get("ttft_ms") is not None
            ]
        ),
        "start_cohort_e2e_ms": summarize(
            [
                float(row["e2e_ms"])
                for row in ok_started
                if row.get("e2e_ms") is not None
            ]
        ),
    }
    metrics["by_traffic_class"] = (
        {
            traffic_class: request_window_metrics(
                [
                    row
                    for row in rows
                    if row.get("traffic_class") == traffic_class
                ],
                start,
                end,
                include_breakdown=False,
            )
            for traffic_class in ("live_initial", "live_revisit", "churn")
        }
        if include_breakdown
        else {}
    )
    return metrics


def state_value(state: Mapping[str, Any], field: str, tier: str) -> float:
    values = state.get(field)
    if not isinstance(values, Mapping):
        return 0.0
    value = values.get(tier)
    return float(value) if isinstance(value, (int, float)) else 0.0


def state_window_metrics(
    samples: Sequence[Mapping[str, Any]], start: float, end: float
) -> dict[str, Any]:
    valid = sorted(
        (
            sample
            for sample in samples
            if isinstance(sample.get("state"), Mapping)
        ),
        key=lambda sample: float(sample["elapsed_seconds"]),
    )
    fields = {
        "dead_physical_bytes": {tier: 0.0 for tier in TIERS},
        "pool_used_bytes": {tier: 0.0 for tier in TIERS},
        "pool_max_subpool_utilization": {tier: 0.0 for tier in TIERS},
    }
    peaks = {
        field: {tier: 0.0 for tier in TIERS}
        for field in fields
    }
    lifecycle_fields = (
        "logical_ended_programs",
        "session_end_acked_programs",
        "session_end_backlog",
    )
    lifecycle_totals = {field: 0.0 for field in lifecycle_fields}
    lifecycle_peaks = {field: 0.0 for field in lifecycle_fields}
    coverage = 0.0
    for index, sample in enumerate(valid):
        left = max(start, float(sample["elapsed_seconds"]))
        next_time = (
            float(valid[index + 1]["elapsed_seconds"])
            if index + 1 < len(valid)
            else end
        )
        right = min(end, next_time)
        if right <= left:
            continue
        seconds = right - left
        coverage += seconds
        state = sample["state"]
        for field in lifecycle_fields:
            value = float(sample.get(field) or 0.0)
            lifecycle_totals[field] += value * seconds
            lifecycle_peaks[field] = max(lifecycle_peaks[field], value)
        for field in fields:
            for tier in TIERS:
                value = state_value(state, field, tier)
                fields[field][tier] += value * seconds
                peaks[field][tier] = max(peaks[field][tier], value)
    averages = {
        field: {
            tier: total / coverage if coverage > 0 else None
            for tier, total in tiers.items()
        }
        for field, tiers in fields.items()
    }
    auc = {
        tier: fields["dead_physical_bytes"][tier]
        for tier in TIERS
    }
    return {
        "window_seconds": end - start,
        "covered_seconds": coverage,
        "coverage_fraction": coverage / (end - start),
        "time_weighted_mean": averages,
        "peak": peaks,
        "dead_byte_seconds": auc,
        "lifecycle": {
            "time_weighted_mean": {
                field: total / coverage if coverage > 0 else None
                for field, total in lifecycle_totals.items()
            },
            "peak": lifecycle_peaks,
        },
    }


def end_metrics(
    rows: Sequence[Mapping[str, Any]], start: float, end: float
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if start <= float(row.get("started_elapsed_seconds") or -1) < end
    ]
    return {
        "n": len(selected),
        "n_ok": sum(bool(row.get("ok")) for row in selected),
        "n_error": sum(not bool(row.get("ok")) for row in selected),
        "latency_ms": summarize(
            [float(row["end_ms"]) for row in selected if row.get("end_ms") is not None]
        ),
        "queue_delay_ms": summarize(
            [
                float(row["queue_delay_ms"])
                for row in selected
                if row.get("queue_delay_ms") is not None
            ]
        ),
        "retry_attempts": sum(
            max(0, len(row.get("attempts") or []) - 1) for row in selected
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "ours"), required=True)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--agentreplay-root", type=pathlib.Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30001/generate")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--session-end-max-concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=7200.0)
    parser.add_argument("--session-end-timeout-s", type=float, default=120.0)
    parser.add_argument("--session-end-retries", type=int, default=2)
    parser.add_argument("--session-end-retry-delay-s", type=float, default=0.25)
    parser.add_argument("--telemetry-interval-s", type=float, default=1.0)
    parser.add_argument("--state-timeout-s", type=float, default=10.0)
    parser.add_argument("--flush-timeout-s", type=float, default=120.0)
    parser.add_argument("--confirm-dedicated-server", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "max_concurrency",
        "session_end_max_concurrency",
        "request_timeout_s",
        "session_end_timeout_s",
        "telemetry_interval_s",
        "state_timeout_s",
        "flush_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.session_end_retries < 0 or args.session_end_retry_delay_s < 0:
        parser.error("SESSION_END retry values cannot be negative")
    if not args.confirm_dedicated_server:
        parser.error(
            "--confirm-dedicated-server is required because this runner flushes cache"
        )
    return args


async def execute(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    driver,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    request_path = out_dir / "requests.jsonl"
    end_path = out_dir / "session_end.jsonl"
    telemetry_path = out_dir / "telemetry.jsonl"
    for path in (request_path, end_path, telemetry_path):
        path.touch(exist_ok=False)

    def append_jsonl(path: pathlib.Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    programs = source_programs(records)
    parents, blocking = blocking_children(programs)
    final_steps = {
        program_id: max(int(row["step"]) for row in rows)
        for program_id, rows in programs.items()
    }
    role_by_program = {
        program_id: str(rows[0]["steady_role"])
        for program_id, rows in programs.items()
    }
    runtime_ids = {
        driver._runtime_program_id(program_id, args.salt)
        for program_id in programs
    }
    logical_ended: set[str] = set()
    logical_ended_source: set[str] = set()
    end_acked: set[str] = set()
    request_rows: list[dict[str, Any]] = []
    end_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    final_request_completed: dict[str, float] = {}
    logical_end_elapsed: dict[str, float] = {}
    run_zero = time.perf_counter()
    base_url = args.url.rsplit("/", 1)[0]
    state_url = base_url + "/aginfer/state"
    end_url = base_url + "/aginfer/session_end"
    stop_sampler = asyncio.Event()

    def elapsed() -> float:
        return time.perf_counter() - run_zero

    def mark_logically_ended(program_id: str) -> None:
        if program_id in logical_ended_source:
            return
        if program_id not in final_request_completed:
            return
        if not blocking.get(program_id, set()).issubset(logical_ended_source):
            return
        logical_ended_source.add(program_id)
        logical_end_elapsed[program_id] = elapsed()
        logical_ended.add(driver._runtime_program_id(program_id, args.salt))
        parent = parents.get(program_id)
        if parent is not None:
            mark_logically_ended(parent)

    async def take_sample(event: str) -> None:
        ended_snapshot = set(logical_ended)
        acked_snapshot = set(end_acked)
        payload, error, status = await asyncio.to_thread(
            telemetry.fetch_json, state_url, args.state_timeout_s
        )
        state = None
        if error is None:
            try:
                state = telemetry.analyze_state(
                    payload, runtime_ids, ended_snapshot
                )
            except Exception as exc:  # telemetry must not stop inference
                error = f"{type(exc).__name__}: {exc}"[:300]
        sample = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "at": utc_now(),
            "elapsed_seconds": elapsed(),
            "state_http_status": status,
            "state_error": error,
            "logical_ended_programs": len(ended_snapshot),
            "session_end_acked_programs": len(acked_snapshot),
            "session_end_backlog": len(ended_snapshot - acked_snapshot),
            "state": state,
        }
        samples.append(sample)
        append_jsonl(telemetry_path, sample)

    async def sample_loop() -> None:
        while not stop_sampler.is_set():
            await take_sample("sample")
            try:
                await asyncio.wait_for(
                    stop_sampler.wait(), timeout=args.telemetry_interval_s
                )
            except TimeoutError:
                pass

    limits = driver.httpx.Limits(max_connections=args.max_concurrency + 8)
    end_limits = driver.httpx.Limits(
        max_connections=args.session_end_max_concurrency + 4
    )
    async with (
        driver.httpx.AsyncClient(
            timeout=driver.httpx.Timeout(args.request_timeout_s), limits=limits
        ) as client,
        driver.httpx.AsyncClient(
            timeout=driver.httpx.Timeout(args.session_end_timeout_s),
            limits=end_limits,
        ) as end_client,
    ):

        async def request_fn(record: Mapping[str, Any]) -> dict[str, Any]:
            started = elapsed()
            row = await driver._one_request(client, args.url, record, args.salt)
            completed = elapsed()
            row["started_elapsed_seconds"] = started
            row["completed_elapsed_seconds"] = completed
            safe = safe_request_row(record, row)
            request_rows.append(safe)
            append_jsonl(request_path, safe)
            program_id = str(record["program_id"])
            if int(record.get("step") or 0) == final_steps[program_id]:
                final_request_completed[program_id] = completed
                mark_logically_ended(program_id)
            return row

        async def end_fn(program: Mapping[str, Any]) -> dict[str, Any]:
            program_id = str(program["program_id"])
            mark_logically_ended(program_id)
            started = elapsed()
            attempts = []
            row: dict[str, Any] = {}
            for attempt in range(args.session_end_retries + 1):
                row = await driver._end_program_http(
                    end_client, end_url, program, args.salt
                )
                attempts.append(
                    {
                        "ok": bool(row.get("ok")),
                        "http_status": row.get("http_status"),
                        "error": row.get("error"),
                    }
                )
                if row.get("ok"):
                    break
                if attempt < args.session_end_retries:
                    await asyncio.sleep(args.session_end_retry_delay_s)
            completed = elapsed()
            row["attempts"] = attempts
            row["started_elapsed_seconds"] = started
            row["completed_elapsed_seconds"] = completed
            request_done = logical_end_elapsed.get(program_id)
            row["queue_delay_ms"] = (
                (started - request_done) * 1000.0
                if request_done is not None
                else None
            )
            row["role"] = role_by_program[program_id]
            row["program_id_sha256"] = hashlib.sha256(
                program_id.encode()
            ).hexdigest()
            end_rows.append(row)
            append_jsonl(
                end_path,
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"program_id", "runtime_program_id"}
                },
            )
            if row.get("ok"):
                end_acked.add(driver._runtime_program_id(program_id, args.salt))
            return row

        sampler_task = asyncio.create_task(sample_loop())
        timings: dict[str, Any] | None = None
        try:
            timings = await driver.replay(
                records,
                request_fn,
                args.max_concurrency,
                None,
                [],
                [],
                end_fn if args.mode == "ours" else None,
                [],
                gap_scale=1.0,
                max_gap_s=None,
                end_max_conc=args.session_end_max_concurrency,
            )
            await take_sample("post_replay")
        finally:
            stop_sampler.set()
            await sampler_task

    windows = manifest["windows"]
    measurement_start = float(windows["measurement_start_seconds"])
    measurement_end = float(windows["measurement_end_seconds"])
    requests_metrics = request_window_metrics(
        request_rows, measurement_start, measurement_end
    )
    state_metrics = state_window_metrics(
        samples, measurement_start, measurement_end
    )
    first_by_program: dict[str, Mapping[str, Any]] = {}
    for row in request_rows:
        digest = str(row["program_id_sha256"])
        if digest not in first_by_program or int(row["step"]) < int(
            first_by_program[digest]["step"]
        ):
            first_by_program[digest] = row
    arrival_delays = [
        max(
            0.0,
            float(row["started_elapsed_seconds"])
            - float(row["scheduled_session_arrival_s"]),
        )
        for row in first_by_program.values()
        if row.get("is_root")
        and measurement_start
        <= float(row["scheduled_session_arrival_s"])
        < measurement_end
    ]

    request_issues = []
    if len(request_rows) != len(records):
        request_issues.append(
            f"request rows={len(request_rows)}, expected={len(records)}"
        )
    for row in request_rows:
        if not row.get("ok"):
            request_issues.append("one or more inference requests failed")
            break
        if int(row.get("n_out") or -1) != int(row.get("want_out") or -2):
            request_issues.append("one or more output lengths differ")
            break
        if row.get("force_exact") is not True:
            request_issues.append("one or more forced outputs lack exact evidence")
            break
    end_issues = []
    if args.mode == "ours":
        if len(end_rows) != len(programs):
            end_issues.append(
                f"SESSION_END rows={len(end_rows)}, expected={len(programs)}"
            )
        if any(not row.get("ok") for row in end_rows):
            end_issues.append("one or more SESSION_END calls failed")
    if state_metrics["coverage_fraction"] < 0.95:
        request_issues.append(
            "state telemetry covered less than 95% of the measurement window"
        )

    return {
        "timings": timings,
        "request_rows": request_rows,
        "end_rows": end_rows,
        "samples": samples,
        "measurement": {
            "requests": requests_metrics,
            "state": state_metrics,
            "session_end": end_metrics(
                end_rows, measurement_start, measurement_end
            ),
            "session_arrival_admission_delay_seconds": summarize(arrival_delays),
        },
        "logical_ended_count": len(logical_ended),
        "end_acked_count": len(end_acked),
        "issues": [*request_issues, *end_issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.trace = args.trace.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.agentreplay_root = args.agentreplay_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    for path in (args.trace, args.manifest):
        if not path.is_file():
            raise SystemExit(f"missing input artifact: {path}")
    if not args.agentreplay_root.is_dir():
        raise SystemExit(
            f"AgentReplay root is not a directory: {args.agentreplay_root}"
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise SystemExit("unsupported steady manifest")
    trace_metadata = manifest.get("steady_trace")
    if not isinstance(trace_metadata, Mapping) or trace_metadata.get(
        "sha256"
    ) != telemetry.sha256_file(args.trace):
        raise SystemExit("steady trace hash does not match manifest")
    windows = manifest.get("windows")
    required_windows = (
        "warmup_seconds",
        "measurement_seconds",
        "cooldown_seconds",
        "measurement_start_seconds",
        "measurement_end_seconds",
    )
    if not isinstance(windows, Mapping) or any(
        name not in windows for name in required_windows
    ):
        raise SystemExit("steady manifest is missing measurement windows")
    records = load_trace(args.trace)
    driver = load_agentreplay(args.agentreplay_root)
    base_url = args.url.rsplit("/", 1)[0]
    state_url = base_url + "/aginfer/state"
    telemetry.validate_state_url(state_url, allow_remote=False)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "mode": args.mode,
        "started_at": utc_now(),
        "valid": False,
        "issues": [],
        "configuration": {
            "url": args.url,
            "max_concurrency": args.max_concurrency,
            "session_end_max_concurrency": args.session_end_max_concurrency,
            "telemetry_interval_s": args.telemetry_interval_s,
            "trace_sha256": telemetry.sha256_file(args.trace),
            "manifest_sha256": telemetry.sha256_file(args.manifest),
            "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        },
        "workload": {
            key: manifest.get(key)
            for key in (
                "arrival_rate_sessions_per_second",
                "arrival_interval_seconds",
                "arrival_duration_seconds",
                "live_fraction_requested",
                "live_revisit_seconds",
                "live_steps",
                "churn_gap_seconds",
                "session_count",
                "program_count",
                "request_count",
                "scheduled_request_rate_per_second",
                "scheduled_forced_output_tokens_per_second",
                "role_session_counts",
                "role_program_counts",
                "windows",
            )
        },
        "cleanup": None,
    }
    started = time.monotonic()
    execution: dict[str, Any] | None = None
    try:
        flush_cache(base_url, args.flush_timeout_s)
        initial = wait_empty(state_url, args.flush_timeout_s)
        summary["initial_state"] = initial
        execution = asyncio.run(execute(args, records, manifest, driver, out_dir))
        summary["timings"] = execution["timings"]
        summary["measurement"] = execution["measurement"]
        summary["logical_ended_count"] = execution["logical_ended_count"]
        summary["session_end_acked_count"] = execution["end_acked_count"]
        summary["issues"].extend(execution["issues"])

        summary["valid"] = not summary["issues"]
    except Exception as exc:
        summary["issues"].append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        cleanup_started = time.monotonic()
        cleanup_error = None
        try:
            flush_cache(base_url, args.flush_timeout_s)
            wait_empty(state_url, args.flush_timeout_s)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"[:500]
            summary["issues"].append("post-measurement cache cleanup failed")
            summary["valid"] = False
        summary["cleanup"] = {
            "cache_flush_attempted": True,
            "ok": cleanup_error is None,
            "error": cleanup_error,
            "elapsed_seconds": time.monotonic() - cleanup_started,
        }
        summary["duration_seconds"] = time.monotonic() - started
        summary["finished_at"] = utc_now()
        atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
