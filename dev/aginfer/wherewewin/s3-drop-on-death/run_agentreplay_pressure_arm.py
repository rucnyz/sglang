#!/usr/bin/env python3
"""Run one real-trace Dead-KV pressure arm on direct SGLang.

The phases are:

1. seed reusable live programs without SESSION_END;
2. churn complete terminal programs for one or more waves, giving every wave a
   distinct derived session namespace (SESSION_END only for ``--mode ours``),
   monitored by ``run_agentreplay_with_telemetry.py`` and followed by a fixed
   retention barrier (30 seconds by default);
3. probe the live programs' final turn (SESSION_END only for ``--mode ours``).

Baseline cleanup sends explicit SESSION_END for every workload program only
after all measured snapshots. Cleanup and cache flush are excluded from metrics.
Every phase must pass AgentReplay's length and exact-token checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_agentreplay_with_telemetry as telemetry  # noqa: E402
import split_agentreplay_pressure_trace as splitter  # noqa: E402

SCHEMA_VERSION = 2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def post_json(url: str, payload: Mapping[str, Any], timeout: float) -> tuple[Any, int]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with telemetry.no_proxy_opener().open(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}") from None
    if not 200 <= status < 300:
        raise RuntimeError(f"POST {url} returned HTTP {status}")
    if not body.strip():
        payload: Any = {}
    else:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"text": body.decode("utf-8", errors="replace")[:300]}
    return payload, status


def trace_programs(
    path: pathlib.Path, *, require_start_at_one: bool = True
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records = splitter.load_trace(path)
    return records, splitter.program_index(
        records, require_start_at_one=require_start_at_one
    )


def token_payload_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash token content while deliberately excluding program identifiers."""
    programs: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        programs.setdefault(str(record["program_id"]), []).append(record)
    payload = sorted(
        json.dumps(
            [
                {
                    "step": record["step"],
                    "input_ids": record["input_ids"],
                    "forced_output_ids": record["forced_output_ids"],
                    "context_reset": bool(record.get("context_reset")),
                }
                for record in sorted(rows, key=lambda row: int(row["step"]))
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        for rows in programs.values()
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_phase_traces(
    seed_path: pathlib.Path,
    terminal_paths: Sequence[pathlib.Path],
    probe_path: pathlib.Path,
) -> dict[str, Any]:
    seed_records, seed_programs = trace_programs(seed_path)
    probe_records, probe_programs = trace_programs(
        probe_path, require_start_at_one=False
    )
    live_ids = set(seed_programs)
    if live_ids != set(probe_programs):
        raise ValueError("live-seed and live-probe program sets differ")
    if not live_ids:
        raise ValueError("live-seed and live-probe traces have no programs")
    for program_id in sorted(live_ids):
        seed_rows = seed_programs[program_id]
        probe_rows = probe_programs[program_id]
        if [row["step"] for row in seed_rows] != [1, 2, 3]:
            raise ValueError(f"live seed {program_id!r} must contain steps 1-3")
        if [row["step"] for row in probe_rows] != [4]:
            raise ValueError(f"live probe {program_id!r} must contain only step 4")
        prefix = seed_rows[-1]["input_ids"] + seed_rows[-1]["forced_output_ids"]
        if (
            probe_rows[0].get("context_reset")
            or probe_rows[0]["input_ids"][: len(prefix)] != prefix
        ):
            raise ValueError(
                f"live probe {program_id!r} does not extend its seed prefix"
            )
    terminal_records_by_wave = []
    terminal_programs_by_wave = []
    terminal_wave_stats = []
    seen_terminal_ids: set[str] = set()
    seen_token_payloads: set[str] = set()
    for wave_number, terminal_path in enumerate(terminal_paths, 1):
        terminal_records, terminal_programs = trace_programs(terminal_path)
        terminal_parents, _ = splitter.graph(terminal_programs)
        terminal_roots = {
            program_id
            for program_id, parent in terminal_parents.items()
            if parent is None
        }
        if not terminal_roots:
            raise ValueError(f"terminal wave {wave_number} has no root programs")
        terminal_ids = set(terminal_programs)
        overlap = (live_ids | seen_terminal_ids) & terminal_ids
        if overlap:
            raise ValueError(
                f"terminal wave {wave_number} overlaps live or earlier program IDs"
            )
        payload_hash = token_payload_hash(terminal_records)
        if payload_hash in seen_token_payloads:
            raise ValueError(
                f"terminal wave {wave_number} duplicates an earlier token payload"
            )
        seen_token_payloads.add(payload_hash)
        seen_terminal_ids.update(terminal_ids)
        terminal_records_by_wave.append(terminal_records)
        terminal_programs_by_wave.append(terminal_programs)
        terminal_wave_stats.append(
            {
                "wave_number": wave_number,
                "basename": terminal_path.name,
                "sha256": telemetry.sha256_file(terminal_path),
                "token_payload_sha256": payload_hash,
                "requests": len(terminal_records),
                "programs": len(terminal_programs),
                "roots": len(terminal_roots),
            }
        )
    terminal_records = [
        record for wave_records in terminal_records_by_wave for record in wave_records
    ]
    terminal_programs = {
        program_id: rows
        for wave_programs in terminal_programs_by_wave
        for program_id, rows in wave_programs.items()
    }
    all_records = [*seed_records, *terminal_records, *probe_records]
    max_request = max(
        len(record["input_ids"]) + len(record["forced_output_ids"])
        for record in all_records
    )
    aggregate_terminal_hash = (
        terminal_wave_stats[0]["sha256"]
        if len(terminal_wave_stats) == 1
        else hashlib.sha256(
            json.dumps(
                [row["sha256"] for row in terminal_wave_stats], separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    terminal_trace = {
        "sha256": aggregate_terminal_hash,
        "wave_count": len(terminal_paths),
        "waves": terminal_wave_stats,
    }
    if len(terminal_paths) == 1:
        terminal_trace["basename"] = terminal_paths[0].name
    return {
        "_live_program_ids": sorted(live_ids),
        "_terminal_program_ids": sorted(terminal_programs),
        "_terminal_program_ids_by_wave": [
            sorted(programs) for programs in terminal_programs_by_wave
        ],
        "live_program_count": len(live_ids),
        "terminal_program_count": len(terminal_programs),
        "all_program_count": len(live_ids | set(terminal_programs)),
        "live_seed_requests": len(seed_records),
        "terminal_requests": len(terminal_records),
        "live_probe_requests": len(probe_records),
        "terminal_wave_count": len(terminal_paths),
        "terminal_roots": sum(row["roots"] for row in terminal_wave_stats),
        "max_request_tokens": max_request,
        "traces": {
            "live_seed": {
                "basename": seed_path.name,
                "sha256": telemetry.sha256_file(seed_path),
            },
            "terminal_churn": terminal_trace,
            "live_probe": {
                "basename": probe_path.name,
                "sha256": telemetry.sha256_file(probe_path),
            },
        },
    }


def replay_command(
    args: argparse.Namespace,
    trace: pathlib.Path,
    result_path: pathlib.Path,
    label: str,
    emit_session_end: bool,
    *,
    salt: str | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "agentreplay",
        "replay",
        "--trace",
        str(trace),
        "--url",
        args.url,
        "--max-concurrency",
        str(max_concurrency if max_concurrency is not None else args.max_concurrency),
        "--stagger",
        str(args.stagger),
        "--gap-scale",
        str(args.gap_scale),
        "--max-gap-s",
        str(args.max_gap_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--salt",
        salt if salt is not None else args.salt,
        "--label",
        label,
        "--out",
        str(result_path),
    ]
    if emit_session_end:
        command += [
            "--emit-session-end",
            "--session-end-max-concurrency",
            str(args.session_end_max_concurrency),
            "--session-end-retries",
            str(args.session_end_retries),
            "--session-end-retry-delay-s",
            str(args.session_end_retry_delay_s),
            "--session-end-timeout-s",
            str(args.session_end_timeout_s),
        ]
    return command


def child_environment(agentreplay_root: pathlib.Path) -> dict[str, str]:
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(agentreplay_root)
        if not old_pythonpath
        else str(agentreplay_root) + os.pathsep + old_pythonpath
    )
    telemetry.append_no_proxy(environment)
    return environment


def run_logged(
    command: Sequence[str],
    cwd: pathlib.Path,
    out_dir: pathlib.Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            check=False,
            shell=False,
        )
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": str(stdout_path.name),
        "stderr": str(stderr_path.name),
        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
    }


def load_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_replay_result(
    result: Any, *, expected_requests: int, expected_programs: int, expect_end: bool
) -> list[str]:
    issues = []
    if not isinstance(result, Mapping):
        return ["result is missing or not an object"]
    expected = {
        "n_requests": expected_requests,
        "n_ok": expected_requests,
        "n_error": 0,
        "len_match_rate": 1.0,
        "force_exact_rate": 1.0,
        "force_exact_failures": 0,
        "force_exact_missing": 0,
        "n_programs": expected_programs,
    }
    for field, wanted in expected.items():
        if result.get(field) != wanted:
            issues.append(f"{field}={result.get(field)!r}, expected {wanted!r}")
    session_end = result.get("session_end")
    if expect_end:
        if (
            not isinstance(session_end, Mapping)
            or session_end.get("enabled") is not True
        ):
            issues.append("SESSION_END was not enabled")
        elif (
            session_end.get("n") != expected_programs
            or session_end.get("n_ok") != expected_programs
            or session_end.get("n_error") != 0
            or session_end.get("remaining_nodes") != 0
        ):
            issues.append("SESSION_END did not complete for every program")
    elif isinstance(session_end, Mapping) and session_end.get("enabled") is not False:
        issues.append("SESSION_END unexpectedly enabled")
    return issues


def terminal_wave_salt(base_salt: str, wave_number: int) -> str:
    """Derive a stable, non-revealing namespace for one terminal wave."""
    digest = hashlib.sha256(
        f"{base_salt}\0terminal-wave\0{wave_number}".encode()
    ).hexdigest()
    return f"tw{wave_number:03d}-{digest[:24]}"


def weighted_mean(values: Sequence[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _value, weight in values if weight > 0)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values if weight > 0) / total_weight


def aggregate_terminal_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the aggregate-only compatibility result for sequential waves.

    Throughput is recomputed from summed output tokens and inference spans.
    END means are weighted by completed calls. Since per-call latency samples
    are intentionally not copied into aggregate artifacts, per-wave p50/p90/
    p99 values are kept separately and never presented as aggregate percentiles.
    """
    if not results:
        raise ValueError("cannot aggregate zero terminal-wave results")

    def total(field: str) -> int | float:
        return sum(
            value
            for result in results
            if isinstance((value := result.get(field)), (int, float))
            and not isinstance(value, bool)
        )

    request_count = int(total("n_requests"))
    prompt_tokens = int(total("total_prompt_tokens"))
    cached_tokens = int(total("total_cached_tokens"))
    output_tokens = int(total("total_out_tokens"))
    inference_span = float(total("inference_makespan_s"))
    pipeline_span = float(total("pipeline_makespan_s"))
    wall_span = float(total("wall_s"))

    aggregate: dict[str, Any] = {
        "label": "terminal-waves-aggregate",
        "n_requests": request_count,
        "n_ok": int(total("n_ok")),
        "n_error": int(total("n_error")),
        "n_programs": int(total("n_programs")),
        "total_out_tokens": output_tokens,
        "total_prompt_tokens": prompt_tokens,
        "total_cached_tokens": cached_tokens,
        "len_match_rate": weighted_mean(
            [
                (float(result["len_match_rate"]), int(result.get("n_requests") or 0))
                for result in results
                if isinstance(result.get("len_match_rate"), (int, float))
            ]
        ),
        "force_exact_rate": weighted_mean(
            [
                (
                    float(result["force_exact_rate"]),
                    int(result.get("n_requests") or 0),
                )
                for result in results
                if isinstance(result.get("force_exact_rate"), (int, float))
            ]
        ),
        "force_exact_failures": int(total("force_exact_failures")),
        "force_exact_missing": int(total("force_exact_missing")),
        "cache_hit": (
            cached_tokens / prompt_tokens
            if prompt_tokens > 0
            else weighted_mean(
                [
                    (float(result["cache_hit"]), int(result.get("n_requests") or 0))
                    for result in results
                    if isinstance(result.get("cache_hit"), (int, float))
                ]
            )
        ),
        "wall_s": wall_span,
        "inference_makespan_s": inference_span,
        "pipeline_makespan_s": pipeline_span,
        "inference_throughput_tok_s": (
            output_tokens / inference_span if inference_span > 0 else None
        ),
        "pipeline_throughput_tok_s": (
            output_tokens / pipeline_span if pipeline_span > 0 else None
        ),
        "wave_count": len(results),
    }
    cached_detail_rows = []
    for result in results:
        safe = telemetry.safe_result(result)
        row = safe.get("cached_tokens_details") if safe is not None else None
        if isinstance(row, Mapping):
            cached_detail_rows.append(row)
    if cached_detail_rows:
        aggregate["cached_tokens_details"] = {
            tier: sum(int(row.get(tier) or 0) for row in cached_detail_rows)
            for tier in ("device", "host", "storage")
        }

    ends = [result.get("session_end") for result in results]
    end_rows = [end for end in ends if isinstance(end, Mapping)]
    enabled = bool(end_rows) and all(end.get("enabled") is True for end in end_rows)
    end_count = sum(int(end.get("n") or 0) for end in end_rows)
    latency_rows = [(end.get("latency_ms"), int(end.get("n") or 0)) for end in end_rows]
    latency: dict[str, Any] = {}
    mean_values = [
        (float(stats["mean"]), weight)
        for stats, weight in latency_rows
        if isinstance(stats, Mapping) and isinstance(stats.get("mean"), (int, float))
    ]
    latency["mean"] = weighted_mean(mean_values)
    latency["aggregation"] = "weighted_by_completed_calls"
    worst_wave_latency = {}
    for percentile in ("p50", "p90", "p99"):
        values = [
            float(stats[percentile])
            for stats, _weight in latency_rows
            if isinstance(stats, Mapping)
            and isinstance(stats.get(percentile), (int, float))
        ]
        worst_wave_latency[percentile] = max(values) if values else None
        if len(results) == 1:
            latency[percentile] = worst_wave_latency[percentile]
    aggregate["session_end"] = {
        "enabled": enabled,
        "n": end_count,
        "n_ok": sum(int(end.get("n_ok") or 0) for end in end_rows),
        "n_error": sum(int(end.get("n_error") or 0) for end in end_rows),
        "remaining_nodes": sum(
            int(end.get("remaining_nodes") or 0) for end in end_rows
        ),
        "retry_attempts": sum(int(end.get("retry_attempts") or 0) for end in end_rows),
        "latency_ms": latency if enabled and end_count > 0 else {"n": 0},
        "worst_wave_latency_ms": (
            worst_wave_latency if enabled and end_count > 0 else {"n": 0}
        ),
    }
    for field in (
        "freed_units",
        "matched_nodes",
        "holders_removed",
        "released_nodes",
        "released_hbm_tokens",
        "released_dram_tokens",
    ):
        aggregate["session_end"][field] = sum(
            int(end.get(field) or 0) for end in end_rows
        )
    return aggregate


def aggregate_terminal_telemetry(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate safe per-wave telemetry without reading raw telemetry JSONL."""
    result: dict[str, Any] = {
        "child_returncode": max(
            (int(summary.get("child_returncode") or 0) for summary in summaries),
            default=0,
        ),
        "sample_count": sum(
            int(summary.get("sample_count") or 0) for summary in summaries
        ),
        "state_error_count": sum(
            int(summary.get("state_error_count") or 0) for summary in summaries
        ),
        "gpu_error_count": sum(
            int(summary.get("gpu_error_count") or 0) for summary in summaries
        ),
        "wave_count": len(summaries),
    }
    for field in (
        "peak_pool_used_bytes",
        "peak_radix_physical_bytes",
        "peak_residual_holder_bytes",
        "peak_dead_physical_bytes",
    ):
        result[field] = {
            tier: max(
                (
                    int(value)
                    for summary in summaries
                    if isinstance(summary.get(field), Mapping)
                    and isinstance((value := summary[field].get(tier)), (int, float))
                ),
                default=0,
            )
            for tier in telemetry.TIERS
        }
    result["peak_gpu_used_bytes_total"] = max(
        (int(summary.get("peak_gpu_used_bytes_total") or 0) for summary in summaries),
        default=0,
    )
    return result


def state_summary(
    state_url: str,
    timeout: float,
    tracked_programs: set[str],
    ended_programs: set[str] | None,
) -> dict[str, Any]:
    payload, error, status = telemetry.fetch_json(state_url, timeout)
    if error is not None:
        raise RuntimeError(f"GET state failed: status={status}, error={error}")
    return telemetry.analyze_state(payload, tracked_programs, ended_programs)


def wait_state(
    state_url: str,
    timeout: float,
    interval: float,
    tracked_programs: set[str],
    ended_programs: set[str] | None,
    predicate,
    description: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        state = state_summary(
            state_url, min(timeout, 10.0), tracked_programs, ended_programs
        )
        if predicate(state):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {description}; last state={state}"
            )
        time.sleep(interval)


def end_program(
    end_url: str, program_id: str, timeout: float, retries: int, retry_delay: float
) -> dict[str, Any]:
    attempts = []
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            payload, status = post_json(end_url, {"program_id": program_id}, timeout)
            results = payload.get("per_rank") if isinstance(payload, Mapping) else None
            if not isinstance(results, list):
                results = (
                    payload.get("results") if isinstance(payload, Mapping) else None
                )
            complete = (
                isinstance(payload, Mapping)
                and payload.get("ok") is True
                and isinstance(results, list)
                and bool(results)
                and all(
                    isinstance(item, Mapping)
                    and item.get("ok") is True
                    and not item.get("deferred")
                    and int(item.get("remaining_nodes") or 0) == 0
                    for item in results
                )
            )
            row = {
                "ok": complete,
                "http_status": status,
                "elapsed_seconds": time.monotonic() - started,
            }
            attempts.append(row)
            if complete:
                return {"ok": True, "attempts": attempts}
        except Exception as exc:
            attempts.append(
                {
                    "ok": False,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
        if attempt < retries:
            time.sleep(retry_delay)
    return {"ok": False, "attempts": attempts}


def flush_cache(base_url: str, timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + f"/flush_cache?timeout={timeout:g}"
    started = time.monotonic()
    payload, status = post_json(url, {}, timeout + 5)
    return {
        "ok": True,
        "http_status": status,
        "elapsed_seconds": time.monotonic() - started,
        "response_ok": payload.get("ok") if isinstance(payload, Mapping) else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "ours"), required=True)
    parser.add_argument("--live-seed", type=pathlib.Path, required=True)
    parser.add_argument(
        "--terminal-churn",
        type=pathlib.Path,
        action="append",
        required=True,
        help=(
            "one distinct terminal-wave trace; repeat this option for multiple "
            "waves (the runner rejects overlapping IDs and duplicate token payloads)"
        ),
    )
    parser.add_argument("--live-probe", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--python", type=pathlib.Path, default=pathlib.Path(sys.executable)
    )
    parser.add_argument("--agentreplay-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--telemetry-script",
        type=pathlib.Path,
        default=SCRIPT_DIR / "run_agentreplay_with_telemetry.py",
    )
    parser.add_argument("--url", default="http://127.0.0.1:30001/generate")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--seed-concurrency",
        type=int,
        default=None,
        help="seed-phase concurrency (defaults to --max-concurrency)",
    )
    parser.add_argument(
        "--terminal-concurrency",
        type=int,
        default=None,
        help="terminal-wave concurrency (defaults to --max-concurrency)",
    )
    parser.add_argument(
        "--probe-concurrency",
        type=int,
        default=None,
        help="live-probe concurrency (defaults to --max-concurrency)",
    )
    parser.add_argument("--stagger", type=float, default=0.5)
    parser.add_argument("--gap-scale", type=float, default=0.01)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--barrier-seconds", type=float, default=30.0)
    parser.add_argument("--request-timeout-s", type=float, default=7200.0)
    parser.add_argument("--session-end-timeout-s", type=float, default=60.0)
    parser.add_argument("--session-end-retries", type=int, default=2)
    parser.add_argument("--session-end-retry-delay-s", type=float, default=0.25)
    parser.add_argument("--session-end-max-concurrency", type=int, default=1)
    parser.add_argument("--state-wait-timeout-s", type=float, default=120.0)
    parser.add_argument("--state-poll-interval-s", type=float, default=0.2)
    parser.add_argument("--allow-nonempty-start", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        "max_concurrency",
        "request_timeout_s",
        "session_end_timeout_s",
        "session_end_max_concurrency",
        "state_wait_timeout_s",
        "state_poll_interval_s",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    for name in ("seed_concurrency", "terminal_concurrency", "probe_concurrency"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if (
        args.stagger < 0
        or args.gap_scale < 0
        or args.max_gap_s < 0
        or args.barrier_seconds < 0
    ):
        parser.error("stagger/gap/barrier values cannot be negative")
    if args.session_end_retries < 0 or args.session_end_retry_delay_s < 0:
        parser.error("SESSION_END retry values cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in ("live_seed", "live_probe", "python", "telemetry_script"):
        path = getattr(args, field).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"missing file for --{field.replace('_', '-')}: {path}")
        setattr(args, field, path)
    terminal_paths = [path.expanduser().resolve() for path in args.terminal_churn]
    for path in terminal_paths:
        if not path.is_file():
            raise SystemExit(f"missing file for --terminal-churn: {path}")
    if len(set(terminal_paths)) != len(terminal_paths):
        raise SystemExit("duplicate --terminal-churn path")
    args.terminal_churn = terminal_paths
    args.seed_concurrency = args.seed_concurrency or args.max_concurrency
    args.terminal_concurrency = args.terminal_concurrency or args.max_concurrency
    args.probe_concurrency = args.probe_concurrency or args.max_concurrency
    args.agentreplay_root = args.agentreplay_root.expanduser().resolve()
    if not args.agentreplay_root.is_dir():
        raise SystemExit(
            f"AgentReplay root is not a directory: {args.agentreplay_root}"
        )
    telemetry.validate_state_url(
        args.url.rsplit("/", 1)[0] + "/aginfer/state", allow_remote=False
    )
    phase_validation = validate_phase_traces(
        args.live_seed, args.terminal_churn, args.live_probe
    )
    source_live = set(phase_validation.pop("_live_program_ids"))
    source_terminal = set(phase_validation.pop("_terminal_program_ids"))
    source_terminal_by_wave = [
        set(programs)
        for programs in phase_validation.pop("_terminal_program_ids_by_wave")
    ]
    phase_manifest = phase_validation
    if phase_manifest["max_request_tokens"] > 131072:
        raise SystemExit(
            f"trace request length {phase_manifest['max_request_tokens']} exceeds 131072"
        )

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to use non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    environment = child_environment(args.agentreplay_root)
    base_url = args.url.rsplit("/", 1)[0]
    state_url = base_url + "/aginfer/state"
    end_url = base_url + "/aginfer/session_end"
    live_runtime = {telemetry.runtime_program_id(pid, args.salt) for pid in source_live}
    terminal_wave_salts = (
        [args.salt]
        if len(args.terminal_churn) == 1
        else [
            terminal_wave_salt(args.salt, wave_number)
            for wave_number in range(1, len(args.terminal_churn) + 1)
        ]
    )
    terminal_runtime_by_wave = [
        {
            telemetry.runtime_program_id(program_id, wave_salt)
            for program_id in program_ids
        }
        for program_ids, wave_salt in zip(source_terminal_by_wave, terminal_wave_salts)
    ]
    terminal_runtime = set().union(*terminal_runtime_by_wave)
    all_runtime = live_runtime | terminal_runtime
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "mode": args.mode,
        "started_at": utc_now(),
        "run_salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        "trace": phase_manifest["traces"]["terminal_churn"],
        "phase_manifest": phase_manifest,
        "configuration": {
            "url": args.url,
            "max_concurrency": args.max_concurrency,
            "seed_concurrency": args.seed_concurrency,
            "terminal_concurrency": args.terminal_concurrency,
            "probe_concurrency": args.probe_concurrency,
            "stagger": args.stagger,
            "gap_scale": args.gap_scale,
            "max_gap_s": args.max_gap_s,
            "barrier_seconds": args.barrier_seconds,
            "session_end_max_concurrency": args.session_end_max_concurrency,
            "terminal_wave_count": len(args.terminal_churn),
        },
        "phases": {},
        "states": {},
        "cleanup": None,
        "issues": [],
        "valid": False,
    }
    started = time.monotonic()

    def save() -> None:
        summary["duration_seconds"] = time.monotonic() - started
        atomic_write_json(summary_path, summary)

    save()
    try:
        initial = state_summary(state_url, 10.0, all_runtime, None)
        summary["states"]["initial"] = initial
        if not args.allow_nonempty_start and (
            initial["unit_count"] != 0 or initial["program_usage_count"] != 0
        ):
            raise RuntimeError("server state is not empty at arm start")

        seed_dir = out_dir / "seed"
        seed_result_path = seed_dir / "result.json"
        seed_command = replay_command(
            args,
            args.live_seed,
            seed_result_path,
            args.label + "-seed",
            False,
            max_concurrency=args.seed_concurrency,
        )
        seed_run = run_logged(
            seed_command, args.agentreplay_root, seed_dir, environment
        )
        seed_result = (
            load_json(seed_result_path) if seed_result_path.is_file() else None
        )
        seed_issues = verify_replay_result(
            seed_result,
            expected_requests=phase_manifest["live_seed_requests"],
            expected_programs=len(source_live),
            expect_end=False,
        )
        if seed_run["returncode"] != 0:
            seed_issues.append(f"seed process returncode={seed_run['returncode']}")
        summary["phases"]["seed"] = {
            **seed_run,
            "result": telemetry.safe_result(seed_result),
            "issues": seed_issues,
        }
        if seed_issues:
            raise RuntimeError("seed phase failed exactness/completion checks")
        after_seed = wait_state(
            state_url,
            args.state_wait_timeout_s,
            args.state_poll_interval_s,
            live_runtime,
            None,
            lambda state: state["tracked_programs_present_count"] == len(live_runtime),
            "all live programs to appear after seed",
        )
        summary["states"]["after_seed"] = after_seed
        save()

        terminal_results = []
        terminal_summaries = []
        terminal_wave_runs = []
        cumulative_ended: set[str] = set()
        terminal_wave_stats = phase_manifest["traces"]["terminal_churn"]["waves"]
        final_control_span = 0.0
        for wave_number, (
            terminal_path,
            wave_salt,
            wave_runtime,
            wave_stats,
        ) in enumerate(
            zip(
                args.terminal_churn,
                terminal_wave_salts,
                terminal_runtime_by_wave,
                terminal_wave_stats,
            ),
            1,
        ):
            wave_name = (
                "terminal"
                if len(args.terminal_churn) == 1
                else f"terminal-wave-{wave_number:03d}"
            )
            terminal_dir = out_dir / wave_name
            terminal_result_path = terminal_dir / "result.json"
            terminal_replay = replay_command(
                args,
                terminal_path,
                terminal_result_path,
                args.label + f"-{wave_name}",
                args.mode == "ours",
                salt=wave_salt,
                max_concurrency=args.terminal_concurrency,
            )
            terminal_wrapper = [
                str(args.python),
                str(args.telemetry_script),
                "--out-dir",
                str(terminal_dir),
                "--state-url",
                state_url,
                "--poll-interval",
                "1",
                "--post-seconds",
                "0",
                "--state-timeout",
                "10",
                "--gpu-timeout",
                "10",
                "--trace",
                str(terminal_path),
                "--run-salt",
                wave_salt,
                "--result-json",
                str(terminal_result_path),
                "--child-cwd",
                str(args.agentreplay_root),
                "--",
                *terminal_replay,
            ]
            terminal_run = run_logged(
                terminal_wrapper,
                args.agentreplay_root,
                out_dir / f"{wave_name}-wrapper",
                environment,
            )
            terminal_result = (
                load_json(terminal_result_path)
                if terminal_result_path.is_file()
                else None
            )
            terminal_summary_path = terminal_dir / "summary.json"
            terminal_summary = (
                load_json(terminal_summary_path)
                if terminal_summary_path.is_file()
                else None
            )
            terminal_issues = verify_replay_result(
                terminal_result,
                expected_requests=int(wave_stats["requests"]),
                expected_programs=len(source_terminal_by_wave[wave_number - 1]),
                expect_end=args.mode == "ours",
            )
            if terminal_run["returncode"] != 0:
                terminal_issues.append(
                    f"terminal wave {wave_number} telemetry/replay "
                    f"returncode={terminal_run['returncode']}"
                )
            wave_row = {
                "wave_number": wave_number,
                **terminal_run,
                "result": telemetry.safe_result(terminal_result),
                "telemetry_summary": terminal_summary,
                "issues": terminal_issues,
            }
            terminal_wave_runs.append(wave_row)
            summary["phases"]["terminal_waves"] = terminal_wave_runs
            if terminal_issues or not isinstance(terminal_summary, Mapping):
                raise RuntimeError(
                    f"terminal wave {wave_number} failed "
                    "telemetry/exactness/completion checks"
                )
            assert isinstance(terminal_result, Mapping)
            terminal_results.append(terminal_result)
            terminal_summaries.append(terminal_summary)
            cumulative_ended.update(wave_runtime)
            inference_span = float(terminal_result.get("inference_makespan_s") or 0.0)
            pipeline_span = float(terminal_result.get("pipeline_makespan_s") or 0.0)
            final_control_span = max(0.0, pipeline_span - inference_span)
            summary["states"][f"after_terminal_wave_{wave_number:03d}"] = {
                "workload": state_summary(
                    state_url, 10.0, all_runtime, cumulative_ended
                ),
                "live": state_summary(state_url, 10.0, live_runtime, None),
            }
            save()

        if len(terminal_results) == 1:
            terminal_result = terminal_results[0]
            terminal_summary = terminal_summaries[0]
        else:
            terminal_result = aggregate_terminal_results(terminal_results)
            terminal_summary = aggregate_terminal_telemetry(terminal_summaries)
        terminal_issues = verify_replay_result(
            terminal_result,
            expected_requests=phase_manifest["terminal_requests"],
            expected_programs=len(source_terminal),
            expect_end=args.mode == "ours",
        )
        if len(terminal_wave_runs) == 1:
            aggregate_run = {
                key: terminal_wave_runs[0][key]
                for key in (
                    "returncode",
                    "elapsed_seconds",
                    "stdout",
                    "stderr",
                    "command_sha256",
                )
            }
        else:
            aggregate_run = {
                "returncode": max(
                    int(run.get("returncode") or 0) for run in terminal_wave_runs
                ),
                "elapsed_seconds": sum(
                    float(run.get("elapsed_seconds") or 0.0)
                    for run in terminal_wave_runs
                ),
                "command_sha256": hashlib.sha256(
                    json.dumps(
                        [run.get("command_sha256") for run in terminal_wave_runs],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        summary["phases"]["terminal"] = {
            **aggregate_run,
            "result": telemetry.safe_result(terminal_result),
            "telemetry_summary": terminal_summary,
            "issues": terminal_issues,
        }
        if terminal_issues:
            raise RuntimeError("terminal-wave aggregation failed completion checks")
        control_span = final_control_span
        if control_span > args.barrier_seconds + 0.5:
            raise RuntimeError(
                "final SESSION_END control span exceeded the fixed post-inference "
                "barrier: "
                f"{control_span:.3f}s > {args.barrier_seconds:.3f}s"
            )
        barrier_sleep = max(0.0, args.barrier_seconds - control_span)
        time.sleep(barrier_sleep)
        after_terminal = state_summary(
            state_url,
            10.0,
            terminal_runtime,
            terminal_runtime,
        )
        summary["barrier"] = {
            "target_seconds_after_inference": args.barrier_seconds,
            "control_span_seconds": control_span,
            "additional_sleep_seconds": barrier_sleep,
        }
        summary["states"]["after_terminal_barrier"] = after_terminal
        save()

        # At this point every terminal program has logically completed while the
        # live programs must remain reusable. Classify terminal-only holders as
        # dead without misclassifying units shared with a live program.
        pre_probe = state_summary(state_url, 10.0, all_runtime, terminal_runtime)
        summary["states"]["before_live_probe"] = pre_probe
        summary["states"]["live_before_live_probe"] = state_summary(
            state_url, 10.0, live_runtime, None
        )
        probe_dir = out_dir / "probe"
        probe_result_path = probe_dir / "result.json"
        probe_command = replay_command(
            args,
            args.live_probe,
            probe_result_path,
            args.label + "-probe",
            args.mode == "ours",
            max_concurrency=args.probe_concurrency,
        )
        probe_run = run_logged(
            probe_command, args.agentreplay_root, probe_dir, environment
        )
        probe_result = (
            load_json(probe_result_path) if probe_result_path.is_file() else None
        )
        probe_issues = verify_replay_result(
            probe_result,
            expected_requests=phase_manifest["live_probe_requests"],
            expected_programs=len(source_live),
            expect_end=args.mode == "ours",
        )
        if probe_run["returncode"] != 0:
            probe_issues.append(f"probe process returncode={probe_run['returncode']}")
        summary["phases"]["probe"] = {
            **probe_run,
            "result": telemetry.safe_result(probe_result),
            "issues": probe_issues,
        }
        if probe_issues:
            raise RuntimeError("probe phase failed exactness/completion checks")
        post_probe = state_summary(state_url, 10.0, all_runtime, all_runtime)
        summary["states"]["after_live_probe"] = post_probe
        if args.mode == "ours" and post_probe["tracked_programs_present_count"] != 0:
            raise RuntimeError("Ours retained workload holders after probe SESSION_END")

        # A single wave preserves the legacy top-level compatibility view used
        # by analyze_agentreplay_realtrace.py. Multi-wave summaries use the same
        # aggregate keys, but percentile-only latency fields are intentionally
        # omitted when they cannot be reconstructed exactly.
        summary["agentreplay_result"] = telemetry.safe_result(terminal_result)
        summary["end_state"] = after_terminal
        for field in (
            "peak_pool_used_bytes",
            "peak_radix_physical_bytes",
            "peak_residual_holder_bytes",
            "peak_dead_physical_bytes",
            "peak_gpu_used_bytes_total",
            "sample_count",
            "state_error_count",
            "gpu_error_count",
        ):
            summary[field] = terminal_summary.get(field)
        summary["live_probe_result"] = telemetry.safe_result(probe_result)
        summary["valid"] = True
    except Exception as exc:
        summary["issues"].append(f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_rows = []
        should_explicitly_clean = args.mode == "baseline" or not summary["valid"]
        if should_explicitly_clean:
            for program_id in sorted(all_runtime):
                cleanup_rows.append(
                    {
                        "program_id_sha256": hashlib.sha256(
                            program_id.encode()
                        ).hexdigest(),
                        **end_program(
                            end_url,
                            program_id,
                            args.session_end_timeout_s,
                            args.session_end_retries,
                            args.session_end_retry_delay_s,
                        ),
                    }
                )
        cleanup: dict[str, Any] = {
            "explicit_end_attempted": should_explicitly_clean,
            "programs_targeted": len(all_runtime) if should_explicitly_clean else 0,
            "programs_attempted": len(cleanup_rows),
            "programs_ok": sum(bool(row["ok"]) for row in cleanup_rows),
            "programs_error": sum(not bool(row["ok"]) for row in cleanup_rows),
            "retry_attempts": sum(
                max(0, len(row.get("attempts", [])) - 1) for row in cleanup_rows
            ),
            "all_end_calls_ok": all(row["ok"] for row in cleanup_rows),
        }
        if cleanup["programs_error"]:
            summary["issues"].append(
                "explicit post-measurement SESSION_END cleanup failed"
            )
        try:
            cleanup["flush"] = flush_cache(base_url, args.state_wait_timeout_s)
            cleanup["final_state"] = wait_state(
                state_url,
                args.state_wait_timeout_s,
                args.state_poll_interval_s,
                all_runtime,
                all_runtime,
                lambda state: state["unit_count"] == 0
                and state["program_usage_count"] == 0,
                "post-arm cleanup to empty all cache/program state",
            )
        except Exception as exc:
            cleanup["error"] = f"{type(exc).__name__}: {exc}"[:500]
            summary["issues"].append("post-arm cleanup failed")
        summary["cleanup"] = cleanup
        summary["finished_at"] = utc_now()
        summary["valid"] = bool(summary["valid"] and not summary["issues"])
        summary["child_returncode"] = 0 if summary["valid"] else 1
        save()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(summary["child_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
