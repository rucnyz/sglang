#!/usr/bin/env python3
"""Run a fixed-playlist, closed-loop saturated AgentReplay Dead-KV arm.

Unlike the open-loop steady runner, this scheduler ignores wall-clock arrival
times after using them to define a deterministic playlist order.  It keeps up to
``--max-concurrency`` inference requests in flight: every completion immediately
admits the next ready request.  Churn programs advance turn by turn and end when
their fixed trace is exhausted.  Live programs seed their first turn early, then
revisit on deterministic churn-program completion thresholds.

Baseline and Ours execute the same request scheduler.  The only treatment is
that Ours asynchronously sends SESSION_END when a program finishes.  Parent and
child programs in a replicated root bundle are independent playlist programs in
this capacity benchmark; each program's own turn order and token-exact prefixes
remain intact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import statistics
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_agentreplay_steady_arm as steady  # noqa: E402


SCHEMA_VERSION = 1


def playlist_order(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    return sorted(
        programs,
        key=lambda program_id: (
            float(
                programs[program_id][0].get("scheduled_session_arrival_s") or 0.0
            ),
            str(programs[program_id][0].get("steady_root_id") or ""),
            program_id,
        ),
    )


def live_thresholds(
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
    churn_program_count: int,
    revisit_every_churn: int | None,
) -> tuple[dict[tuple[str, int], int], int]:
    live_ids = [
        program_id
        for program_id in playlist_order(programs)
        if programs[program_id][0].get("steady_role") == "live"
    ]
    if not live_ids:
        raise ValueError("saturated playlist has no live programs")
    maximum_steps = max(len(programs[program_id]) for program_id in live_ids)
    if maximum_steps < 2:
        raise ValueError("live programs need at least two turns")
    interval = revisit_every_churn or max(1, churn_program_count // maximum_steps)
    if interval <= 0:
        raise ValueError("live revisit churn count must be positive")
    if maximum_steps * interval > churn_program_count:
        raise ValueError(
            "live revisit interval is too large for the fixed churn playlist: "
            f"steps={maximum_steps}, interval={interval}, churn={churn_program_count}"
        )

    thresholds: dict[tuple[str, int], int] = {}
    for rank, program_id in enumerate(live_ids):
        phase = int(rank * interval / len(live_ids))
        for step_index in range(1, len(programs[program_id])):
            thresholds[(program_id, step_index)] = min(
                churn_program_count,
                step_index * interval + phase,
            )
    return thresholds, interval


def concurrency_metrics(
    events: Sequence[tuple[float, int]], makespan: float, target: int
) -> dict[str, Any]:
    if makespan <= 0:
        return {
            "target": target,
            "time_weighted_mean": 0.0,
            "peak": 0,
            "full_concurrency_fraction": 0.0,
            "underfilled_seconds": 0.0,
        }
    ordered = [
        event
        for _index, event in sorted(
            enumerate(events), key=lambda item: (item[1][0], item[0])
        )
    ]
    active = 0
    previous = 0.0
    active_seconds = 0.0
    full_seconds = 0.0
    underfilled_seconds = 0.0
    peak = 0
    for at, value in ordered:
        right = min(max(at, previous), makespan)
        seconds = right - previous
        active_seconds += active * seconds
        if active >= target:
            full_seconds += seconds
        else:
            underfilled_seconds += seconds
        active = value
        peak = max(peak, active)
        previous = right
        if previous >= makespan:
            break
    if previous < makespan:
        seconds = makespan - previous
        active_seconds += active * seconds
        if active >= target:
            full_seconds += seconds
        else:
            underfilled_seconds += seconds
    return {
        "target": target,
        "time_weighted_mean": active_seconds / makespan,
        "peak": peak,
        "full_concurrency_fraction": full_seconds / makespan,
        "underfilled_seconds": underfilled_seconds,
    }


def request_metrics(
    rows: Sequence[Mapping[str, Any]], makespan: float
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("ok")]
    total_output = sum(int(row.get("n_out") or 0) for row in ok)
    live_revisit = [
        row
        for row in ok
        if row.get("traffic_class") == "live_revisit"
    ]
    prompt = sum(int(row.get("prompt") or 0) for row in live_revisit)
    cached = sum(int(row.get("cached") or 0) for row in live_revisit)
    return {
        "n_requests": len(rows),
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "total_output_tokens": total_output,
        "inference_makespan_s": makespan,
        "inference_throughput_tok_s": (
            total_output / makespan if makespan > 0 else 0.0
        ),
        "live_revisit": {
            "n": len(live_revisit),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cache_hit": cached / prompt if prompt else 0.0,
            "ttft_ms": steady.summarize(
                [
                    float(row["ttft_ms"])
                    for row in live_revisit
                    if row.get("ttft_ms") is not None
                ]
            ),
            "e2e_ms": steady.summarize(
                [
                    float(row["e2e_ms"])
                    for row in live_revisit
                    if row.get("e2e_ms") is not None
                ]
            ),
        },
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
    parser.add_argument("--live-revisit-every-churn", type=int, default=None)
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
    if args.live_revisit_every_churn is not None and args.live_revisit_every_churn <= 0:
        parser.error("--live-revisit-every-churn must be positive")
    if args.session_end_retries < 0 or args.session_end_retry_delay_s < 0:
        parser.error("SESSION_END retry values cannot be negative")
    if not args.confirm_dedicated_server:
        parser.error("--confirm-dedicated-server is required")
    return args


async def execute(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    driver,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    programs = steady.source_programs(records)
    order = playlist_order(programs)
    churn_ids = [
        program_id
        for program_id in order
        if programs[program_id][0].get("steady_role") == "churn"
    ]
    churn_id_set = set(churn_ids)
    live_ids = [program_id for program_id in order if program_id not in churn_id_set]
    if len(churn_ids) < args.max_concurrency:
        raise ValueError("fixed playlist has fewer churn programs than concurrency")
    thresholds, revisit_interval = live_thresholds(
        programs, len(churn_ids), args.live_revisit_every_churn
    )
    playlist_descriptor = [
        {
            "program_id": program_id,
            "role": programs[program_id][0]["steady_role"],
            "steps": len(programs[program_id]),
            "thresholds": [
                thresholds.get((program_id, step_index))
                for step_index in range(1, len(programs[program_id]))
            ],
        }
        for program_id in order
    ]
    playlist_sha256 = hashlib.sha256(
        json.dumps(
            playlist_descriptor, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    request_path = out_dir / "requests.jsonl"
    end_path = out_dir / "session_end.jsonl"
    telemetry_path = out_dir / "telemetry.jsonl"
    for path in (request_path, end_path, telemetry_path):
        path.touch(exist_ok=False)

    def append_jsonl(path: pathlib.Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    base_url = args.url.rsplit("/", 1)[0]
    state_url = base_url + "/aginfer/state"
    end_url = base_url + "/aginfer/session_end"
    runtime_ids = {
        driver._runtime_program_id(program_id, args.salt) for program_id in programs
    }
    request_rows: list[dict[str, Any]] = []
    end_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    logical_ended: set[str] = set()
    end_acked: set[str] = set()
    active_events: list[tuple[float, int]] = [(0.0, 0)]
    active = 0
    completed_requests = 0
    churn_completed = 0
    waiting_live: dict[str, int] = {}
    queued: set[tuple[str, int]] = set()
    ready: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
    pending_churn = deque(churn_ids)
    control_tasks: set[asyncio.Task[Any]] = set()
    stop_sampler = asyncio.Event()
    run_zero = time.perf_counter()
    inference_done_s = 0.0
    ready_queue_peak = 0
    waiting_live_peak = 0

    def elapsed() -> float:
        return time.perf_counter() - run_zero

    def enqueue(program_id: str, step_index: int) -> None:
        nonlocal ready_queue_peak
        key = (program_id, step_index)
        if key in queued:
            return
        queued.add(key)
        ready.put_nowait(key)
        ready_queue_peak = max(ready_queue_peak, ready.qsize())

    # Seed every long-lived program near the beginning. Churn programs then
    # enter in fixed playlist order only as slots become available; their next
    # turn has priority over admitting another program, so terminal completions
    # and SESSION_END events occur continuously instead of only at the tail.
    for program_id in live_ids:
        enqueue(program_id, 0)

    def fill_ready_to_target() -> None:
        while active + ready.qsize() < args.max_concurrency and pending_churn:
            enqueue(pending_churn.popleft(), 0)

    fill_ready_to_target()

    async def take_sample(event: str) -> None:
        ended_snapshot = set(logical_ended)
        acked_snapshot = set(end_acked)
        payload, error, status = await asyncio.to_thread(
            steady.telemetry.fetch_json, state_url, args.state_timeout_s
        )
        state = None
        if error is None:
            try:
                state = steady.telemetry.analyze_state(
                    payload, runtime_ids, ended_snapshot
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
        sample = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "at": steady.utc_now(),
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

    end_semaphore = asyncio.Semaphore(args.session_end_max_concurrency)
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

        async def end_program(program_id: str) -> None:
            async with end_semaphore:
                started = elapsed()
                attempts = []
                row: dict[str, Any] = {}
                for attempt in range(args.session_end_retries + 1):
                    row = await driver._end_program_http(
                        end_client,
                        end_url,
                        {"program_id": program_id},
                        args.salt,
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
                row["attempts"] = attempts
                row["started_elapsed_seconds"] = started
                row["completed_elapsed_seconds"] = elapsed()
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
                    end_acked.add(
                        driver._runtime_program_id(program_id, args.salt)
                    )

        def release_waiting_live() -> None:
            for program_id, step_index in list(waiting_live.items()):
                threshold = thresholds[(program_id, step_index)]
                if churn_completed >= threshold:
                    del waiting_live[program_id]
                    enqueue(program_id, step_index)

        async def worker() -> None:
            nonlocal active, churn_completed, completed_requests, inference_done_s
            nonlocal waiting_live_peak
            while True:
                item = await ready.get()
                if item is None:
                    ready.task_done()
                    return
                program_id, step_index = item
                record = programs[program_id][step_index]
                active += 1
                active_events.append((elapsed(), active))
                started = elapsed()
                row = await driver._one_request(client, args.url, record, args.salt)
                completed = elapsed()
                active -= 1
                active_events.append((completed, active))
                row["started_elapsed_seconds"] = started
                row["completed_elapsed_seconds"] = completed
                safe = steady.safe_request_row(record, row)
                request_rows.append(safe)
                append_jsonl(request_path, safe)
                completed_requests += 1

                next_index = step_index + 1
                role = str(record["steady_role"])
                if next_index < len(programs[program_id]):
                    if role == "live":
                        threshold = thresholds[(program_id, next_index)]
                        if churn_completed >= threshold:
                            enqueue(program_id, next_index)
                        else:
                            waiting_live[program_id] = next_index
                            waiting_live_peak = max(
                                waiting_live_peak, len(waiting_live)
                            )
                    else:
                        enqueue(program_id, next_index)
                else:
                    runtime_id = driver._runtime_program_id(program_id, args.salt)
                    logical_ended.add(runtime_id)
                    if role == "churn":
                        churn_completed += 1
                        release_waiting_live()
                    if args.mode == "ours":
                        task = asyncio.create_task(end_program(program_id))
                        control_tasks.add(task)
                        task.add_done_callback(control_tasks.discard)

                fill_ready_to_target()

                if completed_requests == len(records):
                    inference_done_s = completed
                    for _ in range(args.max_concurrency):
                        ready.put_nowait(None)
                ready.task_done()

        sampler_task = asyncio.create_task(sample_loop())
        workers = [asyncio.create_task(worker()) for _ in range(args.max_concurrency)]
        try:
            await asyncio.gather(*workers)
            if control_tasks:
                await asyncio.gather(*list(control_tasks))
            pipeline_done_s = elapsed()
            await take_sample("post_pipeline")
        finally:
            stop_sampler.set()
            await sampler_task

    inference_metrics = request_metrics(request_rows, inference_done_s)
    inference_state = steady.state_window_metrics(samples, 0.0, inference_done_s)
    pipeline_state = steady.state_window_metrics(samples, 0.0, pipeline_done_s)
    control_metrics = steady.end_metrics(end_rows, 0.0, pipeline_done_s)
    total_output = int(inference_metrics["total_output_tokens"])
    issues = []
    if len(request_rows) != len(records):
        issues.append(f"requests={len(request_rows)}, expected={len(records)}")
    if any(not row.get("ok") for row in request_rows):
        issues.append("one or more inference requests failed")
    if any(
        int(row.get("n_out") if row.get("n_out") is not None else -1)
        != int(row.get("want_out") if row.get("want_out") is not None else -2)
        or row.get("force_exact") is not True
        for row in request_rows
    ):
        issues.append("one or more inference requests failed exactness")
    if args.mode == "ours" and (
        len(end_rows) != len(programs) or any(not row.get("ok") for row in end_rows)
    ):
        issues.append("SESSION_END did not complete for every program")
    if inference_state["coverage_fraction"] < 0.95:
        issues.append("state telemetry covered less than 95% of inference makespan")
    concurrency = concurrency_metrics(
        active_events, inference_done_s, args.max_concurrency
    )
    concurrency.update(
        {
            "ready_queue_peak": ready_queue_peak,
            "waiting_live_programs_peak": waiting_live_peak,
        }
    )
    if (
        len(records) >= args.max_concurrency
        and concurrency["peak"] != args.max_concurrency
    ):
        issues.append("inference never reached target concurrency")

    return {
        "playlist_sha256": playlist_sha256,
        "live_revisit_every_churn": revisit_interval,
        "program_count": len(programs),
        "churn_program_count": len(churn_ids),
        "live_program_count": len(live_ids),
        "inference": inference_metrics,
        "pipeline": {
            "makespan_s": pipeline_done_s,
            "throughput_tok_s": (
                total_output / pipeline_done_s if pipeline_done_s > 0 else 0.0
            ),
        },
        "session_end": control_metrics,
        "concurrency": concurrency,
        "inference_state": inference_state,
        "pipeline_state": pipeline_state,
        "logical_ended_count": len(logical_ended),
        "session_end_acked_count": len(end_acked),
        "issues": issues,
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

    manifest = steady.load_json(args.manifest)
    trace_metadata = (
        manifest.get("steady_trace") if isinstance(manifest, Mapping) else None
    )
    if not isinstance(trace_metadata, Mapping) or trace_metadata.get(
        "sha256"
    ) != steady.telemetry.sha256_file(args.trace):
        raise SystemExit("steady trace hash does not match manifest")
    records = steady.load_trace(args.trace)
    driver = steady.load_agentreplay(args.agentreplay_root)
    base_url = args.url.rsplit("/", 1)[0]
    state_url = base_url + "/aginfer/state"
    steady.telemetry.validate_state_url(state_url, allow_remote=False)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scheduler": "closed_loop_saturated",
        "label": args.label,
        "mode": args.mode,
        "started_at": steady.utc_now(),
        "valid": False,
        "issues": [],
        "configuration": {
            "dispatch_mode": "fixed_playlist_closed_loop",
            "root_stagger_s": 0.0,
            "url": args.url,
            "max_concurrency": args.max_concurrency,
            "requested_live_revisit_every_churn": args.live_revisit_every_churn,
            "session_end_max_concurrency": args.session_end_max_concurrency,
            "request_timeout_s": args.request_timeout_s,
            "session_end_timeout_s": args.session_end_timeout_s,
            "session_end_retries": args.session_end_retries,
            "session_end_retry_delay_s": args.session_end_retry_delay_s,
            "telemetry_interval_s": args.telemetry_interval_s,
            "trace_sha256": steady.telemetry.sha256_file(args.trace),
            "manifest_sha256": steady.telemetry.sha256_file(args.manifest),
            "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        },
        "cleanup": None,
    }
    started = time.monotonic()
    try:
        steady.flush_cache(base_url, args.flush_timeout_s)
        summary["initial_state"] = steady.wait_empty(
            state_url, args.flush_timeout_s
        )
        result = asyncio.run(execute(args, records, driver, out_dir))
        summary.update({key: value for key, value in result.items() if key != "issues"})
        summary["configuration"]["playlist_sha256"] = result["playlist_sha256"]
        summary["configuration"]["effective_live_revisit_every_churn"] = result[
            "live_revisit_every_churn"
        ]
        summary["issues"].extend(result["issues"])
        summary["valid"] = not summary["issues"]
    except Exception as exc:
        summary["issues"].append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        cleanup_error = None
        cleanup_started = time.monotonic()
        try:
            steady.flush_cache(base_url, args.flush_timeout_s)
            steady.wait_empty(state_url, args.flush_timeout_s)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"[:500]
            summary["issues"].append("post-measurement cleanup failed")
            summary["valid"] = False
        summary["cleanup"] = {
            "ok": cleanup_error is None,
            "error": cleanup_error,
            "elapsed_seconds": time.monotonic() - cleanup_started,
        }
        summary["duration_seconds"] = time.monotonic() - started
        summary["finished_at"] = steady.utc_now()
        steady.atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
