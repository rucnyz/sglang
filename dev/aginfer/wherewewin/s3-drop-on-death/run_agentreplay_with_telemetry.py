#!/usr/bin/env python3
"""Run an AgentReplay command while sampling direct-SGLang cache/GPU telemetry.

The wrapper is deliberately standard-library-only and never records raw SGLang
state, prompts, input token IDs, or forced output token IDs.  It writes:

* ``telemetry.jsonl``: one summarized start/running/post/end sample per interval;
* ``start.json`` and ``end.json``: convenient summarized snapshots;
* ``summary.json``: child return code, sampled peaks, final residual/dead bytes,
  and a safe subset of the AgentReplay result JSON.

Pass the child command after ``--``.  The child is launched without a shell and
its return code is returned unchanged.  For exact dead-byte classification after
the replay, pass ``--trace`` and ``--run-salt`` (or include AgentReplay's
``--trace``, ``--salt``, and ``--out`` arguments in the child command).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

TIERS = ("HBM", "DRAM", "DISK")
SCHEMA_VERSION = 1


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


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_option(command: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(command):
        if value == name and index + 1 < len(command):
            return command[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def resolve_child_path(
    value: str | None, child_cwd: pathlib.Path
) -> pathlib.Path | None:
    if not value:
        return None
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else child_cwd / path


def runtime_program_id(program_id: str, run_salt: str) -> str:
    value = f"{program_id}#{run_salt}"
    if len(value) <= 64:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return f"{value[:47]}#{digest}"


def load_trace_program_ids(path: pathlib.Path) -> set[str]:
    programs: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid trace JSON at line {line_number}: {exc}"
                ) from exc
            program_id = (
                record.get("program_id") if isinstance(record, Mapping) else None
            )
            if not isinstance(program_id, str) or not program_id:
                raise ValueError(f"trace line {line_number} has no string program_id")
            programs.add(program_id)
    if not programs:
        raise ValueError("trace has no programs")
    return programs


def no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_json(url: str, timeout: float) -> tuple[Any | None, str | None, int | None]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "agentreplay-telemetry/1"},
        method="GET",
    )
    try:
        with no_proxy_opener().open(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}", int(exc.code)
    except Exception as exc:  # telemetry must not hide the child result
        return None, f"{type(exc).__name__}: {exc}"[:300], None
    try:
        return json.loads(body), None, status
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}", status


def direct_states(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("state response is not an object")
    per_rank = payload.get("per_rank")
    if per_rank is None:
        states: list[Any] = [payload]
    elif isinstance(per_rank, list) and per_rank:
        states = per_rank
    else:
        raise ValueError("state response has invalid per_rank")
    if not all(isinstance(state, Mapping) for state in states):
        raise ValueError("state response contains a non-object rank")
    return states


def numeric_int(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def unit_holders(unit: Mapping[str, Any]) -> set[str]:
    raw = unit.get("session_ids")
    if not isinstance(raw, list):
        return set()
    return {str(value) for value in raw if value not in (None, "")}


def unit_tier_bytes(unit: Mapping[str, Any]) -> dict[str, int]:
    result = {tier: 0 for tier in TIERS}
    raw = unit.get("n_bytes")
    if not isinstance(raw, Mapping):
        return result
    for tier in TIERS:
        subpools = raw.get(tier)
        if isinstance(subpools, Mapping):
            result[tier] = sum(numeric_int(amount) for amount in subpools.values())
    return result


def add_tiers(target: dict[str, int], source: Mapping[str, int]) -> None:
    for tier in TIERS:
        target[tier] += int(source.get(tier, 0) or 0)


def analyze_state(
    payload: Any,
    tracked_programs: set[str] | None,
    ended_programs: set[str] | None,
) -> dict[str, Any]:
    states = direct_states(payload)
    pool_used = {tier: 0 for tier in TIERS}
    pool_cap = {tier: 0 for tier in TIERS}
    radix_bytes = {tier: 0 for tier in TIERS}
    residual_holder_bytes = {tier: 0 for tier in TIERS}
    tracked_bytes = {tier: 0 for tier in TIERS}
    dead_bytes = {tier: 0 for tier in TIERS} if ended_programs is not None else None
    holder_programs: set[str] = set()
    tracked_present: set[str] = set()
    ended_present: set[str] = set()
    usage_programs: set[str] = set()
    unit_count = 0
    residual_unit_count = 0
    tracked_unit_count = 0
    dead_unit_count = 0

    for state in states:
        units = state.get("units")
        if not isinstance(units, list):
            raise ValueError("rank state lacks units[]")
        pool_usage = state.get("pool_usage")
        if not isinstance(pool_usage, Mapping):
            raise ValueError("rank state lacks pool_usage")
        usage = state.get("per_program_usage")
        if isinstance(usage, Mapping):
            usage_programs.update(str(program) for program in usage)
        for tier in TIERS:
            tier_entry = pool_usage.get(tier)
            subpools = (
                tier_entry.get("subpools") if isinstance(tier_entry, Mapping) else None
            )
            if isinstance(subpools, Mapping):
                for fields in subpools.values():
                    if isinstance(fields, Mapping):
                        pool_used[tier] += numeric_int(fields.get("used_bytes"))
                        pool_cap[tier] += numeric_int(fields.get("cap_bytes"))
        unit_count += len(units)
        for unit in units:
            if not isinstance(unit, Mapping):
                continue
            holders = unit_holders(unit)
            amounts = unit_tier_bytes(unit)
            add_tiers(radix_bytes, amounts)
            holder_programs.update(holders)
            if holders:
                residual_unit_count += 1
                add_tiers(residual_holder_bytes, amounts)
            if tracked_programs is not None and holders & tracked_programs:
                tracked_unit_count += 1
                tracked_present.update(holders & tracked_programs)
                add_tiers(tracked_bytes, amounts)
            if (
                ended_programs is not None
                and holders
                and holders.issubset(ended_programs)
            ):
                dead_unit_count += 1
                ended_present.update(holders)
                add_tiers(dead_bytes, amounts)  # type: ignore[arg-type]

    return {
        "rank_count": len(states),
        "unit_count": unit_count,
        "residual_unit_count": residual_unit_count,
        "tracked_unit_count": tracked_unit_count,
        "dead_unit_count": dead_unit_count if dead_bytes is not None else None,
        "holder_program_count": len(holder_programs),
        "program_usage_count": len(usage_programs),
        "tracked_programs_present_count": len(tracked_present),
        "ended_programs_present_count": (
            len(ended_present) if dead_bytes is not None else None
        ),
        "external_holder_count": (
            len(holder_programs - tracked_programs)
            if tracked_programs is not None
            else None
        ),
        "pool_used_bytes": pool_used,
        "pool_cap_bytes": pool_cap,
        "radix_physical_bytes": radix_bytes,
        "residual_holder_bytes": residual_holder_bytes,
        "tracked_physical_bytes": (
            tracked_bytes if tracked_programs is not None else None
        ),
        "dead_physical_bytes": dead_bytes,
        "dead_physical_bytes_total": (
            sum(dead_bytes.values()) if dead_bytes is not None else None
        ),
    }


def parse_nvidia_smi(text: str) -> list[dict[str, int]]:
    devices = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            devices.append(
                {
                    "index": int(fields[0]),
                    "used_bytes": int(float(fields[1]) * 1024**2),
                    "total_bytes": int(float(fields[2]) * 1024**2),
                }
            )
        except ValueError:
            continue
    return devices


_NVITOP_BRACKET_INDEX = re.compile(r"^\s*\[\s*(?P<index>\d+)\]")
_NVITOP_TABLE_INDEX = re.compile(r"^\s*\|\s*(?P<index>\d+)\s+")
_NVITOP_MEMORY = re.compile(
    r"(?P<used>\d+(?:\.\d+)?)\s*(?P<used_unit>[KMGT]i?B|[KMGT]B|B)?\s*/\s*"
    r"(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGT]i?B|[KMGT]B|B)",
    re.IGNORECASE,
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def bytes_from_unit(value: str, unit: str) -> int:
    normalized = unit.upper()
    powers = {
        "B": 0,
        "KB": 1,
        "KIB": 1,
        "MB": 2,
        "MIB": 2,
        "GB": 3,
        "GIB": 3,
        "TB": 4,
        "TIB": 4,
    }
    power = powers[normalized]
    base = 1024 if "I" in normalized else 1000
    return int(float(value) * base**power)


def parse_nvitop(text: str) -> list[dict[str, int]]:
    devices = []
    pending_index: int | None = None
    for raw_line in text.splitlines():
        line = _ANSI.sub("", raw_line)
        index_match = _NVITOP_BRACKET_INDEX.search(line) or _NVITOP_TABLE_INDEX.search(
            line
        )
        if index_match:
            pending_index = int(index_match.group("index"))
        match = _NVITOP_MEMORY.search(line)
        if not match or pending_index is None:
            continue
        total_unit = match.group("total_unit")
        used_unit = match.group("used_unit") or total_unit
        devices.append(
            {
                "index": pending_index,
                "used_bytes": bytes_from_unit(match.group("used"), used_unit),
                "total_bytes": bytes_from_unit(match.group("total"), total_unit),
            }
        )
        pending_index = None
    return devices


def gpu_sample(timeout: float) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    errors: list[str] = []
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            devices = parse_nvidia_smi(result.stdout)
            if result.returncode == 0 and devices:
                return gpu_summary("nvidia-smi", devices, None)
            errors.append(f"nvidia-smi rc={result.returncode}, parsed={len(devices)}")
        except Exception as exc:
            errors.append(f"nvidia-smi {type(exc).__name__}: {exc}")

    nvitop = shutil.which("nvitop")
    if nvitop:
        for arguments in (
            [nvitop, "-1", "--no-unicode", "--readonly"],
            [nvitop, "--once", "--no-unicode", "--readonly"],
        ):
            try:
                env = dict(os.environ)
                env.update({"NO_COLOR": "1", "TERM": "dumb"})
                result = subprocess.run(
                    arguments,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                devices = parse_nvitop(result.stdout)
                if result.returncode == 0 and devices:
                    return gpu_summary("nvitop", devices, None)
                errors.append(
                    f"{' '.join(arguments[1:])} rc={result.returncode}, parsed={len(devices)}"
                )
            except Exception as exc:
                errors.append(f"nvitop {type(exc).__name__}: {exc}")
    return gpu_summary(
        None, [], "; ".join(errors)[:500] or "no GPU telemetry tool found"
    )


def gpu_summary(
    source: str | None, devices: list[dict[str, int]], error: str | None
) -> dict[str, Any]:
    return {
        "source": source,
        "devices": sorted(devices, key=lambda device: device["index"]),
        "used_bytes_total": sum(device["used_bytes"] for device in devices),
        "total_bytes_total": sum(device["total_bytes"] for device in devices),
        "error": error,
    }


def safe_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    allowed = {
        "label",
        "n_requests",
        "n_ok",
        "n_error",
        "total_out_tokens",
        "len_match_rate",
        "force_exact_rate",
        "force_exact_failures",
        "force_exact_missing",
        "cache_hit",
        "total_prompt_tokens",
        "total_cached_tokens",
        "wall_s",
        "throughput_tok_s",
        "makespan_s",
        "inference_makespan_s",
        "pipeline_makespan_s",
        "inference_throughput_tok_s",
        "pipeline_throughput_tok_s",
        "n_programs",
        "max_output_tokens",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "program_e2e_s",
    }
    sanitized = {key: result[key] for key in allowed if key in result}
    session_end = result.get("session_end")
    if isinstance(session_end, Mapping):
        end_allowed = {
            "enabled",
            "n",
            "n_ok",
            "n_error",
            "latency_ms",
            "freed_bytes",
            "freed_units",
            "matched_nodes",
            "holders_removed",
            "released_nodes",
            "released_hbm_tokens",
            "released_dram_tokens",
            "remaining_nodes",
            "deferred",
            "retry_attempts",
        }
        sanitized["session_end"] = {
            key: session_end[key] for key in end_allowed if key in session_end
        }
    return sanitized


def load_result(path: pathlib.Path | None) -> tuple[Any | None, str | None]:
    if path is None:
        return None, "AgentReplay result path was not provided or discovered"
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def result_proves_all_programs_ended(
    result: Any, child_returncode: int, source_program_count: int
) -> bool:
    if child_returncode != 0 or not isinstance(result, Mapping):
        return False
    return (
        numeric_int(result.get("n_error")) == 0
        and numeric_int(result.get("n_programs")) == source_program_count
        and float(result.get("len_match_rate") or 0.0) == 1.0
    )


def append_no_proxy(environment: dict[str, str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        values = [
            value.strip()
            for value in environment.get(key, "").split(",")
            if value.strip()
        ]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in values:
                values.append(host)
        environment[key] = ",".join(values)


def validate_state_url(url: str, allow_remote: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--state-url must be an http(s) URL")
    loopback = {"127.0.0.1", "localhost", "::1"}
    if not allow_remote and parsed.hostname not in loopback:
        raise ValueError(
            "remote state URL refused; pass --allow-remote-state-url explicitly"
        )


def peak_tiers(samples: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    peaks = {tier: 0 for tier in TIERS}
    for sample in samples:
        state = sample.get("state")
        values = state.get(field) if isinstance(state, Mapping) else None
        if isinstance(values, Mapping):
            for tier in TIERS:
                peaks[tier] = max(peaks[tier], numeric_int(values.get(tier)))
    return peaks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--state-url",
        default="http://127.0.0.1:30001/aginfer/state",
        help="direct SGLang state endpoint (loopback only unless explicitly allowed)",
    )
    parser.add_argument("--allow-remote-state-url", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--post-seconds", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--gpu-timeout", type=float, default=10.0)
    parser.add_argument("--trace", type=pathlib.Path, default=None)
    parser.add_argument("--run-salt", default=None)
    parser.add_argument("--result-json", type=pathlib.Path, default=None)
    parser.add_argument("--child-cwd", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument(
        "--record-command",
        action="store_true",
        help="record child argv verbatim; off by default because argv can contain secrets",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    if args.poll_interval <= 0 or args.state_timeout <= 0 or args.gpu_timeout <= 0:
        parser.error("poll interval and telemetry timeouts must be positive")
    if args.post_seconds < 0:
        parser.error("--post-seconds cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_state_url(args.state_url, args.allow_remote_state_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry.jsonl"
    summary_path = out_dir / "summary.json"
    start_path = out_dir / "start.json"
    end_path = out_dir / "end.json"
    existing = [
        path
        for path in (telemetry_path, summary_path, start_path, end_path)
        if path.exists()
    ]
    if existing:
        raise SystemExit(
            "refusing to overwrite telemetry artifacts: "
            + ", ".join(map(str, existing))
        )

    child_cwd = args.child_cwd.expanduser().resolve()
    trace_path = args.trace
    if trace_path is None:
        trace_path = resolve_child_path(
            child_option(args.command, "--trace"), child_cwd
        )
    else:
        trace_path = trace_path.expanduser().resolve()
    result_path = args.result_json
    if result_path is None:
        result_path = resolve_child_path(child_option(args.command, "--out"), child_cwd)
    else:
        result_path = result_path.expanduser().resolve()
    if result_path in {telemetry_path, summary_path, start_path, end_path}:
        raise SystemExit(
            "--result-json/child --out conflicts with a telemetry artifact"
        )
    run_salt = args.run_salt or child_option(args.command, "--salt")

    warnings: list[str] = []
    for unsafe in ("--limit", "--max-output-tokens"):
        if child_option(args.command, unsafe) is not None:
            warnings.append(
                f"formal replay command contains {unsafe}; workload may be truncated"
            )
    source_programs: set[str] | None = None
    trace_metadata: dict[str, Any] | None = None
    if trace_path is not None:
        source_programs = load_trace_program_ids(trace_path)
        trace_metadata = {
            "basename": trace_path.name,
            "sha256": sha256_file(trace_path),
            "program_count": len(source_programs),
        }
    else:
        warnings.append(
            "trace path unavailable; tracked/dead bytes cannot be classified"
        )
    if not run_salt:
        warnings.append(
            "run salt unavailable during replay; tracked holders cannot be classified live"
        )

    tracked_programs = (
        {runtime_program_id(program, run_salt) for program in source_programs}
        if source_programs is not None and run_salt
        else None
    )
    samples: list[dict[str, Any]] = []
    started_wall = utc_now()
    started = time.monotonic()
    child: subprocess.Popen[Any] | None = None
    child_returncode: int | None = None
    result: Any = None
    result_error: str | None = None
    ended_programs: set[str] | None = None
    launch_error: str | None = None

    def take_sample(event: str, phase: str) -> dict[str, Any]:
        payload, state_error, http_status = fetch_json(
            args.state_url, args.state_timeout
        )
        state_summary = None
        if state_error is None:
            try:
                state_summary = analyze_state(payload, tracked_programs, ended_programs)
            except Exception as exc:
                state_error = f"{type(exc).__name__}: {exc}"[:300]
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sequence": len(samples),
            "event": event,
            "phase": phase,
            "at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "child_running": child is not None and child.poll() is None,
            "child_returncode": child_returncode,
            "ended_program_ids_known": ended_programs is not None,
            "state_http_status": http_status,
            "state_error": state_error,
            "state": state_summary,
            "gpu": gpu_sample(args.gpu_timeout),
        }
        samples.append(sample)
        telemetry_stream.write(json.dumps(sample, sort_keys=True) + "\n")
        telemetry_stream.flush()
        return sample

    command_digest = hashlib.sha256("\0".join(args.command).encode()).hexdigest()
    environment = dict(os.environ)
    append_no_proxy(environment)
    interrupted = False

    with telemetry_path.open("x", encoding="utf-8", buffering=1) as telemetry_stream:
        start_sample = take_sample("start", "pre_child")
        atomic_write_json(start_path, start_sample)
        try:
            child = subprocess.Popen(
                args.command, cwd=child_cwd, env=environment, shell=False
            )
            next_sample = time.monotonic()
            while child.poll() is None:
                now = time.monotonic()
                if now >= next_sample:
                    take_sample("sample", "running")
                    next_sample = time.monotonic() + args.poll_interval
                sleep_for = max(0.01, min(0.1, next_sample - time.monotonic()))
                time.sleep(sleep_for)
            child_returncode = int(child.returncode)
        except KeyboardInterrupt:
            interrupted = True
            if child is not None and child.poll() is None:
                child.send_signal(2)
                try:
                    child_returncode = int(child.wait(timeout=30))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child_returncode = int(child.wait(timeout=30))
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child_returncode = int(child.wait())
            elif child is not None:
                child_returncode = int(child.returncode)
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"[:300]
            child_returncode = 127
            warnings.append(f"child launch failed: {launch_error}")

        result, result_error = load_result(result_path)
        if (
            not run_salt
            and isinstance(result, Mapping)
            and isinstance(result.get("run_salt"), str)
        ):
            run_salt = str(result["run_salt"])
            tracked_programs = (
                {runtime_program_id(program, run_salt) for program in source_programs}
                if source_programs is not None
                else None
            )
        if (
            source_programs is not None
            and tracked_programs is not None
            and child_returncode is not None
            and result_proves_all_programs_ended(
                result, child_returncode, len(source_programs)
            )
        ):
            ended_programs = set(tracked_programs)
        else:
            warnings.append(
                "could not prove every trace program ended successfully; dead bytes are left null"
            )

        take_sample("child_exit", "post_child")
        post_deadline = time.monotonic() + (
            0.0 if launch_error is not None else args.post_seconds
        )
        next_sample = time.monotonic() + args.poll_interval
        while time.monotonic() < post_deadline:
            delay = next_sample - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, post_deadline - time.monotonic()))
            if time.monotonic() >= next_sample and time.monotonic() < post_deadline:
                take_sample("sample", "retention")
                next_sample = time.monotonic() + args.poll_interval
        end_sample = take_sample("end", "retention_end")
        atomic_write_json(end_path, end_sample)

    state_errors = sum(1 for sample in samples if sample.get("state_error"))
    gpu_errors = sum(1 for sample in samples if (sample.get("gpu") or {}).get("error"))
    if state_errors:
        warnings.append(
            f"{state_errors} telemetry sample(s) lacked usable SGLang state"
        )
    if gpu_errors:
        warnings.append(f"{gpu_errors} telemetry sample(s) lacked usable GPU HBM data")
    gpu_peak = max(
        (
            numeric_int((sample.get("gpu") or {}).get("used_bytes_total"))
            for sample in samples
        ),
        default=0,
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_wall,
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "child_returncode": child_returncode,
        "child_launch_error": launch_error,
        "interrupted": interrupted,
        "command_sha256": command_digest,
        "command_argc": len(args.command),
        "state_url": args.state_url,
        "poll_interval_seconds": args.poll_interval,
        "post_seconds": args.post_seconds,
        "sample_count": len(samples),
        "state_error_count": state_errors,
        "gpu_error_count": gpu_errors,
        "trace": trace_metadata,
        "result_basename": result_path.name if result_path is not None else None,
        "result_error": result_error,
        "agentreplay_result": safe_result(result),
        "run_salt_sha256": (
            hashlib.sha256(run_salt.encode()).hexdigest() if run_salt else None
        ),
        "tracked_program_count": (
            len(tracked_programs) if tracked_programs is not None else None
        ),
        "ended_program_ids_known": ended_programs is not None,
        "peak_pool_used_bytes": peak_tiers(samples, "pool_used_bytes"),
        "peak_radix_physical_bytes": peak_tiers(samples, "radix_physical_bytes"),
        "peak_residual_holder_bytes": peak_tiers(samples, "residual_holder_bytes"),
        "peak_dead_physical_bytes": peak_tiers(samples, "dead_physical_bytes"),
        "peak_gpu_used_bytes_total": gpu_peak,
        "start_state": samples[0].get("state") if samples else None,
        "end_state": samples[-1].get("state") if samples else None,
        "warnings": list(dict.fromkeys(warnings)),
        "artifacts": {
            "telemetry_jsonl": telemetry_path.name,
            "start_json": start_path.name,
            "end_json": end_path.name,
        },
    }
    if args.record_command:
        summary["command"] = args.command
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if child_returncode is None:
        return 130 if interrupted else 1
    return child_returncode


if __name__ == "__main__":
    raise SystemExit(main())
