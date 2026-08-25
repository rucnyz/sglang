#!/usr/bin/env python3
"""Analyze paired single- or multi-wave AgentReplay Dead-KV pressure campaigns.

Each positional input is one model directory containing::

    baseline-rN/summary.json
    ours-rN/summary.json

Only the aggregate arm summaries are opened. Token traces, raw request results,
telemetry JSONL, and logs are never read or copied to the output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import random
import re
import statistics
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 2
RUN_RE = re.compile(r"^(baseline|ours)-r(.+)$")
PHASES = ("seed", "terminal", "probe")
TRACE_PHASES = ("live_seed", "terminal_churn", "live_probe")

# name, display label, unit, higher-is-better, has a meaningful paired delta
METRICS: tuple[tuple[str, str, str, bool, bool], ...] = (
    ("pre_probe_dead_hbm_bytes", "Pre-probe dead HBM", "bytes", False, True),
    ("pre_probe_dead_dram_bytes", "Pre-probe dead DRAM", "bytes", False, True),
    ("pre_probe_pool_hbm_utilization", "Pre-probe HBM pool use", "ratio", False, True),
    (
        "pre_probe_pool_dram_utilization",
        "Pre-probe DRAM pool use",
        "ratio",
        False,
        True,
    ),
    (
        "pre_probe_live_program_presence",
        "Pre-probe live programs with any holder",
        "ratio",
        True,
        True,
    ),
    (
        "pre_probe_live_hbm_bytes",
        "Pre-probe live KV (HBM)",
        "bytes",
        True,
        True,
    ),
    (
        "pre_probe_live_dram_bytes",
        "Pre-probe live KV (DRAM)",
        "bytes",
        True,
        True,
    ),
    ("live_probe_cache_hit", "Live-probe cache hit", "ratio", True, True),
    (
        "live_probe_device_cache_hit",
        "Live-probe device cache hit",
        "ratio",
        True,
        True,
    ),
    (
        "live_probe_host_cache_hit",
        "Live-probe host cache hit",
        "ratio",
        True,
        True,
    ),
    (
        "live_probe_storage_cache_hit",
        "Live-probe storage cache hit",
        "ratio",
        True,
        True,
    ),
    ("live_probe_ttft_mean_ms", "Live-probe TTFT mean", "ms", False, True),
    ("live_probe_ttft_p50_ms", "Live-probe TTFT p50", "ms", False, True),
    ("live_probe_ttft_p90_ms", "Live-probe TTFT p90", "ms", False, True),
    (
        "terminal_inference_throughput_tok_s",
        "Terminal inference throughput",
        "tok/s",
        True,
        True,
    ),
    (
        "terminal_pipeline_throughput_tok_s",
        "Terminal full-pipeline throughput",
        "tok/s",
        True,
        True,
    ),
    ("terminal_end_latency_mean_ms", "Terminal END latency mean", "ms", False, False),
    ("terminal_end_latency_p50_ms", "Terminal END latency p50", "ms", False, False),
    ("terminal_end_latency_p90_ms", "Terminal END latency p90", "ms", False, False),
    (
        "terminal_end_worst_wave_p50_ms",
        "Terminal END worst-wave p50",
        "ms",
        False,
        False,
    ),
    (
        "terminal_end_worst_wave_p90_ms",
        "Terminal END worst-wave p90",
        "ms",
        False,
        False,
    ),
    ("terminal_end_retry_attempts", "Terminal END retries", "count", False, False),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def get_path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return None


def ratio(used: Any, capacity: Any) -> float | None:
    used_number = number(used)
    capacity_number = number(capacity)
    if used_number is None or capacity_number is None or capacity_number <= 0:
        return None
    return float(used_number) / float(capacity_number)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric_value(summary: Mapping[str, Any], name: str) -> float | int | None:
    pre_probe = get_path(summary, "states", "before_live_probe")
    terminal_end_state = summary.get("end_state")
    probe = summary.get("live_probe_result")
    if not isinstance(probe, Mapping):
        probe = get_path(summary, "phases", "probe", "result")
    terminal = summary.get("agentreplay_result")
    if not isinstance(terminal, Mapping):
        terminal = get_path(summary, "phases", "terminal", "result")
    if not isinstance(pre_probe, Mapping):
        pre_probe = {}
    if not isinstance(terminal_end_state, Mapping):
        terminal_end_state = {}
    if not isinstance(probe, Mapping):
        probe = {}
    if not isinstance(terminal, Mapping):
        terminal = {}

    if name == "pre_probe_live_program_presence":
        live_count = number(get_path(summary, "phase_manifest", "live_program_count"))
        if live_count is None or live_count <= 0:
            return None
        pre_probe_live = get_path(
            summary,
            "states",
            "live_before_live_probe",
            "tracked_programs_present_count",
        )
        return ratio(pre_probe_live, live_count)

    live_tiers = {
        "pre_probe_live_hbm_bytes": "HBM",
        "pre_probe_live_dram_bytes": "DRAM",
    }
    if name in live_tiers:
        return number(
            get_path(
                summary,
                "states",
                "live_before_live_probe",
                "tracked_physical_bytes",
                live_tiers[name],
            )
        )

    cache_tiers = {
        "live_probe_device_cache_hit": "device",
        "live_probe_host_cache_hit": "host",
        "live_probe_storage_cache_hit": "storage",
    }
    if name in cache_tiers:
        if get_path(probe, "cached_tokens_details", "coverage_complete") is not True:
            return None
        return ratio(
            get_path(probe, "cached_tokens_details", cache_tiers[name]),
            probe.get("total_prompt_tokens"),
        )

    if (
        name.startswith("terminal_end_")
        and get_path(terminal, "session_end", "enabled") is not True
    ):
        return None

    paths: dict[str, tuple[Any, tuple[str, ...]]] = {
        "pre_probe_dead_hbm_bytes": (
            (
                pre_probe
                if get_path(pre_probe, "dead_physical_bytes", "HBM") is not None
                else terminal_end_state
            ),
            ("dead_physical_bytes", "HBM"),
        ),
        "pre_probe_dead_dram_bytes": (
            (
                pre_probe
                if get_path(pre_probe, "dead_physical_bytes", "DRAM") is not None
                else terminal_end_state
            ),
            ("dead_physical_bytes", "DRAM"),
        ),
        "live_probe_cache_hit": (probe, ("cache_hit",)),
        "live_probe_ttft_mean_ms": (probe, ("ttft_ms", "mean")),
        "live_probe_ttft_p50_ms": (probe, ("ttft_ms", "p50")),
        "live_probe_ttft_p90_ms": (probe, ("ttft_ms", "p90")),
        "terminal_inference_throughput_tok_s": (
            terminal,
            ("inference_throughput_tok_s",),
        ),
        "terminal_pipeline_throughput_tok_s": (
            terminal,
            ("pipeline_throughput_tok_s",),
        ),
        "terminal_end_latency_mean_ms": (
            terminal,
            ("session_end", "latency_ms", "mean"),
        ),
        "terminal_end_latency_p50_ms": (
            terminal,
            ("session_end", "latency_ms", "p50"),
        ),
        "terminal_end_latency_p90_ms": (
            terminal,
            ("session_end", "latency_ms", "p90"),
        ),
        "terminal_end_worst_wave_p50_ms": (
            terminal,
            ("session_end", "worst_wave_latency_ms", "p50"),
        ),
        "terminal_end_worst_wave_p90_ms": (
            terminal,
            ("session_end", "worst_wave_latency_ms", "p90"),
        ),
        "terminal_end_retry_attempts": (
            terminal,
            ("session_end", "retry_attempts"),
        ),
    }
    if name == "pre_probe_pool_hbm_utilization":
        constrained = number(get_path(pre_probe, "pool_max_subpool_utilization", "HBM"))
        if constrained is not None:
            return constrained
        return ratio(
            get_path(pre_probe, "pool_used_bytes", "HBM"),
            get_path(pre_probe, "pool_cap_bytes", "HBM"),
        )
    if name == "pre_probe_pool_dram_utilization":
        constrained = number(
            get_path(pre_probe, "pool_max_subpool_utilization", "DRAM")
        )
        if constrained is not None:
            return constrained
        return ratio(
            get_path(pre_probe, "pool_used_bytes", "DRAM"),
            get_path(pre_probe, "pool_cap_bytes", "DRAM"),
        )
    source_path = paths.get(name)
    if source_path is None:
        return None
    source, path = source_path
    return number(get_path(source, *path))


def result_issues(
    result: Any,
    *,
    phase: str,
    expected_requests: int | None,
    expected_programs: int | None,
    expect_end: bool,
) -> list[str]:
    if not isinstance(result, Mapping):
        return [f"{phase}: missing aggregate result"]
    issues = []
    expected = {
        "n_requests": expected_requests,
        "n_ok": expected_requests,
        "n_error": 0,
        "n_programs": expected_programs,
        "len_match_rate": 1.0,
        "force_exact_rate": 1.0,
        "force_exact_failures": 0,
        "force_exact_missing": 0,
    }
    for field, wanted in expected.items():
        if wanted is not None and number(result.get(field)) != wanted:
            issues.append(
                f"{phase}: {field}={result.get(field)!r}, expected {wanted!r}"
            )
    end = result.get("session_end")
    if not isinstance(end, Mapping):
        issues.append(f"{phase}: missing session_end summary")
        return issues
    if expect_end:
        if end.get("enabled") is not True:
            issues.append(f"{phase}: SESSION_END not enabled")
        if (
            number(end.get("n")) != expected_programs
            or number(end.get("n_ok")) != expected_programs
            or number(end.get("n_error")) != 0
            or number(end.get("remaining_nodes")) != 0
        ):
            issues.append(f"{phase}: incomplete SESSION_END completion")
    elif end.get("enabled") is not False:
        issues.append(f"{phase}: SESSION_END unexpectedly enabled")
    return issues


def run_issues(summary: Mapping[str, Any], arm: str) -> list[str]:
    issues = []
    if summary.get("valid") is not True:
        issues.append(f"summary valid={summary.get('valid')!r}, expected true")
    if summary.get("mode") != arm:
        issues.append(f"summary mode={summary.get('mode')!r}, expected {arm!r}")
    if number(summary.get("child_returncode")) != 0:
        issues.append(f"child_returncode={summary.get('child_returncode')!r}")
    summary_issues = summary.get("issues")
    if not isinstance(summary_issues, list) or summary_issues:
        issues.append("summary issues are missing or non-empty")

    manifest = summary.get("phase_manifest")
    if not isinstance(manifest, Mapping):
        return [*issues, "missing phase_manifest"]
    live_count = number(manifest.get("live_program_count"))
    terminal_count = number(manifest.get("terminal_program_count"))
    # Accept schema-v1 summaries produced before program IDs were removed from
    # aggregate artifacts, but never copy those IDs to the analysis output.
    live_ids = manifest.get("live_program_ids")
    terminal_ids = manifest.get("terminal_program_ids")
    if live_count is None and isinstance(live_ids, list):
        live_count = len(live_ids)
    if terminal_count is None and isinstance(terminal_ids, list):
        terminal_count = len(terminal_ids)
    if live_count is None or live_count <= 0:
        issues.append("phase manifest has no live programs")
    if terminal_count is None or terminal_count <= 0:
        issues.append("phase manifest has no terminal programs")
    for field in (
        "live_seed_requests",
        "terminal_requests",
        "live_probe_requests",
    ):
        value = number(manifest.get(field))
        if value is None or value <= 0:
            issues.append(f"phase manifest has invalid {field}")
    expected_by_phase = {
        "seed": (number(manifest.get("live_seed_requests")), live_count, False),
        "terminal": (
            number(manifest.get("terminal_requests")),
            terminal_count,
            arm == "ours",
        ),
        "probe": (
            number(manifest.get("live_probe_requests")),
            live_count,
            arm == "ours",
        ),
    }
    phases = summary.get("phases")
    if not isinstance(phases, Mapping):
        return [*issues, "missing phases"]
    for phase in PHASES:
        value = phases.get(phase)
        if not isinstance(value, Mapping):
            issues.append(f"{phase}: missing phase summary")
            continue
        if number(value.get("returncode")) != 0:
            issues.append(f"{phase}: returncode={value.get('returncode')!r}")
        phase_issues = value.get("issues")
        if not isinstance(phase_issues, list) or phase_issues:
            issues.append(f"{phase}: issues are missing or non-empty")
        expected_requests, expected_programs, expect_end = expected_by_phase[phase]
        issues.extend(
            result_issues(
                value.get("result"),
                phase=phase,
                expected_requests=(
                    int(expected_requests) if expected_requests is not None else None
                ),
                expected_programs=(
                    int(expected_programs) if expected_programs is not None else None
                ),
                expect_end=expect_end,
            )
        )

    terminal_telemetry = get_path(summary, "phases", "terminal", "telemetry_summary")
    if not isinstance(terminal_telemetry, Mapping):
        issues.append("terminal: missing telemetry summary")
    else:
        if number(terminal_telemetry.get("child_returncode")) != 0:
            issues.append("terminal: telemetry child return code is non-zero")
        if number(terminal_telemetry.get("state_error_count")) != 0:
            issues.append("terminal: state telemetry errors present")

    wave_specs = get_path(manifest, "traces", "terminal_churn", "waves")
    wave_runs = get_path(summary, "phases", "terminal_waves")
    wave_count = number(manifest.get("terminal_wave_count"))
    if isinstance(wave_specs, list):
        if wave_count != len(wave_specs):
            issues.append("terminal wave manifest count mismatch")
        wave_hashes = [
            spec.get("sha256") for spec in wave_specs if isinstance(spec, Mapping)
        ]
        payload_hashes = [
            spec.get("token_payload_sha256")
            for spec in wave_specs
            if isinstance(spec, Mapping)
        ]
        if any(not isinstance(value, str) or not value for value in wave_hashes):
            issues.append("terminal wave manifest has an invalid trace hash")
        elif len(set(wave_hashes)) != len(wave_hashes):
            issues.append("terminal wave manifest repeats a trace hash")
        if any(not isinstance(value, str) or not value for value in payload_hashes):
            issues.append("terminal wave manifest has an invalid token-payload hash")
        elif len(set(payload_hashes)) != len(payload_hashes):
            issues.append("terminal wave manifest repeats a token payload")
        if not isinstance(wave_runs, list) or len(wave_runs) != len(wave_specs):
            issues.append("terminal wave run count mismatch")
        else:
            for index, (spec, wave) in enumerate(zip(wave_specs, wave_runs), 1):
                if not isinstance(spec, Mapping) or not isinstance(wave, Mapping):
                    issues.append(f"terminal wave {index}: invalid summary")
                    continue
                issues.extend(
                    result_issues(
                        wave.get("result"),
                        phase=f"terminal wave {index}",
                        expected_requests=(
                            int(value)
                            if (value := number(spec.get("requests"))) is not None
                            else None
                        ),
                        expected_programs=(
                            int(value)
                            if (value := number(spec.get("programs"))) is not None
                            else None
                        ),
                        expect_end=arm == "ours",
                    )
                )
                wave_telemetry = wave.get("telemetry_summary")
                if not isinstance(wave_telemetry, Mapping):
                    issues.append(f"terminal wave {index}: missing telemetry summary")
                elif (
                    number(wave_telemetry.get("child_returncode")) != 0
                    or number(wave_telemetry.get("state_error_count")) != 0
                ):
                    issues.append(f"terminal wave {index}: telemetry errors present")
                live_state = get_path(
                    summary,
                    "states",
                    f"after_terminal_wave_{index:03d}",
                    "live",
                )
                retained = get_path(live_state, "tracked_programs_present_count")
                if (
                    not isinstance(live_state, Mapping)
                    or number(retained) is None
                    or float(retained) < 0
                    or (live_count is not None and float(retained) > float(live_count))
                ):
                    issues.append(
                        f"terminal wave {index}: invalid live-retention state"
                    )
        pre_probe_live = get_path(summary, "states", "live_before_live_probe")
        retained = get_path(pre_probe_live, "tracked_programs_present_count")
        if (
            not isinstance(pre_probe_live, Mapping)
            or number(retained) is None
            or float(retained) < 0
            or (live_count is not None and float(retained) > float(live_count))
        ):
            issues.append("invalid pre-probe live-retention state")

    pre_probe = get_path(summary, "states", "before_live_probe")
    if not isinstance(pre_probe, Mapping):
        issues.append("missing pre-probe state")
    probe_result = get_path(summary, "phases", "probe", "result")
    cache_details = get_path(probe_result, "cached_tokens_details")
    if (
        isinstance(cache_details, Mapping)
        and cache_details.get("coverage_complete") is not True
    ):
        issues.append("probe cached-token source breakdown is incomplete")
    cleanup = summary.get("cleanup")
    if not isinstance(cleanup, Mapping):
        issues.append("missing cleanup summary")
    else:
        expected_explicit = arm == "baseline"
        if cleanup.get("explicit_end_attempted") is not expected_explicit:
            issues.append("cleanup explicit-END mode mismatch")
        if expected_explicit and cleanup.get("all_end_calls_ok") is not True:
            issues.append("baseline explicit-END cleanup was incomplete")
        expected_total = (
            live_count + terminal_count
            if live_count is not None and terminal_count is not None
            else None
        )
        if expected_explicit and expected_total is not None:
            if number(cleanup.get("programs_targeted")) != expected_total:
                issues.append("baseline cleanup did not target every workload program")
            if number(cleanup.get("programs_ok")) != expected_total:
                issues.append("baseline cleanup did not END every workload program")
        final_state = cleanup.get("final_state")
        if not isinstance(final_state, Mapping) or (
            number(final_state.get("unit_count")) != 0
            or number(final_state.get("program_usage_count")) != 0
        ):
            issues.append("post-arm state was not empty")
    return issues


def load_run(path: pathlib.Path, model_dir: pathlib.Path) -> dict[str, Any]:
    match = RUN_RE.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"unexpected run directory name: {path.parent.name}")
    arm, pair_id = match.groups()
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, Mapping):
        raise ValueError(f"summary is not an object: {path}")
    manifest = summary.get("phase_manifest")
    traces = manifest.get("traces") if isinstance(manifest, Mapping) else None
    trace_hashes = {phase: get_path(traces, phase, "sha256") for phase in TRACE_PHASES}
    configuration = summary.get("configuration")
    configuration_hash = (
        canonical_hash(configuration) if isinstance(configuration, Mapping) else None
    )
    metrics = {name: metric_value(summary, name) for name, *_ in METRICS}
    issues = run_issues(summary, arm)
    optional_metrics = {
        "pre_probe_live_program_presence",
        "pre_probe_live_hbm_bytes",
        "pre_probe_live_dram_bytes",
        "live_probe_device_cache_hit",
        "live_probe_host_cache_hit",
        "live_probe_storage_cache_hit",
        "terminal_end_latency_mean_ms",
        "terminal_end_latency_p50_ms",
        "terminal_end_latency_p90_ms",
        "terminal_end_worst_wave_p50_ms",
        "terminal_end_worst_wave_p90_ms",
        "terminal_end_retry_attempts",
    }
    required_metrics = [
        name for name, *_rest in METRICS if name not in optional_metrics
    ]
    if arm == "ours":
        required_metrics += [
            "terminal_end_latency_mean_ms",
            "terminal_end_retry_attempts",
        ]
    missing = [name for name in required_metrics if metrics.get(name) is None]
    if missing:
        issues.append("missing metrics: " + ", ".join(missing))
    for name in (
        "pre_probe_pool_hbm_utilization",
        "pre_probe_pool_dram_utilization",
        "pre_probe_live_program_presence",
        "live_probe_cache_hit",
        "live_probe_device_cache_hit",
        "live_probe_host_cache_hit",
        "live_probe_storage_cache_hit",
    ):
        value = number(metrics.get(name))
        if value is not None and not 0.0 <= float(value) <= 1.0:
            issues.append(f"{name} outside [0, 1]")
    for name, value in metrics.items():
        numeric = number(value)
        if numeric is not None and numeric < 0:
            issues.append(f"{name} is negative")
    if not isinstance(configuration, Mapping):
        issues.append("missing configuration")
    run_salt_sha256 = summary.get("run_salt_sha256")
    if not isinstance(run_salt_sha256, str):
        legacy_salt = summary.get("run_salt")
        if isinstance(legacy_salt, str) and legacy_salt:
            run_salt_sha256 = hashlib.sha256(legacy_salt.encode()).hexdigest()
    if (
        not isinstance(run_salt_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", run_salt_sha256) is None
    ):
        issues.append("missing or invalid run salt SHA256")
    if any(not isinstance(value, str) or not value for value in trace_hashes.values()):
        issues.append("missing one or more phase trace SHA256 values")
    terminal_sha = get_path(summary, "trace", "sha256")
    if terminal_sha != trace_hashes["terminal_churn"]:
        issues.append("top-level terminal trace SHA does not match phase manifest")
    terminal_end = get_path(summary, "phases", "terminal", "result", "session_end")
    wave_count_value = number(
        manifest.get("terminal_wave_count") if isinstance(manifest, Mapping) else None
    )
    wave_count = int(wave_count_value) if wave_count_value is not None else 0
    live_count_value = number(
        manifest.get("live_program_count") if isinstance(manifest, Mapping) else None
    )
    live_retention_by_wave = []
    for wave_number in range(1, wave_count + 1):
        live_state = get_path(
            summary,
            "states",
            f"after_terminal_wave_{wave_number:03d}",
            "live",
        )
        present = number(get_path(live_state, "tracked_programs_present_count"))
        live_retention_by_wave.append(
            {
                "wave_number": wave_number,
                "programs_with_any_holder": present,
                "presence_fraction": ratio(present, live_count_value),
                "hbm_bytes": number(
                    get_path(live_state, "tracked_physical_bytes", "HBM")
                ),
                "dram_bytes": number(
                    get_path(live_state, "tracked_physical_bytes", "DRAM")
                ),
                "hbm_bytes_vs_seed": ratio(
                    get_path(live_state, "tracked_physical_bytes", "HBM"),
                    get_path(
                        summary,
                        "states",
                        "after_seed",
                        "tracked_physical_bytes",
                        "HBM",
                    ),
                ),
                "dram_bytes_vs_seed": ratio(
                    get_path(live_state, "tracked_physical_bytes", "DRAM"),
                    get_path(
                        summary,
                        "states",
                        "after_seed",
                        "tracked_physical_bytes",
                        "DRAM",
                    ),
                ),
            }
        )
    return {
        "model": model_dir.name,
        "pair_id": pair_id,
        "arm": arm,
        "source": str(path.relative_to(model_dir)),
        "valid": not issues,
        "issues": issues,
        "run_salt_sha256": run_salt_sha256,
        "trace_hashes": trace_hashes,
        "configuration_sha256": configuration_hash,
        "configuration_field_sha256": (
            {str(key): canonical_hash(value) for key, value in configuration.items()}
            if isinstance(configuration, Mapping)
            else None
        ),
        "terminal_end_enabled": (
            terminal_end.get("enabled") if isinstance(terminal_end, Mapping) else None
        ),
        "terminal_end_ok": (
            terminal_end.get("n_ok") if isinstance(terminal_end, Mapping) else None
        ),
        "terminal_end_attempted": (
            terminal_end.get("n") if isinstance(terminal_end, Mapping) else None
        ),
        "terminal_wave_count": wave_count,
        "live_retention_by_wave": live_retention_by_wave,
        "metrics": metrics,
    }


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    )


def configuration_diff_keys(left: Any, right: Any) -> list[str]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return ["<configuration>"]
    return sorted(
        str(key) for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def pair_runs(
    runs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    indexed: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    warnings = []
    for run in runs:
        key = (str(run["model"]), str(run["pair_id"]))
        arm = str(run["arm"])
        if arm in indexed.setdefault(key, {}):
            raise ValueError(f"duplicate {arm} run for {key[0]}/r{key[1]}")
        indexed[key][arm] = run
    metric_specs = {
        name: (label, unit, higher, paired)
        for name, label, unit, higher, paired in METRICS
    }
    pairs = []
    for (model, pair_id), arms in sorted(
        indexed.items(), key=lambda item: (item[0][0], natural_key(item[0][1]))
    ):
        if set(arms) != {"baseline", "ours"}:
            warnings.append(f"unpaired run: {model}/r{pair_id} has {sorted(arms)}")
            continue
        baseline = arms["baseline"]
        ours = arms["ours"]
        issues = []
        if baseline.get("trace_hashes") != ours.get("trace_hashes"):
            issues.append("phase trace SHA mismatch")
        if baseline.get("run_salt_sha256") != ours.get("run_salt_sha256"):
            issues.append("run salt fingerprint mismatch")
        if baseline.get("configuration_sha256") != ours.get("configuration_sha256"):
            keys = configuration_diff_keys(
                baseline.get("configuration_field_sha256"),
                ours.get("configuration_field_sha256"),
            )
            issues.append("configuration mismatch: " + ", ".join(keys))
        if not baseline.get("valid") or not ours.get("valid"):
            issues.append("one or both runs failed validation")
        paired_metrics = {}
        for name, (label, unit, higher_better, supports_delta) in metric_specs.items():
            base = number(get_path(baseline, "metrics", name))
            treatment = number(get_path(ours, "metrics", name))
            delta = (
                float(treatment) - float(base)
                if not issues
                and supports_delta
                and base is not None
                and treatment is not None
                else None
            )
            delta_percent = (
                100.0 * delta / float(base)
                if delta is not None and float(base) != 0.0
                else None
            )
            improvement = (
                delta_percent
                if higher_better
                else -delta_percent if delta_percent is not None else None
            )
            paired_metrics[name] = {
                "label": label,
                "unit": unit,
                "higher_is_better": higher_better,
                "supports_paired_delta": supports_delta,
                "baseline": base,
                "ours": treatment,
                "delta_ours_minus_baseline": delta,
                "delta_percent": delta_percent,
                "improvement_percent": improvement,
            }
        pairs.append(
            {
                "model": model,
                "pair_id": pair_id,
                "comparable": not issues,
                "issues": issues,
                "baseline_source": baseline["source"],
                "ours_source": ours["source"],
                "metrics": paired_metrics,
            }
        )
    return pairs, warnings


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = quantile * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = index - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def bootstrap_mean_ci(
    values: Sequence[float], samples: int, seed: int
) -> dict[str, Any] | None:
    if len(values) < 3:
        return None
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    return {
        "method": "nonparametric percentile bootstrap of the mean",
        "n": len(values),
        "resamples": samples,
        "low": percentile(means, 0.025),
        "high": percentile(means, 0.975),
    }


def metric_seed(seed: int, model: str, metric: str, kind: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{model}\0{metric}\0{kind}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def aggregate_pairs(
    pairs: Sequence[Mapping[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for pair in pairs:
        by_model.setdefault(str(pair["model"]), []).append(pair)
    aggregates = []
    for model in sorted(by_model):
        model_pairs = by_model[model]
        comparable = [pair for pair in model_pairs if pair.get("comparable")]
        metrics = {}
        for name, label, unit, higher_better, supports_delta in METRICS:
            entries = [get_path(pair, "metrics", name) for pair in comparable]
            entries = [entry for entry in entries if isinstance(entry, Mapping)]
            baseline_values = [
                float(value)
                for entry in entries
                if (value := number(entry.get("baseline"))) is not None
            ]
            ours_values = [
                float(value)
                for entry in entries
                if (value := number(entry.get("ours"))) is not None
            ]
            deltas = [
                float(value)
                for entry in entries
                if (value := number(entry.get("delta_ours_minus_baseline"))) is not None
            ]
            base_mean = mean_or_none(baseline_values)
            ours_mean = mean_or_none(ours_values)
            mean_delta = mean_or_none(deltas)
            delta_of_means = (
                100.0 * (ours_mean - base_mean) / base_mean
                if supports_delta
                and base_mean not in (None, 0.0)
                and ours_mean is not None
                else None
            )
            improvement = (
                delta_of_means
                if higher_better
                else -delta_of_means if delta_of_means is not None else None
            )
            metrics[name] = {
                "label": label,
                "unit": unit,
                "higher_is_better": higher_better,
                "supports_paired_delta": supports_delta,
                "n_baseline": len(baseline_values),
                "n_ours": len(ours_values),
                "n_paired_delta": len(deltas),
                "baseline_mean": base_mean,
                "ours_mean": ours_mean,
                "mean_paired_delta": mean_delta,
                "delta_of_means_percent": delta_of_means,
                "improvement_of_means_percent": improvement,
                "mean_paired_delta_ci95": (
                    bootstrap_mean_ci(
                        deltas,
                        bootstrap_samples,
                        metric_seed(seed, model, name, "delta"),
                    )
                    if supports_delta
                    else None
                ),
                "ours_mean_ci95": (
                    bootstrap_mean_ci(
                        ours_values,
                        bootstrap_samples,
                        metric_seed(seed, model, name, "ours"),
                    )
                    if not supports_delta
                    else None
                ),
            }
        aggregates.append(
            {
                "model": model,
                "pair_count": len(model_pairs),
                "comparable_pair_count": len(comparable),
                "metrics": metrics,
            }
        )
    return aggregates


def format_value(value: Any, unit: str) -> str:
    numeric = number(value)
    if numeric is None:
        return "—"
    if unit == "bytes":
        return f"{float(numeric) / 1024**2:.2f} MiB"
    if unit == "ratio":
        return f"{100.0 * float(numeric):.2f}%"
    if unit == "ms":
        return f"{float(numeric):.2f} ms"
    if unit == "tok/s":
        return f"{float(numeric):.2f} tok/s"
    if unit == "count":
        return f"{float(numeric):.2f}"
    return f"{float(numeric):.4g}"


def format_percent(value: Any) -> str:
    numeric = number(value)
    return "—" if numeric is None else f"{float(numeric):+.2f}%"


def format_ci(ci: Any, unit: str) -> str:
    if not isinstance(ci, Mapping):
        return "—"
    return (
        f"[{format_value(ci.get('low'), unit)}, {format_value(ci.get('high'), unit)}]"
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# AgentReplay high-pressure A/B results",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Pre-probe metrics are sampled after terminal churn and its retention barrier, "
        "immediately before the live step-4 probe. Positive improvement means Ours "
        "moved in the desired direction. Only aggregate summaries were read; raw "
        "prompts, token arrays, telemetry JSONL, and logs are excluded.",
        "For multi-wave runs, END mean is weighted by completed calls. Exact "
        "cross-wave percentiles cannot be reconstructed from aggregate-only inputs, "
        "so p50/p90 are left blank and worst-wave p50/p90 are labelled separately.",
        "",
        "## Validation",
        "",
        "| Model | Pair | Arm | Valid | Terminal END | Source |",
        "|---|---:|---|---:|---|---|",
    ]
    for run in report["runs"]:
        end = "disabled"
        if run["terminal_end_enabled"] is True:
            end = f"{run['terminal_end_ok']}/{run['terminal_end_attempted']}"
        lines.append(
            f"| {run['model']} | {run['pair_id']} | {run['arm']} | "
            f"{'yes' if run['valid'] else 'NO'} | {end} | `{run['source']}` |"
        )

    lines += ["", "## Per-run metrics", ""]
    for model in sorted({str(run["model"]) for run in report["runs"]}):
        lines += [
            f"### {model}",
            "",
            "| Pair | Arm | Dead HBM | Dead DRAM | HBM pool | DRAM pool | Live holders | Live HBM | Live DRAM | Probe hit | Device hit | Host hit | Storage hit | Probe TTFT mean | p50 | p90 | Inference tok/s | Pipeline tok/s | END mean | END p50 | END p90 | Worst-wave END p50 | Worst-wave END p90 | END retries |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for run in report["runs"]:
            if run["model"] != model:
                continue
            metrics = run["metrics"]
            ordered = [
                ("pre_probe_dead_hbm_bytes", "bytes"),
                ("pre_probe_dead_dram_bytes", "bytes"),
                ("pre_probe_pool_hbm_utilization", "ratio"),
                ("pre_probe_pool_dram_utilization", "ratio"),
                ("pre_probe_live_program_presence", "ratio"),
                ("pre_probe_live_hbm_bytes", "bytes"),
                ("pre_probe_live_dram_bytes", "bytes"),
                ("live_probe_cache_hit", "ratio"),
                ("live_probe_device_cache_hit", "ratio"),
                ("live_probe_host_cache_hit", "ratio"),
                ("live_probe_storage_cache_hit", "ratio"),
                ("live_probe_ttft_mean_ms", "ms"),
                ("live_probe_ttft_p50_ms", "ms"),
                ("live_probe_ttft_p90_ms", "ms"),
                ("terminal_inference_throughput_tok_s", "tok/s"),
                ("terminal_pipeline_throughput_tok_s", "tok/s"),
                ("terminal_end_latency_mean_ms", "ms"),
                ("terminal_end_latency_p50_ms", "ms"),
                ("terminal_end_latency_p90_ms", "ms"),
                ("terminal_end_worst_wave_p50_ms", "ms"),
                ("terminal_end_worst_wave_p90_ms", "ms"),
                ("terminal_end_retry_attempts", "count"),
            ]
            values = [format_value(metrics[name], unit) for name, unit in ordered]
            lines.append(
                "| "
                + " | ".join([str(run["pair_id"]), str(run["arm"]), *values])
                + " |"
            )
        lines.append("")

    wave_rows = [
        (run, wave)
        for run in report["runs"]
        for wave in run.get("live_retention_by_wave", [])
    ]
    if wave_rows:
        lines += [
            "## Live retention by terminal wave",
            "",
            "| Model | Pair | Arm | Wave | Programs with any holder | Presence | Live HBM | HBM bytes / seed | Live DRAM | DRAM bytes / seed |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for run, wave in wave_rows:
            lines.append(
                f"| {run['model']} | {run['pair_id']} | {run['arm']} | "
                f"{wave['wave_number']} | "
                f"{format_value(wave['programs_with_any_holder'], 'count')} | "
                f"{format_value(wave['presence_fraction'], 'ratio')} | "
                f"{format_value(wave['hbm_bytes'], 'bytes')} | "
                f"{format_value(wave['hbm_bytes_vs_seed'], 'ratio')} | "
                f"{format_value(wave['dram_bytes'], 'bytes')} | "
                f"{format_value(wave['dram_bytes_vs_seed'], 'ratio')} |"
            )
        lines.append("")

    lines += ["## Paired deltas", ""]
    for pair in report["pairs"]:
        lines += [
            f"### {pair['model']} / r{pair['pair_id']}",
            "",
            f"Comparable: `{'yes' if pair['comparable'] else 'no'}`",
            "",
            "| Metric | Baseline | Ours | Ours − Baseline | Relative delta | Improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, label, unit, _higher, _paired in METRICS:
            metric = pair["metrics"][name]
            lines.append(
                f"| {label} | {format_value(metric['baseline'], unit)} | "
                f"{format_value(metric['ours'], unit)} | "
                f"{format_value(metric['delta_ours_minus_baseline'], unit)} | "
                f"{format_percent(metric['delta_percent'])} | "
                f"{format_percent(metric['improvement_percent'])} |"
            )
        if pair["issues"]:
            lines += ["", "Issues: " + "; ".join(pair["issues"])]
        lines.append("")

    lines += ["## Means and bootstrap confidence intervals", ""]
    for model in report["models"]:
        lines += [
            f"### {model['model']}",
            "",
            f"Pairs: `{model['pair_count']}`; comparable: `{model['comparable_pair_count']}`. "
            "A 95% percentile bootstrap CI is shown only with at least three values.",
            "",
            "| Metric | Baseline mean | Ours mean | Mean paired delta | 95% bootstrap CI | Delta of means | Improvement |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, label, unit, _higher, supports_delta in METRICS:
            metric = model["metrics"][name]
            ci = (
                metric["mean_paired_delta_ci95"]
                if supports_delta
                else metric["ours_mean_ci95"]
            )
            ci_label = format_ci(ci, unit)
            if ci_label != "—" and not supports_delta:
                ci_label += " (Ours mean)"
            lines.append(
                f"| {label} | {format_value(metric['baseline_mean'], unit)} | "
                f"{format_value(metric['ours_mean'], unit)} | "
                f"{format_value(metric['mean_paired_delta'], unit)} | {ci_label} | "
                f"{format_percent(metric['delta_of_means_percent'])} | "
                f"{format_percent(metric['improvement_of_means_percent'])} |"
            )
        lines.append("")
    if report["warnings"]:
        lines += ["## Validation failures", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--json-name", default="pressure-analysis.json")
    parser.add_argument("--markdown-name", default="pressure-analysis.md")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    for name in (args.json_name, args.markdown_name):
        if pathlib.Path(name).name != name:
            parser.error("output names must be basenames, not paths")
    if args.json_name == args.markdown_name:
        parser.error("--json-name and --markdown-name must differ")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dirs = [path.expanduser().resolve() for path in args.model_dirs]
    if len(set(model_dirs)) != len(model_dirs):
        raise SystemExit("duplicate model directory")
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise SystemExit(f"model directory does not exist: {model_dir}")
    names = [path.name for path in model_dirs]
    if len(set(names)) != len(names):
        raise SystemExit("model directory basenames must be unique")
    if args.out_dir is None:
        parents = {path.parent for path in model_dirs}
        if len(parents) != 1:
            raise SystemExit("model directories have different parents; pass --out-dir")
        out_dir = next(iter(parents))
    else:
        out_dir = args.out_dir.expanduser().resolve()
    json_out = out_dir / args.json_name
    markdown_out = out_dir / args.markdown_name

    inputs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for model_dir in model_dirs:
        paths = sorted(model_dir.glob("baseline-r*/summary.json")) + sorted(
            model_dir.glob("ours-r*/summary.json")
        )
        if not paths:
            raise SystemExit(f"no baseline-rN/ours-rN summaries in {model_dir}")
        inputs.extend((path, model_dir) for path in paths)
    if json_out in {path for path, _ in inputs} or markdown_out in {
        path for path, _ in inputs
    }:
        raise SystemExit("refusing to overwrite an input summary")

    runs = sorted(
        (load_run(path, model_dir) for path, model_dir in inputs),
        key=lambda run: (run["model"], natural_key(run["pair_id"]), run["arm"]),
    )
    pairs, warnings = pair_runs(runs)
    for run in runs:
        warnings.extend(
            f"{run['model']}/{run['arm']}-r{run['pair_id']}: {issue}"
            for issue in run["issues"]
        )
    if not pairs:
        warnings.append("no complete baseline/Ours pair found")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "bootstrap": {
            "method": "nonparametric percentile bootstrap of the mean",
            "confidence": 0.95,
            "resamples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "minimum_pairs": 3,
        },
        "model_count": len(model_dirs),
        "run_count": len(runs),
        "pair_count": len(pairs),
        "runs": runs,
        "pairs": pairs,
        "models": aggregate_pairs(pairs, args.bootstrap_samples, args.bootstrap_seed),
        "warnings": warnings,
    }
    atomic_write(json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(markdown_out, markdown_report(report) + "\n")
    print(f"wrote {json_out}")
    print(f"wrote {markdown_out}")
    invalid = bool(warnings) or any(not pair["comparable"] for pair in pairs)
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
