#!/usr/bin/env python3
"""Analyze paired steady-state AgentReplay Dead-KV summaries.

Each input is a model directory containing ``baseline-rN/summary.json`` and
``ours-rN/summary.json``.  Only those aggregate summaries are read; private
token traces and per-request/telemetry JSONL artifacts are never opened.
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


SCHEMA_VERSION = 1
RUN_RE = re.compile(r"^(baseline|ours)-r(.+)$")

# name, label, unit, higher-is-better, paired comparison
METRICS: tuple[tuple[str, str, str, bool, bool], ...] = (
    ("completion_goodput", "Completion goodput", "tok/s", True, True),
    ("completion_request_rate", "Completion request rate", "req/s", True, True),
    ("inference_makespan", "Final inference makespan", "s", False, True),
    ("pipeline_makespan", "Final pipeline makespan", "s", False, True),
    ("inference_drain", "Inference drain after final root arrival", "s", False, True),
    ("live_revisit_cache_hit", "Live-revisit cache hit", "ratio", True, True),
    ("live_revisit_ttft_mean", "Live-revisit TTFT mean", "ms", False, True),
    ("live_revisit_ttft_p90", "Live-revisit TTFT p90", "ms", False, True),
    ("live_revisit_e2e_mean", "Live-revisit E2E mean", "ms", False, True),
    ("live_revisit_e2e_p90", "Live-revisit E2E p90", "ms", False, True),
    ("dead_hbm_auc", "Dead HBM AUC", "byte-s", False, True),
    ("dead_dram_auc", "Dead DRAM AUC", "byte-s", False, True),
    ("hbm_utilization_mean", "HBM pool utilization mean", "ratio", False, True),
    ("hbm_utilization_peak", "HBM pool utilization peak", "ratio", False, True),
    (
        "dram_utilization_mean",
        "DRAM pool utilization mean",
        "ratio",
        False,
        True,
    ),
    ("dram_utilization_peak", "DRAM pool utilization peak", "ratio", False, True),
    ("end_backlog_mean", "Unreclaimed/END backlog mean", "count", False, True),
    ("end_backlog_peak", "Unreclaimed/END backlog peak", "count", False, True),
    ("arrival_delay_mean", "Root admission delay mean", "s", False, True),
    ("arrival_delay_p90", "Root admission delay p90", "s", False, True),
    (
        "arrival_delay_growth",
        "Root admission delay late-half minus early-half",
        "s",
        False,
        True,
    ),
    (
        "arrival_delay_slope",
        "Root admission delay linear slope",
        "s/s",
        False,
        True,
    ),
    ("end_latency_mean", "SESSION_END latency mean", "ms", False, False),
    ("end_latency_p90", "SESSION_END latency p90", "ms", False, False),
    ("end_queue_delay_mean", "SESSION_END queue delay mean", "ms", False, False),
    ("end_queue_delay_p90", "SESSION_END queue delay p90", "ms", False, False),
    ("end_retries", "SESSION_END retries", "count", False, False),
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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric_value(summary: Mapping[str, Any], name: str) -> float | int | None:
    if name == "inference_drain":
        makespan = number(get_path(summary, "timings", "inference_makespan_s"))
        interval = number(get_path(summary, "workload", "arrival_interval_seconds"))
        sessions = number(get_path(summary, "workload", "session_count"))
        if makespan is None or interval is None or sessions is None or sessions <= 0:
            return None
        final_arrival = (float(sessions) - 1) * float(interval)
        return max(0.0, float(makespan) - final_arrival)
    paths = {
        "completion_goodput": (
            "measurement",
            "requests",
            "completion_accounted_goodput_tok_s",
        ),
        "completion_request_rate": (
            "measurement",
            "requests",
            "completion_request_rate_per_second",
        ),
        "inference_makespan": ("timings", "inference_makespan_s"),
        "pipeline_makespan": ("timings", "pipeline_makespan_s"),
        "live_revisit_cache_hit": (
            "measurement",
            "requests",
            "by_traffic_class",
            "live_revisit",
            "start_cohort_cache_hit",
        ),
        "live_revisit_ttft_mean": (
            "measurement",
            "requests",
            "by_traffic_class",
            "live_revisit",
            "start_cohort_ttft_ms",
            "mean",
        ),
        "live_revisit_ttft_p90": (
            "measurement",
            "requests",
            "by_traffic_class",
            "live_revisit",
            "start_cohort_ttft_ms",
            "p90",
        ),
        "live_revisit_e2e_mean": (
            "measurement",
            "requests",
            "by_traffic_class",
            "live_revisit",
            "start_cohort_e2e_ms",
            "mean",
        ),
        "live_revisit_e2e_p90": (
            "measurement",
            "requests",
            "by_traffic_class",
            "live_revisit",
            "start_cohort_e2e_ms",
            "p90",
        ),
        "dead_hbm_auc": (
            "measurement",
            "state",
            "dead_byte_seconds",
            "HBM",
        ),
        "dead_dram_auc": (
            "measurement",
            "state",
            "dead_byte_seconds",
            "DRAM",
        ),
        "hbm_utilization_mean": (
            "measurement",
            "state",
            "time_weighted_mean",
            "pool_max_subpool_utilization",
            "HBM",
        ),
        "hbm_utilization_peak": (
            "measurement",
            "state",
            "peak",
            "pool_max_subpool_utilization",
            "HBM",
        ),
        "dram_utilization_mean": (
            "measurement",
            "state",
            "time_weighted_mean",
            "pool_max_subpool_utilization",
            "DRAM",
        ),
        "dram_utilization_peak": (
            "measurement",
            "state",
            "peak",
            "pool_max_subpool_utilization",
            "DRAM",
        ),
        "end_backlog_mean": (
            "measurement",
            "state",
            "lifecycle",
            "time_weighted_mean",
            "session_end_backlog",
        ),
        "end_backlog_peak": (
            "measurement",
            "state",
            "lifecycle",
            "peak",
            "session_end_backlog",
        ),
        "arrival_delay_mean": (
            "measurement",
            "session_arrival_admission_delay_seconds",
            "mean",
        ),
        "arrival_delay_p90": (
            "measurement",
            "session_arrival_admission_delay_seconds",
            "p90",
        ),
        "arrival_delay_growth": (
            "measurement",
            "session_arrival_admission_delay_seconds",
            "late_minus_early_mean_seconds",
        ),
        "arrival_delay_slope": (
            "measurement",
            "session_arrival_admission_delay_seconds",
            "linear_slope_delay_seconds_per_scheduled_second",
        ),
        "end_latency_mean": (
            "measurement",
            "session_end",
            "latency_ms",
            "mean",
        ),
        "end_latency_p90": (
            "measurement",
            "session_end",
            "latency_ms",
            "p90",
        ),
        "end_queue_delay_mean": (
            "measurement",
            "session_end",
            "queue_delay_ms",
            "mean",
        ),
        "end_queue_delay_p90": (
            "measurement",
            "session_end",
            "queue_delay_ms",
            "p90",
        ),
        "end_retries": ("measurement", "session_end", "retry_attempts"),
    }
    path = paths.get(name)
    return number(get_path(summary, *path)) if path is not None else None


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def run_issues(summary: Mapping[str, Any], arm: str) -> list[str]:
    issues = []
    if summary.get("valid") is not True:
        issues.append(f"summary valid={summary.get('valid')!r}, expected true")
    if summary.get("mode") != arm:
        issues.append(f"summary mode={summary.get('mode')!r}, expected {arm!r}")
    recorded_issues = summary.get("issues")
    if not isinstance(recorded_issues, list) or recorded_issues:
        issues.append("summary issues are missing or non-empty")

    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        issues.append("missing configuration")
    else:
        for field in ("trace_sha256", "manifest_sha256", "salt_sha256"):
            if not valid_sha256(configuration.get(field)):
                issues.append(f"missing or invalid configuration {field}")
        for field in (
            "max_concurrency",
            "session_end_max_concurrency",
            "request_timeout_s",
            "session_end_timeout_s",
            "telemetry_interval_s",
        ):
            value = number(configuration.get(field))
            if value is None:
                issues.append(
                    f"missing configuration {field}; rerun this legacy arm "
                    "with the current steady runner"
                )
            elif value <= 0:
                issues.append(f"invalid configuration {field}")
        retries = number(configuration.get("session_end_retries"))
        retry_delay = number(configuration.get("session_end_retry_delay_s"))
        if retries is None:
            issues.append(
                "missing configuration session_end_retries; rerun this legacy "
                "arm with the current steady runner"
            )
        elif retries < 0:
            issues.append("invalid configuration session_end_retries")
        if retry_delay is None:
            issues.append(
                "missing configuration session_end_retry_delay_s; rerun this "
                "legacy arm with the current steady runner"
            )
        elif retry_delay < 0:
            issues.append("invalid configuration session_end_retry_delay_s")

    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        issues.append("missing workload")
    else:
        for field in ("session_count", "program_count", "request_count"):
            value = number(workload.get(field))
            if value is None or value <= 0:
                issues.append(f"invalid workload {field}")
        windows = workload.get("windows")
        if not isinstance(windows, Mapping):
            issues.append("missing workload windows")
        else:
            duration = number(windows.get("measurement_seconds"))
            start = number(windows.get("measurement_start_seconds"))
            end = number(windows.get("measurement_end_seconds"))
            if (
                duration is None
                or duration <= 0
                or start is None
                or end is None
                or not math.isclose(float(end) - float(start), float(duration))
            ):
                issues.append("invalid measurement window")

    measurement = summary.get("measurement")
    if not isinstance(measurement, Mapping):
        issues.append("missing measurement")
    else:
        coverage = number(get_path(measurement, "state", "coverage_fraction"))
        if coverage is None or coverage < 0.95 or coverage > 1.01:
            issues.append("measurement state coverage is outside [0.95, 1.01]")
        live_revisits = number(
            get_path(
                measurement,
                "requests",
                "by_traffic_class",
                "live_revisit",
                "requests_started",
            )
        )
        if live_revisits is None or live_revisits <= 0:
            issues.append("measurement has no live-revisit requests")

    program_count = number(get_path(summary, "workload", "program_count"))
    if program_count is not None:
        if number(summary.get("logical_ended_count")) != program_count:
            issues.append("not every workload program reached logical end")
        expected_acks = program_count if arm == "ours" else 0
        if number(summary.get("session_end_acked_count")) != expected_acks:
            issues.append("SESSION_END acknowledged-program count mismatch")

    cleanup = summary.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("ok") is not True:
        issues.append("post-measurement cleanup did not complete")

    metrics = {name: metric_value(summary, name) for name, *_ in METRICS}
    paired_required = [name for name, *_rest, paired in METRICS if paired]
    ours_required = [name for name, *_rest, paired in METRICS if not paired]
    required = paired_required + (ours_required if arm == "ours" else [])
    missing = [name for name in required if metrics[name] is None]
    if missing:
        issues.append("missing metrics: " + ", ".join(missing))
    for name in (
        "live_revisit_cache_hit",
        "hbm_utilization_mean",
        "hbm_utilization_peak",
        "dram_utilization_mean",
        "dram_utilization_peak",
    ):
        value = number(metrics.get(name))
        if value is not None and not 0 <= float(value) <= 1:
            issues.append(f"{name} outside [0, 1]")
    for name, value in metrics.items():
        numeric = number(value)
        if (
            numeric is not None
            and numeric < 0
            and name not in {"arrival_delay_growth", "arrival_delay_slope"}
        ):
            issues.append(f"{name} is negative")
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
    configuration = summary.get("configuration")
    workload = summary.get("workload")
    metrics = {name: metric_value(summary, name) for name, *_ in METRICS}
    return {
        "model": model_dir.name,
        "pair_id": pair_id,
        "arm": arm,
        "source": str(path.relative_to(model_dir)),
        "valid": not (issues := run_issues(summary, arm)),
        "issues": issues,
        "trace_sha256": get_path(configuration, "trace_sha256"),
        "manifest_sha256": get_path(configuration, "manifest_sha256"),
        "salt_sha256": get_path(configuration, "salt_sha256"),
        "configuration_sha256": (
            canonical_hash(configuration)
            if isinstance(configuration, Mapping)
            else None
        ),
        "configuration_fields": (
            dict(configuration) if isinstance(configuration, Mapping) else None
        ),
        "workload_sha256": (
            canonical_hash(workload) if isinstance(workload, Mapping) else None
        ),
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
        for field, label in (
            ("trace_sha256", "trace SHA"),
            ("manifest_sha256", "manifest SHA"),
            ("salt_sha256", "salt SHA"),
            ("workload_sha256", "workload"),
        ):
            if baseline.get(field) != ours.get(field):
                issues.append(f"{label} mismatch")
        if baseline.get("configuration_sha256") != ours.get("configuration_sha256"):
            keys = configuration_diff_keys(
                baseline.get("configuration_fields"), ours.get("configuration_fields")
            )
            issues.append("configuration mismatch: " + ", ".join(keys))
        if not baseline.get("valid") or not ours.get("valid"):
            issues.append("one or both runs failed validation")

        metrics = {}
        for name, (label, unit, higher_better, paired) in metric_specs.items():
            base = number(get_path(baseline, "metrics", name))
            treatment = number(get_path(ours, "metrics", name))
            delta = (
                float(treatment) - float(base)
                if not issues
                and paired
                and base is not None
                and treatment is not None
                else None
            )
            delta_percent = (
                100.0 * delta / float(base)
                if delta is not None and float(base) != 0
                else None
            )
            improvement = (
                delta_percent
                if higher_better
                else -delta_percent if delta_percent is not None else None
            )
            metrics[name] = {
                "label": label,
                "unit": unit,
                "higher_is_better": higher_better,
                "paired": paired,
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
                "metrics": metrics,
            }
        )
    return pairs, warnings


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


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
    models = []
    for model in sorted(by_model):
        model_pairs = by_model[model]
        comparable = [pair for pair in model_pairs if pair.get("comparable")]
        metrics = {}
        for name, label, unit, higher_better, paired in METRICS:
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
            baseline_mean = mean_or_none(baseline_values)
            ours_mean = mean_or_none(ours_values)
            mean_delta = mean_or_none(deltas)
            delta_percent = (
                100 * (ours_mean - baseline_mean) / baseline_mean
                if paired
                and baseline_mean not in (None, 0)
                and ours_mean is not None
                else None
            )
            improvement = (
                delta_percent
                if higher_better
                else -delta_percent if delta_percent is not None else None
            )
            metrics[name] = {
                "label": label,
                "unit": unit,
                "higher_is_better": higher_better,
                "paired": paired,
                "n_baseline": len(baseline_values),
                "n_ours": len(ours_values),
                "n_paired_delta": len(deltas),
                "baseline_mean": baseline_mean,
                "ours_mean": ours_mean,
                "mean_paired_delta": mean_delta,
                "delta_of_means_percent": delta_percent,
                "improvement_of_means_percent": improvement,
                "mean_paired_delta_ci95": (
                    bootstrap_mean_ci(
                        deltas,
                        bootstrap_samples,
                        metric_seed(seed, model, name, "delta"),
                    )
                    if paired
                    else None
                ),
                "ours_mean_ci95": (
                    bootstrap_mean_ci(
                        ours_values,
                        bootstrap_samples,
                        metric_seed(seed, model, name, "ours"),
                    )
                    if not paired
                    else None
                ),
            }
        models.append(
            {
                "model": model,
                "pair_count": len(model_pairs),
                "comparable_pair_count": len(comparable),
                "metrics": metrics,
            }
        )
    return models


def format_value(value: Any, unit: str) -> str:
    numeric = number(value)
    if numeric is None:
        return "—"
    numeric = float(numeric)
    if unit == "byte-s":
        return f"{numeric / 1024**2:.2f} MiB·s"
    if unit == "ratio":
        return f"{100 * numeric:.2f}%"
    if unit == "ms":
        return f"{numeric:.2f} ms"
    if unit == "s":
        return f"{numeric:.3f} s"
    if unit == "s/s":
        return f"{numeric:.4f} s/s"
    if unit == "tok/s":
        return f"{numeric:.2f} tok/s"
    if unit == "req/s":
        return f"{numeric:.3f} req/s"
    if unit == "count":
        return f"{numeric:.2f}"
    return f"{numeric:.4g}"


def format_percent(value: Any) -> str:
    numeric = number(value)
    return "—" if numeric is None else f"{float(numeric):+.2f}%"


def format_ci(ci: Any, unit: str) -> str:
    if not isinstance(ci, Mapping):
        return "—"
    low = format_value(ci.get("low"), unit)
    high = format_value(ci.get("high"), unit)
    return f"[{low}, {high}]"


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# AgentReplay steady-state Dead-KV A/B results",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Throughput is completion-accounted output goodput over the fixed measurement "
        "window. Cache hit, TTFT, and E2E use requests admitted in that window; the "
        "headline latency/cache metrics select live revisits only. Positive "
        "improvement means Ours moved in the desired direction.",
        "",
        "## Validation",
        "",
        "| Model | Pair | Arm | Valid | Source |",
        "|---|---:|---|---:|---|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['model']} | {run['pair_id']} | {run['arm']} | "
            f"{'yes' if run['valid'] else 'NO'} | `{run['source']}` |"
        )

    lines += ["", "## Paired results", ""]
    for pair in report["pairs"]:
        lines += [
            f"### {pair['model']} / r{pair['pair_id']}",
            "",
            f"Comparable: `{'yes' if pair['comparable'] else 'no'}`",
            "",
            "| Metric | Baseline | Ours | Ours − Baseline | Relative | Improvement |",
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

    lines += ["## Aggregate means and bootstrap confidence intervals", ""]
    for model in report["models"]:
        lines += [
            f"### {model['model']}",
            "",
            f"Pairs: `{model['pair_count']}`; comparable: "
            f"`{model['comparable_pair_count']}`. A 95% percentile bootstrap CI is "
            "shown with at least three values.",
            "",
            "| Metric | Baseline mean | Ours mean | Mean paired delta | 95% CI | "
            "Improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, label, unit, _higher, paired in METRICS:
            metric = model["metrics"][name]
            ci = (
                metric["mean_paired_delta_ci95"]
                if paired
                else metric["ours_mean_ci95"]
            )
            ci_text = format_ci(ci, unit)
            if ci_text != "—" and not paired:
                ci_text += " (Ours mean)"
            lines.append(
                f"| {label} | {format_value(metric['baseline_mean'], unit)} | "
                f"{format_value(metric['ours_mean'], unit)} | "
                f"{format_value(metric['mean_paired_delta'], unit)} | {ci_text} | "
                f"{format_percent(metric['improvement_of_means_percent'])} |"
            )
        lines.append("")
    if report["warnings"]:
        lines += ["## Validation failures", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--json-name", default="steady-analysis.json")
    parser.add_argument("--markdown-name", default="steady-analysis.md")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    for name in (args.json_name, args.markdown_name):
        if pathlib.Path(name).name != name:
            parser.error("output names must be basenames")
    if args.json_name == args.markdown_name:
        parser.error("JSON and Markdown output names must differ")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dirs = [path.expanduser().resolve() for path in args.model_dirs]
    if len(set(model_dirs)) != len(model_dirs):
        raise SystemExit("duplicate model directory")
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise SystemExit(f"model directory does not exist: {model_dir}")
    if len({path.name for path in model_dirs}) != len(model_dirs):
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
            raise SystemExit(f"no paired summaries found in {model_dir}")
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
