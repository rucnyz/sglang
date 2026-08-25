#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "analyze_agentreplay_steady.py"
SPEC = importlib.util.spec_from_file_location("steady_analyzer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


HASH_A = "a" * 64
HASH_B = "b" * 64


def summary(arm: str, pair: int, offset: float = 0.0) -> dict:
    ours = arm == "ours"
    goodput = 100.0 + offset + (10.0 if ours else 0.0)
    hit = 0.70 + offset / 1000 + (0.20 if ours else 0.0)
    ttft = 300.0 + offset - (100.0 if ours else 0.0)
    backlog = 1.0 if ours else 20.0 + offset
    return {
        "schema_version": 1,
        "label": f"{arm}-r{pair}",
        "mode": arm,
        "valid": True,
        "issues": [],
        "configuration": {
            "url": "http://127.0.0.1:30001/generate",
            "max_concurrency": 8,
            "session_end_max_concurrency": 1,
            "request_timeout_s": 7200,
            "session_end_timeout_s": 120,
            "session_end_retries": 2,
            "session_end_retry_delay_s": 0.25,
            "telemetry_interval_s": 1.0,
            "trace_sha256": HASH_A,
            "manifest_sha256": HASH_B,
            "salt_sha256": hashlib_for_pair(pair),
        },
        "workload": {
            "arrival_rate_sessions_per_second": 0.1,
            "arrival_interval_seconds": 10.0,
            "arrival_duration_seconds": 2400,
            "live_fraction_requested": 0.25,
            "live_revisit_seconds": 60,
            "live_steps": 4,
            "churn_gap_seconds": 0,
            "session_count": 240,
            "program_count": 260,
            "request_count": 500,
            "role_session_counts": {"live": 60, "churn": 180},
            "role_program_counts": {"live": 60, "churn": 200},
            "windows": {
                "warmup_seconds": 300,
                "measurement_seconds": 1800,
                "cooldown_seconds": 300,
                "measurement_start_seconds": 300,
                "measurement_end_seconds": 2100,
            },
        },
        "timings": {
            "inference_makespan_s": 2800 + offset - (100 if ours else 0),
            "pipeline_makespan_s": 2810 + offset - (100 if ours else 0),
        },
        "measurement": {
            "requests": {
                "completion_accounted_goodput_tok_s": goodput,
                "completion_request_rate_per_second": 1.5 + offset / 100,
                "by_traffic_class": {
                    "live_revisit": {
                        "requests_started": 100,
                        "start_cohort_cache_hit": hit,
                        "start_cohort_ttft_ms": {
                            "n": 100,
                            "mean": ttft,
                            "p90": ttft + 50,
                        },
                        "start_cohort_e2e_ms": {
                            "n": 100,
                            "mean": ttft + 500,
                            "p90": ttft + 700,
                        },
                    }
                },
            },
            "state": {
                "coverage_fraction": 1.0,
                "dead_byte_seconds": {
                    "HBM": 0 if ours else 1000 + offset,
                    "DRAM": 0 if ours else 2000 + offset,
                },
                "time_weighted_mean": {
                    "pool_max_subpool_utilization": {
                        "HBM": 0.6 if ours else 0.9,
                        "DRAM": 0.5 if ours else 0.8,
                    }
                },
                "peak": {
                    "pool_max_subpool_utilization": {
                        "HBM": 0.7 if ours else 0.95,
                        "DRAM": 0.6 if ours else 0.9,
                    }
                },
                "lifecycle": {
                    "time_weighted_mean": {"session_end_backlog": backlog},
                    "peak": {"session_end_backlog": backlog + 2},
                },
            },
            "session_end": {
                "n": 100 if ours else 0,
                "n_ok": 100 if ours else 0,
                "n_error": 0,
                "latency_ms": (
                    {"n": 100, "mean": 20 + offset, "p90": 30 + offset}
                    if ours
                    else {"n": 0}
                ),
                "queue_delay_ms": (
                    {"n": 100, "mean": 2 + offset, "p90": 3 + offset}
                    if ours
                    else {"n": 0}
                ),
                "retry_attempts": pair if ours else 0,
            },
            "session_arrival_admission_delay_seconds": {
                "n": 180,
                "mean": 0.1 + offset / 100,
                "p90": 0.2 + offset / 100,
                "late_minus_early_mean_seconds": 0.05 + offset / 100,
                "linear_slope_delay_seconds_per_scheduled_second": (
                    0.001 + offset / 10000
                ),
            },
        },
        "logical_ended_count": 260,
        "session_end_acked_count": 260 if ours else 0,
        "cleanup": {"ok": True},
    }


def hashlib_for_pair(pair: int) -> str:
    return f"{pair:064x}"


def write_summary(model_dir: pathlib.Path, arm: str, pair: int) -> pathlib.Path:
    path = model_dir / f"{arm}-r{pair}" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary(arm, pair, pair)), encoding="utf-8")
    return path


class SteadyAnalyzerTests(unittest.TestCase):
    def test_load_and_pair_extract_headline_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = pathlib.Path(temporary) / "qwen"
            baseline = analyzer.load_run(
                write_summary(model_dir, "baseline", 1), model_dir
            )
            ours = analyzer.load_run(write_summary(model_dir, "ours", 1), model_dir)
            pairs, warnings = analyzer.pair_runs([baseline, ours])

            self.assertFalse(warnings)
            self.assertTrue(baseline["valid"])
            self.assertTrue(ours["valid"])
            self.assertTrue(pairs[0]["comparable"])
            self.assertEqual(
                pairs[0]["metrics"]["completion_goodput"][
                    "delta_ours_minus_baseline"
                ],
                10,
            )
            self.assertAlmostEqual(
                pairs[0]["metrics"]["live_revisit_cache_hit"][
                    "improvement_percent"
                ],
                100 * 0.2 / 0.701,
            )

    def test_pair_rejects_salt_or_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = pathlib.Path(temporary) / "qwen"
            baseline_path = write_summary(model_dir, "baseline", 1)
            ours_path = write_summary(model_dir, "ours", 1)
            ours_summary = json.loads(ours_path.read_text())
            ours_summary["configuration"]["salt_sha256"] = "f" * 64
            ours_summary["configuration"]["max_concurrency"] = 16
            ours_path.write_text(json.dumps(ours_summary), encoding="utf-8")
            pairs, _warnings = analyzer.pair_runs(
                [
                    analyzer.load_run(baseline_path, model_dir),
                    analyzer.load_run(ours_path, model_dir),
                ]
            )

            self.assertFalse(pairs[0]["comparable"])
            self.assertIn("salt SHA mismatch", pairs[0]["issues"])
            self.assertTrue(
                any(
                    "configuration mismatch" in issue
                    for issue in pairs[0]["issues"]
                )
            )

    def test_legacy_summary_without_timeout_fields_is_invalid(self):
        value = summary("baseline", 1)
        for field in (
            "request_timeout_s",
            "session_end_timeout_s",
            "session_end_retries",
            "session_end_retry_delay_s",
        ):
            del value["configuration"][field]

        issues = analyzer.run_issues(value, "baseline")

        self.assertTrue(any("legacy arm" in issue for issue in issues))

    def test_three_pairs_produce_paired_and_ours_only_bootstrap_ci(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = pathlib.Path(temporary) / "qwen"
            runs = []
            for pair in range(1, 4):
                for arm in ("baseline", "ours"):
                    runs.append(
                        analyzer.load_run(
                            write_summary(model_dir, arm, pair), model_dir
                        )
                    )
            pairs, warnings = analyzer.pair_runs(runs)
            aggregate = analyzer.aggregate_pairs(pairs, 500, 7)[0]

            self.assertFalse(warnings)
            self.assertEqual(aggregate["comparable_pair_count"], 3)
            self.assertIsNotNone(
                aggregate["metrics"]["completion_goodput"][
                    "mean_paired_delta_ci95"
                ]
            )
            self.assertIsNotNone(
                aggregate["metrics"]["end_latency_mean"]["ours_mean_ci95"]
            )

    def test_main_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_dir = root / "qwen"
            for pair in range(1, 4):
                write_summary(model_dir, "baseline", pair)
                write_summary(model_dir, "ours", pair)

            returncode = analyzer.main(
                [
                    str(model_dir),
                    "--out-dir",
                    str(root / "analysis"),
                    "--bootstrap-samples",
                    "100",
                ]
            )

            self.assertEqual(returncode, 0)
            report = json.loads(
                (root / "analysis" / "steady-analysis.json").read_text()
            )
            markdown = (root / "analysis" / "steady-analysis.md").read_text()
            self.assertEqual(report["pair_count"], 3)
            self.assertEqual(report["models"][0]["comparable_pair_count"], 3)
            self.assertIn("Completion goodput", markdown)
            self.assertIn("SESSION_END latency mean", markdown)


if __name__ == "__main__":
    unittest.main()
