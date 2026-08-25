#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("analyze_agentreplay_pressure.py")


def phase_result(
    phase: str,
    arm: str,
    pair_number: int,
    requests: int,
    programs: int,
) -> dict:
    end_enabled = arm == "ours" and phase in {"terminal", "probe"}
    session_end = {
        "enabled": end_enabled,
        "n": programs if end_enabled else 0,
        "n_ok": programs if end_enabled else 0,
        "n_error": 0,
        "remaining_nodes": 0,
        "retry_attempts": pair_number % 2 if end_enabled else 0,
        "latency_ms": (
            {
                "mean": 4.0 + pair_number,
                "p50": 3.0 + pair_number,
                "p90": 7.0 + pair_number,
                "p99": 9.0 + pair_number,
            }
            if end_enabled
            else {"n": 0}
        ),
    }
    return {
        "n_requests": requests,
        "n_ok": requests,
        "n_error": 0,
        "n_programs": programs,
        "len_match_rate": 1.0,
        "force_exact_rate": 1.0,
        "force_exact_failures": 0,
        "force_exact_missing": 0,
        "cache_hit": (
            0.90 + (0.03 if arm == "ours" else 0.0) if phase == "probe" else 0.8
        ),
        "ttft_ms": {
            "mean": 20.0 + pair_number - (2.0 if arm == "ours" else 0.0),
            "p50": 18.0 + pair_number - (2.0 if arm == "ours" else 0.0),
            "p90": 30.0 + pair_number - (2.0 if arm == "ours" else 0.0),
        },
        "inference_throughput_tok_s": (
            100.0 + pair_number + (10.0 if arm == "ours" else 0.0)
        ),
        "session_end": session_end,
    }


def summary(arm: str, pair_number: int) -> dict:
    seed = phase_result("seed", arm, pair_number, 6, 2)
    terminal = phase_result("terminal", arm, pair_number, 12, 3)
    probe = phase_result("probe", arm, pair_number, 2, 2)
    dead = 0 if arm == "ours" else 100 + pair_number * 10
    used_hbm = 800 if arm == "ours" else 900
    used_dram = 600 if arm == "ours" else 700
    return {
        "prompt": "must-not-leak",
        "input_ids": [314159265],
        "valid": True,
        "mode": arm,
        "child_returncode": 0,
        "issues": [],
        "run_salt": f"pair-{pair_number}",
        "trace": {"sha256": f"terminal-{pair_number}"},
        "configuration": {
            "url": "http://127.0.0.1:30001/generate",
            "max_concurrency": 4,
            "barrier_seconds": 30.0,
            "private_note": "must-not-leak-config",
        },
        "phase_manifest": {
            "live_program_count": 2,
            "terminal_program_count": 3,
            "all_program_count": 5,
            "live_seed_requests": 6,
            "terminal_requests": 12,
            "live_probe_requests": 2,
            "traces": {
                "live_seed": {"sha256": f"seed-{pair_number}"},
                "terminal_churn": {"sha256": f"terminal-{pair_number}"},
                "live_probe": {"sha256": f"probe-{pair_number}"},
            },
        },
        "phases": {
            "seed": {"returncode": 0, "issues": [], "result": seed},
            "terminal": {
                "returncode": 0,
                "issues": [],
                "result": terminal,
                "telemetry_summary": {
                    "child_returncode": 0,
                    "state_error_count": 0,
                },
            },
            "probe": {"returncode": 0, "issues": [], "result": probe},
        },
        "states": {
            "before_live_probe": {
                "dead_physical_bytes": {"HBM": dead, "DRAM": dead * 2},
                "pool_used_bytes": {"HBM": used_hbm, "DRAM": used_dram},
                "pool_cap_bytes": {"HBM": 1000, "DRAM": 1000},
                "pool_max_subpool_utilization": {
                    "HBM": 0.83 if arm == "ours" else 0.91,
                    "DRAM": 0.62 if arm == "ours" else 0.74,
                },
            }
        },
        "agentreplay_result": terminal,
        "live_probe_result": probe,
        "cleanup": {
            "explicit_end_attempted": arm == "baseline",
            "all_end_calls_ok": True,
            "programs_targeted": 5 if arm == "baseline" else 0,
            "programs_ok": 5 if arm == "baseline" else 0,
            "final_state": {"unit_count": 0, "program_usage_count": 0},
        },
    }


def write_run(model_dir: pathlib.Path, arm: str, pair_number: int, value: dict) -> None:
    run_dir = model_dir / f"{arm}-r{pair_number}"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(value) + "\n", encoding="utf-8")


class PressureAnalyzerTests(unittest.TestCase):
    def test_three_pairs_emit_paired_bootstrap_and_redact_raw_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_dir = root / "deployment-a"
            for pair_number in range(1, 4):
                write_run(
                    model_dir, "baseline", pair_number, summary("baseline", pair_number)
                )
                write_run(model_dir, "ours", pair_number, summary("ours", pair_number))
            output = root / "report"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(model_dir),
                    "--out-dir",
                    str(output),
                    "--bootstrap-samples",
                    "500",
                    "--bootstrap-seed",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report_text = (output / "pressure-analysis.json").read_text(
                encoding="utf-8"
            )
            report = json.loads(report_text)
            self.assertEqual(report["pair_count"], 3)
            self.assertTrue(all(pair["comparable"] for pair in report["pairs"]))
            first = report["pairs"][0]["metrics"]
            self.assertEqual(
                first["pre_probe_dead_hbm_bytes"]["delta_ours_minus_baseline"],
                -110.0,
            )
            self.assertAlmostEqual(
                first["pre_probe_pool_hbm_utilization"]["delta_ours_minus_baseline"],
                -0.08,
            )
            aggregate = report["models"][0]["metrics"]
            self.assertIsNotNone(
                aggregate["pre_probe_dead_hbm_bytes"]["mean_paired_delta_ci95"]
            )
            self.assertIsNotNone(
                aggregate["terminal_end_latency_mean_ms"]["ours_mean_ci95"]
            )
            self.assertNotIn("input_ids", report_text)
            self.assertNotIn("must-not-leak", report_text)
            self.assertNotIn("pair-1", report_text)
            markdown = (output / "pressure-analysis.md").read_text(encoding="utf-8")
            self.assertIn("Pre-probe dead HBM", markdown)
            self.assertIn("95% bootstrap CI", markdown)

    def test_mismatched_pair_is_nonzero_and_not_comparable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_dir = root / "deployment-b"
            baseline = summary("baseline", 1)
            ours = summary("ours", 1)
            del baseline["states"]["before_live_probe"]["pool_max_subpool_utilization"]
            del ours["states"]["before_live_probe"]["pool_max_subpool_utilization"]
            ours["run_salt"] = "different-salt"
            ours["phase_manifest"]["traces"]["live_probe"]["sha256"] = "different"
            ours["configuration"]["max_concurrency"] = 8
            ours["valid"] = False
            write_run(model_dir, "baseline", 1, baseline)
            write_run(model_dir, "ours", 1, ours)
            output = root / "report"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(model_dir), "--out-dir", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            report = json.loads((output / "pressure-analysis.json").read_text())
            pair = report["pairs"][0]
            self.assertFalse(pair["comparable"])
            issues = "; ".join(pair["issues"])
            self.assertIn("phase trace SHA mismatch", issues)
            self.assertIn("run salt fingerprint mismatch", issues)
            self.assertIn("configuration mismatch: max_concurrency", issues)
            self.assertIn("failed validation", issues)
            utilization = pair["metrics"]["pre_probe_pool_hbm_utilization"]
            self.assertEqual(utilization["baseline"], 0.9)
            self.assertEqual(utilization["ours"], 0.8)


if __name__ == "__main__":
    unittest.main()
