#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import tempfile
import types
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "run_agentreplay_saturated_arm.py"
SPEC = importlib.util.spec_from_file_location("saturated_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def records(program_id: str, role: str, steps: int, arrival: float) -> list[dict]:
    prompt = [1, 2, int(program_id.rsplit("-", 1)[-1])]
    result = []
    for step in range(1, steps + 1):
        output = [10 + step]
        result.append(
            {
                "program_id": program_id,
                "step": step,
                "input_ids": list(prompt),
                "forced_output_ids": output,
                "steady_role": role,
                "steady_root_id": program_id,
                "steady_is_root": True,
                "scheduled_session_arrival_s": arrival,
            }
        )
        prompt.extend(output)
    return result


class SaturatedRunnerTests(unittest.TestCase):
    def test_execute_keeps_workers_busy_and_ends_every_program(self):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class FakeDriver:
            httpx = types.SimpleNamespace(
                AsyncClient=FakeClient,
                Timeout=lambda value: value,
                Limits=lambda **kwargs: kwargs,
            )

            @staticmethod
            def _runtime_program_id(program_id, salt):
                return f"{program_id}#{salt}"

            @staticmethod
            async def _one_request(_client, _url, record, _salt):
                await asyncio.sleep(0.01)
                wanted = len(record["forced_output_ids"])
                return {
                    "ok": True,
                    "n_out": wanted,
                    "want_out": wanted,
                    "force_exact": True,
                    "prompt": len(record["input_ids"]),
                    "cached": len(record["input_ids"]) - 1,
                    "ttft_ms": 1.0,
                    "e2e_ms": 10.0,
                }

            @staticmethod
            async def _end_program_http(_client, _url, program, salt):
                await asyncio.sleep(0)
                return {
                    "program_id": program["program_id"],
                    "runtime_program_id": f"{program['program_id']}#{salt}",
                    "ok": True,
                    "end_ms": 1.0,
                }

        rows = []
        for index in range(4):
            rows.extend(records(f"steady-{index}", "churn", 2, float(index)))
        rows.extend(records("steady-4", "live", 4, 4.0))
        rows.extend(records("steady-5", "live", 4, 5.0))
        args = types.SimpleNamespace(
            salt="test",
            url="http://127.0.0.1:30001/generate",
            max_concurrency=2,
            live_revisit_every_churn=None,
            session_end_max_concurrency=1,
            request_timeout_s=10,
            session_end_timeout_s=10,
            session_end_retries=0,
            session_end_retry_delay_s=0,
            telemetry_interval_s=0.01,
            state_timeout_s=1,
            mode="ours",
        )
        state = {
            "pool_used_bytes": {tier: 0 for tier in runner.steady.TIERS},
            "pool_max_subpool_utilization": {
                tier: 0 for tier in runner.steady.TIERS
            },
            "dead_physical_bytes": {tier: 0 for tier in runner.steady.TIERS},
        }
        original_fetch = runner.steady.telemetry.fetch_json
        original_analyze = runner.steady.telemetry.analyze_state
        runner.steady.telemetry.fetch_json = lambda *_args: ({}, None, 200)
        runner.steady.telemetry.analyze_state = lambda *_args: state
        try:
            with tempfile.TemporaryDirectory() as temporary:
                result = asyncio.run(
                    runner.execute(
                        args,
                        rows,
                        FakeDriver,
                        pathlib.Path(temporary),
                    )
                )
        finally:
            runner.steady.telemetry.fetch_json = original_fetch
            runner.steady.telemetry.analyze_state = original_analyze

        self.assertFalse(result["issues"])
        self.assertEqual(result["inference"]["n_requests"], len(rows))
        self.assertEqual(result["session_end"]["n_ok"], 6)
        self.assertEqual(result["concurrency"]["peak"], 2)
        self.assertGreater(result["concurrency"]["full_concurrency_fraction"], 0.5)
        self.assertGreaterEqual(result["concurrency"]["ready_queue_peak"], 2)
        self.assertGreaterEqual(
            result["concurrency"]["waiting_live_programs_peak"], 1
        )

    def test_playlist_order_uses_original_deterministic_arrivals(self):
        programs = {
            "steady-2": records("steady-2", "churn", 2, 2.0),
            "steady-0": records("steady-0", "live", 4, 0.0),
            "steady-1": records("steady-1", "churn", 2, 1.0),
        }

        self.assertEqual(
            runner.playlist_order(programs),
            ["steady-0", "steady-1", "steady-2"],
        )

    def test_live_revisits_are_spread_by_churn_completion_count(self):
        programs = {
            **{
                f"steady-{index}": records(
                    f"steady-{index}", "churn", 2, float(index)
                )
                for index in range(8)
            },
            "steady-8": records("steady-8", "live", 4, 8.0),
            "steady-9": records("steady-9", "live", 4, 9.0),
        }

        thresholds, interval = runner.live_thresholds(programs, 8, None)

        self.assertEqual(interval, 2)
        self.assertEqual(
            [thresholds[("steady-8", step)] for step in (1, 2, 3)],
            [2, 4, 6],
        )
        self.assertEqual(
            [thresholds[("steady-9", step)] for step in (1, 2, 3)],
            [3, 5, 7],
        )

    def test_concurrency_metrics_separate_saturated_body_from_tail(self):
        metrics = runner.concurrency_metrics(
            [(0, 0), (0, 1), (0, 2), (8, 1), (10, 0)],
            makespan=10,
            target=2,
        )

        self.assertEqual(metrics["peak"], 2)
        self.assertAlmostEqual(metrics["time_weighted_mean"], 1.8)
        self.assertAlmostEqual(metrics["full_concurrency_fraction"], 0.8)
        self.assertAlmostEqual(metrics["underfilled_seconds"], 2.0)

    def test_request_metrics_use_fixed_workload_makespan(self):
        rows = [
            {
                "ok": True,
                "n_out": 100,
                "traffic_class": "churn",
            },
            {
                "ok": True,
                "n_out": 50,
                "traffic_class": "live_revisit",
                "prompt": 200,
                "cached": 150,
                "ttft_ms": 20,
                "e2e_ms": 100,
            },
        ]

        metrics = runner.request_metrics(rows, makespan=10)

        self.assertEqual(metrics["total_output_tokens"], 150)
        self.assertEqual(metrics["inference_throughput_tok_s"], 15)
        self.assertEqual(metrics["live_revisit"]["cache_hit"], 0.75)
        self.assertEqual(metrics["live_revisit"]["ttft_ms"]["mean"], 20)


if __name__ == "__main__":
    unittest.main()
