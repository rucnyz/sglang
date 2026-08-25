#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("steady_builder", "build_agentreplay_steady_trace.py")
runner = load_module("steady_runner", "run_agentreplay_steady_arm.py")


def program(program_id: str, unique: int, steps: int = 4) -> list[dict]:
    prompt = [10, 11, unique]
    records = []
    for step in range(1, steps + 1):
        output = [100 + unique + step]
        records.append(
            {
                "t": float(step - 1),
                "program_id": program_id,
                "step": step,
                "input_ids": list(prompt),
                "forced_output_ids": output,
            }
        )
        prompt.extend(output)
        prompt.append(200 + unique + step)
    return records


class SteadyTraceBuilderTests(unittest.TestCase):
    def test_deterministic_schedule_has_distinct_prefixes_and_revisits(self):
        rows = [*program("a", 1), *program("b", 2)]
        programs = builder.program_index(rows)
        first, manifest = builder.build_schedule(
            programs,
            arrival_rate=2.0,
            total_seconds=5.0,
            live_fraction=0.5,
            live_steps=4,
            revisit_seconds=7.0,
            churn_gap_seconds=0.0,
            seed=9,
            identity_width=8,
            identity_insertion_offset=2,
        )
        second, second_manifest = builder.build_schedule(
            programs,
            arrival_rate=2.0,
            total_seconds=5.0,
            live_fraction=0.5,
            live_steps=4,
            revisit_seconds=7.0,
            churn_gap_seconds=0.0,
            seed=9,
            identity_width=8,
            identity_insertion_offset=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(manifest["session_count"], 10)
        self.assertEqual(sum(manifest["role_session_counts"].values()), 10)
        self.assertEqual(manifest["role_session_counts"], {"live": 5, "churn": 5})
        self.assertEqual(manifest["live_fraction_actual"], 0.5)
        by_program = runner.source_programs(first)
        first_inputs = [
            tuple(records[0]["input_ids"]) for records in by_program.values()
        ]
        self.assertEqual(len(first_inputs), len(set(first_inputs)))
        for records in by_program.values():
            if records[0]["steady_role"] == "live":
                self.assertEqual(len(records), 4)
                self.assertTrue(
                    all(row["tool_gap_after"] == 7.0 for row in records[:-1])
                )
            else:
                self.assertEqual(len(records), 4)
            for previous, current in zip(records, records[1:]):
                prefix = previous["input_ids"] + previous["forced_output_ids"]
                self.assertEqual(current["input_ids"][: len(prefix)], prefix)

    def test_root_bundle_clone_preserves_parent_topology(self):
        root = program("root", 1)
        child = program("child", 2, steps=2)
        for row in child:
            row["parent_program_id"] = "root"
            row["spawned_at_step"] = 2
            row["background_spawn"] = False
        programs = builder.program_index([*root, *child])
        _parents, children = builder.program_graph(programs)
        source_ids = builder.root_closure("root", children)
        cloned = builder.clone_bundle(
            programs,
            source_program_ids=source_ids,
            source_root_id="root",
            replica_index=3,
            arrival_seconds=5.0,
            role="churn",
            live_steps=4,
            live_revisit_seconds=7.0,
            churn_gap_seconds=0.0,
            insertion_offset=2,
            identity=[10, 11, 10],
        )

        by_program = runner.source_programs(cloned)
        self.assertEqual(len(by_program), 2)
        roots = [
            program_id
            for program_id, records in by_program.items()
            if records[0].get("parent_program_id") is None
        ]
        self.assertEqual(len(roots), 1)
        child_rows = next(
            records
            for records in by_program.values()
            if records[0].get("parent_program_id") is not None
        )
        self.assertEqual(child_rows[0]["parent_program_id"], roots[0])
        self.assertEqual(child_rows[0]["steady_root_id"], roots[0])


class SteadyMetricTests(unittest.TestCase):
    def test_request_metrics_use_fixed_window_and_separate_roles(self):
        rows = [
            {
                "role": "live",
                "traffic_class": "live_revisit",
                "ok": True,
                "started_elapsed_seconds": 2.0,
                "completed_elapsed_seconds": 3.0,
                "n_out": 30,
                "prompt": 100,
                "cached": 90,
                "ttft_ms": 10,
                "e2e_ms": 100,
            },
            {
                "role": "churn",
                "traffic_class": "churn",
                "ok": True,
                "started_elapsed_seconds": 4.0,
                "completed_elapsed_seconds": 6.0,
                "n_out": 20,
                "prompt": 100,
                "cached": 20,
                "ttft_ms": 30,
                "e2e_ms": 200,
            },
            {
                "role": "churn",
                "traffic_class": "churn",
                "ok": True,
                "started_elapsed_seconds": 7.5,
                "completed_elapsed_seconds": 8.5,
                "n_out": 40,
                "prompt": 100,
                "cached": 10,
                "ttft_ms": 50,
                "e2e_ms": 300,
            },
        ]
        metrics = runner.request_window_metrics(rows, 2.0, 8.0)

        self.assertEqual(metrics["completion_accounted_output_tokens"], 50)
        self.assertAlmostEqual(metrics["completion_accounted_goodput_tok_s"], 50 / 6)
        self.assertEqual(metrics["requests_started"], 3)
        self.assertAlmostEqual(metrics["start_cohort_cache_hit"], 0.4)
        self.assertEqual(
            metrics["by_traffic_class"]["live_revisit"]["requests_started"], 1
        )
        self.assertEqual(
            metrics["by_traffic_class"]["churn"]["requests_started"], 2
        )

    def test_state_metrics_integrate_dead_bytes_and_occupancy(self):
        def sample(at: float, dead: float, utilization: float) -> dict:
            return {
                "elapsed_seconds": at,
                "logical_ended_programs": dead,
                "session_end_acked_programs": 0,
                "session_end_backlog": dead,
                "state": {
                    "dead_physical_bytes": {
                        "HBM": dead,
                        "DRAM": dead * 2,
                        "DISK": 0,
                    },
                    "pool_used_bytes": {
                        "HBM": dead * 10,
                        "DRAM": dead * 20,
                        "DISK": 0,
                    },
                    "pool_max_subpool_utilization": {
                        "HBM": utilization,
                        "DRAM": utilization / 2,
                        "DISK": 0,
                    },
                },
            }

        metrics = runner.state_window_metrics(
            [sample(0, 10, 0.5), sample(5, 20, 0.8), sample(10, 30, 0.9)],
            2,
            8,
        )

        self.assertEqual(metrics["covered_seconds"], 6)
        self.assertEqual(metrics["coverage_fraction"], 1)
        self.assertEqual(metrics["dead_byte_seconds"]["HBM"], 90)
        self.assertEqual(
            metrics["time_weighted_mean"]["dead_physical_bytes"]["HBM"], 15
        )
        self.assertEqual(
            metrics["peak"]["pool_max_subpool_utilization"]["HBM"], 0.8
        )
        self.assertEqual(
            metrics["lifecycle"]["time_weighted_mean"]["session_end_backlog"],
            15,
        )


if __name__ == "__main__":
    unittest.main()
