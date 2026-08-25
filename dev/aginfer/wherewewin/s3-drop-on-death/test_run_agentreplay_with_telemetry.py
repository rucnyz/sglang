#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = pathlib.Path(__file__).with_name("run_agentreplay_with_telemetry.py")
SPEC = importlib.util.spec_from_file_location("agentreplay_telemetry", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


STATE = {
    "per_rank": [
        {
            "pool_usage": {
                "HBM": {
                    "subpools": {
                        "full": {"used_bytes": 100, "cap_bytes": 1000},
                        "swa": {"used_bytes": 90, "cap_bytes": 100},
                    },
                    "token_usage": 0.9,
                },
                "DRAM": {"subpools": {"kv": {"used_bytes": 50, "cap_bytes": 2000}}},
                "DISK": {"subpools": {}},
            },
            "per_program_usage": {"p1#s": {}, "p2#s": {}},
            "units": [
                {
                    "session_ids": ["p1#s"],
                    "n_bytes": {"HBM": {"kv": 60}, "DRAM": {"kv": 20}},
                },
                {
                    "session_ids": ["p1#s", "p2#s"],
                    "n_bytes": {"HBM": {"kv": 30}, "DRAM": {"kv": 10}},
                },
                {"session_ids": [], "n_bytes": {"HBM": {"kv": 5}}},
            ],
        }
    ]
}


class StateHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps(STATE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class TelemetryTests(unittest.TestCase):
    def test_state_analysis(self):
        observed = MODULE.analyze_state(STATE, {"p1#s", "p2#s"}, {"p1#s"})
        self.assertEqual(
            observed["pool_used_bytes"], {"HBM": 190, "DRAM": 50, "DISK": 0}
        )
        self.assertEqual(observed["pool_max_subpool_utilization"]["HBM"], 0.9)
        self.assertEqual(observed["pool_reported_token_usage"]["HBM"], 0.9)
        self.assertEqual(observed["residual_holder_bytes"]["HBM"], 90)
        self.assertEqual(observed["tracked_physical_bytes"]["HBM"], 90)
        self.assertEqual(observed["dead_physical_bytes"]["HBM"], 60)
        self.assertEqual(observed["dead_unit_count"], 1)

    def test_pool_utilization_uses_max_across_ranks_and_subpools(self):
        payload = {
            "per_rank": [
                {
                    "pool_usage": {
                        "HBM": {
                            "subpools": {
                                "full": {"used_bytes": 80, "cap_bytes": 100},
                                "swa": {"used_bytes": 10, "cap_bytes": 100},
                            },
                            "token_usage": 0.77,
                        },
                        "DRAM": {"subpools": {}},
                        "DISK": {"subpools": {}},
                    },
                    "per_program_usage": {},
                    "units": [],
                },
                {
                    "pool_usage": {
                        "HBM": {
                            "subpools": {"full": {"used_bytes": 9, "cap_bytes": 10}},
                            "token_usage": 0.88,
                        },
                        "DRAM": {
                            "subpools": {
                                "full": {"used_bytes": 50, "cap_bytes": 100},
                                "small": {"used_bytes": 1, "cap_bytes": 4},
                            },
                            "token_usage": 0.45,
                        },
                        "DISK": {"subpools": {}},
                    },
                    "per_program_usage": {},
                    "units": [],
                },
            ]
        }
        observed = MODULE.analyze_state(payload, None, None)
        self.assertEqual(observed["pool_max_subpool_utilization"]["HBM"], 0.9)
        self.assertEqual(observed["pool_max_subpool_utilization"]["DRAM"], 0.5)
        self.assertEqual(observed["pool_reported_token_usage"]["HBM"], 0.88)
        self.assertEqual(observed["pool_reported_token_usage"]["DRAM"], 0.45)
        self.assertIsNone(observed["pool_max_subpool_utilization"]["DISK"])
        self.assertIsNone(observed["pool_reported_token_usage"]["DISK"])

    def test_gpu_parsers(self):
        smi = MODULE.parse_nvidia_smi("0, 1024, 2048\n1, 512, 2048\n")
        self.assertEqual(smi[0]["used_bytes"], 1024 * 1024**2)
        nvitop = MODULE.parse_nvitop(
            "|   0  GB300                 On | 00000008:06:00.0 Off | Disabled |\n"
            "| N/A   61C   P0   922W / 1400W |  221.9GiB / 277.5GiB | 100% |\n"
        )
        self.assertEqual(nvitop[0]["total_bytes"], int(277.5 * 1024**3))

    def test_runtime_program_id_is_bounded(self):
        value = MODULE.runtime_program_id("p" * 100, "salt")
        self.assertLessEqual(len(value), 64)
        self.assertEqual(value, MODULE.runtime_program_id("p" * 100, "salt"))

    def test_integration_preserves_child_rc_and_redacts_trace(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StateHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                trace = root / "trace.jsonl"
                trace.write_text(
                    json.dumps(
                        {
                            "t": 0,
                            "program_id": "p1",
                            "step": 1,
                            "input_ids": [111, 222],
                            "forced_output_ids": [333],
                        }
                    )
                    + "\n"
                )
                result = root / "result.json"
                result.write_text(
                    json.dumps(
                        {
                            "run_salt": "s",
                            "n_error": 1,
                            "n_programs": 1,
                            "len_match_rate": 0,
                            "n_requests": 1,
                        }
                    )
                )
                out_dir = root / "telemetry"
                command = [
                    sys.executable,
                    str(SCRIPT),
                    "--out-dir",
                    str(out_dir),
                    "--state-url",
                    f"http://127.0.0.1:{server.server_port}/aginfer/state",
                    "--poll-interval",
                    "0.05",
                    "--post-seconds",
                    "0",
                    "--state-timeout",
                    "1",
                    "--gpu-timeout",
                    "1",
                    "--trace",
                    str(trace),
                    "--run-salt",
                    "s",
                    "--result-json",
                    str(result),
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; sys.exit(7)",
                ]
                completed = subprocess.run(
                    command, check=False, capture_output=True, text=True
                )
                self.assertEqual(completed.returncode, 7, completed.stderr)
                summary = json.loads((out_dir / "summary.json").read_text())
                self.assertEqual(summary["child_returncode"], 7)
                self.assertFalse(summary["ended_program_ids_known"])
                telemetry = (out_dir / "telemetry.jsonl").read_text()
                self.assertNotIn("input_ids", telemetry)
                self.assertNotIn("111", telemetry)
                self.assertTrue((out_dir / "start.json").is_file())
                self.assertTrue((out_dir / "end.json").is_file())
        finally:
            server.shutdown()
            server.server_close()

    def test_successful_result_enables_final_dead_byte_classification(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StateHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                trace = root / "trace.jsonl"
                trace.write_text(
                    json.dumps(
                        {
                            "t": 0,
                            "program_id": "p1",
                            "step": 1,
                            "input_ids": [111],
                            "forced_output_ids": [222],
                        }
                    )
                    + "\n"
                )
                result = root / "result.json"
                result.write_text(
                    json.dumps(
                        {
                            "run_salt": "s",
                            "n_error": 0,
                            "n_programs": 1,
                            "len_match_rate": 1.0,
                            "n_requests": 1,
                        }
                    )
                )
                out_dir = root / "telemetry"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--out-dir",
                        str(out_dir),
                        "--state-url",
                        f"http://127.0.0.1:{server.server_port}/aginfer/state",
                        "--poll-interval",
                        "0.05",
                        "--post-seconds",
                        "0",
                        "--state-timeout",
                        "1",
                        "--gpu-timeout",
                        "1",
                        "--trace",
                        str(trace),
                        "--run-salt",
                        "s",
                        "--result-json",
                        str(result),
                        "--",
                        sys.executable,
                        "-c",
                        "pass",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                summary = json.loads((out_dir / "summary.json").read_text())
                self.assertTrue(summary["ended_program_ids_known"])
                self.assertEqual(summary["end_state"]["dead_physical_bytes"]["HBM"], 60)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
