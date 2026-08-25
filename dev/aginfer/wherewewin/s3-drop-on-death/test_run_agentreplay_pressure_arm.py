#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "run_agentreplay_pressure_arm.py"


def rows(program_id: str, steps: range, secret: int) -> list[dict]:
    prompt = [secret]
    result = []
    for step in range(1, 5):
        output = [secret + step]
        if step in steps:
            result.append(
                {
                    "t": float(step),
                    "program_id": program_id,
                    "step": step,
                    "input_ids": list(prompt),
                    "forced_output_ids": output,
                }
            )
        prompt = [*prompt, *output, secret + step + 10]
    return result


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


FAKE_AGENTREPLAY = r"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request


def post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request) as response:
        return json.loads(response.read())


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
replay = subparsers.add_parser("replay")
replay.add_argument("--trace", required=True)
replay.add_argument("--url", required=True)
replay.add_argument("--salt", required=True)
replay.add_argument("--label", required=True)
replay.add_argument("--out", required=True)
replay.add_argument("--emit-session-end", action="store_true")
args, _ = parser.parse_known_args()
records = [json.loads(line) for line in pathlib.Path(args.trace).read_text().splitlines() if line]
programs = sorted({record["program_id"] for record in records})
runtime_ids = [program + "#" + args.salt for program in programs]
base_url = args.url.rsplit("/", 1)[0]
for program_id in runtime_ids:
    post(base_url + "/fake/register", {"program_id": program_id})
if args.emit_session_end:
    for program_id in runtime_ids:
        post(base_url + "/aginfer/session_end", {"program_id": program_id})
phase = (
    "terminal"
    if "-terminal-wave-" in args.label
    else args.label.rsplit("-", 1)[-1]
)
cache_hit = {"seed": 0.5, "terminal": 0.8, "probe": 0.95}[phase]
session_end = {
    "enabled": args.emit_session_end,
    "n": len(programs) if args.emit_session_end else 0,
    "n_ok": len(programs) if args.emit_session_end else 0,
    "n_error": 0,
    "remaining_nodes": 0,
    "latency_ms": {"mean": 2.0, "p50": 2.0, "p90": 3.0, "p99": 4.0},
}
result = {
    "label": args.label,
    "run_salt": args.salt,
    "n_requests": len(records),
    "n_ok": len(records),
    "n_error": 0,
    "n_programs": len(programs),
    "len_match_rate": 1.0,
    "force_exact_rate": 1.0,
    "force_exact_failures": 0,
    "force_exact_missing": 0,
    "cache_hit": cache_hit,
    "total_prompt_tokens": len(records) * 100,
    "total_cached_tokens": int(len(records) * 100 * cache_hit),
    "cached_tokens_details": {
        "device": int(len(records) * 100 * cache_hit) - len(records),
        "host": len(records),
        "storage": 0,
    },
    "total_out_tokens": len(records),
    "inference_makespan_s": 1.0,
    "pipeline_makespan_s": 1.1,
    "inference_throughput_tok_s": 10.0,
    "pipeline_throughput_tok_s": 9.0,
    "ttft_ms": {"mean": 1.0, "p50": 1.0, "p90": 2.0, "p99": 3.0},
    "session_end": session_end,
}
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result) + "\n")
"""


class CacheState:
    def __init__(self) -> None:
        self.holders: set[str] = set()
        self.lock = threading.Lock()

    def register(self, program_id: str) -> None:
        with self.lock:
            self.holders.add(program_id)

    def end(self, program_id: str) -> None:
        with self.lock:
            self.holders.discard(program_id)

    def clear(self) -> None:
        with self.lock:
            self.holders.clear()

    def state(self) -> dict:
        with self.lock:
            holders = sorted(self.holders)
        return {
            "per_rank": [
                {
                    "pool_usage": {
                        "HBM": {
                            "subpools": {
                                "kv": {
                                    "used_bytes": 100 * len(holders),
                                    "cap_bytes": 10_000,
                                }
                            }
                        },
                        "DRAM": {
                            "subpools": {
                                "kv": {
                                    "used_bytes": 50 * len(holders),
                                    "cap_bytes": 20_000,
                                }
                            }
                        },
                        "DISK": {"subpools": {}},
                    },
                    "per_program_usage": {program_id: {} for program_id in holders},
                    "units": [
                        {
                            "session_ids": [program_id],
                            "n_bytes": {"HBM": {"kv": 100}, "DRAM": {"kv": 50}},
                        }
                        for program_id in holders
                    ],
                }
            ]
        }


class PressureArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = CacheState()
        cache = self.cache

        class Handler(BaseHTTPRequestHandler):
            def send_json(self, value: dict) -> None:
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/aginfer/state":
                    self.send_json(cache.state())
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/fake/register":
                    cache.register(payload["program_id"])
                    self.send_json({"ok": True})
                elif self.path == "/aginfer/session_end":
                    cache.end(payload["program_id"])
                    self.send_json(
                        {
                            "ok": True,
                            "per_rank": [
                                {"ok": True, "deferred": False, "remaining_nodes": 0}
                            ],
                        }
                    )
                elif self.path.startswith("/flush_cache"):
                    cache.clear()
                    body = b"Cache flushed."
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def run_arm(
        self, root: pathlib.Path, mode: str, *, terminal_waves: int = 1
    ) -> tuple[subprocess.CompletedProcess, dict]:
        fake_root = root / "fake-agentreplay"
        package = fake_root / "agentreplay"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(FAKE_AGENTREPLAY, encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        nvidia_smi = fake_bin / "nvidia-smi"
        nvidia_smi.write_text("#!/bin/sh\necho '0, 100, 1000'\n", encoding="utf-8")
        nvidia_smi.chmod(0o755)

        seed = root / "live-seed.jsonl"
        terminals = []
        probe = root / "live-probe.jsonl"
        write_jsonl(
            seed,
            [
                *rows("live-a", range(1, 4), 314159265),
                *rows("live-b", range(1, 4), 271828182),
            ],
        )
        for wave_number in range(1, terminal_waves + 1):
            terminal = root / f"terminal-churn-{wave_number:03d}.jsonl"
            write_jsonl(
                terminal,
                rows(
                    f"ended-program-sentinel-{wave_number}",
                    range(1, 5),
                    161803398 + wave_number * 100,
                ),
            )
            terminals.append(terminal)
        write_jsonl(
            probe,
            [
                *rows("live-a", range(4, 5), 314159265),
                *rows("live-b", range(4, 5), 271828182),
            ],
        )

        out_dir = root / f"{mode}-r1"
        command = [
            sys.executable,
            str(SCRIPT),
            "--mode",
            mode,
            "--live-seed",
            str(seed),
            "--live-probe",
            str(probe),
            "--out-dir",
            str(out_dir),
            "--salt",
            "pair-r1",
            "--label",
            mode + "-r1",
            "--python",
            sys.executable,
            "--agentreplay-root",
            str(fake_root),
            "--url",
            f"http://127.0.0.1:{self.server.server_port}/generate",
            "--barrier-seconds",
            "0.05",
            "--state-wait-timeout-s",
            "2",
            "--state-poll-interval-s",
            "0.01",
        ]
        for terminal in terminals:
            command.extend(["--terminal-churn", str(terminal)])
        environment = dict(os.environ)
        environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        return completed, summary

    def test_baseline_retains_terminal_then_explicitly_cleans(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, summary = self.run_arm(pathlib.Path(temporary), "baseline")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["end_state"]["dead_physical_bytes"]["HBM"], 100)
            self.assertEqual(
                summary["states"]["before_live_probe"]["dead_physical_bytes"]["HBM"],
                100,
            )
            self.assertEqual(
                summary["states"]["before_live_probe"]["holder_program_count"], 3
            )
            self.assertTrue(summary["cleanup"]["explicit_end_attempted"])
            self.assertEqual(summary["cleanup"]["programs_ok"], 3)
            self.assertFalse(self.cache.holders)
            serialized = json.dumps(summary)
            self.assertNotIn("input_ids", serialized)
            self.assertNotIn("314159265", serialized)
            self.assertNotIn("pair-r1", serialized)
            self.assertNotIn("live-a", serialized)
            self.assertNotIn("ended-program-sentinel", serialized)

    def test_distinct_terminal_waves_accumulate_only_in_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, summary = self.run_arm(
                pathlib.Path(temporary), "baseline", terminal_waves=2
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["configuration"]["terminal_wave_count"], 2)
            self.assertEqual(summary["phase_manifest"]["terminal_wave_count"], 2)
            self.assertEqual(summary["phase_manifest"]["terminal_program_count"], 2)
            self.assertEqual(summary["phases"]["terminal"]["result"]["n_requests"], 8)
            self.assertEqual(len(summary["phases"]["terminal_waves"]), 2)
            self.assertEqual(
                summary["states"]["after_terminal_wave_001"]["live"][
                    "tracked_programs_present_count"
                ],
                2,
            )
            self.assertEqual(
                summary["states"]["before_live_probe"]["dead_physical_bytes"]["HBM"],
                200,
            )
            self.assertEqual(summary["cleanup"]["programs_ok"], 4)
            serialized = json.dumps(summary)
            self.assertNotIn("pair-r1", serialized)
            self.assertNotIn("ended-program-sentinel", serialized)

    def test_rejects_duplicate_token_payload_waves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fake_root = root / "fake-agentreplay"
            fake_root.mkdir()
            seed = root / "live-seed.jsonl"
            probe = root / "live-probe.jsonl"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(
                seed,
                [
                    *rows("live-a", range(1, 4), 314159265),
                    *rows("live-b", range(1, 4), 271828182),
                ],
            )
            write_jsonl(
                probe,
                [
                    *rows("live-a", range(4, 5), 314159265),
                    *rows("live-b", range(4, 5), 271828182),
                ],
            )
            payload = rows("terminal-a", range(1, 5), 161803398)
            write_jsonl(first, payload)
            renamed = [dict(row, program_id="terminal-b") for row in payload]
            write_jsonl(second, renamed)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "baseline",
                    "--live-seed",
                    str(seed),
                    "--terminal-churn",
                    str(first),
                    "--terminal-churn",
                    str(second),
                    "--live-probe",
                    str(probe),
                    "--out-dir",
                    str(root / "out"),
                    "--salt",
                    "private-salt",
                    "--label",
                    "duplicate",
                    "--agentreplay-root",
                    str(fake_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicates an earlier token payload", completed.stderr)

    def test_ours_reclaims_each_distinct_terminal_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, summary = self.run_arm(
                pathlib.Path(temporary), "ours", terminal_waves=2
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(summary["valid"])
            terminal = summary["phases"]["terminal"]["result"]
            self.assertEqual(terminal["n_programs"], 2)
            self.assertEqual(terminal["session_end"]["n_ok"], 2)
            self.assertEqual(
                terminal["session_end"]["latency_ms"]["aggregation"],
                "weighted_by_completed_calls",
            )
            self.assertNotIn("p90", terminal["session_end"]["latency_ms"])
            self.assertEqual(
                terminal["session_end"]["worst_wave_latency_ms"]["p90"], 3.0
            )
            for wave_number in (1, 2):
                state = summary["states"][f"after_terminal_wave_{wave_number:03d}"][
                    "workload"
                ]
                self.assertEqual(state["dead_physical_bytes"]["HBM"], 0)
            self.assertFalse(summary["cleanup"]["explicit_end_attempted"])
            self.assertFalse(self.cache.holders)

    def test_ours_ends_terminal_and_live_programs(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, summary = self.run_arm(pathlib.Path(temporary), "ours")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["end_state"]["dead_physical_bytes"]["HBM"], 0)
            self.assertEqual(
                summary["states"]["before_live_probe"]["dead_physical_bytes"]["HBM"],
                0,
            )
            self.assertEqual(
                summary["states"]["after_live_probe"]["tracked_programs_present_count"],
                0,
            )
            self.assertFalse(summary["cleanup"]["explicit_end_attempted"])
            self.assertFalse(self.cache.holders)


if __name__ == "__main__":
    unittest.main()
