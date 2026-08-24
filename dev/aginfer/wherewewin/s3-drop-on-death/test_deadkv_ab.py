"""Tiny stdlib mock used to smoke-test deadkv_ab.py without a GPU server."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    lock = threading.Lock()
    programs: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True})
            return
        if self.path == "/aginfer/state":
            with State.lock:
                holders = sorted(State.programs)
                units = []
                if holders:
                    units.append(
                        {
                            "hash": "shared",
                            "session_ids": holders,
                            "hit_count": len(State.seen),
                            "n_bytes": {"HBM": {"full": 4096}, "DRAM": {"full": 4096}},
                        }
                    )
                for program in holders:
                    units.append(
                        {
                            "hash": "tail-" + program,
                            "session_ids": [program],
                            "hit_count": 1,
                            "n_bytes": {"HBM": {"full": 2048}, "DRAM": {"full": 2048}},
                        }
                    )
                used = (
                    sum(
                        sum(sum(sp.values()) for sp in unit["n_bytes"].values())
                        for unit in units
                    )
                    // 2
                )
                state = {
                    "units": units,
                    "per_program_usage": {program: {} for program in holders},
                    "pool_usage": {
                        "HBM": {
                            "subpools": {
                                "full": {
                                    "used_bytes": used,
                                    "cap_bytes": 1 << 30,
                                    "available_bytes": (1 << 30) - used,
                                    "evictable_bytes": used,
                                    "page_bytes": 65536,
                                    "decode_bytes_per_token": 1024,
                                }
                            }
                        },
                        "DRAM": {
                            "subpools": {
                                "full": {
                                    "used_bytes": used,
                                    "cap_bytes": 1 << 31,
                                    "available_bytes": (1 << 31) - used,
                                    "evictable_bytes": used,
                                    "page_bytes": 65536,
                                    "decode_bytes_per_token": 1024,
                                }
                            }
                        },
                        "DISK": {
                            "subpools": {
                                "full": {
                                    "used_bytes": 0,
                                    "cap_bytes": 0,
                                    "available_bytes": 0,
                                    "evictable_bytes": 0,
                                    "page_bytes": 65536,
                                    "decode_bytes_per_token": 1024,
                                }
                            }
                        },
                    },
                }
            self._json(200, {"per_rank": [state]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/flush_cache"):
            with State.lock:
                State.programs.clear()
                State.seen.clear()
            self._json(200, {"ok": True})
            return
        if self.path == "/aginfer/session_end":
            program = payload["program_id"]
            with State.lock:
                existed = program in State.programs
                State.programs.pop(program, None)
            self._json(
                200,
                {
                    "ok": True,
                    "program_id": program,
                    "per_rank": [
                        {
                            "ok": True,
                            "deferred": False,
                            "status": "completed" if existed else "already_absent",
                            "matched_nodes": int(existed),
                            "holders_removed": int(existed),
                            "remaining_nodes": 0,
                            "released_nodes": int(existed),
                            "released_hbm_tokens": int(existed),
                            "released_dram_tokens": int(existed),
                        }
                    ],
                },
            )
            return
        if self.path == "/generate":
            program = payload["program_id"]
            tokens = payload.get("input_ids") or [1, 2, 3]
            key = tuple(tokens)
            with State.lock:
                cached = (
                    len(tokens)
                    if key in State.seen
                    else (len(tokens) // 2 if State.programs else 0)
                )
                State.programs[program] = list(tokens)
                State.seen.add(key)
            if payload.get("stream"):
                event = {
                    "text": "ok",
                    "output_ids": [42],
                    "meta_info": {
                        "prompt_tokens": len(tokens),
                        "cached_tokens": cached,
                        "completion_tokens": 1,
                    },
                }
                body = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(
                    200,
                    {
                        "text": "ok",
                        "output_ids": [42],
                        "meta_info": {
                            "prompt_tokens": len(tokens),
                            "cached_tokens": cached,
                            "completion_tokens": 1,
                        },
                    },
                )
            return
        self._json(404, {"error": "not found"})


def run_mock_test() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory(prefix="deadkv-ab-mock-") as directory:
        command = [
            sys.executable,
            str(pathlib.Path(__file__).with_name("deadkv_ab.py")),
            "--backend",
            "direct",
            "--server-url",
            f"http://127.0.0.1:{server.server_port}",
            "--repeats",
            "2",
            "--live-sessions",
            "2",
            "--dead-per-epoch",
            "2",
            "--epochs",
            "2",
            "--concurrency",
            "2",
            "--shared-pages",
            "0",
            "--tail-pages",
            "2",
            "--max-tokens",
            "1",
            "--warmup-requests",
            "1",
            "--retention-seconds",
            "0.05",
            "--poll-interval",
            "0.01",
            "--artifact-dir",
            directory,
            "--confirm-dedicated-server",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)
        summaries = list(pathlib.Path(directory).glob("*/summary.json"))
        assert len(summaries) == 1
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        assert summary["status"] == "passed"
        assert len(summary["trials"]) == 4
        baseline = summary["aggregate"]["conditions"]["baseline"]
        ours = summary["aggregate"]["conditions"]["ours"]
        assert baseline["end_dead_bytes_total"]["median"] > 0
        assert ours["end_dead_bytes_total"]["median"] == 0

        bootstrap_path = pathlib.Path(directory) / "paired_bootstrap.json"
        analyzed = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("analyze_results.py")),
                str(summaries[0]),
                "--samples",
                "100",
                "--output",
                str(bootstrap_path),
            ],
            text=True,
            capture_output=True,
        )
        if analyzed.returncode:
            print(analyzed.stdout)
            print(analyzed.stderr, file=sys.stderr)
            raise SystemExit(analyzed.returncode)
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        assert bootstrap["pair_count"] == 2
        assert "dead_kv_reduction_pct" in bootstrap["metrics"]

        verifier_artifacts = pathlib.Path(directory) / "e2e"
        verified = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("verify_dead_kv_e2e.py")),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--shared-repeats",
                "1",
                "--tail-repeats",
                "1",
                "--artifact-dir",
                str(verifier_artifacts),
                "--confirm-dedicated-server",
            ],
            text=True,
            capture_output=True,
        )
        if verified.returncode:
            print(verified.stdout)
            print(verified.stderr, file=sys.stderr)
            raise SystemExit(verified.returncode)
        verifier_result = json.loads(
            (verifier_artifacts / "result.json").read_text(encoding="utf-8")
        )
        assert verifier_result["status"] == "PASS"
        print(completed.stdout, end="")
        print(verified.stdout, end="")
        print("mock smoke test passed")
    server.shutdown()


def test_deadkv_ab_mock() -> None:
    run_mock_test()


def main() -> int:
    run_mock_test()
    return 0


if __name__ == "__main__":
    main()
