"""T43 verify — `fatal(reason, **context)` helper + call sites.

In-process / subprocess hybrid.  ``fatal()`` itself terminates the
process (``sys.exit(1)``), so every stage that exercises it runs in a
subprocess and inspects:

  * exit code == 1
  * stderr has a critical-level line naming the forensic file path
  * the forensic JSON file exists under ``<data_dir>/forensic/``
  * the JSON deserialises to a dict with the contract keys
    (``reason``, ``timestamp_unix``, ``traceback``, ``context``) and
    that the supplied ``event``/``state``/``candidates``/``dp_inputs``
    /extra-kwargs round-trip into ``context``

Then the existing daemon call sites are exercised by feeding malformed
state into ``build_paper_state``: cross-rank subpool key mismatch,
``peak_bw_bps == 0``, missing required field — each should subprocess-
fatal with a recognisable reason.

Usage:
    python dev/aginfer/verify/t43/verify.py
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent          # dev/aginfer
_SGLANG_ROOT = _AGINFER_ROOT.parent.parent   # repo root
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_PYTHON = sys.executable


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------------- helpers


def _run_subprocess_fatal(
    body: str,
    *,
    data_dir: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a Python snippet that is expected to call fatal()."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_AGINFER_ROOT)
    env["AGINFER_DATA_DIR"] = str(data_dir)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [_PYTHON, "-c", body],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_forensic_file(
    data_dir: Path,
    reason: str,
    stderr: str,
    expected_context_subset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assert that exactly one forensic file matches ``reason`` and that
    its path appears in ``stderr``.  Returns the parsed JSON."""
    forensic_dir = data_dir / "forensic"
    if not forensic_dir.exists():
        raise StageFail(f"forensic dir not created at {forensic_dir}")
    matches = sorted(forensic_dir.glob(f"{reason}_*.json"))
    if not matches:
        raise StageFail(
            f"no forensic file for reason {reason!r} in {forensic_dir} "
            f"(found: {list(forensic_dir.iterdir())})"
        )
    if len(matches) > 1:
        raise StageFail(
            f"expected exactly one forensic file for {reason!r}, "
            f"got {len(matches)}: {[m.name for m in matches]}"
        )
    forensic_path = matches[0]
    # The file path must appear in stderr (the fatal-level log line
    # promised by §10).
    if str(forensic_path) not in stderr:
        raise StageFail(
            f"stderr does not name the forensic path {forensic_path}; "
            f"stderr was:\n{stderr}"
        )
    payload = json.loads(forensic_path.read_text())
    for key in ("reason", "timestamp_unix", "traceback", "context"):
        if key not in payload:
            raise StageFail(
                f"forensic payload missing {key!r}; keys={list(payload)}"
            )
    if payload["reason"] != reason:
        raise StageFail(
            f"forensic payload reason mismatch: "
            f"want {reason!r}, got {payload['reason']!r}"
        )
    if expected_context_subset:
        for k, want in expected_context_subset.items():
            if k not in payload["context"]:
                raise StageFail(
                    f"context missing key {k!r}; context={payload['context']}"
                )
            got = payload["context"][k]
            if got != want:
                raise StageFail(
                    f"context[{k!r}] mismatch: want {want!r}, got {got!r}"
                )
    return payload


# ----------------------------------------------------------------- stages


def stage_0_helper_contract() -> None:
    """Direct call: fatal('schema_sanity', foo=1, bar=[1,2,3]) →
    exit 1; <data_dir>/forensic/schema_sanity_*.json contains
    {reason, timestamp_unix, traceback, context: {foo:1, bar:[1,2,3]}}.
    stderr names the file path with a CRITICAL-level marker."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_subprocess_fatal(
            "from daemon._fatal import fatal\n"
            "fatal('schema_sanity', foo=1, bar=[1, 2, 3])\n",
            data_dir=data_dir,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        _assert_forensic_file(
            data_dir, "schema_sanity", result.stderr,
            expected_context_subset={"foo": 1, "bar": [1, 2, 3]},
        )
        # CRITICAL-level marker (§10: "logs a fatal-level line").
        if "CRITICAL" not in result.stderr and "fatal" not in result.stderr.lower():
            raise StageFail(
                f"stderr does not look like a fatal-level log line: "
                f"{result.stderr}"
            )


def stage_1_traceback_captured() -> None:
    """fatal() captures the call-site traceback even though no exception
    was raised.  We snake-stack through `outer → inner → fatal` and
    assert both frame names appear in the dumped traceback."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_subprocess_fatal(
            "from daemon._fatal import fatal\n"
            "def outer_frame_name():\n"
            "    inner_frame_name()\n"
            "def inner_frame_name():\n"
            "    fatal('tb_check')\n"
            "outer_frame_name()\n",
            data_dir=data_dir,
        )
        if result.returncode != 1:
            raise StageFail(f"expected exit=1; got {result.returncode}")
        payload = _assert_forensic_file(
            data_dir, "tb_check", result.stderr
        )
        tb = "\n".join(payload["traceback"])
        for needle in ("outer_frame_name", "inner_frame_name"):
            if needle not in tb:
                raise StageFail(
                    f"traceback does not contain frame {needle!r}; "
                    f"traceback was:\n{tb}"
                )


def stage_2_unserialisable_context_falls_back_to_repr() -> None:
    """A context value that is not JSON-serialisable (e.g. a socket
    object) must NOT cause fatal() to itself raise.  It should record
    repr(value) and dump the rest of the context.  Forensic
    preservation under bug conditions is the whole point."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_subprocess_fatal(
            "from daemon._fatal import fatal\n"
            "import socket\n"
            "sock = socket.socket()\n"
            "fatal('unser_check', bad=sock, good='still here')\n",
            data_dir=data_dir,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        payload = _assert_forensic_file(
            data_dir, "unser_check", result.stderr,
            expected_context_subset={"good": "still here"},
        )
        if "socket" not in str(payload["context"]["bad"]):
            raise StageFail(
                f"context.bad should be a repr of the socket; "
                f"got {payload['context']['bad']!r}"
            )


# ---------- call-site stages (kv_scheduler) ----------


def _seed_valid_state() -> Dict[str, Any]:
    """Minimal happy-path single-rank state-dump dict that
    ``build_paper_state`` accepts.  Stages 3+ mutate copies of this to
    drive each fatal path."""
    return {
        "pool_usage": {
            "HBM":  {"subpools": {"attn": {
                "used_bytes": 1024, "cap_bytes": 65536,
                "available_bytes": 64512, "evictable_bytes": 0,
                "page_bytes": 16,
            }}},
            "DRAM": {"subpools": {"attn": {
                "used_bytes": 0, "cap_bytes": 1048576,
                "available_bytes": 1048576, "evictable_bytes": 0,
                "page_bytes": 16,
            }}},
            "DISK": {"subpools": {"attn": {
                "used_bytes": 0, "cap_bytes": 0,
                "available_bytes": 0, "evictable_bytes": 0,
                "page_bytes": 16,
            }}},
        },
        "link_stats": {
            "HBM->DRAM": {
                "peak_bw_bps": 12_000_000_000,
                "recent_throughput_bps": 0,
                "time_since_last_sample_s": 99.0,
            },
            "DRAM->HBM": {
                "peak_bw_bps": 12_000_000_000,
                "recent_throughput_bps": 0,
                "time_since_last_sample_s": 99.0,
            },
            "DRAM->DISK": {
                "peak_bw_bps": 1_000_000_000,
                "recent_throughput_bps": 0,
                "time_since_last_sample_s": 99.0,
            },
            "DISK->DRAM": {
                "peak_bw_bps": 1_000_000_000,
                "recent_throughput_bps": 0,
                "time_since_last_sample_s": 99.0,
            },
        },
        "tier_holding_cost": {
            "HBM":  {"attn": {"h_max_per_byte_sec": 1.0e-9}},
            "DRAM": {"attn": {"h_max_per_byte_sec": 1.0e-10}},
            "DISK": {"attn": {"h_max_per_byte_sec": 1.0e-11}},
        },
        "throughput_ema": {
            "prefill_bps": 1.0e8,
            "decode_per_program": {},
        },
        "per_program_usage": {},
        "units": [],
        "time_counter": 0,
    }


_BUILD_PAPER_STATE_HARNESS = """\
import json, sys
from daemon.kv_scheduler import build_paper_state
from daemon.events import Event, EventKind
from daemon.program_tracker import ProgramTracker
state_json = json.loads(sys.argv[1])
evt = Event(
    kind=EventKind.MEMORY_PRESSURE,
    session=None,
    payload={},
)
build_paper_state(
    state_json,
    event=evt,
    tracker=ProgramTracker(),
    unknown_tier_log=set(),
)
print("UNEXPECTED-SUCCESS", file=sys.stderr)
"""


def _run_build_paper_state(
    state_json: Dict[str, Any], *, data_dir: Path
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_AGINFER_ROOT)
    env["AGINFER_DATA_DIR"] = str(data_dir)
    return subprocess.run(
        [_PYTHON, "-c", _BUILD_PAPER_STATE_HARNESS, json.dumps(state_json)],
        env=env, capture_output=True, text=True, timeout=30,
    )


def stage_3_cross_rank_subpool_key_mismatch() -> None:
    """Multi-rank state with rank-0 HBM subpools = {attn} and rank-1
    HBM subpools = {attn, moe_expert} → fatal('subpool_key_mismatch')."""
    rank0 = _seed_valid_state()
    rank1 = copy.deepcopy(rank0)
    rank1["pool_usage"]["HBM"]["subpools"]["moe_expert"] = {
        "used_bytes": 0, "cap_bytes": 4096,
        "available_bytes": 4096, "evictable_bytes": 0, "page_bytes": 16,
    }
    multi_rank = {"per_rank": [rank0, rank1]}
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(multi_rank, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail("build_paper_state did not raise/fatal")
        _assert_forensic_file(
            data_dir, "subpool_key_mismatch_across_ranks", result.stderr,
        )


def stage_4_peak_bw_bps_zero() -> None:
    """A link with peak_bw_bps == 0 is a deployment bug
    (DESIGN §10 "Required positivity")."""
    state = _seed_valid_state()
    state["link_stats"]["HBM->DRAM"]["peak_bw_bps"] = 0
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail("build_paper_state did not raise/fatal")
        _assert_forensic_file(
            data_dir, "peak_bw_bps_non_positive", result.stderr,
        )


def stage_5_missing_throughput_ema() -> None:
    """Missing ``throughput_ema`` block → fatal('missing_state_field')."""
    state = _seed_valid_state()
    del state["throughput_ema"]
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail("build_paper_state did not raise/fatal")
        _assert_forensic_file(
            data_dir, "missing_state_field", result.stderr,
        )


def stage_6_per_rank_empty() -> None:
    """``per_rank=[]`` is a deployment bug, not a workload reality."""
    multi_rank = {"per_rank": []}
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(multi_rank, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail("build_paper_state did not raise/fatal")
        _assert_forensic_file(
            data_dir, "per_rank_empty", result.stderr,
        )


def stage_7_unsupported_tree_cache() -> None:
    """``unsupported_tree_cache`` field in state → fatal."""
    state = _seed_valid_state()
    state["unsupported_tree_cache"] = "HiRadixCache"
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        _assert_forensic_file(
            data_dir, "unsupported_tree_cache", result.stderr,
        )


def stage_8_happy_path_does_not_fatal() -> None:
    """Sanity: the seed-valid state does NOT trigger fatal() — i.e. we
    have not over-tightened the new checks into the green path."""
    state = _seed_valid_state()
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 0:
            raise StageFail(
                f"expected exit=0 on happy path; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        forensic_dir = data_dir / "forensic"
        if forensic_dir.exists() and any(forensic_dir.iterdir()):
            raise StageFail(
                f"happy path created a forensic file: "
                f"{list(forensic_dir.iterdir())}"
            )


# ----------------------------------------------------------------- run


_STAGES = [
    ("0  fatal helper contract",              stage_0_helper_contract),
    ("1  traceback captured (no exception)",  stage_1_traceback_captured),
    ("2  unserialisable context falls back to repr",
                                              stage_2_unserialisable_context_falls_back_to_repr),
    ("3  cross-rank subpool key mismatch",    stage_3_cross_rank_subpool_key_mismatch),
    ("4  peak_bw_bps non-positive",           stage_4_peak_bw_bps_zero),
    ("5  missing throughput_ema field",       stage_5_missing_throughput_ema),
    ("6  per_rank empty list",                stage_6_per_rank_empty),
    ("7  unsupported_tree_cache field",       stage_7_unsupported_tree_cache),
    ("8  happy-path sanity (no fatal)",       stage_8_happy_path_does_not_fatal),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT43 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT43 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
