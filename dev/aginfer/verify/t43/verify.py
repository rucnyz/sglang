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
    # B8 + B9: every payload must carry the full contract field set.
    # Asserting these here means every subsequent stage gets the
    # presence check for free, so an impl regression (e.g. someone
    # drops the pid field) is caught by ANY stage, not just S0.
    for key in ("reason", "timestamp_unix", "timestamp_iso", "pid",
                "traceback", "context"):
        if key not in payload:
            raise StageFail(
                f"forensic payload missing {key!r}; keys={list(payload)}"
            )
    if payload["reason"] != reason:
        raise StageFail(
            f"forensic payload reason mismatch: "
            f"want {reason!r}, got {payload['reason']!r}"
        )
    if not isinstance(payload["pid"], int) or payload["pid"] <= 0:
        raise StageFail(
            f"forensic payload pid not a positive int: {payload['pid']!r}"
        )
    iso = payload["timestamp_iso"]
    if not isinstance(iso, str) or not re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", iso):
        raise StageFail(
            f"timestamp_iso does not look like YYYY-MM-DDTHH:MM:SS: {iso!r}"
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
    object) must NOT cause fatal() to itself raise.  Recursive
    ``_to_jsonable`` should record repr(value) and dump the rest of
    the context.  Two sub-cases:

      (a) top-level bad value: ``bad=sock`` direct kwarg
      (b) nested bad value: ``nested={"deep": [{"bad": sock}]}`` —
          exercises the dict→list→dict recursion path; a regression
          that special-cases only top-level kwargs would silently
          drop nested badness on the floor.
    """
    # (a) top-level bad value
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
                f"(a) top-level: expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        payload = _assert_forensic_file(
            data_dir, "unser_check", result.stderr,
            expected_context_subset={"good": "still here"},
        )
        if "socket" not in str(payload["context"]["bad"]):
            raise StageFail(
                f"(a) context.bad should be a repr of the socket; "
                f"got {payload['context']['bad']!r}"
            )

    # (b) nested bad value
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_subprocess_fatal(
            "from daemon._fatal import fatal\n"
            "import socket\n"
            "sock = socket.socket()\n"
            "fatal('unser_nested',\n"
            "      nested={'deep': [{'bad': sock, 'keep': 'me'}]},\n"
            "      sibling={'plain': 42})\n",
            data_dir=data_dir,
        )
        if result.returncode != 1:
            raise StageFail(
                f"(b) nested: expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        payload = _assert_forensic_file(
            data_dir, "unser_nested", result.stderr,
        )
        deep_dict = payload["context"]["nested"]["deep"][0]
        if "socket" not in str(deep_dict["bad"]):
            raise StageFail(
                f"(b) context.nested.deep[0].bad should be socket repr; "
                f"got {deep_dict['bad']!r}"
            )
        if deep_dict.get("keep") != "me":
            raise StageFail(
                f"(b) sibling-in-same-dict 'keep' got dropped: "
                f"{deep_dict!r}"
            )
        if payload["context"]["sibling"] != {"plain": 42}:
            raise StageFail(
                f"(b) JSON-safe sibling kwarg got mangled: "
                f"{payload['context']['sibling']!r}"
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
    """A link with peak_bw_bps <= 0 is a deployment bug
    (DESIGN §10 "Required positivity").  Tests both 0 and -1 to
    defend against a regression that tightens to ``>= 0``."""
    for bad_value in (0, -1_000_000_000):
        state = _seed_valid_state()
        state["link_stats"]["HBM->DRAM"]["peak_bw_bps"] = bad_value
        with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
            data_dir = Path(td)
            result = _run_build_paper_state(state, data_dir=data_dir)
            if result.returncode != 1:
                raise StageFail(
                    f"peak_bw_bps={bad_value}: expected exit=1; "
                    f"got {result.returncode}; stderr={result.stderr}"
                )
            if "UNEXPECTED-SUCCESS" in result.stderr:
                raise StageFail(
                    f"peak_bw_bps={bad_value}: build_paper_state did "
                    f"not raise/fatal"
                )
            _assert_forensic_file(
                data_dir, "peak_bw_bps_non_positive", result.stderr,
            )


def stage_5_missing_required_fields_parametrized() -> None:
    """Missing any of the 7 top-level required fields →
    fatal('missing_state_field') with context.missing naming the
    dropped field.

    Parametrised across all 7 so an impl regression (e.g. someone
    shrinks the required-list to 4 fields) is caught immediately
    instead of slipping through a single-field smoke."""
    required = (
        "pool_usage", "link_stats", "tier_holding_cost",
        "throughput_ema", "per_program_usage", "units", "time_counter",
    )
    for field in required:
        state = _seed_valid_state()
        del state[field]
        with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
            data_dir = Path(td)
            result = _run_build_paper_state(state, data_dir=data_dir)
            if result.returncode != 1:
                raise StageFail(
                    f"missing {field!r}: expected exit=1; "
                    f"got {result.returncode}; stderr={result.stderr}"
                )
            if "UNEXPECTED-SUCCESS" in result.stderr:
                raise StageFail(
                    f"missing {field!r}: build_paper_state did not "
                    f"raise/fatal"
                )
            payload = _assert_forensic_file(
                data_dir, "missing_state_field", result.stderr,
            )
            if payload["context"].get("missing") != field:
                raise StageFail(
                    f"missing {field!r}: context.missing mismatch — "
                    f"got {payload['context'].get('missing')!r}"
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


def stage_9_h_max_per_byte_sec_non_positive() -> None:
    """DESIGN §10 line 2319 positivity invariant: every
    ``tier_holding_cost[τ][sp].h_max_per_byte_sec > 0``.  Zero is a
    deployment bug (operator forgot to set ``h_max`` for a subpool)."""
    state = _seed_valid_state()
    state["tier_holding_cost"]["HBM"]["attn"]["h_max_per_byte_sec"] = 0.0
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1 on h_max=0; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail("build_paper_state did not raise/fatal")
        _assert_forensic_file(
            data_dir, "holding_cost_non_positive", result.stderr,
        )
    # Negative is also a deployment bug (and must not regress to
    # ``>=0`` silently if someone tightens the check later).
    state["tier_holding_cost"]["HBM"]["attn"]["h_max_per_byte_sec"] = -1.0e-9
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1 on h_max=-1e-9; got {result.returncode}"
            )
        _assert_forensic_file(
            data_dir, "holding_cost_non_positive", result.stderr,
        )


def stage_10_prefill_bps_positivity_conditional() -> None:
    """DESIGN §10 line 2319: ``prefill_bps > 0`` ONCE ANY PREFILL HAS
    RUN.  Three sub-cases:

      (a) prefill_bps == 0 + no units + time_counter == 0
          → startup; NO fatal.
      (b) prefill_bps < 0 → fatal regardless (negative throughput is
          structurally nonsense; can never be a startup state).
      (c) prefill_bps == 0 + units present → fatal (we have evidence
          that prefill has run because there are committed units,
          but the EMA reports zero throughput → bug).
    """
    # (a) startup: prefill_bps=0 + no units → NO fatal
    state_a = _seed_valid_state()
    state_a["throughput_ema"]["prefill_bps"] = 0.0
    # seed already has units=[] and time_counter=0
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state_a, data_dir=data_dir)
        if result.returncode != 0:
            raise StageFail(
                f"(a) startup: expected exit=0; got {result.returncode}; "
                f"stderr={result.stderr}"
            )

    # (b) negative: unconditional fatal
    state_b = _seed_valid_state()
    state_b["throughput_ema"]["prefill_bps"] = -1.0
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state_b, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"(b) negative: expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        _assert_forensic_file(
            data_dir, "prefill_bps_non_positive_with_traffic", result.stderr,
        )

    # (c) zero with units present: fatal
    state_c = _seed_valid_state()
    state_c["throughput_ema"]["prefill_bps"] = 0.0
    state_c["units"] = [{
        "hash": "node-1",
        "residence": ["HBM"],
        "n_tokens": 256,
        "n_bytes": {"HBM": {"attn": 4096}},
        "last_access_time": 0,
        "hit_count": 1,
        "session_ids": [],
    }]
    state_c["time_counter"] = 1
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(state_c, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"(c) zero+units: expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        _assert_forensic_file(
            data_dir, "prefill_bps_non_positive_with_traffic", result.stderr,
        )


def stage_11_cross_rank_n_bytes_disagreement() -> None:
    """DESIGN §6 line 736: ``units[i].n_bytes[τ][sp]`` is identical
    across ranks (derived from architecture); cross-rank disagreement
    is a deployment bug → fatal().

    Pre-T43 ``_flatten_per_rank`` took ``max(rank0, rank1)`` over
    disagreeing values, silently absorbing the bug.  This stage feeds
    a 2-rank state where the same hash reports
    ``n_bytes[HBM][attn] = 4096`` on rank-0 and ``8192`` on rank-1
    and asserts fatal."""
    rank0 = _seed_valid_state()
    rank0["units"] = [{
        "hash": "node-replica",
        "residence": ["HBM"],
        "n_tokens": 256,
        "n_bytes": {"HBM": {"attn": 4096}},
        "last_access_time": 0,
        "hit_count": 1,
        "session_ids": [],
    }]
    rank0["time_counter"] = 1
    rank1 = copy.deepcopy(rank0)
    rank1["units"][0]["n_bytes"]["HBM"]["attn"] = 8192  # disagrees
    multi_rank = {"per_rank": [rank0, rank1]}
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        result = _run_build_paper_state(multi_rank, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1 on n_bytes disagreement; "
                f"got {result.returncode}; stderr={result.stderr}"
            )
        if "UNEXPECTED-SUCCESS" in result.stderr:
            raise StageFail(
                "build_paper_state silently absorbed a cross-rank "
                "n_bytes disagreement instead of fataling"
            )
        _assert_forensic_file(
            data_dir, "n_bytes_disagreement_across_ranks", result.stderr,
        )


def stage_12_exception_context_traceback() -> None:
    """``fatal()`` called inside an ``except`` block must capture the
    active exception's traceback via ``sys.exc_info()``, not just
    ``format_stack()`` (the no-exception fallback).  DESIGN §10
    contract: forensic payload includes 'the Python traceback'.

    Probe: ``raise RuntimeError`` → catch → ``fatal('exc_check')``.
    Expect payload.traceback to contain both the exception class +
    message AND the raising frame name."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        body = (
            "from daemon._fatal import fatal\n"
            "def the_raising_frame():\n"
            "    raise RuntimeError('boom-from-test')\n"
            "try:\n"
            "    the_raising_frame()\n"
            "except RuntimeError:\n"
            "    fatal('exc_check', note='inside-except')\n"
        )
        result = _run_subprocess_fatal(body, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        payload = _assert_forensic_file(
            data_dir, "exc_check", result.stderr,
            expected_context_subset={"note": "inside-except"},
        )
        tb_joined = "\n".join(payload["traceback"])
        for needle in ("RuntimeError", "boom-from-test",
                       "the_raising_frame"):
            if needle not in tb_joined:
                raise StageFail(
                    f"traceback missing {needle!r}; trace was:\n{tb_joined}"
                )


def stage_13_full_context_contract_round_trip() -> None:
    """DESIGN §10 L2302 enumerates four contract context keys: 'event,
    state, candidates, dp_inputs'.  This stage feeds a REAL ``Event``
    (frozen dataclass with an ``EventKind`` Enum), a nested ``state``
    dict, a list of dataclass candidates, and a numeric ``dp_inputs``
    dict — then verifies every field round-trips correctly through
    ``_to_jsonable``.

    A regression that (e.g.) drops the dataclass-asdict path or
    leaves Enums as ``EventKind.MEMORY_PRESSURE`` strings would be
    caught here but missed by the flat-primitive Stage 0."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        body = (
            "import dataclasses\n"
            "from daemon._fatal import fatal\n"
            "from daemon.events import Event, EventKind\n"
            "\n"
            "@dataclasses.dataclass\n"
            "class CandidateStub:\n"
            "    hash: str\n"
            "    source_tier: str\n"
            "    target_tier: str\n"
            "    expected_bytes: int\n"
            "\n"
            "evt = Event(\n"
            "    kind=EventKind.MEMORY_PRESSURE,\n"
            "    session='prog-x',\n"
            "    payload={'detail': 'test', 'occ': 0.92},\n"
            ")\n"
            "state = {\n"
            "    'pool_usage': {'HBM': {'subpools': {\n"
            "        'attn': {'cap_bytes': 65536, 'used_bytes': 60000}}}},\n"
            "    'time_counter': 42,\n"
            "}\n"
            "candidates = [\n"
            "    CandidateStub('h1', 'HBM', 'DRAM', 4096),\n"
            "    CandidateStub('h2', 'DRAM', 'HBM', 8192),\n"
            "]\n"
            "dp_inputs = {\n"
            "    'bytes_needed': {'HBM:attn': 12288},\n"
            "    'cap_left':     {'DRAM:attn': 1048576},\n"
            "    'bucket_size':  4096,\n"
            "    'dp_size':      37,\n"
            "}\n"
            "fatal('full_ctx_check',\n"
            "      event=evt, state=state,\n"
            "      candidates=candidates, dp_inputs=dp_inputs)\n"
        )
        result = _run_subprocess_fatal(body, data_dir=data_dir)
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr}"
            )
        payload = _assert_forensic_file(
            data_dir, "full_ctx_check", result.stderr,
        )
        ctx = payload["context"]
        # ---- event: dataclass + Enum coercion ----
        if ctx["event"].get("kind") != "memory_pressure":
            raise StageFail(
                f"event.kind should be Enum.value 'memory_pressure'; "
                f"got {ctx['event'].get('kind')!r}"
            )
        if ctx["event"].get("session") != "prog-x":
            raise StageFail(f"event.session: got {ctx['event'].get('session')!r}")
        if ctx["event"].get("payload") != {"detail": "test", "occ": 0.92}:
            raise StageFail(f"event.payload: got {ctx['event'].get('payload')!r}")
        # ---- state: deep nested dict ----
        if ctx["state"]["pool_usage"]["HBM"]["subpools"]["attn"]["cap_bytes"] != 65536:
            raise StageFail(f"state deep cap_bytes: {ctx['state']!r}")
        if ctx["state"]["time_counter"] != 42:
            raise StageFail(f"state.time_counter: {ctx['state'].get('time_counter')!r}")
        # ---- candidates: list of dataclass via asdict() ----
        if not isinstance(ctx["candidates"], list) or len(ctx["candidates"]) != 2:
            raise StageFail(f"candidates shape: {ctx['candidates']!r}")
        c0 = ctx["candidates"][0]
        if (c0.get("hash") != "h1" or c0.get("source_tier") != "HBM"
                or c0.get("target_tier") != "DRAM"
                or c0.get("expected_bytes") != 4096):
            raise StageFail(f"candidate[0] round-trip: {c0!r}")
        # ---- dp_inputs: numeric dict ----
        if ctx["dp_inputs"].get("bytes_needed") != {"HBM:attn": 12288}:
            raise StageFail(f"dp_inputs.bytes_needed: {ctx['dp_inputs']!r}")
        if ctx["dp_inputs"].get("bucket_size") != 4096:
            raise StageFail(f"dp_inputs.bucket_size: {ctx['dp_inputs']!r}")
        if ctx["dp_inputs"].get("dp_size") != 37:
            raise StageFail(f"dp_inputs.dp_size: {ctx['dp_inputs']!r}")


def stage_14_unwritable_data_dir_degraded_path() -> None:
    """If ``$AGINFER_DATA_DIR`` is unwritable (or points at a non-
    directory like ``/proc/cmdline``), ``fatal()`` must NOT itself
    raise; it must log the full payload to stderr (the degraded-log
    branch) and still ``sys.exit(1)``.  Forensic preservation under
    bug conditions includes "the file system is broken" conditions.
    """
    # Use /proc/cmdline (regular file) as the data_dir.  mkdir(
    # /proc/cmdline/forensic, parents=True) will raise NotADirectoryError
    # since the parent is not a directory — exercising the OSError
    # branch in ``fatal()``.
    if not Path("/proc/cmdline").exists():
        # Non-Linux sandbox: skip with a soft pass instead of failing
        # the suite for an env-specific reason.
        print(f"  (skip) /proc/cmdline unavailable; degraded-path probe needs Linux proc")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_AGINFER_ROOT)
    env["AGINFER_DATA_DIR"] = "/proc/cmdline"
    result = subprocess.run(
        [_PYTHON, "-c",
         "from daemon._fatal import fatal\n"
         "fatal('unwritable_dir_check', note='degraded')\n"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 1:
        raise StageFail(
            f"expected exit=1; got {result.returncode}; "
            f"stderr={result.stderr}"
        )
    # Must mention the reason in stderr (fallback log line includes it).
    if "unwritable_dir_check" not in result.stderr:
        raise StageFail(
            f"stderr missing reason marker; stderr={result.stderr}"
        )
    # Fallback log line names the payload — must include both the
    # 'no forensic file written' marker AND the supplied note kwarg
    # (proving the payload was actually emitted, not just the
    # reason).
    if "no forensic file written" not in result.stderr:
        raise StageFail(
            f"stderr missing degraded marker; stderr={result.stderr}"
        )
    if "degraded" not in result.stderr:
        raise StageFail(
            f"stderr missing payload-note kwarg; stderr={result.stderr}"
        )


def stage_15_concurrent_fatals_no_clobber() -> None:
    """Filename design ``<reason>_<ns_ts>_<pid>.json`` must prevent
    two concurrent fatals (e.g. multi-rank race) from overwriting
    each other's forensic file.  Spawn 2 subprocesses on the SAME
    data_dir simultaneously; expect 2 distinct files."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t43_") as td:
        data_dir = Path(td)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_AGINFER_ROOT)
        env["AGINFER_DATA_DIR"] = str(data_dir)
        body = (
            "from daemon._fatal import fatal\n"
            "fatal('race_test', who='subprocess')\n"
        )
        procs = [
            subprocess.Popen(
                [_PYTHON, "-c", body],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                raise StageFail(
                    "concurrent fatal subprocess timed out"
                )
            if p.returncode != 1:
                raise StageFail(
                    f"concurrent fatal exit code: got {p.returncode}"
                )
        files = sorted((data_dir / "forensic").glob("race_test_*.json"))
        if len(files) != 2:
            raise StageFail(
                f"expected 2 distinct forensic files, got {len(files)}: "
                f"{[f.name for f in files]}"
            )
        # Pid must differ between the two files (the two subprocesses
        # have distinct PIDs by construction).
        pids = [json.loads(f.read_text())["pid"] for f in files]
        if pids[0] == pids[1]:
            raise StageFail(f"two forensic files share a pid: {pids}")


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
    ("4  peak_bw_bps non-positive (0 and negative)",
                                              stage_4_peak_bw_bps_zero),
    ("5  missing required fields (parametrized × 7)",
                                              stage_5_missing_required_fields_parametrized),
    ("6  per_rank empty list",                stage_6_per_rank_empty),
    ("7  unsupported_tree_cache field",       stage_7_unsupported_tree_cache),
    ("9  h_max_per_byte_sec non-positive (DESIGN §10 positivity)",
                                              stage_9_h_max_per_byte_sec_non_positive),
    ("10 prefill_bps positivity (conditional on prefill having run)",
                                              stage_10_prefill_bps_positivity_conditional),
    ("11 cross-rank n_bytes disagreement (DESIGN §6 L736)",
                                              stage_11_cross_rank_n_bytes_disagreement),
    ("12 exception-context traceback (sys.exc_info path)",
                                              stage_12_exception_context_traceback),
    ("13 full context contract round-trip (event/state/candidates/dp_inputs)",
                                              stage_13_full_context_contract_round_trip),
    ("14 unwritable data dir degraded path",  stage_14_unwritable_data_dir_degraded_path),
    ("15 concurrent fatals no clobber",       stage_15_concurrent_fatals_no_clobber),
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
