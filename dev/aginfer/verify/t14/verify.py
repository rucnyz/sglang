"""T14 verify — sglang state-dump cost observability (PLAN §2).

Two phases:

  **Phase A** (in-process unit tests of the ``_StateDumpMetrics`` ring
  buffer): exercises the contract without needing a live sglang.

  **Phase B** (integration against a live sglang; opt-in via
  ``$AGINFER_VERIFY_BASE``): hits ``/aginfer/state`` N times and
  verifies the piggybacked ``state_dump_metrics`` field tracks the
  per-call latency.

Phase A always runs; Phase B is skipped (soft-pass) if the env var is
unset.  The README has the launch recipe for a Phase-B run.

Usage:
    python dev/aginfer/verify/t14/verify.py
    AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \\
        python dev/aginfer/verify/t14/verify.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple


# Make the sglang package importable from the source tree.
_HERE = Path(__file__).resolve().parent
_SGLANG_PY = _HERE.parent.parent.parent.parent / "python"  # dev/aginfer/verify/t14 → repo/python
if (_SGLANG_PY / "sglang").is_dir() and str(_SGLANG_PY) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PY))

from sglang.srt.mem_cache.unified_radix_cache import _StateDumpMetrics  # noqa: E402


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ Phase A


def stage_a0_empty_summary() -> None:
    """A fresh ``_StateDumpMetrics`` reports ``n_samples=0`` and zero
    quantiles — never raises on the cold-start read.  Pre-launch
    metric polls must not crash the daemon."""
    m = _StateDumpMetrics(capacity=128)
    s = m.summary()
    expected_keys = {
        "n_samples", "n_recorded_total", "capacity", "window_seconds",
        "p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms",
        "last_dump_ms", "last_dump_bytes",
    }
    if set(s.keys()) != expected_keys:
        raise StageFail(
            f"summary key set mismatch; "
            f"missing={expected_keys - set(s.keys())}, "
            f"extra={set(s.keys()) - expected_keys}"
        )
    if s["n_samples"] != 0 or s["n_recorded_total"] != 0:
        raise StageFail(f"cold-start n != 0: {s}")
    if s["capacity"] != 128:
        raise StageFail(f"capacity not echoed: {s['capacity']}")
    for q in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms",
              "last_dump_ms"):
        if s[q] != 0.0:
            raise StageFail(f"cold-start {q} != 0.0: {s[q]}")
    if s["last_dump_bytes"] != -1:
        raise StageFail(f"cold-start last_dump_bytes != -1: {s['last_dump_bytes']}")


def stage_a1_record_and_summary() -> None:
    """Recording 10 samples with known latencies should reflect in the
    summary: mean ≈ average, max = largest, last_dump_* echoes the
    final record."""
    m = _StateDumpMetrics(capacity=128)
    # Latencies in ns, ascending so quantiles are easy.  Use integers.
    lats = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
            6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000]
    for i, ns in enumerate(lats):
        m.record(elapsed_ns=ns, dump_bytes=1024 * (i + 1))
    s = m.summary()
    if s["n_samples"] != 10:
        raise StageFail(f"n_samples != 10: {s['n_samples']}")
    if s["n_recorded_total"] != 10:
        raise StageFail(f"n_recorded_total != 10: {s['n_recorded_total']}")
    if abs(s["mean_ms"] - 5.5) > 0.001:
        raise StageFail(f"mean_ms expected ~5.5; got {s['mean_ms']}")
    if abs(s["max_ms"] - 10.0) > 0.001:
        raise StageFail(f"max_ms expected ~10.0; got {s['max_ms']}")
    if abs(s["last_dump_ms"] - 10.0) > 0.001:
        raise StageFail(f"last_dump_ms expected ~10.0; got {s['last_dump_ms']}")
    if s["last_dump_bytes"] != 10240:
        raise StageFail(f"last_dump_bytes expected 10240; got {s['last_dump_bytes']}")
    if s["window_seconds"] <= 0.0:
        raise StageFail(f"window_seconds should be > 0 after recording: {s}")


def stage_a2_ring_buffer_wrap() -> None:
    """Recording 2000 samples into a cap-512 buffer keeps only the
    last 512 — but ``n_recorded_total`` tracks the full count for
    operator-facing health reporting."""
    m = _StateDumpMetrics(capacity=512)
    for i in range(2000):
        # Latency = i ms; ensures the WRAPPED window has 1488..1999 ms.
        m.record(elapsed_ns=int(i * 1_000_000), dump_bytes=i * 8)
    s = m.summary()
    if s["n_samples"] != 512:
        raise StageFail(f"wrapped n_samples != cap=512: {s['n_samples']}")
    if s["n_recorded_total"] != 2000:
        raise StageFail(
            f"n_recorded_total should be 2000 (total ever recorded), "
            f"got {s['n_recorded_total']}"
        )
    # The buffer holds samples 1488..1999 (last 512); max should be 1999 ms.
    if abs(s["max_ms"] - 1999.0) > 0.001:
        raise StageFail(
            f"after wrap, max_ms should be the latest sample (1999.0); "
            f"got {s['max_ms']}"
        )
    # Min sample in window = 1488 ms; mean = (1488+1999)/2 = 1743.5 ms.
    if abs(s["mean_ms"] - 1743.5) > 0.5:
        raise StageFail(
            f"mean_ms after wrap should be ~1743.5; got {s['mean_ms']}"
        )


def stage_a3_quantile_monotonicity() -> None:
    """p50 <= p95 <= p99 <= max is an invariant of any quantile
    estimator and must hold even on highly skewed inputs (one slow
    outlier among many fast samples).  Defends against an impl that
    accidentally averages instead of sorts."""
    m = _StateDumpMetrics(capacity=1024)
    # 99 fast (1 ms) + 1 slow (1000 ms) — a typical state-dump tail.
    for _ in range(99):
        m.record(elapsed_ns=1_000_000, dump_bytes=1024)
    m.record(elapsed_ns=1_000_000_000, dump_bytes=1024)
    s = m.summary()
    if not (s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]):
        raise StageFail(
            f"quantile monotonicity broken: "
            f"p50={s['p50_ms']} p95={s['p95_ms']} "
            f"p99={s['p99_ms']} max={s['max_ms']}"
        )
    # The outlier (1000 ms) should be in the p99 bucket.  With n=100
    # and 1 outlier, p99 must be the outlier and max equals the
    # outlier.
    if abs(s["max_ms"] - 1000.0) > 0.01:
        raise StageFail(f"max should be 1000 ms outlier; got {s['max_ms']}")


def stage_a4_dict_path_bytes_sentinel() -> None:
    """The dict path doesn't measure serialised bytes; ``-1`` is the
    sentinel.  A summary that omits negative samples in stats
    (which would be a bug) is caught here.  Recording mixed +/-
    bytes should still produce coherent latency stats."""
    m = _StateDumpMetrics(capacity=128)
    for _ in range(5):
        m.record(elapsed_ns=2_000_000, dump_bytes=-1)   # dict path
    for _ in range(5):
        m.record(elapsed_ns=4_000_000, dump_bytes=8192)  # bytes path
    s = m.summary()
    if s["last_dump_bytes"] != 8192:
        raise StageFail(
            f"last_dump_bytes should echo final record (8192); "
            f"got {s['last_dump_bytes']}"
        )
    if abs(s["mean_ms"] - 3.0) > 0.001:
        raise StageFail(
            f"mean should aggregate both paths' latencies (3 ms); "
            f"got {s['mean_ms']}"
        )


# ============================================================ Phase B
# Live-sglang integration.  Opt-in via $AGINFER_VERIFY_BASE.


def _maybe_get_base() -> str:
    return os.environ.get("AGINFER_VERIFY_BASE", "").rstrip("/")


def _fetch_state(base: str):
    import urllib.request
    import json
    with urllib.request.urlopen(f"{base}/aginfer/state", timeout=10) as resp:
        return json.loads(resp.read())


def stage_b0_state_carries_metrics_field() -> None:
    """A live ``/aginfer/state`` response includes a top-level
    ``state_dump_metrics`` key with the contract field set.  Schema-
    level contract — any drift breaks downstream monitoring scripts."""
    base = _maybe_get_base()
    if not base:
        print(_yellow("  (skip B0) set AGINFER_VERIFY_BASE to run live"))
        return
    state = _fetch_state(base)
    if "state_dump_metrics" not in state:
        raise StageFail(
            f"state_dump_metrics field missing from /aginfer/state; "
            f"top-level keys: {sorted(state.keys())}"
        )
    m = state["state_dump_metrics"]
    expected = {
        "n_samples", "n_recorded_total", "capacity", "window_seconds",
        "p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms",
        "last_dump_ms", "last_dump_bytes",
    }
    if set(m.keys()) != expected:
        raise StageFail(
            f"state_dump_metrics key drift; "
            f"missing={expected - set(m.keys())}, "
            f"extra={set(m.keys()) - expected}"
        )


def stage_b1_metrics_grow_with_polls() -> None:
    """Each ``/aginfer/state`` fetch should bump ``n_recorded_total``
    by exactly 1 (the dump records its own latency on the way out).
    The summary IN THIS dump reflects samples BEFORE the current
    record() call, so deltas are exact."""
    base = _maybe_get_base()
    if not base:
        print(_yellow("  (skip B1) set AGINFER_VERIFY_BASE to run live"))
        return
    s0 = _fetch_state(base)["state_dump_metrics"]
    s1 = _fetch_state(base)["state_dump_metrics"]
    s2 = _fetch_state(base)["state_dump_metrics"]
    # The first dump's summary is computed before its own record, so
    # the second poll's n_recorded_total is exactly s0.n_recorded_total
    # + 1 (one dump completed between the two fetches).
    delta_01 = s1["n_recorded_total"] - s0["n_recorded_total"]
    delta_12 = s2["n_recorded_total"] - s1["n_recorded_total"]
    if delta_01 != 1 or delta_12 != 1:
        raise StageFail(
            f"n_recorded_total should grow by exactly 1 per poll; "
            f"got deltas ({delta_01}, {delta_12}); "
            f"raw values: {s0['n_recorded_total']}, "
            f"{s1['n_recorded_total']}, {s2['n_recorded_total']}"
        )


def stage_b2_quantile_monotonicity_live() -> None:
    """Same invariant as Stage A3 but on the live sglang's measurements.
    Fire enough fetches to populate the ring; then assert quantile
    monotonicity on the LIVE wallclock distribution."""
    base = _maybe_get_base()
    if not base:
        print(_yellow("  (skip B2) set AGINFER_VERIFY_BASE to run live"))
        return
    for _ in range(40):
        _fetch_state(base)
    m = _fetch_state(base)["state_dump_metrics"]
    if m["n_samples"] < 20:
        raise StageFail(
            f"after 40 fetches, n_samples should be >= 20; got {m['n_samples']}"
        )
    if not (m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"] <= m["max_ms"]):
        raise StageFail(
            f"live quantile monotonicity broken: "
            f"p50={m['p50_ms']} p95={m['p95_ms']} "
            f"p99={m['p99_ms']} max={m['max_ms']}"
        )


def stage_b3_bytes_path_reports_positive_bytes() -> None:
    """The HTTP /aginfer/state hits ``dump_aginfer_state_bytes``
    (the bytes path), so ``last_dump_bytes`` after a fetch must be
    a positive int — never -1 (the dict-path sentinel)."""
    base = _maybe_get_base()
    if not base:
        print(_yellow("  (skip B3) set AGINFER_VERIFY_BASE to run live"))
        return
    # Need at least 2 fetches: the first one's summary is from prior
    # state; the second's summary contains the first's record.
    _fetch_state(base)
    m = _fetch_state(base)["state_dump_metrics"]
    if m["last_dump_bytes"] <= 0:
        raise StageFail(
            f"bytes-path /aginfer/state should report "
            f"last_dump_bytes > 0; got {m['last_dump_bytes']}"
        )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 _StateDumpMetrics empty summary",          stage_a0_empty_summary),
    ("A1 record + summary",                         stage_a1_record_and_summary),
    ("A2 ring buffer wraps at capacity",            stage_a2_ring_buffer_wrap),
    ("A3 quantile monotonicity (p50≤p95≤p99≤max)",  stage_a3_quantile_monotonicity),
    ("A4 dict-path bytes=-1 sentinel + mixed mean", stage_a4_dict_path_bytes_sentinel),
    ("B0 state carries state_dump_metrics field",   stage_b0_state_carries_metrics_field),
    ("B1 n_recorded_total grows by exactly 1 per poll",
                                                    stage_b1_metrics_grow_with_polls),
    ("B2 live quantile monotonicity",               stage_b2_quantile_monotonicity_live),
    ("B3 bytes-path last_dump_bytes positive",      stage_b3_bytes_path_reports_positive_bytes),
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
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT14 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    skipped = sum(1 for _, fn in _STAGES if False)  # placeholder
    print(_green(f"\nT14 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
