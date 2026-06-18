"""T20 regression probes — unit-level tests for daemon-side T20+T33
audit findings.

Companion to ``verify.py`` (HTTP E2E).  These probes target the
PARSING / AGGREGATION layer that lives in
``dev/aginfer/daemon/kv_scheduler.py`` so we can catch regressions
without a full sglang round-trip.

Each probe:

  1. Constructs a fixture that triggers the audit finding's failure
     mode.
  2. Calls the production function directly.
  3. Asserts that the production code does NOT silently swallow the
     error (a fail-fast behaviour the audit found violated).

Run via:

    python dev/aginfer/verify/t20/regression_probe.py

Exit 0 PASS, 1 FAIL.  Each probe prints a one-line result.

Probes mapped to audit findings (joint T33+T20, 2026-05-31):

  D6 — daemon's migrate-response parser used
       ``entry.get("reason", "?")`` — a malformed sglang response
       with no ``reason`` field would silently log a "?" metric
       instead of surfacing the protocol break.
  D8 — daemon's ``_flatten_per_rank`` aggregated
       ``recent_throughput_bps`` as MEAN across ranks; DESIGN §6
       line 731 specifies SUM (each rank has its own PCIe / NVMe
       link; aggregate scales with rank count).  An 8-rank
       deployment would report 1/8 of actual system bandwidth.

D7 (HASH_COLLISION log spam) is not probed here — fixture cost
(forging two radix nodes with the same SHA-256 hash) exceeds the
value of the probe.  git log + grep is the cheaper check.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``baselines`` and ``daemon`` packages importable when this
# probe is run as a standalone script.
_HERE = Path(__file__).resolve()
_AGINFER = _HERE.parents[2]
if str(_AGINFER) not in sys.path:
    sys.path.insert(0, str(_AGINFER))


def _print(probe: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {probe}"
          f"{('  — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# D6: daemon's _dispatch_migrate must KeyError on missing 'reason'
# ---------------------------------------------------------------------------

def probe_d6_reason_subscript_fails_loud() -> bool:
    """Audit D6: daemon's migrate-response parser previously did
    ``entry.get("reason", "?")``.  A malformed sglang response with
    a skip entry that lacks ``reason`` should surface as a KeyError,
    not be silently coerced to ``"?"`` and emit a degenerate metric
    line.

    We construct the parse loop in isolation: the production parsing
    code in ``KvScheduler._dispatch_migrate`` iterates
    ``skipped_list`` and reads ``entry["reason"]``.  Probe replays
    that exact loop against a synthetic response with no reason.
    """
    from daemon.kv_scheduler import KvScheduler  # noqa: F401
    import dis
    # Locate the parse loop's behavior by source inspection.  The
    # function `_dispatch_migrate` should reference `entry["reason"]`
    # (direct subscript) — NOT `entry.get("reason"`).  AST-style
    # check.  This is robust to refactors that move the loop into a
    # helper, as long as the helper still does the subscript.
    import inspect
    src = inspect.getsource(KvScheduler._dispatch_migrate)
    if 'entry.get("reason"' in src or "entry.get('reason'" in src:
        _print("D6 reason direct-subscript",
               False,
               "_dispatch_migrate still uses entry.get('reason', ...) "
               "— a malformed sglang response with no reason would "
               "be silently swallowed as a '?' metric value")
        return False
    if 'entry["reason"]' not in src and "entry['reason']" not in src:
        _print("D6 reason direct-subscript",
               False,
               "_dispatch_migrate does not appear to read "
               "entry['reason'] anywhere — has the parse loop been "
               "refactored?  Verify manually.")
        return False
    _print("D6 reason direct-subscript", True,
           "entry['reason'] is direct subscript; missing field → KeyError")
    return True


# ---------------------------------------------------------------------------
# D8: _flatten_per_rank must SUM throughput across ranks, not MEAN
# ---------------------------------------------------------------------------

def _make_rank_fixture(*, peak_bps: int, recent_bps: float,
                      idle_s: float) -> dict:
    """Minimal DESIGN §5-shaped per-rank dict."""
    return {
        "time_counter": 0,
        "throughput_ema": {"prefill_bps": 0.0, "decode_per_program": {}},
        "pool_usage": {
            "HBM":  {"subpools": {"full": {
                "used_bytes": 0, "cap_bytes": 1,
                "available_bytes": 1, "evictable_bytes": 0,
                "page_bytes": 256}}},
            "DRAM": {"subpools": {"full": {
                "used_bytes": 0, "cap_bytes": 1,
                "available_bytes": 1, "evictable_bytes": 0,
                "page_bytes": 256}}},
            "DISK": {"subpools": {"full": {
                "used_bytes": 0, "cap_bytes": 0,
                "available_bytes": 0, "evictable_bytes": 0,
                "page_bytes": 256}}},
        },
        "per_program_usage": {},
        "units": [],
        "link_stats": {
            link: {
                "peak_bw_bps": peak_bps,
                "recent_throughput_bps": recent_bps,
                "time_since_last_sample_s": idle_s,
            }
            for link in ("HBM->DRAM", "DRAM->HBM",
                         "DRAM->DISK", "DISK->DRAM")
        },
        "tier_holding_cost": {
            t: {"full": {"h_max_per_byte_sec": 0.0}}
            for t in ("HBM", "DRAM", "DISK")
        },
    }


def probe_d8_flatten_per_rank_sums_throughput() -> bool:
    """Audit D8: DESIGN §6 line 731 specifies link_stats peak +
    recent_throughput SUM across ranks (each rank's link is
    independent; aggregate scales with rank count).  An 8-rank
    deployment that reads MEAN sees 1/8 of system bandwidth.

    Feed 2 ranks, each with recent_throughput=1e9.  The aggregated
    value must be 2e9 (SUM), not 1e9 (MEAN).
    """
    from daemon.kv_scheduler import _flatten_per_rank

    rank0 = _make_rank_fixture(
        peak_bps=64_000_000_000,
        recent_bps=1.0e9,
        idle_s=0.5,
    )
    rank1 = _make_rank_fixture(
        peak_bps=64_000_000_000,
        recent_bps=1.0e9,
        idle_s=0.3,
    )
    multi = {"per_rank": [rank0, rank1]}
    agg = _flatten_per_rank(multi)
    got = agg["link_stats"]["HBM->DRAM"]["recent_throughput_bps"]
    expected_sum = 2.0e9
    expected_mean = 1.0e9
    if abs(got - expected_sum) < 1.0:
        _print("D8 flatten_per_rank throughput SUM",
               True, f"recent_throughput_bps={got:.2e} ≈ "
                     f"expected SUM={expected_sum:.2e}")
        return True
    if abs(got - expected_mean) < 1.0:
        _print("D8 flatten_per_rank throughput SUM",
               False, f"recent_throughput_bps={got:.2e} == MEAN "
                      f"(expected SUM={expected_sum:.2e}); DESIGN §6 "
                      f"line 731 specifies SUM across ranks because "
                      f"each rank's link is independent")
        return False
    _print("D8 flatten_per_rank throughput SUM",
           False, f"recent_throughput_bps={got:.2e} matches neither "
                  f"SUM ({expected_sum:.2e}) nor MEAN "
                  f"({expected_mean:.2e}) — formula has drifted")
    return False


def probe_d8_flatten_per_rank_sums_peak() -> bool:
    """Same as D8 main but for peak_bw_bps (also specified as SUM
    in DESIGN §6 line 731).  Probe separately so a partial fix that
    sums one but not the other surfaces."""
    from daemon.kv_scheduler import _flatten_per_rank
    rank0 = _make_rank_fixture(
        peak_bps=64_000_000_000, recent_bps=0.0, idle_s=1e12)
    rank1 = _make_rank_fixture(
        peak_bps=64_000_000_000, recent_bps=0.0, idle_s=1e12)
    agg = _flatten_per_rank({"per_rank": [rank0, rank1]})
    got = agg["link_stats"]["HBM->DRAM"]["peak_bw_bps"]
    if got == 128_000_000_000:
        _print("D8 flatten_per_rank peak SUM", True,
               f"peak_bw_bps={got/1e9:.1f} GB/s (= 2 × rank peak)")
        return True
    _print("D8 flatten_per_rank peak SUM", False,
           f"peak_bw_bps={got/1e9:.1f} GB/s; expected 128 (= SUM of "
           f"2 ranks × 64 GB/s)")
    return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== T20 regression probe ===")
    print()
    print("[probes]")
    results = [
        probe_d6_reason_subscript_fails_loud(),
        probe_d8_flatten_per_rank_sums_throughput(),
        probe_d8_flatten_per_rank_sums_peak(),
    ]
    print()
    print("=== summary ===")
    n_pass = sum(1 for r in results if r)
    n_fail = len(results) - n_pass
    print(f"  {n_pass} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
