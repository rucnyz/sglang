"""#231 verify: replay metrics comparison (pure summarize + verdict + load).

  A _band_verdict: disjoint mean±std bands -> stable; overlap -> within-noise
  B summarize: groups arms, computes mean±std, flags a stable regression
  C load_dir: parses metrics_<arm>_c<i>.json, groups by arm

Usage:  python dev/aginfer/verify/replay_compare/verify.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

_REPLAY_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "replay"
sys.path.insert(0, str(_REPLAY_DIR))

from compare import _band_verdict, summarize, load_dir  # noqa: E402

_FAILS: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


def test_verdict() -> None:
    print("A. _band_verdict")
    lo = {"mean": 10.0, "std": 1.0}
    hi = {"mean": 20.0, "std": 1.0}
    check(_band_verdict(lo, hi, lower_is_better=True) == "ours STABLY LOWER",
          "A ours band fully below -> STABLY LOWER")
    check("regression" in _band_verdict(hi, lo, lower_is_better=True),
          "A ours band fully above -> regression")
    overlap_a = {"mean": 15.0, "std": 5.0}
    overlap_b = {"mean": 16.0, "std": 5.0}
    check(_band_verdict(overlap_a, overlap_b, lower_is_better=True) == "within-noise",
          "A overlapping bands -> within-noise")


def _m(ttft_p99: float, e2e_p50: float, thr: float) -> dict:
    return {
        "ttft_ms": {"p50": ttft_p99 / 2, "p99": ttft_p99},
        "tpot_ms": {"mean": 5.0, "p99": 7.0},
        "e2e_ms": {"p50": e2e_p50, "p99": e2e_p50 * 2},
        "throughput_tok_s": thr,
        "len_match_rate": 1.0,
        "total_out_tokens": 1000,
        "n_error": 0,
    }


def test_summarize() -> None:
    print("B. summarize")
    # ours ttft_p99 stably HIGHER than baseline -> regression flagged
    by_arm = {
        "a3": [_m(200, 100, 50), _m(205, 102, 49), _m(202, 101, 51)],
        "a3_kvoff": [_m(100, 99, 60), _m(102, 101, 61), _m(101, 100, 59)],
    }
    s = summarize(by_arm)
    check(s["arms"] == {"a3": 3, "a3_kvoff": 3}, "B arm trial counts")
    ttft99 = next(r for r in s["latency"] if r["metric"] == "ttft_ms.p99")
    check(abs(ttft99["a3"]["mean"] - 202.33) < 0.5, "B a3 ttft_p99 mean ~202")
    check("regression" in ttft99["verdict"], "B ttft_p99 ours-higher -> regression verdict")
    # e2e overlaps -> within-noise
    e2e50 = next(r for r in s["latency"] if r["metric"] == "e2e_ms.p50")
    check(e2e50["verdict"] == "within-noise", "B e2e_p50 overlap -> within-noise")
    # sanity carries len_match
    check(s["sanity"]["a3"]["len_match_rate"]["mean"] == 1.0, "B sanity len_match captured")


def test_load_dir() -> None:
    print("C. load_dir")
    with tempfile.TemporaryDirectory() as d:
        for arm, i in [("a3", 1), ("a3", 2), ("a3_kvoff", 1)]:
            with open(os.path.join(d, f"metrics_{arm}_c{i}.json"), "w") as fh:
                json.dump({"metrics": _m(100, 50, 40)}, fh)
        by = load_dir(d)
        check(set(by.keys()) == {"a3", "a3_kvoff"}, "C arms parsed")
        check(len(by["a3"]) == 2 and len(by["a3_kvoff"]) == 1, "C trial grouping by arm")


def main() -> int:
    test_verdict()
    test_summarize()
    test_load_dir()
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for f in _FAILS:
            print("  - " + f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
