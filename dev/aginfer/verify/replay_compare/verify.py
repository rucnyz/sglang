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

from compare import _band_verdict, summarize, load_dir, sanity_check  # noqa: E402

_FAILS: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


def test_verdict() -> None:
    print("A. _band_verdict")
    lo = {"mean": 10.0, "std": 1.0, "n": 3}
    hi = {"mean": 20.0, "std": 1.0, "n": 3}
    check(_band_verdict(lo, hi, lower_is_better=True) == "ours STABLY LOWER",
          "A ours band fully below -> STABLY LOWER")
    check("regression" in _band_verdict(hi, lo, lower_is_better=True),
          "A ours band fully above -> regression")
    overlap_a = {"mean": 15.0, "std": 5.0, "n": 3}
    overlap_b = {"mean": 16.0, "std": 5.0, "n": 3}
    check(_band_verdict(overlap_a, overlap_b, lower_is_better=True) == "within-noise",
          "A overlapping bands -> within-noise")
    # M5: n<2 or unequal n must REFUSE a verdict (no false stable bands)
    check("insufficient" in _band_verdict({"mean": 10, "std": float("nan"), "n": 1},
                                          {"mean": 20, "std": 1.0, "n": 3},
                                          lower_is_better=True),
          "A M5 n<2 -> insufficient samples (no verdict)")
    check("unequal" in _band_verdict({"mean": 10, "std": 1.0, "n": 2},
                                     {"mean": 20, "std": 1.0, "n": 3},
                                     lower_is_better=True),
          "A M5 unequal n -> refused")


def test_sanity() -> None:
    print("C2. sanity_check gates the verdict")
    # clean: len_match 1.0, no errors, equal tokens -> ok
    ok = summarize({"a3": [_m(100, 50, 40)] * 3, "a3_kvoff": [_m(100, 50, 40)] * 3})
    check(sanity_check(ok)["ok"], "C2 identical work -> sanity ok")
    # low len_match -> invalid
    bad = _m(100, 50, 40); bad["len_match_rate"] = 0.5
    s_bad = summarize({"a3": [bad] * 3, "a3_kvoff": [_m(100, 50, 40)] * 3})
    res = sanity_check(s_bad)
    check(not res["ok"] and any("len_match" in r for r in res["reasons"]),
          "C2 low len_match -> COMPARISON INVALID")
    # divergent total tokens -> invalid
    hi = _m(100, 50, 40); hi["total_out_tokens"] = 5000
    lo = _m(100, 50, 40); lo["total_out_tokens"] = 1000
    s_div = summarize({"a3": [hi] * 3, "a3_kvoff": [lo] * 3})
    res2 = sanity_check(s_div)
    check(not res2["ok"] and any("total_out_tokens" in r for r in res2["reasons"]),
          "C2 divergent tokens -> COMPARISON INVALID")


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


def _ms(makespan: float) -> dict:
    m = _m(100, 50, 40)
    m["sessions"] = {"makespan_s": makespan, "session_e2e_s": {"p50": makespan / 2, "p99": makespan},
                     "n_sessions": 5, "total_steps": 20}
    return m


def test_endtoend() -> None:
    print("D. closed-loop end-to-end comparison")
    # ours makespan stably LOWER -> benefit
    by_arm = {
        "a3": [_ms(80), _ms(82), _ms(81)],
        "a3_kvoff": [_ms(100), _ms(101), _ms(99)],
    }
    s = summarize(by_arm)
    check("endtoend" in s, "D endtoend block present when sessions carried")
    ms = next(r for r in s["endtoend"] if r["metric"] == "makespan_s")
    check("STABLY LOWER" in ms["verdict"], "D ours makespan stably lower -> benefit")
    # no sessions -> no endtoend block
    s2 = summarize({"a3": [_m(100, 50, 40)], "a3_kvoff": [_m(100, 50, 40)]})
    check("endtoend" not in s2, "D no endtoend block without session data")


def main() -> int:
    test_verdict()
    test_sanity()
    test_summarize()
    test_load_dir()
    test_endtoend()
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
