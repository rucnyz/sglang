"""#231 — compare replay metrics across arms (ours a3 vs baseline a3_kvoff).

Reads the per-trial ``metrics_<arm>_c<i>.json`` written by replay_driver,
groups by arm, and reports mean±std per latency/throughput metric across
trials.  The do-no-harm verdict is per-metric and uses disjoint mean±std
bands (same significance bar as the campaign): ours is "stably worse" on a
latency metric only if its whole band sits above baseline's.

Because both arms replay the SAME trace with output length forced, the
len_match_rate sanity (≈1.0 both arms, equal total tokens) is what licenses
the comparison in the first place — it is printed up front.

Usage:  python compare.py <results_dir>
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional


# Metrics we compare.  (key path into the metrics dict, lower-is-better?)
_LATENCY = [
    ("ttft_ms", "p50"),
    ("ttft_ms", "p99"),
    ("tpot_ms", "mean"),
    ("tpot_ms", "p99"),
    ("e2e_ms", "p50"),
    ("e2e_ms", "p99"),
]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    # M6: SAMPLE std (÷ n-1), not population (÷ n).  Population std on N=3
    # trials under-states spread and makes the do-no-harm "stably worse"
    # bands trigger-happy — the opposite of what a do-no-harm gate wants.
    # n=1 → std is undefined; report NaN so _band_verdict refuses a verdict.
    if len(xs) >= 2:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        std = math.sqrt(var)
    else:
        std = float("nan")
    return {"mean": m, "std": std, "n": len(xs)}


def _get(metrics: Dict[str, Any], grp: str, stat: str) -> Optional[float]:
    g = metrics.get(grp)
    if isinstance(g, dict):
        v = g.get(stat)
        return float(v) if isinstance(v, (int, float)) else None
    return None


def summarize(metrics_by_arm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Pure: {arm -> [per-trial metrics dict]} -> structured comparison."""
    out: Dict[str, Any] = {"arms": {}, "latency": [], "throughput": {}, "sanity": {}}
    for arm, trials in metrics_by_arm.items():
        out["arms"][arm] = len(trials)
        out["sanity"][arm] = {
            "len_match_rate": _mean_std([t.get("len_match_rate") for t in trials]),
            "total_out_tokens": _mean_std([t.get("total_out_tokens") for t in trials]),
            "n_error": _mean_std([t.get("n_error") for t in trials]),
        }
    out["throughput"] = {
        arm: _mean_std([t.get("throughput_tok_s") for t in trials])
        for arm, trials in metrics_by_arm.items()
    }
    # Closed-loop (session mode) end-to-end metrics, if present.  makespan
    # + session-e2e are THE headline for benefit: lower = the workload
    # finishes sooner.  Only emitted when trials carry a "sessions" block.
    if any("sessions" in t for trials in metrics_by_arm.values() for t in trials):
        ms_rows: List[Dict[str, Any]] = []
        for metric, lower in [("makespan_s", True), ("session_e2e_s.p50", True),
                              ("session_e2e_s.p99", True)]:
            row: Dict[str, Any] = {"metric": metric}
            for arm, trials in metrics_by_arm.items():
                vals = []
                for t in trials:
                    sess = t.get("sessions") or {}
                    if metric == "makespan_s":
                        vals.append(sess.get("makespan_s"))
                    else:
                        stat = metric.split(".")[1]
                        vals.append((sess.get("session_e2e_s") or {}).get(stat))
                row[arm] = _mean_std(vals)
            a, b = row.get("a3"), row.get("a3_kvoff")
            if a and b and a["n"] and b["n"]:
                row["verdict"] = _band_verdict(a, b, lower_is_better=lower)
            ms_rows.append(row)
        out["endtoend"] = ms_rows
    for grp, stat in _LATENCY:
        row: Dict[str, Any] = {"metric": f"{grp}.{stat}"}
        for arm, trials in metrics_by_arm.items():
            row[arm] = _mean_std([_get(t, grp, stat) for t in trials])
        # verdict (ours a3 vs baseline a3_kvoff), if both present
        a, b = row.get("a3"), row.get("a3_kvoff")
        if a and b and a["n"] and b["n"]:
            row["verdict"] = _band_verdict(a, b, lower_is_better=True)
        out["latency"].append(row)
    return out


def _band_verdict(ours: Dict[str, float], base: Dict[str, float], *, lower_is_better: bool) -> str:
    """Disjoint mean±std bands -> stable better/worse; else within-noise.

    Refuses a verdict (M5) when either arm has <2 trials (std undefined,
    NaN) or the two arms have unequal trial counts — comparing a 3-sample
    band against a 1-sample (std=0) band manufactures false stable verdicts.
    """
    if ours.get("n", 0) < 2 or base.get("n", 0) < 2:
        return "insufficient samples (need n>=2/arm)"
    if ours["n"] != base["n"]:
        return f"unequal samples (a3 n={ours['n']} vs base n={base['n']})"
    if ours["std"] != ours["std"] or base["std"] != base["std"]:  # NaN
        return "insufficient samples"
    o_lo, o_hi = ours["mean"] - ours["std"], ours["mean"] + ours["std"]
    b_lo, b_hi = base["mean"] - base["std"], base["mean"] + base["std"]
    if o_hi < b_lo:
        return "ours STABLY LOWER" if lower_is_better else "ours STABLY WORSE"
    if o_lo > b_hi:
        return "ours STABLY HIGHER (regression)" if lower_is_better else "ours STABLY BETTER"
    return "within-noise"


def sanity_check(summary: Dict[str, Any], *, min_len_match: float = 0.98) -> Dict[str, Any]:
    """C2 — verify the arms actually did identical work before any verdict
    is trustworthy.  Returns {ok, reasons}.  Fails if (a) either arm's
    len_match_rate < min_len_match, (b) either arm saw errors, or (c) the
    two arms' total generated tokens are stably different (disjoint
    mean±std bands) — any of which means the forced-length invariant
    broke and the comparison is INVALID, not 'do-no-harm holds'.
    """
    reasons: List[str] = []
    san = summary.get("sanity", {})
    for arm, d in san.items():
        lm = d.get("len_match_rate", {})
        if lm.get("n") and lm["mean"] < min_len_match:
            reasons.append(f"{arm} len_match_rate {lm['mean']:.3f} < {min_len_match}")
        ne = d.get("n_error", {})
        if ne.get("n") and ne["mean"] > 0.5:
            reasons.append(f"{arm} mean n_error {ne['mean']:.1f} > 0")
    a = san.get("a3", {}).get("total_out_tokens")
    b = san.get("a3_kvoff", {}).get("total_out_tokens")
    if a and b and a.get("n", 0) >= 2 and b.get("n", 0) >= 2:
        v = _band_verdict(a, b, lower_is_better=True)
        if "STABLY" in v:
            reasons.append(
                f"total_out_tokens diverged across arms (a3={a['mean']:.0f} "
                f"vs base={b['mean']:.0f}) — forced-length invariant broke"
            )
    return {"ok": not reasons, "reasons": reasons}


def load_dir(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for pf in sorted(glob.glob(os.path.join(results_dir, "metrics_*.json"))):
        base = os.path.basename(pf)[len("metrics_"):-len(".json")]  # <arm>_c<i>
        arm = base.rsplit("_c", 1)[0]
        try:
            doc = json.load(open(pf))
        except Exception:
            continue
        m = doc.get("metrics", doc)
        by_arm.setdefault(arm, []).append(m)
    return by_arm


def _fmt(d: Dict[str, float]) -> str:
    if not d.get("n"):
        return "    n/a   "
    return f"{d['mean']:8.1f}±{d['std']:5.1f}"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare.py <results_dir>")
        return 2
    by_arm = load_dir(sys.argv[1])
    if not by_arm:
        print("no metrics_*.json found")
        return 1
    s = summarize(by_arm)

    print(f"arms: " + ", ".join(f"{a}(n={n})" for a, n in s["arms"].items()))
    print("\n=== sanity (must hold to license the comparison) ===")
    for arm, san in s["sanity"].items():
        print(f"  {arm:9s} len_match={_fmt(san['len_match_rate'])}  "
              f"total_tok={_fmt(san['total_out_tokens'])}  n_err={_fmt(san['n_error'])}")

    print("\n=== throughput tok/s (higher better) ===")
    for arm, d in s["throughput"].items():
        print(f"  {arm:9s} {_fmt(d)}")

    if s.get("endtoend"):
        print("\n=== END-TO-END closed-loop (lower better) — the benefit metric ===")
        print(f"  {'metric':18s} {'a3 (ours)':16s} {'a3_kvoff (base)':16s}  verdict")
        for row in s["endtoend"]:
            a = _fmt(row.get("a3", {"n": 0}))
            b = _fmt(row.get("a3_kvoff", {"n": 0}))
            print(f"  {row['metric']:18s} {a:16s} {b:16s}  {row.get('verdict','')}")

    print("\n=== latency (lower better) — ours=a3 vs baseline=a3_kvoff ===")
    print(f"  {'metric':12s} {'a3 (ours)':16s} {'a3_kvoff (base)':16s}  verdict")
    for row in s["latency"]:
        a = _fmt(row.get("a3", {"n": 0}))
        b = _fmt(row.get("a3_kvoff", {"n": 0}))
        print(f"  {row['metric']:12s} {a:16s} {b:16s}  {row.get('verdict','')}")

    # C2 — the verdict is only trustworthy if the arms did identical work.
    sanity = sanity_check(s)
    print()
    if not sanity["ok"]:
        print("COMPARISON INVALID — arms did not do identical work:")
        for r in sanity["reasons"]:
            print("  - " + r)
        print("  (forced-length / fairness invariant broke; verdict suppressed)")
        return 1

    regressions = [r["metric"] for r in s["latency"] + s.get("endtoend", [])
                   if "regression" in str(r.get("verdict", ""))]
    if regressions:
        print(f"DO-NO-HARM: VIOLATED on {regressions}")
    else:
        print("DO-NO-HARM: HOLDS (no latency/end-to-end metric stably worse than baseline)")
    # Benefit call-out: closed-loop makespan stably lower = the workload
    # finishes sooner with the daemon on.
    for r in s.get("endtoend", []):
        if r["metric"] == "makespan_s" and "STABLY LOWER" in str(r.get("verdict", "")):
            print("BENEFIT: makespan STABLY LOWER — daemon finishes the workload sooner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
