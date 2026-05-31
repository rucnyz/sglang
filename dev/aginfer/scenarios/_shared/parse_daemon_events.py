#!/usr/bin/env python3
"""Parse `aginfer_metric` structured-log lines from a daemon.log.

Usage:
    python verify/t9/parse_daemon_events.py <cycle_dir>
    python verify/t9/parse_daemon_events.py <matrix_root>/

Emits per-cycle counters answering:
* G1 — admission pauses / resumes
* G2 — memory_pressure event count (received vs acted)
* G3 — kv_scheduler dispatch breakdown
        (empty_decision_set / policy_declined / dispatched)
        + migrate POST status + applied/skipped
* G5 — HBM occupancy time series (head + tail samples printed)
* G6 — list of paused program_ids
* G9 — wasted RTT in occ ∈ [0.7, 0.85): count of
        `admission_pressure occ=X will_act=false occ < theta_hi`

Output: human-readable summary on stdout.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

METRIC_RE = re.compile(r"aginfer_metric (.+?)\s*$")


def parse_kv(rest: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in rest.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def iter_metrics(log_path: Path):
    with log_path.open(errors="ignore") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if not m:
                continue
            kv = parse_kv(m.group(1))
            yield kv


def summarize_cycle(log_path: Path) -> Dict[str, object]:
    received = Counter()       # event_received by kind
    kv_outcome = Counter()      # kv_decide outcome
    kv_by_kind = defaultdict(Counter)  # kv_decide by (kind, outcome)
    migrate_status = Counter()  # migrate_post status
    skip_reasons = Counter()    # migrate_skipped reason
    migrate_applied = 0
    migrate_skipped = 0
    pauses = []                 # (pid, occ)
    resumes = []                # (pid, occ)
    pressure_acted = 0
    pressure_no_act = 0          # G9 wasted RTT
    pressure_occ_distribution = []  # all pressure occ values
    occ_series = []              # (event_idx, occ_hbm)
    program_states = Counter()   # (from, to)
    summary_line: Dict[str, str] = {}

    event_idx = 0
    for kv in iter_metrics(log_path):
        ev = kv.get("event")
        if ev == "event_received":
            received[kv.get("kind", "?")] += 1
        elif ev == "kv_decide":
            outcome = kv.get("outcome", "?")
            kind = kv.get("kind", "?")
            kv_outcome[outcome] += 1
            kv_by_kind[kind][outcome] += 1
        elif ev == "migrate_post":
            migrate_status[kv.get("status", "?")] += 1
            try:
                migrate_applied += int(kv.get("applied", 0))
                migrate_skipped += int(kv.get("skipped", 0))
            except ValueError:
                pass
        elif ev == "migrate_skipped":
            skip_reasons[kv.get("reason", "?")] += 1
        elif ev == "admission_pressure":
            occ = float(kv.get("occ", 0.0))
            pressure_occ_distribution.append(occ)
            if kv.get("will_act") == "false":
                pressure_no_act += 1
            else:
                pressure_acted += 1
        elif ev == "admission_pause":
            pauses.append((kv.get("pid", "?"), float(kv.get("occ", 0))))
        elif ev == "admission_resume":
            resumes.append((kv.get("pid", "?"), float(kv.get("occ", 0))))
        elif ev == "program_state":
            program_states[(kv.get("from_", "?"), kv.get("to", "?"))] += 1
        elif ev == "state_fetched":
            try:
                occ_series.append((event_idx, float(kv.get("occ_hbm", 0))))
            except ValueError:
                pass
            event_idx += 1
        elif ev == "cycle_summary":
            summary_line = kv

    return {
        "received": received,
        "kv_outcome": kv_outcome,
        "kv_by_kind": kv_by_kind,
        "migrate_status": migrate_status,
        "migrate_applied": migrate_applied,
        "migrate_skipped": migrate_skipped,
        "skip_reasons": skip_reasons,
        "pauses": pauses,
        "resumes": resumes,
        "pressure_acted": pressure_acted,
        "pressure_no_act": pressure_no_act,
        "pressure_occ": pressure_occ_distribution,
        "occ_series": occ_series,
        "program_states": program_states,
        "summary_line": summary_line,
    }


def print_summary(name: str, s: dict) -> None:
    print(f"\n## {name}\n")
    if not s["received"] and not s["summary_line"]:
        print("  (no aginfer_metric lines — daemon predates instrumentation)")
        return

    total_events = sum(s["received"].values())
    print(f"### Event flow")
    print(f"  total events received: {total_events}")
    for k, n in s["received"].most_common():
        print(f"    {k}: {n}")

    print(f"\n### kv_scheduler decisions (T7 G3)")
    total_kv = sum(s["kv_outcome"].values())
    print(f"  total kv_decide calls: {total_kv}")
    for outcome, n in s["kv_outcome"].most_common():
        pct = 100 * n / total_kv if total_kv else 0
        print(f"    {outcome}: {n} ({pct:.1f}%)")
    print(f"\n  migrate POSTs by status:")
    for st, n in s["migrate_status"].most_common():
        print(f"    status={st}: {n}")
    print(
        f"  applied total: {s['migrate_applied']}  "
        f"skipped total: {s['migrate_skipped']}"
    )
    if s["skip_reasons"]:
        print(f"  skip reasons:")
        for reason, n in s["skip_reasons"].most_common():
            pct = 100 * n / sum(s["skip_reasons"].values())
            print(f"    {reason}: {n} ({pct:.1f}%)")

    print(f"\n### admission_controller (T8 G1)")
    print(f"  pauses: {len(s['pauses'])}")
    print(f"  resumes: {len(s['resumes'])}")
    if s["pauses"]:
        print(f"  pids paused: {Counter(p for p, _ in s['pauses']).most_common(5)}")

    print(f"\n### memory_pressure analysis (T8 G9 wasted-RTT)")
    print(f"  pressure events where admission ACTED: {s['pressure_acted']}")
    print(f"  pressure events where admission did NOT act (occ<theta_hi): {s['pressure_no_act']}")
    if s["pressure_occ"]:
        n = len(s["pressure_occ"])
        below_085 = sum(1 for o in s["pressure_occ"] if o < 0.85)
        below_07 = sum(1 for o in s["pressure_occ"] if o < 0.7)
        print(
            f"  occ at pressure events: n={n}  "
            f"<0.85: {below_085} ({100*below_085/n:.0f}%)  "
            f"<0.70: {below_07} ({100*below_07/n:.0f}%)"
        )

    print(f"\n### HBM occupancy time series (T9 G5)")
    occ = s["occ_series"]
    if occ:
        vals = [o for _, o in occ]
        print(
            f"  samples: {len(vals)}  "
            f"min: {min(vals):.3f}  max: {max(vals):.3f}  "
            f"p50: {sorted(vals)[len(vals)//2]:.3f}  "
            f"final: {vals[-1]:.3f}"
        )
        ever_above_085 = sum(1 for v in vals if v >= 0.85)
        ever_above_07 = sum(1 for v in vals if v >= 0.7)
        print(
            f"  samples ≥ 0.85 (admission would act): {ever_above_085} ({100*ever_above_085/len(vals):.1f}%)"
        )
        print(
            f"  samples ≥ 0.70 (sglang fires pressure): {ever_above_07} ({100*ever_above_07/len(vals):.1f}%)"
        )

    print(f"\n### program_tracker transitions (T6)")
    for (frm, to), n in s["program_states"].most_common():
        print(f"  {frm} -> {to}: {n}")

    if s["summary_line"]:
        print(f"\n### Daemon-emitted cycle_summary")
        for k, v in s["summary_line"].items():
            if k != "event":
                print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path,
                    help="Single cycle dir OR matrix root (will scan children)")
    args = ap.parse_args()

    p: Path = args.path
    logs: List[Path] = []
    if (p / "daemon.log").exists():
        logs.append(p / "daemon.log")
    else:
        # treat as matrix root: find every */daemon.log inside
        for d in sorted(p.iterdir()):
            if d.is_dir() and (d / "daemon.log").exists():
                logs.append(d / "daemon.log")
            elif d.is_symlink() and (d.resolve() / "daemon.log").exists():
                logs.append(d.resolve() / "daemon.log")

    if not logs:
        print(f"no daemon.log found under {p}", file=sys.stderr)
        sys.exit(1)

    print(f"# daemon-event analysis ({len(logs)} cycle(s))")
    for log_path in logs:
        s = summarize_cycle(log_path)
        print_summary(log_path.parent.name, s)


if __name__ == "__main__":
    main()
