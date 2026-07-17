#!/usr/bin/env python3
"""Curate the three waste-demo workloads from the real cc agentreplay trace.

All three slices are derived from data/traces/cc_qwen3p5_9b.jsonl by selecting,
truncating, or phase-concatenating real records (never synthetic tokens), so
the token-exact replay still teacher-forces real output ids.

The cc trace is purely long-horizon (200 sessions, p50 27 turns, first turn
already ~17k tokens), so:
  case1 (long-horizon, KV-bound): the longest sessions, used as-is.
  case2 (large-concurrency short Q&A, mamba-bound): each session's first turn,
    input truncated to a short question and the forced output to a short answer,
    arriving tightly packed so concurrency (mamba slots) binds while KV stays idle.
  case3 (dynamic): a long-horizon phase followed in time by a short-swarm phase,
    so the binding pool flips across the run.

Record schema preserved: t, program_id, step, parent_program_id, spawned_at_step,
spawn_ts, input_ids, forced_output_ids, tool_gap_after.
"""
import argparse
import copy
import json
import os
from collections import defaultdict

SRC = "/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen3p5_9b.jsonl"
OUT = "/scratch/yuzhou/projects/sglang/reproduce/waste"

# case2 truncation: a short user question + short answer, sized so a swarm of
# them keeps KV far below its ceiling while filling mamba slots.
SHORT_IN = 512
SHORT_OUT = 128


def load_sessions(path):
    """Group records by program_id, ordered by step; return list of sessions."""
    by_id = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["program_id"]].append(r)
    sessions = []
    for pid, recs in by_id.items():
        recs.sort(key=lambda r: r["step"])
        sessions.append(recs)
    return sessions


def write(name, records, fname="trace.jsonl"):
    d = os.path.join(OUT, name, "data")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_sess = len({r["program_id"] for r in records})
    print(f"{name}/{fname}: {len(records)} records, {n_sess} sessions -> {path}")


def make_case1(sessions, n_sessions=40):
    """Longest-context sessions, as-is: long prompts saturate KV, each session
    needs one recurrent slot, so mamba sits idle (the long-horizon regime)."""
    ranked = sorted(sessions, key=lambda s: max(len(r["input_ids"]) for r in s),
                    reverse=True)
    out = []
    for s in ranked[:n_sessions]:
        out.extend(copy.deepcopy(s))
    return out


def make_case2(sessions, dup_to=256):
    """First turn of each session, truncated to a short Q&A, replicated up to
    dup_to requests and packed at t=0 so a high-concurrency swarm of short
    requests fills mamba slots while their tiny KV stays idle."""
    out = []
    base = [s[0] for s in sessions]
    i = 0
    while len(out) < dup_to:
        src = base[i % len(base)]
        r = copy.deepcopy(src)
        r["program_id"] = f"{src['program_id']}__dup{i}"
        r["step"] = 1
        r["parent_program_id"] = None
        r["spawned_at_step"] = None
        r["spawn_ts"] = None
        r["input_ids"] = src["input_ids"][:SHORT_IN]
        r["forced_output_ids"] = src["forced_output_ids"][:SHORT_OUT]
        r["t"] = 0.0
        r["tool_gap_after"] = 0.0
        out.append(r)
        i += 1
    return out


def make_case3_phase_a(sessions, n_long=12):
    """Phase A of the dynamic case: a long-horizon (KV-bound) phase, replayed
    back to back before the short-swarm phase B. Phase B reuses case2's data, so
    case3 = case3/phase_a_long then case2/trace against one server, and the
    binding pool flips mid-run (see run_case.sh)."""
    return make_case1(sessions, n_sessions=n_long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()
    sessions = load_sessions(args.src)
    print(f"loaded {len(sessions)} sessions from {args.src}")
    write("case1", make_case1(sessions))
    write("case2", make_case2(sessions))
    write("case3", make_case3_phase_a(sessions), fname="phase_a_long.jsonl")


if __name__ == "__main__":
    main()
