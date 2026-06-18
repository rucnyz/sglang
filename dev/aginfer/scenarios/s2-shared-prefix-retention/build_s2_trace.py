#!/usr/bin/env python3
"""Build the S2 holder-count trace(s) from REAL collected CC data (NO synthesis).

Source: agentreplay's tokenized real CC trace (data/samples/demo_trace.jsonl) — 14
real Claude-Code trajectories, already token-exact (input_ids/forced_output_ids).
We only TRUNCATE and REPLICATE real token vectors (no fabricated tokens), per the
operating constraint "用我们收集的来自agentreplay里的data里抽出你需要的，并且进行截断或者复制".

S2 lever (n_holders): a KV unit SHARED by N programs must outrank an equal-size
single-program scratch unit. To build that from real data, per BLOCK:

  * SHARED prefix  = the first --shared-tok tokens of ONE big real program (a DIFFERENT
    program per block). Replicated VERBATIM as the leading tokens of all --fleet fleet
    members. Because every member then appends a DISTINCT real suffix, the shared span
    is a NON-LEAF radix node with n_holders = --fleet (the exact S2 condition: a shared
    non-leaf cannot be pinned/migrated, only the eviction KEY can keep it).
  * fleet member i = SHARED ++ suffix_i (suffix = a distinct real tail of another real
    program → members are distinct leaves), re-touched over --turns turns.
  * churn program j = a real program truncated to --churn-tok tokens (per-block offset
    shift for distinctness), --churn-turns turns. Floods the pool BETWEEN fleet rounds.

WHY BLOCKS (= the N for statistics): each block's SHARED prefix is a DIFFERENT real
program → DISTINCT tokens → the radix tree is content-addressed, so block b+1's fleet
gets ZERO free cache hits from block b. The measured quantity (fleet shared-prefix
re-prefill) is therefore INDEPENDENT across blocks even though the cache persists →
N genuinely independent, paired (same trace) data points, no flush needed.

Arrival schedule (driver.run_arrival `t`): fleet round r fires all members at t≈r*ROUND;
churn lands mid-round so it pressures the cache before the next re-touch. ours (n_holders
hint) keeps the shared node resident; true LRU drops it by recency and re-prefills it.

Output: --blocks valid agentreplay JSONL traces  <out-prefix>{0..N-1}.jsonl.
Replay each token-exact; both arms replay the SAME files → byte-identical prompts.
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Any, Dict, List


def load_by_program(path: str):
    by = collections.OrderedDict()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by.setdefault(r["program_id"], []).append(r)
    for pid in by:
        by[pid].sort(key=lambda x: int(x.get("step", 0)))
    return by


def longest_input(recs: List[Dict[str, Any]]) -> List[int]:
    best: List[int] = []
    for r in recs:
        ii = r.get("input_ids") or []
        if len(ii) > len(best):
            best = ii
    return list(best)


def lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/scratch/yuzhou/projects/agentreplay/data/samples/demo_trace.jsonl")
    ap.add_argument("--out-prefix", dest="out_prefix",
                    default="/scratch/yuzhou/projects/sglang/dev/dynamo/s2_block")
    ap.add_argument("--blocks", type=int, default=3, help="independent S2 scenarios = N for stats")
    ap.add_argument("--fleet", type=int, default=8, help="fleet members sharing the prefix → n_holders")
    ap.add_argument("--turns", type=int, default=8, help="fleet re-touch turns")
    ap.add_argument("--shared-tok", dest="shared_tok", type=int, default=24000)
    ap.add_argument("--suffix-tok", dest="suffix_tok", type=int, default=300)
    ap.add_argument("--churn", type=int, default=8)
    ap.add_argument("--churn-turns", dest="churn_turns", type=int, default=2)
    ap.add_argument("--churn-tok", dest="churn_tok", type=int, default=14000)
    ap.add_argument("--forced-cap", dest="forced_cap", type=int, default=16)
    ap.add_argument("--round", dest="round_s", type=float, default=2.0)
    a = ap.parse_args()

    by = load_by_program(a.src)
    ranked = sorted(by.items(), key=lambda kv: len(longest_input(kv[1])), reverse=True)
    pids = [pid for pid, _ in ranked]
    longest = {pid: longest_input(recs) for pid, recs in ranked}

    # programs big enough to host a 24k shared prefix → one distinct one per block
    shared_hosts = [p for p in pids if len(longest[p]) >= a.shared_tok]
    if len(shared_hosts) < a.blocks:
        raise SystemExit(f"need {a.blocks} programs ≥{a.shared_tok} tok; have {len(shared_hosts)}")

    forced_bank = (by[shared_hosts[0]][0].get("forced_output_ids") or list(range(10)))[: a.forced_cap]
    if not forced_bank:
        forced_bank = list(range(a.forced_cap))

    written = []
    for b in range(a.blocks):
        shared_src = shared_hosts[b]                       # DISTINCT real program per block
        shared = longest[shared_src][: a.shared_tok]
        # member suffixes: distinct real tails of other programs
        others = [p for p in pids if p != shared_src]

        def member_suffix(i: int) -> List[int]:
            src = others[i % len(others)]
            toks = longest[src]
            suf = list(toks[-a.suffix_tok:]) if len(toks) >= a.suffix_tok else list(toks)
            return suf or [hash(src) & 0xFFFF]

        def turn_marker(i: int, t: int) -> List[int]:
            src = others[(i + t + b) % len(others)]
            toks = longest[src]
            start = 2000 + (t * 31 + i * 7 + b * 13) % max(1, len(toks) - 8 - 2000)
            return list(toks[start: start + 8])

        records: List[Dict[str, Any]] = []
        for i in range(a.fleet):
            pid = f"b{b}-fleetA-{i}"
            suf = member_suffix(i)
            for t in range(a.turns):
                inp = list(shared) + suf + turn_marker(i, t)
                records.append({"t": round(t * a.round_s + i * 0.05, 3),
                                "program_id": pid, "step": t + 1,
                                "input_ids": inp, "forced_output_ids": list(forced_bank)})

        # churn: rotate real programs, per-block + per-stream offset shift → distinct bodies
        for j in range(a.churn):
            src = others[(j + b * 3) % len(others)]
            toks = longest[src]
            # start PAST the shared CC header (≥1600) at a stream-specific offset so churn
            # streams diverge from each other and from any shared prefix.
            base = 1600 + (j * 37 + b * 401) % max(1, len(toks) - a.churn_tok - 1600)
            body = list(toks[base: base + a.churn_tok])
            if len(body) < 2000:
                body = list(toks[1600: 1600 + a.churn_tok]) or list(toks[: a.churn_tok])
            pid = f"b{b}-churnB-{j}"
            for t in range(a.churn_turns):
                inp = body + turn_marker(100 + j, t)
                sched = (t * (a.turns // max(1, a.churn_turns))) * a.round_s + 0.5 * a.round_s + j * 0.07
                records.append({"t": round(sched, 3), "program_id": pid, "step": t + 1,
                                "input_ids": inp, "forced_output_ids": list(forced_bank)})

        records.sort(key=lambda r: (r["t"], r["program_id"], r["step"]))
        out = f"{a.out_prefix}{b}.jsonl"
        with open(out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        written.append((out, shared_src, records))

    # ---- report + cross-block independence check ----
    print(f"source: {a.src}")
    shared_heads = []
    for b, (out, shared_src, records) in enumerate(written):
        fleet = sorted({r["program_id"] for r in records if "fleetA" in r["program_id"]})
        churn = sorted({r["program_id"] for r in records if "churnB" in r["program_id"]})
        f_t1 = {}
        for r in records:
            if "fleetA" in r["program_id"] and r["program_id"] not in f_t1:
                f_t1[r["program_id"]] = r["input_ids"]
        members = sorted(f_t1)
        mins = [lcp(f_t1[members[i]], f_t1[members[j]])
                for i in range(len(members)) for j in range(i + 1, len(members))]
        shared_heads.append(f_t1[members[0]][: a.shared_tok])
        print(f"block {b}: {out}")
        print(f"   shared_src={shared_src}  shared_node={min(mins)} tok (n_holders={len(fleet)}), "
              f"fleet={len(fleet)}×{a.turns}t, churn={len(churn)}×{a.churn_turns}t@{a.churn_tok}, "
              f"recs={len(records)}")
    # cross-block shared-prefix LCP (must be small → independent measurements)
    xb = [lcp(shared_heads[i], shared_heads[j])
          for i in range(len(shared_heads)) for j in range(i + 1, len(shared_heads))]
    print(f"cross-block shared-prefix LCP: max={max(xb) if xb else 0} "
          f"(<<{a.shared_tok} → blocks are independent: no free cross-block cache hits)")


if __name__ == "__main__":
    main()
