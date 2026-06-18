"""Build a mixed workload trace that amplifies LPB vs LRU differentiation.

Design: 10 'valuable' deep multi-turn roots (full 40 turns, prefix reuse across
turns) + 80 'interference' first-turn-only extracts from other roots, staggered
so interference arrives BETWEEN valuable turns. Under cache pressure:
  - LPB: keeps valuable prefixes cached (high access frequency -> high benefit)
  - LRU: evicts valuable prefixes when interference is more recent -> cache miss
         on next turn -> full re-prefill -> lower throughput + higher TTFT

Token-exactness: every token is a verbatim slice of a real CC record (prefix
truncation for interference, full records for valuable). No fabricated tokens.
"""
import json
import os
from collections import defaultdict

AR = "/scratch/yuzhou/projects/agentreplay"
CORPUS = os.path.join(AR, "data/traces/cc_qwen3p5_9b.jsonl")
OUT = "reproduce/RQ1/case1/data/cc_qwen_case1_lpb_win.jsonl"

N_VALUABLE = 30
N_INTERF = 120
INTERF_MAX_PROMPT = 25000

def load_corpus():
    roots = defaultdict(list)
    with open(CORPUS) as f:
        for line in f:
            rec = json.loads(line)
            rid = rec.get("root_id", rec.get("program_id", ""))
            roots[rid].append(rec)
    for recs in roots.values():
        recs.sort(key=lambda r: (r.get("step", 0), r.get("t", 0)))
    return roots

def select_valuable(roots, n):
    candidates = []
    for rid, recs in roots.items():
        if len(recs) >= 30:
            max_p = max(len(r.get("input_ids", [])) for r in recs)
            if 50000 < max_p < 140000:
                candidates.append((rid, len(recs), max_p))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return [c[0] for c in candidates[:n]]

def select_interference(roots, valuable_set, n, max_prompt):
    candidates = []
    for rid, recs in roots.items():
        if rid in valuable_set:
            continue
        first = min(recs, key=lambda r: r.get("step", 0))
        fp = len(first.get("input_ids", []))
        if fp <= max_prompt:
            candidates.append((rid, fp))
    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:n]]

def build_trace(roots, valuable_ids, interf_ids):
    records = []
    pid_counter = 0

    for rid in valuable_ids:
        for rec in roots[rid]:
            rec_copy = dict(rec)
            records.append(rec_copy)

    stagger_base = 0.5
    for i, rid in enumerate(interf_ids):
        recs = roots[rid]
        first = min(recs, key=lambda r: r.get("step", 0))
        rec_copy = dict(first)
        rec_copy["program_id"] = f"interf_{pid_counter}"
        rec_copy["root_id"] = f"interf_{pid_counter}"
        rec_copy["step"] = 1
        rec_copy.pop("parent_id", None)
        fo = rec_copy.get("forced_output_ids", [])
        if len(fo) > 512:
            rec_copy["forced_output_ids"] = fo[:512]
        pid_counter += 1
        records.append(rec_copy)

    print(f"Total records: {len(records)}")
    print(f"  valuable: {sum(1 for r in records if r.get('root_id','') not in [f'interf_{i}' for i in range(pid_counter)])}")
    print(f"  interference: {pid_counter}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Written to {OUT}")
    sz = os.path.getsize(OUT)
    print(f"Size: {sz / 1e6:.1f} MB")

if __name__ == "__main__":
    roots = load_corpus()
    valuable_ids = select_valuable(roots, N_VALUABLE)
    print(f"Valuable roots ({len(valuable_ids)}):")
    for rid in valuable_ids:
        recs = roots[rid]
        max_p = max(len(r.get("input_ids", [])) for r in recs)
        print(f"  {rid[:20]}  turns={len(recs)}  max_prompt={max_p}")

    interf_ids = select_interference(roots, set(valuable_ids), N_INTERF, INTERF_MAX_PROMPT)
    print(f"\nInterference roots ({len(interf_ids)}):")
    for rid in interf_ids[:5]:
        recs = roots[rid]
        first = min(recs, key=lambda r: r.get("step", 0))
        fp = len(first.get("input_ids", []))
        print(f"  {rid[:20]}  first_prompt={fp}")
    print(f"  ... ({len(interf_ids)} total)")

    build_trace(roots, valuable_ids, interf_ids)
