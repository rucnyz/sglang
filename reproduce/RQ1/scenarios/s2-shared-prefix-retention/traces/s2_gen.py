"""S2 — shared-prefix-retention workload generator.

Builds a replay trace that engineers the divergence the scenario needs:
  * a FLEET of agents sharing ONE identical ~16K system prefix, active in BURSTS
    (each fleet agent re-touches the shared prefix once per burst);
  * heavy CONTINUOUS BACKGROUND churn (agents with UNIQUE prefixes, no sharing) that
    floods the cache during the fleet's idle gaps.

Goal: during each fleet-idle gap the background evicts the fleet's continuations →
the shared prefix becomes a stale leaf → LRU drops it → the fleet's NEXT burst
recomputes the 16K shared prefix. Ours (holder-count value) keeps it pinned.

Output: one JSONL record per (program, turn), replay_driver format.
Usage: python s2_gen.py <out.jsonl> [fleet=12] [bursts=8] [gap_s=25] [bg_per_gap=14]
                                     [shared_tok=16000] [fleet_scratch=4000] [bg_scratch=14000]
"""
import json, sys

OUT = sys.argv[1]
FLEET       = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BURSTS      = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GAP_S       = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
BG_PER_GAP  = int(sys.argv[5]) if len(sys.argv) > 5 else 14
SHARED_TOK  = int(sys.argv[6]) if len(sys.argv) > 6 else 16000
FLEET_SCR   = int(sys.argv[7]) if len(sys.argv) > 7 else 4000
BG_SCR      = int(sys.argv[8]) if len(sys.argv) > 8 else 14000

# ~4 chars/token. Build a DETERMINISTIC, identical shared system prefix (the fleet prefix).
def words(n_tok, seed):
    # deterministic pseudo-text of ~n_tok tokens; seed makes background unique, fleet identical
    out = []
    x = seed * 2654435761 % (2**31)
    vocab = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india",
             "juliet","kilo","lima","mike","november","oscar","papa","quebec","romeo",
             "sierra","tango","uniform","victor","whiskey","xray","yankee","zulu"]
    for _ in range(n_tok):
        x = (x * 1103515245 + 12345) % (2**31)
        out.append(vocab[x % len(vocab)])
    return " ".join(out)

SHARED = "SYSTEM PROMPT v1. You are a shared fleet agent. Tool definitions follow.\n" + words(SHARED_TOK, seed=1)

def rec(pid, t, sys_text, user_text, out_len):
    return {"t": round(t, 3), "program_id": pid,
            "body": {"messages": [{"role": "system", "content": sys_text},
                                   {"role": "user", "content": user_text}],
                     "model": "x", "temperature": 0.0},
            "output_len": out_len,
            "ref_e2e_ms": round((len(sys_text)+len(user_text))//4 * 0.1 + out_len*15)}

records = []
# FLEET: each agent re-touches SHARED once per burst (bursts spaced GAP_S apart)
for b in range(BURSTS):
    t0 = b * GAP_S
    for f in range(FLEET):
        # divergent per-agent scratch (unique), but SHARED system prefix identical
        scratch = words(FLEET_SCR, seed=1000 + f*BURSTS + b)
        records.append(rec(f"fleet-{f}", t0 + f*0.05, SHARED, scratch, out_len=64))

# BACKGROUND: unique-prefix agents filling each idle gap with heavy churn
bg_id = 0
for b in range(BURSTS):
    t0 = b * GAP_S + 1.0  # start just after the burst, fill the gap
    for k in range(BG_PER_GAP):
        uniq_sys = f"BG AGENT {bg_id} UNIQUE SYSTEM. " + words(2000, seed=50000 + bg_id)
        scratch  = words(BG_SCR, seed=90000 + bg_id)
        t = t0 + k * (GAP_S - 2.0) / max(1, BG_PER_GAP)
        records.append(rec(f"bg-{bg_id}", t, uniq_sys, scratch, out_len=64))
        bg_id += 1

records.sort(key=lambda r: r["t"])
with open(OUT, "w") as fh:
    fh.write("\n".join(json.dumps(r) for r in records) + "\n")

n_fleet = sum(1 for r in records if r["program_id"].startswith("fleet-"))
n_bg = sum(1 for r in records if r["program_id"].startswith("bg-"))
span = max(r["t"] for r in records)
bg_working = n_bg * (2000 + BG_SCR) // 1
print(f"S2 trace -> {OUT}")
print(f"  fleet={FLEET} agents x {BURSTS} bursts = {n_fleet} reqs (shared {SHARED_TOK//1000}K prefix)")
print(f"  background={n_bg} reqs (unique prefixes, ~{BG_SCR//1000}K scratch each)")
print(f"  span={span:.0f}s, bg working-set ~{bg_working//1000}K tok (flood the {0} pool)")
print(f"  shared prefix tokens (system msg): ~{len(SHARED)//4} tok")
