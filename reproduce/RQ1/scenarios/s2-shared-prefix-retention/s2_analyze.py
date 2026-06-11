"""S2 analysis — shared-prefix retention.

The fleet (program_id 'fleet-*') all share an identical ~SHARED_TOK system prefix. A
fleet request that HITS the shared prefix has cached_tokens >= ~SHARED_TOK; one that
MISSES (LRU dropped it under background churn) recomputes it (cached < SHARED_TOK).

Reports, per arm (ours=a3, baseline=a3_kvoff): how many fleet requests recomputed the
shared prefix, total shared-prefix tokens recomputed, and fleet reuse-TTFT. The win =
ours keeps the prefix (few recomputes) vs B drops+recomputes it under churn.

Usage: python s2_analyze.py <results_dir> [arm=a3_kvoff] [shared_tok=24000]
"""
import sys, os, json, glob, statistics

RESDIR = sys.argv[1]
ARM = sys.argv[2] if len(sys.argv) > 2 else "a3_kvoff"
SHARED = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
HIT_THRESH = int(SHARED * 0.85)   # >= 85% of shared cached -> counts as "kept"


def load(arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(RESDIR, f"metrics_{arm}_c*.json"))):
        d = json.load(open(f))
        for r in d.get("rows", []):
            if r.get("program_id") and r.get("prompt_tokens") is not None:
                rows.append(r)
    return rows


rows = load(ARM)
fleet = [r for r in rows if str(r.get("program_id")).startswith("fleet-")]
bg = [r for r in rows if str(r.get("program_id")).startswith("bg-")]
if not fleet:
    print(f"no fleet rows for arm={ARM} in {RESDIR}")
    sys.exit(0)

# a fleet request "kept" the shared prefix if cached >= 85% of SHARED
kept = [r for r in fleet if int(r.get("cached_tokens") or 0) >= HIT_THRESH]
miss = [r for r in fleet if int(r.get("cached_tokens") or 0) < HIT_THRESH]
recomputed_shared = sum(max(0, SHARED - int(r.get("cached_tokens") or 0)) for r in miss)
fleet_ttft = [r["ttft_ms"] for r in fleet if r.get("ok") and r.get("ttft_ms") is not None]
# exclude the very first cold touch per agent (unavoidable cold-start)

print(f"=== S2 shared-prefix retention — arm={ARM} ({os.path.basename(RESDIR)}) ===")
print(f"  shared prefix ~{SHARED} tok; fleet reqs={len(fleet)}, bg reqs={len(bg)}")
print(f"  fleet reqs that KEPT shared (cached>=85%): {len(kept)}/{len(fleet)}")
print(f"  fleet reqs that RECOMPUTED shared (cached<85%): {len(miss)}/{len(fleet)}  <- the LRU penalty")
print(f"  total shared-prefix tokens RECOMPUTED: {recomputed_shared}")
if fleet_ttft:
    print(f"  fleet TTFT: p50={statistics.median(fleet_ttft):.0f}ms  mean={statistics.mean(fleet_ttft):.0f}ms  "
          f"p90={sorted(fleet_ttft)[int(len(fleet_ttft)*0.9)]:.0f}ms")
# cached distribution
import collections
buckets = collections.Counter()
for r in fleet:
    c = int(r.get("cached_tokens") or 0)
    buckets[f"{c//5000*5}-{c//5000*5+5}K"] += 1
print("  fleet cached_tokens histogram:", dict(sorted(buckets.items())))
