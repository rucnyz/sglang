"""S1 high-concurrency win analysis. Reads the enhanced replay driver's per-request
rows (program_id, prompt_tokens, cached_tokens) for the baseline (a3_kvoff) and ours
(a3) on the SAME trace, and reports:

  * per-arm cache-hit + re-prefilled tokens (the cost);
  * the RECOVERABLE eviction-reuse in the baseline (a prefix that was cacheable but
    got evicted then reused) = the opportunity;
  * ours' realized reduction in re-prefill = the S1 win on a realistic high-
    concurrency agent mix.

recoverable per request (ordered per program):
  expected = max(this program's prior prompt, SHARED if any earlier program ran),
             capped at this prompt; recoverable = max(0, expected - cached).

Usage: python analyze_opportunity.py <results_dir> [shared_sys_tokens=16000]
"""
import sys, os, json, glob
from collections import defaultdict

RESDIR = sys.argv[1]
SHARED = int(sys.argv[2]) if len(sys.argv) > 2 else 16000


def load_rows(arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(RESDIR, f"metrics_{arm}_c*.json"))):
        d = json.load(open(f))
        for r in d.get("rows", []):
            if r.get("prompt_tokens") is not None:
                rows.append((r.get("program_id"), int(r["prompt_tokens"]),
                             int(r.get("cached_tokens") or 0)))
    return rows


def analyze(arm):
    rows = load_rows(arm)
    if not rows:
        return None
    tot_p = tot_c = recover = 0
    prior = {}
    seen = False
    per_prog = defaultdict(int)
    for pid, pt, ct in rows:
        exp = prior.get(pid, 0)
        if seen:
            exp = max(exp, SHARED)
        exp = min(exp, pt)
        rec = max(0, exp - ct)
        tot_p += pt; tot_c += ct; recover += rec
        if rec > 1000:
            per_prog[pid] += rec
        prior[pid] = max(prior.get(pid, 0), pt)
        seen = True
    return {"arm": arm, "n": len(rows), "prompt": tot_p, "cached": tot_c,
            "reprefill": tot_p - tot_c, "hit": tot_c / tot_p if tot_p else 0,
            "recoverable": recover, "per_prog": per_prog}


base = analyze("a3_kvoff")
ours = analyze("a3")
print(f"=== S1 high-concurrency: ours (a3) vs baseline (a3_kvoff) on {os.path.basename(RESDIR)} ===")
for r in (base, ours):
    if r:
        print(f"  {r['arm']:<9} n={r['n']} prompt={r['prompt']} cached={r['cached']} "
              f"hit={r['hit']*100:.1f}%  reprefill={r['reprefill']}  recoverable={r['recoverable']}")
if base and ours:
    saved = base["reprefill"] - ours["reprefill"]
    print(f"\n  OURS re-prefill reduction = {saved} tok "
          f"({saved/base['reprefill']*100:.1f}% of baseline re-prefill, "
          f"{saved/base['recoverable']*100:.1f}% of the recoverable opportunity)")
    print(f"  cache-hit: baseline {base['hit']*100:.1f}% -> ours {ours['hit']*100:.1f}%  "
          f"(+{(ours['hit']-base['hit'])*100:.1f} pts)")
elif base:
    print(f"\n  baseline only: recoverable opportunity = {base['recoverable']} tok "
          f"({base['recoverable']/base['prompt']*100:.1f}% of prompt tokens)")
    top = sorted(base["per_prog"].items(), key=lambda x: -x[1])[:6]
    for p, v in top:
        print(f"    {str(p)[:26]:<26} {v}")
