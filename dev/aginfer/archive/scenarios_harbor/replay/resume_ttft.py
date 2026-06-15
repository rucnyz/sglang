"""Resume-request TTFT — the metric that actually isolates the S1 win.

S1's predictive promote helps the RESUME requests (a program's 2nd+ turn, after a
tool gap): ours pre-stages the prefix so the resume is a cache hit (fast TTFT), while
the baseline evicted it (recompute, slow TTFT). Aggregate cache-hit dilutes this; here
we isolate resume requests and compare ours (a3) vs baseline (a3_kvoff) TTFT, N trials,
mean±std.

A "resume" = the 2nd+ request of a program (turn-index >= 1, by per-program order).
Turn 0 = the cold establish (recompute for both arms — not where S1 helps).

Usage: python resume_ttft.py <results_dir_ours> [results_dir_base]
  (if one dir holds both a3 and a3_kvoff, pass it once)
"""
import sys, os, json, glob
from collections import defaultdict
from statistics import median, mean, pstdev

DIR_OURS = sys.argv[1]
DIR_BASE = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]


def trial_resume_ttfts(path):
    """Return (resume_ttfts, establish_ttfts) for one trial file."""
    d = json.load(open(path))
    rows = d.get("rows", [])
    seen = defaultdict(int)
    resume, establish = [], []
    for r in rows:
        if not r.get("ok") or r.get("ttft_ms") is None:
            continue
        pid = r.get("program_id")
        turn = seen[pid]
        seen[pid] += 1
        (resume if turn >= 1 else establish).append(float(r["ttft_ms"]))
    return resume, establish


def arm(dirpath, arm_name):
    files = sorted(glob.glob(os.path.join(dirpath, f"metrics_{arm_name}_c*.json")))
    if not files:
        return None
    per_trial_resume_med, per_trial_resume_mean, n_res = [], [], 0
    all_resume = []
    for f in files:
        res, est = trial_resume_ttfts(f)
        if res:
            per_trial_resume_med.append(median(res))
            per_trial_resume_mean.append(mean(res))
            all_resume += res
            n_res = len(res)
    if not per_trial_resume_med:
        return None
    return {
        "arm": arm_name, "trials": len(files),
        "resume_per_trial": n_res,
        "resume_med_mean": mean(per_trial_resume_med),
        "resume_med_std": pstdev(per_trial_resume_med) if len(per_trial_resume_med) > 1 else 0.0,
        "resume_mean_mean": mean(per_trial_resume_mean),
        "resume_p50_all": median(all_resume),
    }


ours = arm(DIR_OURS, "a3")
base = arm(DIR_BASE, "a3_kvoff")
print("=== Resume-request TTFT (S1's actual win metric): ours (a3) vs baseline (a3_kvoff) ===")
for r in (base, ours):
    if r:
        print(f"  {r['arm']:<9} trials={r['trials']} resume_reqs/trial={r['resume_per_trial']}  "
              f"resume-TTFT median = {r['resume_med_mean']:.0f} ± {r['resume_med_std']:.0f} ms  "
              f"(mean {r['resume_mean_mean']:.0f}, pooled-p50 {r['resume_p50_all']:.0f})")
if ours and base:
    d = base["resume_med_mean"] - ours["resume_med_mean"]
    pct = d / base["resume_med_mean"] * 100 if base["resume_med_mean"] else 0
    verdict = "OURS WINS" if d > 0 else ("FLAT" if abs(pct) < 5 else "OURS LOSES")
    print(f"\n  resume-TTFT: ours is {d:+.0f} ms vs baseline ({pct:+.1f}%) -> {verdict}")
    # crude significance: gap vs combined std
    cstd = (ours["resume_med_std"]**2 + base["resume_med_std"]**2) ** 0.5
    print(f"  (gap {abs(d):.0f} ms vs combined std {cstd:.0f} ms — "
          f"{'separable' if abs(d) > cstd else 'within noise'})")
