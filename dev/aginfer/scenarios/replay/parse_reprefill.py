#!/usr/bin/env python3
"""#231 benefit mechanism — re-prefilled tokens / cache-hit rate per arm.

The daemon's value (vs LRU) is keeping high-value reused KV resident so it is
NOT re-prefilled. That shows directly in sglang's per-request logging:

    prompt_tokens  = total prompt length
    cached_tokens  = prefix served from the radix cache (a hit, no prefill)
    re-prefilled   = prompt_tokens - cached_tokens  (recomputed)

Lower re-prefilled tokens / higher cache-hit rate = the benefit, independent of
whether throughput is compute-bound. We read each trial's
results/run_K_<arm>_benefit_<label>_c<i>/sglang_v4flash.log.

Usage: python parse_reprefill.py <benefit_results_dir>
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = "/scratch/yuzhou/projects/sglang/dev/aginfer"

_PROMPT = re.compile(r'"?prompt_tokens"?[:=]\s*(\d+)')
_CACHED = re.compile(r'"?cached_tokens"?[:=]\s*(\d+)')


def trial_dirs(arm_label):
    # cycle dirs are results/run_K_<variant>_benefit_<label>_c<i>
    pats = [
        os.path.join(ROOT, "results", f"run_K_*_benefit_{arm_label}_c*"),
    ]
    out = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return out


def parse_log(path):
    """Sum prompt + cached tokens over all requests in a sglang log."""
    if not os.path.isfile(path):
        return None
    p_sum = c_sum = n = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "prompt_tokens" not in line:
                continue
            mp = _PROMPT.search(line)
            mc = _CACHED.search(line)
            if not mp:
                continue
            p = int(mp.group(1))
            c = int(mc.group(1)) if mc else 0
            p_sum += p
            c_sum += c
            n += 1
    if n == 0:
        return None
    return {"n_req": n, "prompt_tokens": p_sum, "cached_tokens": c_sum,
            "reprefilled": p_sum - c_sum,
            "cache_hit_rate": c_sum / p_sum if p_sum else 0.0}


def arm_summary(arm_label):
    rows = []
    for d in trial_dirs(arm_label):
        r = parse_log(os.path.join(d, "sglang_v4flash.log"))
        if r:
            rows.append(r)
    return rows


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return (m, sd, len(xs))


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    # arm labels are the metrics_<label>_c<i>.json stems
    labels = sorted({os.path.basename(f)[len("metrics_"):].rsplit("_c", 1)[0]
                     for f in glob.glob(os.path.join(results_dir, "metrics_*.json"))})
    print(f"=== re-prefill mechanism (arms: {labels}) ===")
    summ = {}
    for lab in labels:
        rows = arm_summary(lab)
        if not rows:
            print(f"  {lab}: no sglang logs found")
            continue
        rep = _ms([r["reprefilled"] for r in rows])
        hit = _ms([r["cache_hit_rate"] for r in rows])
        pt = _ms([r["prompt_tokens"] for r in rows])
        summ[lab] = (rep, hit, pt)
        print(f"  {lab:9s} n={rep[2]}  reprefilled_tok={rep[0]:.0f}±{rep[1]:.0f}  "
              f"cache_hit={hit[0]*100:.1f}%±{hit[1]*100:.1f}  prompt_tok={pt[0]:.0f}")
    # benefit verdict: a3 (ours) should re-prefill FEWER than a3_kvoff (LRU)
    if "a3" in summ and "a3_kvoff" in summ:
        o_rep, o_hit, _ = summ["a3"]
        b_rep, b_hit, _ = summ["a3_kvoff"]
        d_rep = b_rep[0] - o_rep[0]
        d_pct = d_rep / b_rep[0] * 100 if b_rep[0] else 0.0
        print(f"\nBENEFIT (ours=a3 vs LRU=a3_kvoff):")
        print(f"  re-prefilled tokens: ours {o_rep[0]:.0f} vs LRU {b_rep[0]:.0f}  "
              f"-> ours saves {d_rep:.0f} ({d_pct:+.1f}%)")
        print(f"  cache-hit rate: ours {o_hit[0]*100:.1f}% vs LRU {b_hit[0]*100:.1f}%")
        # significance: disjoint mean±std bands
        stable = (o_rep[0] + o_rep[1]) < (b_rep[0] - b_rep[1])
        print(f"  ours re-prefills STABLY FEWER (bands disjoint)? {stable}")


if __name__ == "__main__":
    main()
