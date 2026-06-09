#!/usr/bin/env python3
"""Part A — is in-loop teacher-forcing a no-op vs normal greedy generation?

Method (isolate the MECHANISM from content): run a prompt under greedy
(temp=0) to capture the model's OWN argmax output O*. Then FORCE the model to
emit exactly O* via ForcedDecodingLogitProcessor. Because both emit identical
tokens, any difference is purely the forcing mechanism.

Checks:
  1. correctness — forced output token ids == O* (forcing actually works).
  2. timing     — total latency + TTFT, N reps each, mean±std overlap within noise.

State (radix-tree) identity is checked separately (Part A-state) once the daemon
/aginfer/state is attached; here we validate behaviour + timing on bare sglang.

Run: sglang must be up with --enable-custom-logit-processor (see run_partA.sh).
  python test_partA.py --base-url http://127.0.0.1:30000 --reps 7 --out-len 256
"""
import argparse
import json
import statistics
import sys
import time

import requests

# Faithful teacher-forcing is done by an aginfer override at sglang's
# output-commit point (batch_result_processor._aginfer_force_token), driven by
# sampling_params.custom_params["forced_output_ids"] — NOT a logit processor
# (which added ~30% under the overlap scheduler). No special server flag needed.

PROMPT = (
    "You are a meticulous software engineer. Explain, in detail and step by step, "
    "how a log-structured merge tree works, why it is well suited to write-heavy "
    "workloads, and what the read-amplification trade-offs are. Then sketch how "
    "compaction scheduling interacts with tail latency."
)


def gen(base_url, *, out_len, forced_ids=None):
    """One /generate call. forced_ids set => teacher-force those tokens.
    Returns (output_token_ids, ttft_ms, total_ms)."""
    # ignore_eos so BOTH arms do exactly out_len tokens — otherwise the
    # (non-deterministic) baseline greedy can hit EOS early, do fewer tokens, and
    # look faster, confounding the no-op timing comparison. Forced does exactly
    # len(forced) tokens by construction; matching the baseline's count isolates
    # the override mechanism as the only difference.
    sp = {"temperature": 0.0, "max_new_tokens": out_len, "ignore_eos": True}
    body = {"text": PROMPT, "sampling_params": sp, "return_logprob": True,
            "stream": False}
    if forced_ids is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced_ids)}
    t0 = time.perf_counter()
    r = requests.post(base_url.rstrip("/") + "/generate", json=body, timeout=600)
    total_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    d = r.json()
    mi = d["meta_info"]
    otl = mi.get("output_token_logprobs") or []
    ids = [int(t[1]) for t in otl]
    # sglang reports e2e/ttft in meta_info when available; fall back to wall total
    ttft_ms = mi.get("first_token_latency")
    ttft_ms = float(ttft_ms) * 1000.0 if ttft_ms is not None else float("nan")
    return ids, ttft_ms, total_ms


def band(xs):
    xs = [x for x in xs if x == x]  # drop nan
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (m, s, len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out-len", type=int, default=256)
    a = ap.parse_args()

    # capture O* (the model's own greedy output)
    print("[partA] capturing baseline greedy output O* ...", flush=True)
    O_star, _, _ = gen(a.base_url, out_len=a.out_len)
    print(f"[partA] O* length = {len(O_star)} tokens", flush=True)
    if len(O_star) < a.out_len // 2:
        print(f"WARNING: O* shorter than expected ({len(O_star)} < {a.out_len}); "
              f"model hit EOS — forcing will reproduce only this length", flush=True)

    # warmup one of each (compile / cache) then timed reps
    gen(a.base_url, out_len=len(O_star))
    gen(a.base_url, out_len=len(O_star), forced_ids=O_star)

    base_tot, base_ttft, force_tot, force_ttft = [], [], [], []
    mismatches = 0
    for i in range(a.reps):
        ids_b, ttft_b, tot_b = gen(a.base_url, out_len=len(O_star))
        ids_f, ttft_f, tot_f = gen(a.base_url, out_len=len(O_star), forced_ids=O_star)
        base_tot.append(tot_b); base_ttft.append(ttft_b)
        force_tot.append(tot_f); force_ttft.append(ttft_f)
        if ids_b != O_star:
            print(f"  rep{i}: baseline drifted from O* (nondeterminism?) "
                  f"first-diff at {next((k for k in range(min(len(ids_b),len(O_star))) if ids_b[k]!=O_star[k]), 'len')}",
                  flush=True)
        if ids_f != O_star:
            mismatches += 1
            k = next((k for k in range(min(len(ids_f), len(O_star))) if ids_f[k] != O_star[k]), None)
            print(f"  rep{i}: FORCED output != O*  (first diff idx {k})", flush=True)
        print(f"  rep{i}: base {tot_b:7.1f}ms  forced {tot_f:7.1f}ms", flush=True)

    bt, bs, bn = band(base_tot); ft, fs, fn = band(force_tot)
    btt = band(base_ttft); ftt = band(force_ttft)
    print("\n=== Part A result ===")
    print(f"correctness: forced == O* on {a.reps - mismatches}/{a.reps} reps "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")
    print(f"total latency  base {bt:.1f}±{bs:.1f}ms   forced {ft:.1f}±{fs:.1f}ms   "
          f"Δ {ft-bt:+.1f}ms ({100*(ft-bt)/bt:+.2f}%)")
    print(f"TTFT           base {btt[0]:.1f}±{btt[1]:.1f}ms forced {ftt[0]:.1f}±{ftt[1]:.1f}ms")
    # verdict: forced within base's noise band (2σ) on total latency
    overlap = abs(ft - bt) <= 2 * max(bs, fs, 1e-9)
    print(f"timing no-op (|Δ| ≤ 2σ): {'PASS' if overlap else 'REVIEW'}")
    print(json.dumps({"o_star_len": len(O_star), "reps": a.reps,
                      "force_correct": mismatches == 0,
                      "base_total_ms": [bt, bs], "forced_total_ms": [ft, fs],
                      "delta_pct": 100*(ft-bt)/bt if bt else None,
                      "timing_overlap_2sigma": overlap}))
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
