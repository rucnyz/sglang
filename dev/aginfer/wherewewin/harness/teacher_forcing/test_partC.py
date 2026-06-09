#!/usr/bin/env python3
"""Part C — is the teacher-forcing override a no-op under CONTINUOUS BATCHING?

Part A showed no-op at batch-1; the real replay runs high concurrency, so verify
the override (a per-token int swap inside the existing per-req commit loop) does
not perturb aggregate throughput or per-request latency at batch.

N independent requests (distinct salted prompts → no prefix dedup), each decoding
L tokens, sent concurrently. Two conditions, M rounds each:
  baseline : no forcing (ignore_eos, max_new=L)
  forced   : each request forced to O* (L valid token ids), ignore_eos, max_new=L
Both do N*L decode tokens. Compare throughput (tok/s) + e2e latency. PASS = overlap
within noise.

Run: sglang up (override in-code, no flag). See run_partC.sh.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

P_BASE = list(range(1000, 1000 + 200))
L = 128


def gen(base, input_ids, max_new, forced=None):
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    body = {"input_ids": input_ids, "sampling_params": sp, "stream": False}
    t0 = time.perf_counter()
    r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=600)
    e2e = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    n_out = int(r.json()["meta_info"].get("completion_tokens") or max_new)
    return n_out, e2e


def capture_ostar(base):
    n, _ = 0, None
    body = {"input_ids": P_BASE, "sampling_params":
            {"temperature": 0.0, "max_new_tokens": L, "ignore_eos": True},
            "return_logprob": True, "stream": False}
    r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=600)
    r.raise_for_status()
    otl = r.json()["meta_info"].get("output_token_logprobs") or []
    return [int(t[1]) for t in otl]


def run_batch(base, n_conc, forced_ids):
    """Fire n_conc concurrent requests; return (throughput_tok_s, [e2e_ms...])."""
    # distinct salt per request so prefills don't dedup in the radix cache
    inputs = [[20000 + i] + P_BASE for i in range(n_conc)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_conc) as ex:
        results = list(ex.map(lambda inp: gen(base, inp, L, forced_ids), inputs))
    wall = time.perf_counter() - t0
    total_out = sum(n for n, _ in results)
    e2es = [e for _, e in results]
    return total_out / wall, e2es


def band(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def level(base, n_conc, o_star, rounds):
    """One concurrency level: warm up, then `rounds` timed base/forced batches."""
    # 2 warmups of each to clear cold-start (the round-0 outlier seen at low conc)
    for _ in range(2):
        run_batch(base, n_conc, None)
        run_batch(base, n_conc, o_star)
    base_tps, force_tps, base_p50, force_p50 = [], [], [], []
    for r in range(rounds):
        tps_b, e_b = run_batch(base, n_conc, None)
        tps_f, e_f = run_batch(base, n_conc, o_star)
        base_tps.append(tps_b); force_tps.append(tps_f)
        base_p50.append(statistics.median(e_b)); force_p50.append(statistics.median(e_f))
    bt, bs = band(base_tps); ft, fs = band(force_tps)
    bp, _ = band(base_p50); fp, _ = band(force_p50)
    dpct = 100 * (ft - bt) / bt if bt else 0.0
    overlap = abs(ft - bt) <= 2 * max(bs, fs, 1e-9)
    print(f"  conc={n_conc:4d}: base {bt:7.1f}±{bs:5.1f}  forced {ft:7.1f}±{fs:5.1f} tok/s  "
          f"Δ {dpct:+.2f}%  p50 {bp:.0f}/{fp:.0f}ms  {'PASS' if overlap else 'REVIEW'}")
    return {"concurrency": n_conc, "base_tps": [bt, bs], "forced_tps": [ft, fs],
            "delta_pct": dpct, "overlap_2sigma": overlap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--concurrency", default="96,128,192,256",
                    help="comma-separated concurrency levels")
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    base = a.base_url
    levels = [int(x) for x in a.concurrency.split(",")]
    o_star = capture_ostar(base)
    print(f"[partC] O* len={len(o_star)}, levels={levels}, rounds={a.rounds}, L={L}")
    print("=== Part C result (override no-op vs concurrency) ===")
    rows = [level(base, n, o_star, a.rounds) for n in levels]
    allpass = all(r["overlap_2sigma"] for r in rows)
    print(f"\nbatched no-op across all levels: {'PASS' if allpass else 'REVIEW'}")
    print(json.dumps({"rounds": a.rounds, "levels": rows, "pass": allpass}))
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
