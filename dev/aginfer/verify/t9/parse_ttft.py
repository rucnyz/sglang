#!/usr/bin/env python3
"""Parse per-request sglang JSON logs from the N=3 matrix run.

Usage:
    python verify/t9/parse_ttft.py <matrix_root>

For each cycleN_<config>/, parse sglang_v4flash.log for
`request.finished` events, extract per-request:
  * e2e_latency
  * queue_time
  * prompt_tokens, cached_tokens, completion_tokens
  * cache hit ratio = cached_tokens / prompt_tokens

Compare distributions across baseline vs ours.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


FINISHED_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] (\{.*\})\s*$')


def parse_sglang_json(log_path: Path):
    """Yield per-request records (only `request.finished` events with meta_info)."""
    if not log_path.exists():
        return
    with log_path.open(errors="ignore") as f:
        for line in f:
            m = FINISHED_RE.match(line)
            if not m:
                continue
            try:
                doc = json.loads(m.group(1))
            except Exception:
                continue
            if doc.get("event") != "request.finished":
                continue
            out = doc.get("out", {})
            mi = out.get("meta_info", {})
            if not mi:
                continue
            yield {
                "e2e_latency": float(mi.get("e2e_latency") or 0.0),
                "queue_time": float(mi.get("queue_time") or 0.0),
                "prompt_tokens": int(mi.get("prompt_tokens") or 0),
                "cached_tokens": int(mi.get("cached_tokens") or 0),
                "completion_tokens": int(mi.get("completion_tokens") or 0),
                "request_received_ts": float(mi.get("request_received_ts") or 0.0),
                "request_finished_ts": float(mi.get("request_finished_ts") or 0.0),
                "response_sent_to_client_ts": float(mi.get("response_sent_to_client_ts") or 0.0),
                "api_server_dispatch_finish_ts": float(mi.get("api_server_dispatch_finish_ts") or 0.0),
            }


def stats(xs, label=""):
    if not xs:
        return f"{label}: n=0"
    xs = sorted(xs)
    n = len(xs)
    mean = statistics.mean(xs)
    p50 = xs[n // 2]
    p90 = xs[int(0.9 * n)]
    p99 = xs[int(0.99 * n)]
    std = statistics.stdev(xs) if n > 1 else 0.0
    return f"n={n} mean={mean:.3f} std={std:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f} max={xs[-1]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_root", type=Path)
    args = ap.parse_args()

    root: Path = args.matrix_root
    by_config = defaultdict(list)
    for child in sorted(root.iterdir()):
        m = re.match(r"cycle(\d+)_(\w+)", child.name)
        if not m:
            continue
        cfg = m.group(2)
        target = child.resolve()
        log = target / "sglang_v4flash.log"
        if not log.exists():
            print(f"warn: no sglang_v4flash.log in {target}", file=sys.stderr)
            continue
        n = 0
        for rec in parse_sglang_json(log):
            by_config[cfg].append(rec)
            n += 1
        print(f"{target.name}: {n} requests parsed", file=sys.stderr)

    print(f"# Per-request TTFT/queue analysis across N=3 matrix\n")

    for cfg in sorted(by_config):
        recs = by_config[cfg]
        print(f"## {cfg} (N={len(recs)} requests across 3 cycles)\n")

        e2e = [r["e2e_latency"] for r in recs]
        qt = [r["queue_time"] for r in recs]
        prompt = [r["prompt_tokens"] for r in recs]
        cached = [r["cached_tokens"] for r in recs]
        completion = [r["completion_tokens"] for r in recs]

        # cache hit ratio per request (cached/prompt)
        ratios = [
            r["cached_tokens"] / r["prompt_tokens"]
            for r in recs
            if r["prompt_tokens"] > 0
        ]
        # latency per cached vs not
        e2e_cached = [r["e2e_latency"] for r in recs if r["cached_tokens"] > 0]
        e2e_uncached = [r["e2e_latency"] for r in recs if r["cached_tokens"] == 0]
        # api server dispatch overhead
        dispatch_lag = [
            (r["api_server_dispatch_finish_ts"] - r["request_received_ts"]) * 1000
            for r in recs
            if r["api_server_dispatch_finish_ts"] > 0 and r["request_received_ts"] > 0
        ]
        # response-sent vs finished (post-LLM overhead)
        post_lag = [
            (r["response_sent_to_client_ts"] - r["request_finished_ts"]) * 1000
            for r in recs
            if r["response_sent_to_client_ts"] > 0 and r["request_finished_ts"] > 0
        ]

        print(f"* e2e_latency (s):        {stats(e2e)}")
        print(f"* queue_time (s):         {stats(qt)}")
        print(f"* prompt_tokens:          {stats(prompt)}")
        print(f"* cached_tokens:          {stats(cached)}")
        print(f"* completion_tokens:      {stats(completion)}")
        print(f"* hit_ratio:              {stats(ratios)}")
        print(f"* e2e_latency cached>0:   {stats(e2e_cached)}")
        print(f"* e2e_latency cached=0:   {stats(e2e_uncached)}")
        print(f"* api_dispatch_lag (ms):  {stats(dispatch_lag)}")
        print(f"* post_send_lag (ms):     {stats(post_lag)}")

        # Sum of all e2e latency / per-trial mean wall (for orientation)
        total = sum(e2e)
        per_trial = total / (32 * 3)  # 3 cycles × 32 trials
        print(f"* Σ e2e_latency = {total:.0f} s; per-trial-equivalent = **{per_trial:.1f} s**")
        print()

    # Side-by-side comparison
    if "baseline" in by_config and "ours" in by_config:
        b = by_config["baseline"]
        o = by_config["ours"]
        print(f"## Side-by-side: baseline ({len(b)} reqs) vs ours ({len(o)} reqs)\n")
        for key, label in [
            ("e2e_latency", "e2e_latency (s)"),
            ("queue_time", "queue_time (s)"),
            ("cached_tokens", "cached_tokens"),
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
        ]:
            b_vals = [r[key] for r in b]
            o_vals = [r[key] for r in o]
            b_mean = statistics.mean(b_vals) if b_vals else 0
            o_mean = statistics.mean(o_vals) if o_vals else 0
            delta = o_mean - b_mean
            print(f"* {label}: baseline mean={b_mean:.3f}, ours mean={o_mean:.3f}, Δ={delta:+.3f}")


if __name__ == "__main__":
    main()
