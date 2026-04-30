"""Phase 1 metrics sampling client.

Polls SGLang's /metrics every --interval seconds and writes one JSONL line
per scrape with the 24 signals Phase 2's budgeter cares about.

Usage:
    python dev/1/sample_metrics.py \\
        --host 127.0.0.1 --port 30000 \\
        --interval 1.0 \\
        --out dev/1/<run_name>.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import urllib.request
from typing import Optional

# Map: prom metric name (with sglang: prefix) → JSON output key.
# Gauges: take last value. Counters: take last value too (driver computes deltas).
_METRICS = {
    # paged KV pool
    "sglang:max_total_num_tokens": "kv_max",
    "sglang:kv_used_tokens": "kv_used",
    "sglang:kv_evictable_tokens": "kv_evictable",
    "sglang:kv_available_tokens": "kv_available",
    "sglang:token_usage": "token_usage",
    "sglang:full_token_usage": "full_token_usage",
    "sglang:swa_token_usage": "swa_token_usage",
    # SSM pool
    "sglang:mamba_usage": "mamba_usage",
    # LoRA pool
    "sglang:lora_pool_slots_used": "lora_used",
    "sglang:lora_pool_slots_total": "lora_total",
    "sglang:lora_pool_utilization": "lora_util",
    # prefix cache (logical view of paged KV evictable region)
    "sglang:cache_hit_rate": "cache_hit_rate",
    "sglang:cached_tokens_total": "cached_tokens_total",
    # eviction / preemption
    "sglang:num_retracted_reqs": "num_retracted",
    "sglang:num_paused_reqs": "num_paused",
    "sglang:num_retracted_requests_total": "num_retracted_total",
    "sglang:evicted_tokens_total": "evicted_tokens_total",
    # queue / throughput
    "sglang:num_running_reqs": "num_running",
    "sglang:num_queue_reqs": "num_queue",
    "sglang:gen_throughput": "gen_throughput",
    # latency histograms (we keep cumulative count + sum so consumers can derive rate / mean)
    "sglang:time_to_first_token_seconds_count": "ttft_count",
    "sglang:time_to_first_token_seconds_sum": "ttft_sum",
    # cumulative request counters
    "sglang:prompt_tokens_total": "prompt_tokens_total",
    "sglang:generation_tokens_total": "generation_tokens_total",
}

# Pattern matches `sglang:metric_name{...labels...} VALUE` per line.
_LINE_RE = re.compile(r"^(sglang:[A-Za-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


def scrape(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Fetch /metrics, return {prom_name: last_value}."""
    out: dict[str, float] = {}
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode()
    for line in body.split("\n"):
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if name in _METRICS:
            try:
                # Last value wins if there are multiple labelsets (we only run TP=1)
                out[name] = float(val)
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between scrapes")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="if >0, stop after this many samples (else run until SIGINT)")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/metrics"

    stop = False
    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    n = 0
    consec_errors = 0
    print(f"sampling {url} every {args.interval}s -> {args.out}", file=sys.stderr)
    with open(args.out, "w", buffering=1) as f:
        while not stop:
            t0 = time.time()
            ts = time.time()
            try:
                raw = scrape(url, timeout=max(2.0, args.interval * 0.8))
                consec_errors = 0
            except Exception as e:
                consec_errors += 1
                if consec_errors == 1 or consec_errors % 30 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] scrape err ({consec_errors}): {e}",
                          file=sys.stderr)
                if consec_errors > 600:  # ~10 min of failures
                    print("too many failures, exiting", file=sys.stderr)
                    return 1
                # Sleep and retry
                time.sleep(args.interval)
                continue

            row = {"ts": round(ts, 3)}
            for prom_name, json_key in _METRICS.items():
                if prom_name in raw:
                    row[json_key] = raw[prom_name]
            f.write(json.dumps(row) + "\n")
            n += 1
            if args.max_samples and n >= args.max_samples:
                break

            # Drift-corrected sleep
            elapsed = time.time() - t0
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)

    print(f"wrote {n} samples to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
