#!/usr/bin/env python3
"""B1 phase-shift dispatcher.

Continuously issues requests at a target concurrency level; the prompt
source rotates between four phases of 90 seconds each (mamba-heavy ↔
KV-heavy ↔ mamba-heavy ↔ KV-heavy). No drains between phases — when one
phase ends, new arrivals immediately use the next phase's prompts while
in-flight requests from the prior phase are still being served. This is
the realistic-traffic phase-shift pattern.

Writes metrics.json with mean/median/P99 TTFT and median E2E aggregated
over the full trace plus per-phase splits.
"""
from __future__ import annotations
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import aiohttp


PORT = int(os.environ["PORT"])
MODEL = os.environ["MODEL"]
OUT_DIR = Path(os.environ["OUT_DIR"])
METRICS_PATH = Path(os.environ["METRICS_PATH"])

PHASE_DURATION_S = 90

# In hybrid models every request uses both a mamba SLOT (per-request) and KV
# TOKENS (per input-token + per output-token). Mamba pool is bottlenecked by
# concurrency; KV pool by aggregate tokens. So genuine phase-shift needs:
#   mamba phase: HIGH concurrency × short prompts  → mamba slot pressure
#   kv phase:    LOW  concurrency × long inputs+outputs → KV token pressure
# v7 used uniform 16 concurrency × short prompts in both phases, which kept
# mamba pinned ~0.95 the whole time and KV at ~0.00 — only 1 transfer fired
# and there was nothing more to do. v8 splits per-phase concurrency below.
MAMBA_CONCURRENCY = 32
KV_CONCURRENCY = 4

def make_mamba_heavy_prompt(idx: int) -> tuple[str, int]:
    """Short input, short output. Many concurrent → mamba slot pressure."""
    p = f"Question {idx}: what is {idx % 13 + 1} times {idx % 7 + 2}? Briefly.\n"
    return p, 32

def make_kv_heavy_prompt(idx: int) -> tuple[str, int]:
    """~8K-token unique input + 512-output. Few concurrent but each holds
    ~8.5K KV tokens for ~20-30s of generation → fills KV token budget."""
    rnd = random.Random(idx)
    tokens = [rnd.choice(["alpha", "beta", "gamma", "delta", "epsilon",
                          "zeta", "eta", "theta", "iota", "kappa"])
              for _ in range(8000)]
    p = " ".join(tokens) + f"\n\nGiven the above tokens, list 50 distinct words.\n"
    return p, 512

def phase_at(t: float) -> str:
    n = int(t // PHASE_DURATION_S) % 4
    return "mamba" if n in (0, 2) else "kv"

def concurrency_for_phase(ph: str) -> int:
    return MAMBA_CONCURRENCY if ph == "mamba" else KV_CONCURRENCY


async def fire_one(session, prompt: str, max_tokens: int, started_at: float, results: list,
                   phase_label: str):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    t0 = time.time()
    ttft_ms = None
    e2e_ms = None
    try:
        async with session.post(
            f"http://127.0.0.1:{PORT}/v1/completions",
            json=body, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            await resp.read()
            e2e_ms = (time.time() - t0) * 1000
            ttft_ms = e2e_ms  # without streaming we approximate; same for both arms
    except Exception as e:
        results.append({"phase": phase_label, "error": str(e)})
        return
    results.append({"phase": phase_label, "ttft_ms": ttft_ms, "e2e_ms": e2e_ms,
                    "started_at": started_at})


async def main():
    duration_total = PHASE_DURATION_S * 4  # 6 minutes
    t_start = time.time()
    results: list = []
    in_flight: set[asyncio.Task] = set()
    idx = 0
    async with aiohttp.ClientSession() as session:
        while time.time() - t_start < duration_total:
            t_now = time.time() - t_start
            ph = phase_at(t_now)
            target = concurrency_for_phase(ph)
            # Maintain phase-specific target concurrency.
            while len(in_flight) < target and (time.time() - t_start < duration_total):
                if ph == "mamba":
                    prompt, max_t = make_mamba_heavy_prompt(idx)
                else:
                    prompt, max_t = make_kv_heavy_prompt(idx)
                idx += 1
                started_at = t_now
                task = asyncio.create_task(fire_one(session, prompt, max_t, started_at, results, ph))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            # Brief pause to avoid busy-loop.
            await asyncio.sleep(0.05)
        # Drain remaining in-flight at end of trace.
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
    # Aggregate.
    full = [r for r in results if "ttft_ms" in r]
    mamba_phase = [r for r in full if r["phase"] == "mamba"]
    kv_phase = [r for r in full if r["phase"] == "kv"]

    def stats(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        ttfts = [r["ttft_ms"] for r in rs]
        e2es = [r["e2e_ms"] for r in rs]
        return {
            "n": len(rs),
            "mean_ttft_ms": statistics.mean(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p99_ttft_ms": statistics.quantiles(ttfts, n=100)[98] if len(ttfts) >= 100 else max(ttfts),
            "mean_e2e_ms": statistics.mean(e2es),
            "median_e2e_ms": statistics.median(e2es),
        }
    overall = stats(full)
    overall["input_tps"] = len(full) / duration_total * 100  # not really TPS, but proxy
    overall["phases"] = {"mamba": stats(mamba_phase), "kv": stats(kv_phase)}
    overall["errors"] = sum(1 for r in results if "error" in r)
    # Transfer count from budgeter log
    bud = os.environ.get("SGLANG_BUDGETER_LOG", "")
    xfers = 0
    if bud and os.path.exists(bud):
        with open(bud) as f:
            for line in f:
                if '"xpool_direction":' in line and '"none"' not in line:
                    xfers += 1
    overall["xpool_transfers"] = xfers
    METRICS_PATH.write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))


asyncio.run(main())
