#!/usr/bin/env python3
"""B3 long-multiturn dispatcher — genuine pool-binding shift.

In Qwen3.5-35B-A3B, mamba pool has ~18 slots while KV pool has ~1.26M
tokens. To shift the binding pool, need:
  α phase: HIGH concurrency × short prompts → mamba slot bottleneck,
           KV idle (~1% of token pool).
  β phase: LOW  concurrency × VERY long prompts → KV token bottleneck,
           mamba slack (~30% of slot pool).

Phase α: 24 concurrent × 1K-input × 32-output  (mamba bound)
Phase β:  8 concurrent × 96K-input × 256-output (KV bound, ~60% pool)

Two cycles of (α 90s, β 90s) = 360 s total. Each phase boundary is a
genuine binding-pool shift the L2 planner should detect and act on.
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
ALPHA_CONCURRENCY = 24
BETA_CONCURRENCY = 8


def make_alpha_prompt(idx: int) -> tuple[str, int]:
    """Short input → mamba slot pressure."""
    rnd = random.Random(idx)
    body = " ".join(rnd.choice(["alpha","beta","gamma","delta","eps","zeta",
                                "eta","theta","iota","kappa"]) for _ in range(250))
    return f"Briefly answer #{idx}: {body}\n", 32


def make_beta_prompt(idx: int) -> tuple[str, int]:
    """~96K-token unique input → fills KV pool. With β concurrency=8, peak
    KV = 8 × 96.5K = 772K tokens, ~60% of the 1.26M pool — pushes the
    planner past mamba_high_water on the KV side under L2's edge trigger.
    """
    rnd = random.Random(idx)
    body = " ".join(rnd.choice(["alpha","beta","gamma","delta","eps","zeta",
                                "eta","theta","iota","kappa","lambda","mu"])
                    for _ in range(24000))
    return body + f"\n\nTopic #{idx}: list 50 distinct words.\n", 256


def phase_at(t: float) -> str:
    n = int(t // PHASE_DURATION_S) % 4
    return "alpha" if n in (0, 2) else "beta"


def concurrency_for(ph: str) -> int:
    return ALPHA_CONCURRENCY if ph == "alpha" else BETA_CONCURRENCY


async def fire_one(session, prompt, max_tokens, started_at, results, ph):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0}
    t0 = time.time()
    try:
        async with session.post(
            f"http://127.0.0.1:{PORT}/v1/completions",
            json=body, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            await resp.read()
            e2e = (time.time() - t0) * 1000
        results.append({"phase": ph, "ttft_ms": e2e, "e2e_ms": e2e,
                        "started_at": started_at})
    except Exception as e:
        results.append({"phase": ph, "error": str(e)})


async def main():
    duration_total = PHASE_DURATION_S * 4
    t_start = time.time()
    results: list = []
    in_flight: set[asyncio.Task] = set()
    idx = 0
    async with aiohttp.ClientSession() as session:
        while time.time() - t_start < duration_total:
            t_now = time.time() - t_start
            ph = phase_at(t_now)
            target = concurrency_for(ph)
            while len(in_flight) < target and (time.time() - t_start < duration_total):
                if ph == "alpha":
                    prompt, max_t = make_alpha_prompt(idx)
                else:
                    prompt, max_t = make_beta_prompt(idx)
                idx += 1
                task = asyncio.create_task(
                    fire_one(session, prompt, max_t, t_now, results, ph))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            await asyncio.sleep(0.1)
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    full = [r for r in results if "ttft_ms" in r]
    alpha = [r for r in full if r["phase"] == "alpha"]
    beta = [r for r in full if r["phase"] == "beta"]

    def stats(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        ttfts = [r["ttft_ms"] for r in rs]
        e2es = [r["e2e_ms"] for r in rs]
        return {
            "n": len(rs),
            "mean_ttft_ms": statistics.mean(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p99_ttft_ms": (statistics.quantiles(ttfts, n=100)[98]
                            if len(ttfts) >= 100 else max(ttfts)),
            "mean_e2e_ms": statistics.mean(e2es),
            "median_e2e_ms": statistics.median(e2es),
        }

    overall = stats(full)
    overall["input_tps"] = len(full) / duration_total * 100
    overall["phases"] = {"alpha": stats(alpha), "beta": stats(beta)}
    overall["errors"] = sum(1 for r in results if "error" in r)

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
