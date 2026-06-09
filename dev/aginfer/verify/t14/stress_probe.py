"""T14 stress probe — exercise state-dump latency under realistic
load instead of on a cold empty tree.

Not a pass/fail verify — produces a report.  Pushes a configurable
number of concurrent unique-prefix chats through sglang until the
radix tree settles near cap, while polling ``/aginfer/state`` at a
fixed cadence to capture how ``state_dump_metrics`` quantiles
respond as the tree grows.

Output:
  * a TSV of (t_s, units, hbm_used_frac, dram_used_frac, p50_ms,
    p95_ms, p99_ms, max_ms, last_dump_bytes) per sample;
  * a final summary block (peak n_units, peak p99, peak max, peak
    dump_bytes, and whether the peak p99 stayed under the PLAN
    50 ms F3-revisit threshold).

Usage:
    AGINFER_VERIFY_BASE=http://127.0.0.1:30002 \\
    python dev/aginfer/verify/t14/stress_probe.py \\
        --concurrency 32 --duration 90 --max-tokens 200 \\
        --prefix-min-tokens 256 --prefix-max-tokens 512

The defaults push ~32 concurrent /v1/chat/completions with
program_id-tagged unique long prefixes so each chat anchors its own
chain in the radix tree (instead of all sharing one system prompt).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------- args


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", default=os.environ.get("AGINFER_VERIFY_BASE", ""),
                   help="sglang base URL (default $AGINFER_VERIFY_BASE)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="served model name for /v1/chat/completions")
    p.add_argument("--concurrency", type=int, default=32,
                   help="number of concurrent chats")
    p.add_argument("--duration", type=float, default=90.0,
                   help="seconds to sustain load")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="max_tokens per chat")
    p.add_argument("--prefix-min-tokens", type=int, default=256,
                   help="lower bound on the per-chat unique prefix length")
    p.add_argument("--prefix-max-tokens", type=int, default=512,
                   help="upper bound on the per-chat unique prefix length")
    p.add_argument("--poll-interval", type=float, default=0.15,
                   help="seconds between /aginfer/state polls")
    p.add_argument("--out", default=None,
                   help="optional TSV path for raw samples")
    p.add_argument("--threshold-ms", type=float, default=50.0,
                   help="PLAN T14 F3-revisit p99 threshold (ms)")
    args = p.parse_args()
    if not args.base:
        p.error("--base or $AGINFER_VERIFY_BASE required")
    args.base = args.base.rstrip("/")
    return args


# ----------------------------------------------------------- payload


def _unique_prefix(rng: random.Random, n_tokens_min: int, n_tokens_max: int) -> str:
    """Word-soup unique prefix in the [min, max] token range.  We send
    plain text to /v1/chat/completions; sglang tokenizes server-side,
    so a ~3.5-tokens-per-word English-letter soup hits the range with
    no per-client tokenizer dep."""
    n_tokens = rng.randint(n_tokens_min, n_tokens_max)
    n_words = int(n_tokens / 1.4)
    words: List[str] = []
    for _ in range(n_words):
        wlen = rng.randint(3, 9)
        w = "".join(rng.choices(string.ascii_lowercase, k=wlen))
        words.append(w)
    return " ".join(words)


async def _drive_one_chat(
    session_pool,
    base: str,
    model: str,
    program_id: str,
    prefix: str,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    """Fire one /v1/chat/completions and return a tiny summary.  Errors
    are swallowed (the goal is to keep load up, not to crash on a
    transient sglang queue overflow)."""
    import aiohttp
    url = f"{base}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                f"Context: {prefix}\n\n"
                f"Reply with: ok-{program_id[-6:]}"
            ),
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "program_id": program_id,
    }
    try:
        async with session_pool.post(url, json=body, timeout=120) as resp:
            txt = await resp.text()
            return {
                "program_id": program_id,
                "status": resp.status,
                "n_chars": len(txt),
            }
    except Exception as exc:  # noqa: BLE001
        return {"program_id": program_id, "error": str(exc)[:120]}


# ---------------------------------------------------------- polling


def _poll_state(base: str) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"{base}/aginfer/state", timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        return None


def _pool_used_frac(pool_usage: Dict[str, Any], tier: str) -> float:
    subpools = pool_usage.get(tier, {}).get("subpools", {})
    used = sum(int(sp["used_bytes"]) for sp in subpools.values())
    cap = sum(int(sp["cap_bytes"]) for sp in subpools.values())
    return used / cap if cap > 0 else 0.0


# ---------------------------------------------------------- driver


async def _stress_loop(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Concurrently:
       (a) fire chats in a tight loop until --duration elapses,
           keeping concurrency-many in flight at all times;
       (b) sample /aginfer/state every --poll-interval seconds.
    """
    import aiohttp
    rng = random.Random(0xA61F_E2)
    samples: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    next_pid_n = 0
    in_flight: set = set()

    async def _drive_forever(session: "aiohttp.ClientSession") -> None:
        nonlocal next_pid_n
        while time.perf_counter() - t_start < args.duration:
            program_id = f"t14-stress-{next_pid_n:05d}"
            next_pid_n += 1
            prefix = _unique_prefix(
                rng, args.prefix_min_tokens, args.prefix_max_tokens,
            )
            await _drive_one_chat(
                session, args.base, args.model,
                program_id, prefix, args.max_tokens,
            )

    async def _poll_loop() -> None:
        while time.perf_counter() - t_start < args.duration:
            t = time.perf_counter() - t_start
            state = _poll_state(args.base)
            if state is not None:
                m = state.get("state_dump_metrics", {})
                samples.append({
                    "t_s": round(t, 3),
                    "units": len(state.get("units", [])),
                    "hbm_used_frac": round(
                        _pool_used_frac(state.get("pool_usage", {}), "HBM"), 3),
                    "dram_used_frac": round(
                        _pool_used_frac(state.get("pool_usage", {}), "DRAM"), 3),
                    "p50_ms": m.get("p50_ms", 0.0),
                    "p95_ms": m.get("p95_ms", 0.0),
                    "p99_ms": m.get("p99_ms", 0.0),
                    "max_ms": m.get("max_ms", 0.0),
                    "last_dump_bytes": m.get("last_dump_bytes", -1),
                    "n_samples_in_window": m.get("n_samples", 0),
                    "n_recorded_total": m.get("n_recorded_total", 0),
                })
            await asyncio.sleep(args.poll_interval)

    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(_drive_forever(session))
                 for _ in range(args.concurrency)]
        tasks.append(asyncio.create_task(_poll_loop()))
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for t in tasks:
                t.cancel()
    return samples


# ----------------------------------------------------------- report


def _print_tsv(samples: List[Dict[str, Any]],
               out_path: Optional[str]) -> None:
    header = (
        "t_s\tunits\thbm%\tdram%\tp50_ms\tp95_ms\tp99_ms\tmax_ms\t"
        "dump_bytes\tn_recorded_total"
    )
    lines = [header]
    for s in samples:
        lines.append(
            f"{s['t_s']:.3f}\t{s['units']}\t"
            f"{s['hbm_used_frac']*100:.1f}\t{s['dram_used_frac']*100:.1f}\t"
            f"{s['p50_ms']:.3f}\t{s['p95_ms']:.3f}\t"
            f"{s['p99_ms']:.3f}\t{s['max_ms']:.3f}\t"
            f"{s['last_dump_bytes']}\t{s['n_recorded_total']}"
        )
    text = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(text)
        print(f"raw samples → {out_path}")
    print(text)


def _summary(samples: List[Dict[str, Any]],
             threshold_ms: float) -> Dict[str, Any]:
    if not samples:
        return {"empty": True}
    peak_units = max(s["units"] for s in samples)
    peak_p99 = max(s["p99_ms"] for s in samples)
    peak_max = max(s["max_ms"] for s in samples)
    peak_dump_bytes = max(s["last_dump_bytes"] for s in samples)
    peak_hbm_frac = max(s["hbm_used_frac"] for s in samples)
    peak_dram_frac = max(s["dram_used_frac"] for s in samples)
    p99_median = median(s["p99_ms"] for s in samples)
    return {
        "n_samples_collected": len(samples),
        "duration_s": samples[-1]["t_s"],
        "peak_units": peak_units,
        "peak_hbm_used_frac": peak_hbm_frac,
        "peak_dram_used_frac": peak_dram_frac,
        "peak_dump_bytes": peak_dump_bytes,
        "p99_median_over_run_ms": p99_median,
        "peak_p99_ms": peak_p99,
        "peak_max_ms": peak_max,
        "threshold_ms": threshold_ms,
        "peak_p99_under_threshold": peak_p99 < threshold_ms,
    }


# --------------------------------------------------------------- main


def main() -> int:
    args = _parse_args()
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("stress probe requires aiohttp; "
              "pip install aiohttp", file=sys.stderr)
        return 2

    print(
        f"[t14 stress] base={args.base} model={args.model} "
        f"concurrency={args.concurrency} duration={args.duration}s "
        f"prefix_tokens=[{args.prefix_min_tokens},{args.prefix_max_tokens}] "
        f"max_tokens={args.max_tokens}"
    )

    samples = asyncio.run(_stress_loop(args))
    _print_tsv(samples, args.out)
    summary = _summary(samples, args.threshold_ms)
    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))

    if not summary.get("peak_p99_under_threshold", False):
        print(
            f"\nFAIL: peak p99 ({summary['peak_p99_ms']:.2f} ms) >= "
            f"threshold ({args.threshold_ms} ms) — PLAN T14 F3-revisit "
            f"trigger would fire at this load",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
