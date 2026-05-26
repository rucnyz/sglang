"""T1 verify: GET /aginfer/state.

Runs against a minimal sglang on GPU 7 (Qwen3-0.6B, UnifiedRadixCache
forced via SGLANG_ENABLE_UNIFIED_RADIX_TREE=1).  Asserts the contract
documented in dev/aginfer/verify/t1/README.md.

Usage:
    # assumes sglang is already up at http://127.0.0.1:30001
    python dev/aginfer/verify/t1/verify.py
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

import requests

import os
BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "deepseek-ai/DeepSeek-V4-Flash")


def chat(prompt: str) -> str:
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def fetch_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def validate_schema(state: Dict[str, Any]) -> None:
    assert "tier_usage" in state, "missing tier_usage"
    assert "units" in state, "missing units"
    assert "page_size" in state, "missing page_size"
    assert "bytes_per_token" in state, "missing bytes_per_token"
    tu = state["tier_usage"]
    for tier in ("HBM", "DRAM"):
        assert tier in tu, f"tier_usage missing {tier}"
        assert "used_bytes" in tu[tier], f"tier_usage[{tier}] missing used_bytes"
        assert "cap_bytes" in tu[tier], f"tier_usage[{tier}] missing cap_bytes"
    for u in state["units"]:
        for k in (
            "hash",
            "tier",
            "n_tokens",
            "n_bytes",
            "last_access_time",
            "hit_count",
            "session_ids",
        ):
            assert k in u, f"unit missing field {k}: {u}"


def validate_invariants(state: Dict[str, Any]) -> None:
    # No duplicate hash in same response.
    hashes = [u["hash"] for u in state["units"]]
    assert len(hashes) == len(set(hashes)), (
        f"duplicate hash in single snapshot: {len(hashes) - len(set(hashes))} dups"
    )
    # Sum-of-unit n_bytes per tier should match tier_usage.used_bytes.
    by_tier_bytes = {"HBM": 0, "DRAM": 0}
    for u in state["units"]:
        by_tier_bytes[u["tier"]] = by_tier_bytes.get(u["tier"], 0) + u["n_bytes"]
    bpt = state["bytes_per_token"]
    page_bytes = state["page_size"] * max(1, bpt)
    for tier, want in by_tier_bytes.items():
        got = state["tier_usage"][tier]["used_bytes"]
        # Allow a one-page slack for race between walk + tier_usage read.
        assert abs(want - got) <= page_bytes, (
            f"{tier}: Σ unit.n_bytes={want}, tier_usage.used_bytes={got}"
        )
    # n_bytes consistency: every unit must satisfy n_bytes == n_tokens * bytes_per_token.
    for u in state["units"]:
        expected = u["n_tokens"] * bpt
        assert u["n_bytes"] == expected, (
            f"unit {u['hash']}: n_bytes={u['n_bytes']} != n_tokens*bpt={expected}"
        )


def main() -> None:
    print("=== T1 verify: GET /aginfer/state ===")

    # Stage 1: smoke + small structured tree
    print("\n[1] smoke + small tree (~50 prompts)")
    SHARED = "You are a helpful assistant. Here is a long system prompt: " + (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 30
    )
    for i in range(50):
        chat(f"{SHARED}\n\nUser question {i}: tell me a 1-line fact about prime number {i}.")
    state = fetch_state()
    validate_schema(state)
    validate_invariants(state)
    n_units = len(state["units"])
    print(f"    units count: {n_units}")
    print(f"    HBM used/cap (bytes): {state['tier_usage']['HBM']}")
    print(f"    DRAM used/cap (bytes): {state['tier_usage']['DRAM']}")
    print(f"    page_size: {state['page_size']}, bytes_per_token: {state['bytes_per_token']}")
    assert n_units >= 30, f"expected >= 30 units, got {n_units}"

    # Stage 2: latency at small scale
    print("\n[2] latency on ~50-unit tree (10 calls)")
    lats = []
    for _ in range(10):
        t0 = time.perf_counter()
        s = fetch_state()
        lats.append((time.perf_counter() - t0) * 1000)
    print(f"    p50 = {statistics.median(lats):.1f} ms,  p99 = {max(lats):.1f} ms")

    # Stage 3: stress to ~10k units
    print("\n[3] stress to ~10k units (100 batches of ~100 distinct prompts)")
    t0 = time.perf_counter()
    for batch in range(20):
        if batch % 5 == 0:
            print(f"    batch {batch}/20  units so far: {len(fetch_state()['units'])}")
        for i in range(100):
            chat(f"unique-prefix-{batch}-{i} please answer the question briefly.")
    print(f"    insert wall: {time.perf_counter() - t0:.1f}s")

    big_lats = []
    for _ in range(20):
        t0 = time.perf_counter()
        s = fetch_state()
        big_lats.append((time.perf_counter() - t0) * 1000)
    big_units = len(s["units"])
    print(f"    final units: {big_units}")
    print(f"    p50 = {statistics.median(big_lats):.1f} ms,  p99 = {max(big_lats):.1f} ms")
    validate_schema(s)
    validate_invariants(s)
    # Realistic ceiling: Python dict construction + ZMQ pickle + HTTP
    # serialization stack measures ~100 μs/node empirically. Bound is
    # 100 μs/node + 200 ms baseline. (Production V4 workloads see <1k
    # nodes per Run F's #cached-token 121856 / page_size 256 ≈ 470 leaves;
    # this stress is synthetic.)
    bound_ms = 200 + 0.100 * big_units
    assert max(big_lats) < bound_ms, (
        f"p99 {max(big_lats):.1f} ms too high for {big_units} units (bound {bound_ms:.0f} ms)"
    )

    # Stage 4: WORST CASE (forced) -- concurrent walk + traffic
    print("\n[4] WORST CASE: concurrent walk + concurrent traffic (15 s)")
    import threading
    stop = threading.Event()
    walk_results = {"ok": 0, "fail": 0, "lats": []}

    def walker():
        while not stop.is_set():
            try:
                t0 = time.perf_counter()
                s2 = fetch_state()
                walk_results["lats"].append((time.perf_counter() - t0) * 1000)
                validate_schema(s2)
                validate_invariants(s2)
                walk_results["ok"] += 1
            except Exception as exc:
                walk_results["fail"] += 1
                print(f"      walker fail: {exc}")

    threads = [threading.Thread(target=walker) for _ in range(5)]
    for t in threads:
        t.start()

    end_at = time.time() + 15
    j = 0
    while time.time() < end_at:
        chat(f"concurrent-traffic-{j}: short answer please")
        j += 1
    stop.set()
    for t in threads:
        t.join()
    print(
        f"    walker: {walk_results['ok']} ok, {walk_results['fail']} fail; "
        f"p99={max(walk_results['lats']) if walk_results['lats'] else 0:.1f} ms"
    )
    assert walk_results["fail"] == 0, "torn snapshot detected"

    print("\n=== T1 PASSED ===")


if __name__ == "__main__":
    main()
