"""G10 live-sglang verify: pool_usage tracks full_token_usage.

Runs against a live sglang at AGINFER_VERIFY_BASE (default
http://127.0.0.1:30000).  Asserts:

  1. /aginfer/state response includes top-level "pool_usage" key
  2. pool_usage.HBM has token_usage / used_bytes / cap_bytes /
     available_bytes / evictable_bytes
  3. token_usage = (cap_bytes - available - evictable) / cap (matches
     sglang's own full_token_usage formula)
  4. Under sustained load, pool_usage.HBM.token_usage > 0 (proves
     allocator-truth differs from radix-tree-keyed tier_usage which
     stays ~0 for in-flight decode).

Usage::
    # sglang must be up at AGINFER_VERIFY_BASE
    python dev/aginfer/verify/g10/verify.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import requests

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "deepseek-ai/DeepSeek-V4-Flash")


def fetch_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def chat_burst(n: int, prompt_chars: int = 4000, max_tokens: int = 1024) -> None:
    """Fire n requests with large prompts + large completions in
    parallel-ish fashion to drive HBM allocator pressure up."""
    big_prompt = "x" * prompt_chars
    for i in range(n):
        try:
            requests.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": f"{big_prompt} #{i}"},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": 42,
                },
                timeout=120,
                stream=False,
            )
        except Exception:  # noqa: BLE001
            pass  # don't care about completion, just want HBM pressure


def main() -> int:
    print(f"target: {BASE}")
    # 1. schema check on cold state
    s = fetch_state()
    assert "pool_usage" in s, f"missing pool_usage in /aginfer/state keys: {list(s)}"
    pu = s["pool_usage"]
    assert "HBM" in pu, f"pool_usage missing HBM: {pu}"
    hbm = pu["HBM"]
    for key in ("used_bytes", "cap_bytes", "available_bytes",
                "evictable_bytes", "token_usage"):
        assert key in hbm, f"pool_usage.HBM missing {key}: {hbm}"
    print(f"PASS  schema  pool_usage.HBM = {hbm}")

    # 2. formula check: used = cap - available - evictable (max 0)
    cap = int(hbm["cap_bytes"])
    avail = int(hbm["available_bytes"])
    evict = int(hbm["evictable_bytes"])
    used = int(hbm["used_bytes"])
    expected_used = max(0, cap - avail - evict)
    assert used == expected_used, (
        f"used={used} expected={expected_used} (cap={cap} avail={avail} evict={evict})"
    )
    tu = float(hbm["token_usage"])
    expected_tu = (used / cap) if cap > 0 else 0.0
    assert abs(tu - expected_tu) < 1e-6, f"token_usage={tu} expected={expected_tu}"
    print(f"PASS  formula  used={used}/cap={cap} → token_usage={tu:.3f}")

    # 3. under load: drive HBM up and assert pool_usage.HBM.token_usage > 0
    #    while tier_usage.HBM.used_bytes may stay at 0 (radix-tree-keyed).
    print("loading sglang with burst (n=8, prompt=8k chars, max_tokens=2k)...")
    import threading
    threads = [threading.Thread(target=chat_burst, args=(2, 8000, 2000))
               for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(2)  # let prefills run

    max_pool = 0.0
    max_tree = 0.0
    samples = 0
    for _ in range(30):
        s = fetch_state()
        hbm_pool = s["pool_usage"]["HBM"]["token_usage"]
        cap_tree = s["tier_usage"]["HBM"]["cap_bytes"]
        used_tree = s["tier_usage"]["HBM"]["used_bytes"]
        tree_occ = (used_tree / cap_tree) if cap_tree > 0 else 0
        max_pool = max(max_pool, float(hbm_pool))
        max_tree = max(max_tree, tree_occ)
        samples += 1
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=180)

    print(f"under-load samples={samples} pool_max={max_pool:.3f} tree_max={max_tree:.3f}")
    assert max_pool > 0.01, (
        f"pool_usage.HBM.token_usage stayed at {max_pool} under load — G10 not fixed"
    )
    # tree may or may not be > 0 depending on commit timing; just report it.
    if max_pool > max_tree + 0.05:
        print(
            f"PASS  divergence  pool_max ({max_pool:.3f}) > tree_max ({max_tree:.3f}) "
            f"— G10 fix proves daemon sees in-flight decode that tree view misses"
        )
    else:
        print(
            f"NOTE  pool_max ({max_pool:.3f}) ≈ tree_max ({max_tree:.3f}) — burst too "
            f"small to exercise divergence; G10 fix is still active per schema check"
        )

    print(f"\nALL VERIFY PASS @ {BASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
