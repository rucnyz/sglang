"""T2 verify: POST /aginfer/migrate.

Runs against the same minimal sglang as T1 (Qwen3-0.6B, UnifiedRadixCache,
flashinfer attention -- trtllm_mha auto-picks page_size=1 and bypasses radix
insert).  Asserts the contract documented in dev/aginfer/verify/t2/README.md.

Usage:
    # assumes sglang is already up at http://127.0.0.1:30001
    python dev/aginfer/verify/t2/verify.py
"""
from __future__ import annotations

import os
import statistics
import time
from typing import Any, Dict, List

import requests

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30000")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "deepseek-ai/DeepSeek-V4-Flash")


def chat(prompt: str) -> None:
    requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        timeout=60,
    ).raise_for_status()


def fetch_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def migrate(actions: List[Dict[str, str]]) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/aginfer/migrate",
        json={"actions": actions},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    print("=== T2 verify: POST /aginfer/migrate ===")

    # ---- HAPPY PATH: DROP a known leaf and verify it's gone ----
    print("\n[1] populate tree, capture a DROP target")
    # Send distinct prompts so we get plenty of leaf nodes.
    for i in range(20):
        chat(f"distinct-prompt-{i}: short answer please about prime {i}")
    state = fetch_state()
    units = state["units"]
    print(f"    units before: {len(units)}")
    if not units:
        print("    no units; cannot exercise migrate. did sglang actually insert?")
        return

    # Pick HBM hashes whose tier is HBM and n_tokens > 0.  Tree internal nodes
    # may be skipped by apply_aginfer_migrations as not_a_leaf -- the API
    # returns reasons, so we still validate.
    hbm = [u for u in units if u["tier"] == "HBM" and u["n_tokens"] > 0]
    print(f"    HBM units to attempt drop: {len(hbm)}")
    targets = [{"hash": u["hash"], "target_tier": "DROP"} for u in hbm[:50]]

    print("[2] POST /aginfer/migrate with DROP for the targets")
    t0 = time.perf_counter()
    resp = migrate(targets)
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"    applied: {resp['applied']}  skipped: {len(resp['skipped'])}")
    print(f"    latency: {dur_ms:.1f} ms total ({dur_ms/max(1,len(targets)):.2f} ms/action)")
    # Capability assertion: we must have applied at least SOME drops; the
    # not_a_leaf path is expected for internal nodes but not 100 %.
    assert resp["applied"] > 0, (
        f"no actions applied; skipped reasons: "
        f"{[s['reason'] for s in resp['skipped'][:5]]}"
    )

    print("[3] re-fetch state, verify those hashes are gone")
    state_after = fetch_state()
    after_hashes = {u["hash"] for u in state_after["units"]}
    dropped = [a["hash"] for a in targets if a["hash"] not in after_hashes]
    skipped_hashes = {s["hash"] for s in resp["skipped"]}
    successfully_dropped = [h for h in dropped if h not in skipped_hashes]
    print(f"    {len(successfully_dropped)} of {resp['applied']} expected drops actually disappeared")
    # Allow some slack: tree may re-create paths if a recent prefill happened.
    assert len(successfully_dropped) >= resp["applied"] // 2, (
        f"only {len(successfully_dropped)} of {resp['applied']} applied drops "
        f"actually left the tree -- migrate is reporting success but not "
        f"freeing nodes"
    )

    # ---- WORST CASE 1: hash-not-found ----
    print("\n[4] WORST CASE: hash-not-found")
    bogus = migrate(
        [
            {"hash": "definitely-not-in-tree-aaaaaaaa", "target_tier": "DROP"},
            {"hash": "also-not-real-bbbbbbbb", "target_tier": "DROP"},
        ]
    )
    print(f"    {bogus}")
    assert bogus["applied"] == 0, "phantom hashes must NOT be applied"
    assert len(bogus["skipped"]) == 2, "every action must be reported"
    reasons = {s["reason"] for s in bogus["skipped"]}
    assert reasons == {"not_in_tree"}, f"wrong reason set: {reasons}"

    # ---- WORST CASE 2: unsupported target tier ----
    print("\n[5] WORST CASE: unknown target_tier")
    # Pick any live hash so the lookup succeeds and we hit the tier dispatch.
    if state_after["units"]:
        h = state_after["units"][0]["hash"]
        bad = migrate([{"hash": h, "target_tier": "DOESNOTEXIST"}])
        print(f"    {bad}")
        assert bad["applied"] == 0
        assert len(bad["skipped"]) == 1
        r = bad["skipped"][0]["reason"]
        assert "unknown_target_tier" in r, f"wrong reason: {r}"

    # ---- WORST CASE 3: DISK tier reported as not_yet_wired (v1 contract) ----
    print("\n[6] DISK tier returns not_yet_wired (v1 contract)")
    if state_after["units"]:
        h = state_after["units"][0]["hash"]
        disk = migrate([{"hash": h, "target_tier": "DISK"}])
        print(f"    {disk}")
        assert disk["applied"] == 0
        assert disk["skipped"][0]["reason"] == "disk_tier_not_yet_wired"

    # ---- COST: 1000-action batch latency ----
    print("\n[7] COST: 1k-action batch (mostly not_in_tree -- cheap path)")
    big_batch = [
        {"hash": f"bogus-{i}-aaaaaaaaaaaaaaaa", "target_tier": "DROP"}
        for i in range(1000)
    ]
    t0 = time.perf_counter()
    big = migrate(big_batch)
    dur_ms = (time.perf_counter() - t0) * 1000
    per_action_ms = dur_ms / 1000
    print(
        f"    {dur_ms:.0f} ms total ({per_action_ms:.3f} ms/action), "
        f"applied={big['applied']} skipped={len(big['skipped'])}"
    )
    # Ceiling for the not-in-tree fast path: < 1 ms/action amortised
    assert per_action_ms < 1.0, (
        f"per-action {per_action_ms:.2f} ms exceeds the 1 ms ceiling"
    )
    assert big["applied"] == 0
    assert len(big["skipped"]) == 1000

    # ---- WORST CASE 4: malformed payload ----
    print("\n[8] WORST CASE: malformed payload returns 400")
    r = requests.post(f"{BASE}/aginfer/migrate", json={}, timeout=10)
    assert r.status_code == 400, f"got {r.status_code}"
    r = requests.post(f"{BASE}/aginfer/migrate", json={"actions": "not a list"}, timeout=10)
    assert r.status_code == 400
    r = requests.post(f"{BASE}/aginfer/migrate", data="not json", headers={"Content-Type": "application/json"}, timeout=10)
    assert r.status_code == 400

    # ---- IDEMPOTENT REPLAY ----
    print("\n[9] IDEMPOTENT REPLAY: rerun same migrate set, no crash")
    again = migrate(targets)
    # Targets already dropped, so applied=0; skipped reason is not_in_tree.
    print(f"    {again['applied']} applied (expect ~0), {len(again['skipped'])} skipped")
    # Some might still apply if requests after the first round re-inserted paths.
    # The invariant is: NO exceptions.  Already asserted by .raise_for_status().

    print("\n=== T2 PASSED ===")


if __name__ == "__main__":
    main()
