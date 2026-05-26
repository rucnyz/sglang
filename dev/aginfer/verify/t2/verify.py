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

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")


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


def warm_distinct_leaves(n: int) -> None:
    """Insert n distinct top-level prefixes so we get n addressable leaves.

    Uses single-token user prompts -- short enough that decode finishes quickly,
    distinct enough that no two share a prefix beyond the chat template.
    """
    for i in range(n):
        chat(f"distinct-prompt-{i}: short answer please about prime {i}")


def main() -> None:
    print("=== T2 verify: POST /aginfer/migrate ===")

    # ---- HAPPY PATH: DROP a known leaf and verify it's gone ----
    print("\n[1] populate tree, capture a DROP target")
    warm_distinct_leaves(20)
    state_before = fetch_state()
    units_before = state_before["units"]
    print(f"    units before: {len(units_before)}")
    if not units_before:
        print("    no units; cannot exercise migrate. did sglang actually insert?")
        return

    # We DROP every HBM unit with n_tokens > 0 that exists right now.  Because
    # we filtered transient (hash_value-less) nodes out of /aginfer/state, every
    # hash in this list IS stably addressable.  not_a_leaf is still possible
    # (internal nodes with kids) -- the daemon's job is bottom-up drop.
    hbm = [u for u in units_before if u["tier"] == "HBM" and u["n_tokens"] > 0]
    print(f"    HBM units to attempt drop: {len(hbm)}")
    target_hashes_before = {u["hash"] for u in hbm}
    targets = [{"hash": u["hash"], "target_tier": "DROP"} for u in hbm]

    print("[2] POST /aginfer/migrate with DROP for the targets")
    t0 = time.perf_counter()
    resp = migrate(targets)
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"    applied: {resp['applied']}  skipped: {len(resp['skipped'])}")
    print(f"    latency: {dur_ms:.1f} ms total ({dur_ms/max(1,len(targets)):.2f} ms/action)")
    skipped_reasons = {s["reason"] for s in resp["skipped"]}
    print(f"    skipped reasons: {skipped_reasons}")
    assert resp["applied"] > 0, (
        f"no actions applied; reasons: {skipped_reasons}"
    )
    # Every action must be accounted for (applied + skipped == sent).
    assert resp["applied"] + len(resp["skipped"]) == len(targets)

    print("[3] re-fetch state, verify EXACTLY the applied_hashes are gone")
    state_after = fetch_state()
    after_hashes = {u["hash"] for u in state_after["units"]}
    applied_hashes = set(resp.get("applied_hashes", []))
    # Sanity: server-reported applied_hashes must match `applied` count.
    assert len(applied_hashes) == resp["applied"], (
        f"server inconsistency: applied={resp['applied']} but "
        f"len(applied_hashes)={len(applied_hashes)}"
    )
    # Causal invariant (audit Q2): each hash the server claims it applied
    # must be ABSENT from a snapshot taken immediately after.  No slack.
    still_present = applied_hashes & after_hashes
    assert not still_present, (
        f"{len(still_present)} of {len(applied_hashes)} server-applied DROPs "
        f"did not actually evict the node: {sorted(list(still_present))[:5]}"
    )
    # And the applied_hashes must have BEEN in the snapshot just before
    # we issued the migrate (rules out the server lying with random
    # hashes it pulled from thin air).
    pre_present_applied = applied_hashes & target_hashes_before
    assert pre_present_applied == applied_hashes, (
        f"{len(applied_hashes - target_hashes_before)} applied_hashes were "
        f"not in the pre-migrate snapshot; server fabricated them"
    )
    print(f"    all {len(applied_hashes)} applied hashes removed (causal check ✓)")

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
    state_live = fetch_state()
    if state_live["units"]:
        h = state_live["units"][0]["hash"]
        bad = migrate([{"hash": h, "target_tier": "DOESNOTEXIST"}])
        print(f"    {bad}")
        assert bad["applied"] == 0
        assert len(bad["skipped"]) == 1
        r = bad["skipped"][0]["reason"]
        assert "unknown_target_tier" in r, f"wrong reason: {r}"
    else:
        print("    no live units; skipping (would be a no-op)")

    # ---- WORST CASE 3: DISK tier reported as not_yet_wired (v1 contract) ----
    print("\n[6] DISK tier returns not_yet_wired (v1 contract)")
    state_live = fetch_state()
    if state_live["units"]:
        h = state_live["units"][0]["hash"]
        disk = migrate([{"hash": h, "target_tier": "DISK"}])
        print(f"    {disk}")
        assert disk["applied"] == 0
        assert disk["skipped"][0]["reason"] == "disk_tier_not_yet_wired"

    # ---- COST: 1000-action slow-path batch (real DROPs, not bogus) ----
    # Audit Q3: the original 1000-bogus measurement only times the not_in_tree
    # fast path.  Re-warm and time a real-DROP batch end-to-end.
    print("\n[7] COST: slow-path real-DROP batch")
    warm_distinct_leaves(60)
    state_warm = fetch_state()
    real_targets = [
        {"hash": u["hash"], "target_tier": "DROP"}
        for u in state_warm["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    print(f"    real-DROP batch size: {len(real_targets)}")
    if real_targets:
        t0 = time.perf_counter()
        big = migrate(real_targets)
        dur_ms = (time.perf_counter() - t0) * 1000
        per_action_ms = dur_ms / max(1, len(real_targets))
        print(
            f"    {dur_ms:.0f} ms total ({per_action_ms:.3f} ms/action), "
            f"applied={big['applied']} skipped={len(big['skipped'])}"
        )
        # Slow-path ceiling: < 1 ms/action amortised (= the README claim).
        assert per_action_ms < 1.0, (
            f"slow-path per-action {per_action_ms:.2f} ms exceeds the 1 ms ceiling"
        )
        assert big["applied"] > 0, (
            f"slow-path batch did 0 real DROPs (reasons: "
            f"{ {s['reason'] for s in big['skipped']} })"
        )

    # ---- COST: 1000-action all-bogus fast path ----
    print("\n[8] COST: 1k-action fast path (all not_in_tree, sanity)")
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
    assert per_action_ms < 1.0
    assert big["applied"] == 0
    assert len(big["skipped"]) == 1000

    # ---- WORST CASE 4: malformed payload ----
    print("\n[9] WORST CASE: malformed payload returns 400")
    r = requests.post(f"{BASE}/aginfer/migrate", json={}, timeout=10)
    assert r.status_code == 400, f"got {r.status_code}"
    r = requests.post(
        f"{BASE}/aginfer/migrate", json={"actions": "not a list"}, timeout=10
    )
    assert r.status_code == 400
    r = requests.post(
        f"{BASE}/aginfer/migrate",
        data="not json",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert r.status_code == 400

    # ---- IDEMPOTENT REPLAY: causal, allows cascade ----
    # The auditor flagged "21 applied on replay" as suspicious.  Actual cause:
    # round 1 drops leaves whose parents were skipped as `not_a_leaf`; once
    # the children are gone the parents BECOME leaves and become droppable.
    # That's the daemon "drop bottom-up" semantics, not a bug.  The strict
    # invariants are:
    #   (a) round-1 applied_hashes are STILL absent after replay.
    #   (b) round-2 applied_hashes ⊆ round-1 `not_a_leaf` skip bucket
    #       (replay must not "discover" hashes outside the original request).
    #   (c) round-2 reasons for any new skip are sane.
    # We still ran the unrelated probes (steps [4]-[9]) in between, but
    # those were either bogus hashes or no-op tier dispatches and cannot
    # mutate the tree.
    print("\n[10] IDEMPOTENT REPLAY: cascade-aware causal check")
    not_a_leaf_round1 = {
        s["hash"] for s in resp["skipped"] if s["reason"] == "not_a_leaf"
    }
    again = migrate(targets)
    again_applied = set(again.get("applied_hashes", []))
    again_reasons = {s["reason"] for s in again["skipped"]}
    print(
        f"    round-2 applied={again['applied']} (came from round-1 "
        f"not_a_leaf bucket of size {len(not_a_leaf_round1)}); "
        f"skipped reasons: {again_reasons}"
    )
    # (a) Round-1 applied still gone.
    state_replay = fetch_state()
    replay_present = applied_hashes & {u["hash"] for u in state_replay["units"]}
    assert not replay_present, (
        f"replay state lost {len(replay_present)} originally-dropped hashes!"
    )
    # (b) Round-2 applied ⊆ round-1 not_a_leaf.
    outside_orig = again_applied - not_a_leaf_round1
    assert not outside_orig, (
        f"replay applied {len(outside_orig)} hashes that were NOT in the "
        f"original not_a_leaf bucket: {sorted(list(outside_orig))[:5]}"
    )
    # (c) Sane reasons.
    assert again_reasons <= {"not_in_tree", "no_data", "not_a_leaf"}, (
        f"unexpected replay reasons: {again_reasons}"
    )

    print("\n=== T2 PASSED ===")


if __name__ == "__main__":
    main()
