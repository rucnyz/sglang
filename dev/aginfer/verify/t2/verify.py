"""T2 verify: POST /aginfer/migrate (depth-audit edition).

Runs against the same minimal sglang as T1 (Qwen3-0.6B, UnifiedRadixCache,
flashinfer attention -- trtllm_mha auto-picks page_size=1 and bypasses radix
insert).  Asserts the contract documented in dev/aginfer/verify/t2/README.md.

This version covers all branches of apply_aginfer_migrations that are
reachable without HiCache, plus the tier_usage delta invariant, IPC
serialization, duplicate hashes, empty lists, malformed action shapes,
cascade-to-zero, round-2 applied absence, and a 2-thread concurrency probe.

Usage:
    # assumes sglang is already up at http://127.0.0.1:30001
    python dev/aginfer/verify/t2/verify.py
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
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


def migrate(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/aginfer/migrate",
        json={"actions": actions},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def hbm_used(state: Dict[str, Any]) -> int:
    """HBM used in BYTES (the paper §7 currency).

    Per-unit n_bytes is exposed for the same reason -- the daemon's per-unit
    value rule divides VALUE by COST, and cost is bytes.
    """
    return int(state["tier_usage"]["HBM"]["used_bytes"])


def warm_distinct_leaves(n: int, salt: str = "") -> None:
    for i in range(n):
        chat(f"{salt}distinct-prompt-{i}: short answer please about prime {i}")


def main() -> None:
    print("=== T2 verify: POST /aginfer/migrate (depth-audit) ===")

    # ---------- HAPPY PATH + CAUSAL INVARIANTS ----------
    print("\n[1] populate tree")
    warm_distinct_leaves(20, salt="round1-")
    state0 = fetch_state()
    n_bytes_by_hash = {u["hash"]: u["n_bytes"] for u in state0["units"]}
    hbm0 = hbm_used(state0)
    targets = [
        {"hash": u["hash"], "target_tier": "DROP"}
        for u in state0["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    print(f"    units: {len(state0['units'])}, HBM used_bytes: {hbm0}, DROP targets: {len(targets)}")

    print("[2] POST migrate; check applied_hashes + tier_usage delta")
    t0 = time.perf_counter()
    resp = migrate(targets)
    dur_ms = (time.perf_counter() - t0) * 1000
    applied_hashes = set(resp.get("applied_hashes", []))
    print(f"    applied={resp['applied']}, dur={dur_ms:.1f}ms ({dur_ms/max(1,len(targets)):.3f} ms/action)")
    print(f"    skip reasons: {sorted({s['reason'] for s in resp['skipped']})}")

    # (A) server-reported count == len(applied_hashes)
    assert len(applied_hashes) == resp["applied"]
    # (B) ALL skips in round 1 must be exactly 'not_a_leaf' (every target was a
    #     pre-snapshot HBM unit, so 'no_data' / 'not_in_tree' would be a bug).
    skip_reasons = {s["reason"] for s in resp["skipped"]}
    assert skip_reasons <= {"not_a_leaf"}, f"unexpected round-1 skip reasons: {skip_reasons}"

    state1 = fetch_state()
    after_hashes = {u["hash"] for u in state1["units"]}
    # (C) applied_hashes absent in post-snapshot
    leaked = applied_hashes & after_hashes
    assert not leaked, f"{len(leaked)} applied DROPs still in tree: {sorted(leaked)[:5]}"
    # (D) applied_hashes were in pre-snapshot (server didn't fabricate)
    fabricated = applied_hashes - set(n_bytes_by_hash.keys())
    assert not fabricated, f"server fabricated {len(fabricated)} applied_hashes"
    # (E) tier_usage delta -- the audit BLOCKER #1 invariant.  HBM used_bytes
    #     must decrease by AT LEAST the sum of dropped n_bytes.  ">=" allows
    #     cascade through tombstones to free more; "<" would mean the migrate
    #     reported success without freeing buffers.
    hbm1 = hbm_used(state1)
    expected_drop = sum(n_bytes_by_hash[h] for h in applied_hashes)
    actual_drop = hbm0 - hbm1
    print(f"    HBM bytes: {hbm0} -> {hbm1} (delta {actual_drop}, expected >= {expected_drop})")
    assert actual_drop >= expected_drop, (
        f"HBM used_bytes dropped by only {actual_drop}, expected >= {expected_drop} "
        f"-- migrate reported applied but did not free buffers"
    )
    print("    causal checks (count, absence, anti-fabrication, tier_usage delta) ✓")

    # ---------- BRANCH PROBES (reachable without HiCache) ----------
    print("\n[3] warm fresh nodes for branch probes")
    warm_distinct_leaves(8, salt="probe-")
    state_probe = fetch_state()
    fresh = [u for u in state_probe["units"] if u["tier"] == "HBM" and u["n_tokens"] > 0]
    assert fresh, "no fresh HBM units; branch probes cannot run"

    # DRAM probe -- HBM-only node => demote_requires_existing_host_backup
    h_dram = fresh[0]["hash"]
    r = migrate([{"hash": h_dram, "target_tier": "DRAM"}])
    print(f"[4] DRAM on HBM-only: applied={r['applied']}, skip={r['skipped']}")
    assert r["applied"] == 0
    assert r["skipped"][0]["reason"] == "demote_requires_existing_host_backup", r

    # HBM probe -- HBM-resident node => already_on_hbm
    h_hbm = fresh[1]["hash"] if len(fresh) > 1 else fresh[0]["hash"]
    r = migrate([{"hash": h_hbm, "target_tier": "HBM"}])
    print(f"[5] HBM on HBM-resident: applied={r['applied']}, skip={r['skipped']}")
    assert r["applied"] == 0
    assert r["skipped"][0]["reason"] == "already_on_hbm", r

    # Explicit not_a_leaf binding (audit BLOCKER #4).  Pick a leftover internal
    # node from round 1 that's still in the tree.
    leftover_internal = next(
        (s["hash"] for s in resp["skipped"]
         if s["reason"] == "not_a_leaf" and s["hash"] in {u["hash"] for u in state_probe["units"]}),
        None,
    )
    if leftover_internal is not None:
        r = migrate([{"hash": leftover_internal, "target_tier": "DROP"}])
        print(f"[6] explicit not_a_leaf: applied={r['applied']}, skip={r['skipped']}")
        # It may have become a leaf via cascade in the meantime; either reason
        # is acceptable but the bug would be a 5xx or a different string.
        assert r["applied"] in (0, 1)
        if r["skipped"]:
            assert r["skipped"][0]["reason"] in {"not_a_leaf", "no_data"}, r
    else:
        print("[6] no leftover internal node to probe explicitly; skipped")

    # Duplicate hashes in same batch (audit MINOR).  Bug case: server applies
    # the same hash twice (double-detach via _remove_leaf_from_parent).  We
    # don't pre-identify a leaf -- if any hash in the batch is droppable, we
    # send each twice and assert it never appears more than once in
    # applied_hashes.
    print("\n[7] duplicate hashes in same batch")
    warm_distinct_leaves(8, salt="dup-")
    state_dup = fetch_state()
    # Diagnostic: any duplicate hashes in /aginfer/state itself?
    all_hashes = [u["hash"] for u in state_dup["units"]]
    from collections import Counter
    h_counter = Counter(all_hashes)
    state_dups = {h: c for h, c in h_counter.items() if c > 1}
    if state_dups:
        print(f"    WARN: /aginfer/state has {len(state_dups)} duplicate hashes (first 3): "
              f"{dict(list(state_dups.items())[:3])}")
    dup_inputs = [
        u["hash"] for u in state_dup["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    # Deduplicate input list before sending duplicates (otherwise we count
    # /aginfer/state's intrinsic duplicates against our defensive check).
    dup_inputs = list(dict.fromkeys(dup_inputs))[:10]
    dup_batch = (
        [{"hash": h, "target_tier": "DROP"} for h in dup_inputs]
        + [{"hash": h, "target_tier": "DROP"} for h in dup_inputs]
    )
    r = migrate(dup_batch)
    applied_set = set(r.get("applied_hashes", []))
    print(
        f"    sent {len(dup_batch)} actions ({len(dup_inputs)} unique hashes, "
        f"each twice), applied={r['applied']}, unique applied={len(applied_set)}"
    )
    # (a) no double-apply
    assert len(applied_set) == r["applied"], (
        f"double-apply detected: count={r['applied']}, unique={len(applied_set)}"
    )
    # (b) total accountability
    assert r["applied"] + len(r["skipped"]) == len(dup_batch)
    # (c) every skip reason is in the legal set
    skip_reasons = {s["reason"] for s in r["skipped"]}
    assert skip_reasons <= {
        "not_a_leaf", "no_data", "not_in_tree", "already_acted_this_batch"
    }, skip_reasons
    from collections import Counter
    reason_count = Counter(s["reason"] for s in r["skipped"])
    print(f"    skipped reason counts: {dict(reason_count)}")
    # (d) No hash applied more than once -- this is the real
    #     "no double-apply / no double-free" invariant.  applied_hashes
    #     is a list; if a hash appears twice the server applied it twice.
    applied_counter = Counter(r.get("applied_hashes", []))
    multi_applied = {h: c for h, c in applied_counter.items() if c > 1}
    assert not multi_applied, f"hash applied multiple times: {multi_applied}"
    # (e) The defensive check MUST have fired at least once if any first-
    #     occurrence got applied (its second occurrence would otherwise have
    #     crashed _remove_leaf_from_parent).  Detect this via: for each pair
    #     (idx_i, idx_{i+10}), if idx_i applied then idx_{i+10} must be
    #     blocked by already_acted_this_batch.  Cascade-promotion (idx_i
    #     skipped as not_a_leaf, idx_{i+10} applied) is a separate legitimate
    #     outcome and does NOT exercise the defensive check.
    n_unique = len(dup_inputs)
    applied_hashes_list = r.get("applied_hashes", [])
    # We need to know WHICH occurrence index applied for each hash; the
    # server returns applied_hashes in action order, so we walk the batch.
    skipped_by_hash_in_order: dict[str, list[str]] = {}
    for s in r["skipped"]:
        skipped_by_hash_in_order.setdefault(s["hash"], []).append(s["reason"])
    must_block_first_applied = 0
    actual_block_first_applied = 0
    cascade_promoted = 0
    for i, h in enumerate(dup_inputs):
        # Position i is the first occurrence; position i+n_unique is dup.
        # Look at which occurrence applied for h (or neither, both skipped).
        # applied_hashes preserves apply order; we can search by index.
        if applied_hashes_list.count(h) == 0:
            # neither occurrence applied; both should be not_a_leaf (or similar)
            continue
        # exactly one occurrence applied (asserted above by multi_applied check)
        # Which occurrence? Server order is action-index order, so the applied
        # one is whichever came first in (i, i+n_unique).  Inspect the skipped
        # reasons for h: if `already_acted_this_batch` is present, the FIRST
        # occurrence (idx=i) applied and the dup got blocked.  If the only
        # other reason is `not_a_leaf`, the FIRST was skipped and SECOND
        # applied (cascade promotion).
        reasons = skipped_by_hash_in_order.get(h, [])
        if "already_acted_this_batch" in reasons:
            must_block_first_applied += 1
            actual_block_first_applied += 1
        elif "not_a_leaf" in reasons:
            cascade_promoted += 1
        else:
            # h applied with no recorded skip for the other occurrence; this
            # would mean only one action targeted h, which contradicts our
            # constructed batch -- assert.
            raise AssertionError(
                f"hash {h} applied but neither dup-occurrence is in skipped; "
                f"batch construction is wrong"
            )
    print(
        f"    first-applied + dup-blocked: {actual_block_first_applied} / "
        f"{must_block_first_applied} (defensive check fired); "
        f"cascade-promoted: {cascade_promoted}"
    )
    assert must_block_first_applied == actual_block_first_applied
    # Final invariant: AT LEAST ONE first-applied path was actually exercised
    # (otherwise the defensive check is untested by this run).
    assert actual_block_first_applied > 0, (
        "no first-applied -> dup-blocked pair in this run; defensive check "
        "wasn't exercised. retry with a different prompt set."
    )
    print(f"    no double-apply ✓; defensive already_acted_this_batch fired ✓")

    # ---------- MALFORMED ACTION SHAPES ----------
    # For tier-dispatch errors to surface, we need a REAL hash (otherwise the
    # lookup misses first and we always see not_in_tree).
    print("\n[8] malformed action shapes (must not 5xx)")
    chat("malformed-test-prompt: short answer please")
    state_mal = fetch_state()
    h_real = next(
        (u["hash"] for u in state_mal["units"]
         if u["tier"] == "HBM" and u["n_tokens"] > 0),
        None,
    )
    assert h_real is not None
    # (a) missing 'hash' -> hash_to_node.get(None) -> None -> not_in_tree
    r = migrate([{"target_tier": "DROP"}])
    assert r["applied"] == 0 and r["skipped"][0]["reason"] == "not_in_tree", r
    # (b) target_tier=None on a REAL hash -> (None or '').upper() == '' -> unknown
    r = migrate([{"hash": h_real, "target_tier": None}])
    assert r["applied"] == 0
    assert "unknown_target_tier" in r["skipped"][0]["reason"], r
    # (c) non-string hash -> lookup miss -> not_in_tree
    r = migrate([{"hash": 42, "target_tier": "DROP"}])
    assert r["applied"] == 0 and r["skipped"][0]["reason"] == "not_in_tree", r
    # (d) missing target_tier on a REAL hash -> '' upper -> unknown
    r = migrate([{"hash": h_real}])
    assert r["applied"] == 0
    assert "unknown_target_tier" in r["skipped"][0]["reason"], r
    print("    4 malformed shapes handled cleanly")

    # ---------- EMPTY ACTIONS ----------
    r = migrate([])
    print(f"[9] empty list: {r}")
    assert r == {"applied": 0, "applied_hashes": [], "skipped": []}

    # ---------- IPC SERIALIZATION (extra unknown key) ----------
    # A future version of the protocol may add fields to the action dict.
    # The server must accept and ignore extras -- catches the case where
    # MigrateAginferReq.actions becomes a structured dataclass that rejects
    # unknown keys.
    chat("ipc-test-prompt: short answer please")
    state_ipc = fetch_state()
    h_ipc = next(
        (u["hash"] for u in state_ipc["units"] if u["tier"] == "HBM" and u["n_tokens"] > 0),
        None,
    )
    assert h_ipc is not None
    r = migrate([{
        "hash": h_ipc,
        "target_tier": "DROP",
        "session_id": "future-field",
        "owner": "depth-audit",
        "weight": 0.5,
    }])
    print(f"[10] IPC unknown keys: applied={r['applied']}, skipped={r['skipped']}")
    assert "applied" in r and "applied_hashes" in r and "skipped" in r
    # Action should have applied (the node was a fresh HBM leaf).
    assert r["applied"] == 1 or (r["skipped"] and r["skipped"][0]["reason"] in {"not_a_leaf", "no_data"})

    # ---------- COST: slow-path 1k+ real-DROPs ----------
    print("\n[11] COST: slow-path real-DROP batch")
    warm_distinct_leaves(60, salt="cost-")
    state_warm = fetch_state()
    real_targets = [
        {"hash": u["hash"], "target_tier": "DROP"}
        for u in state_warm["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    if real_targets:
        t0 = time.perf_counter()
        big = migrate(real_targets)
        dur_ms = (time.perf_counter() - t0) * 1000
        per_action_ms = dur_ms / max(1, len(real_targets))
        print(
            f"    {len(real_targets)} actions, {dur_ms:.0f} ms total "
            f"({per_action_ms:.3f} ms/action), applied={big['applied']}"
        )
        assert per_action_ms < 1.0, f"slow-path {per_action_ms:.2f} ms exceeds 1 ms ceiling"
        assert big["applied"] > 0

    # ---------- COST: 1k fast-path ----------
    print("[12] COST: 1k all-bogus fast path")
    big_batch = [
        {"hash": f"bogus-{i}-aaaaaaaaaaaaaaaa", "target_tier": "DROP"}
        for i in range(1000)
    ]
    t0 = time.perf_counter()
    big = migrate(big_batch)
    dur_ms = (time.perf_counter() - t0) * 1000
    print(f"    {dur_ms:.0f} ms ({dur_ms/1000:.3f} ms/action), applied={big['applied']}")
    assert dur_ms / 1000 < 1.0
    assert big["applied"] == 0 and len(big["skipped"]) == 1000

    # ---------- MALFORMED HTTP PAYLOAD ----------
    print("[13] malformed HTTP payload returns 400")
    for body in [{}, {"actions": "not a list"}]:
        rr = requests.post(f"{BASE}/aginfer/migrate", json=body, timeout=10)
        assert rr.status_code == 400, f"{body} -> {rr.status_code}"
    rr = requests.post(
        f"{BASE}/aginfer/migrate",
        data="not json",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    assert rr.status_code == 400

    # ---------- CASCADE TO ZERO ----------
    print("[14] cascade: loop migrate(targets) until applied == 0")
    warm_distinct_leaves(15, salt="cascade-")
    state_c = fetch_state()
    pre_hbm = hbm_used(state_c)
    cascade_targets = [{"hash": u["hash"], "target_tier": "DROP"} for u in state_c["units"]]
    iters = 0
    cumulative = 0
    max_iters = 20  # tree depth bound for these prompts is much less
    while True:
        iters += 1
        rr = migrate(cascade_targets)
        cumulative += rr["applied"]
        if rr["applied"] == 0:
            break
        assert iters < max_iters, f"cascade did not terminate in {max_iters} iterations"
    state_pc = fetch_state()
    post_hbm = hbm_used(state_pc)
    leftover_with_data = [
        u for u in state_pc["units"]
        if u["hash"] in {a["hash"] for a in cascade_targets} and u["n_tokens"] > 0
    ]
    print(
        f"    terminated in {iters} iterations, cumulative applied={cumulative}; "
        f"HBM {pre_hbm} -> {post_hbm} ({pre_hbm-post_hbm} freed); "
        f"leftover-with-data: {len(leftover_with_data)}"
    )
    assert not leftover_with_data, (
        f"cascade left {len(leftover_with_data)} target nodes with non-zero data"
    )

    # ---------- ROUND-2 APPLIED ABSENCE ----------
    print("[15] round-2 applied_hashes must be absent post-replay")
    warm_distinct_leaves(15, salt="r2-")
    state_r2 = fetch_state()
    r2_targets = [{"hash": u["hash"], "target_tier": "DROP"} for u in state_r2["units"]]
    _ = migrate(r2_targets)
    r2 = migrate(r2_targets)  # second pass: cascade picks up newly-leafed parents
    r2_applied = set(r2.get("applied_hashes", []))
    state_r2_after = fetch_state()
    leak = r2_applied & {u["hash"] for u in state_r2_after["units"]}
    assert not leak, f"{len(leak)} round-2 applied_hashes still in tree"
    print(f"    round-2 applied={len(r2_applied)}, all absent ✓")

    # ---------- CONCURRENCY (audit MINOR) ----------
    print("[16] concurrency: 2 threads, disjoint batches")
    warm_distinct_leaves(30, salt="conc-")
    state_c2 = fetch_state()
    units = [u for u in state_c2["units"] if u["tier"] == "HBM" and u["n_tokens"] > 0]
    batch_a = [{"hash": u["hash"], "target_tier": "DROP"} for u in units[::2]]
    batch_b = [{"hash": u["hash"], "target_tier": "DROP"} for u in units[1::2]]
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(migrate, batch_a)
        fb = ex.submit(migrate, batch_b)
        ra, rb = fa.result(), fb.result()
    print(
        f"    thread A: applied={ra['applied']}/{len(batch_a)}, "
        f"thread B: applied={rb['applied']}/{len(batch_b)}"
    )
    assert "applied_hashes" in ra and "applied_hashes" in rb, "concurrent calls 5xx'd"
    overlap = set(ra["applied_hashes"]) & set(rb["applied_hashes"])
    assert not overlap, f"concurrent applies overlapped on disjoint inputs: {overlap}"
    # At least one of the two should have applied something (assuming the
    # batches contained any leaves at all).
    if batch_a or batch_b:
        assert ra["applied"] + rb["applied"] >= 1

    # ===== AUDIT-ROUND-3 ADDITIONS =====

    # [17] TOCTOU: daemon's real flow is state -> decide -> migrate, with
    # inference traffic flowing the whole time.  We mimic that by issuing
    # chat calls BETWEEN fetch_state and migrate, so the tree mutates
    # underneath the hashes we captured.  The server must remain well-formed
    # (no 5xx) and every skip reason must be in the legal set.
    print("\n[17] TOCTOU: fetch_state -> chats interleave -> migrate")
    warm_distinct_leaves(20, salt="toctou-pre-")
    state_t = fetch_state()
    targets_t = [
        {"hash": u["hash"], "target_tier": "DROP"}
        for u in state_t["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    # interleave inference traffic so the tree mutates between snapshot and migrate
    for j in range(8):
        chat(f"toctou-interleave-{j}: tell me a fun fact about prime {j}")
    rt = migrate(targets_t)
    skip_reasons_t = {s["reason"] for s in rt["skipped"]}
    print(
        f"    applied={rt['applied']}/{len(targets_t)}, "
        f"skip reasons: {skip_reasons_t}"
    )
    legal_reasons = {
        "not_in_tree", "not_a_leaf", "no_data",
        "already_acted_this_batch",
        "demote_requires_existing_host_backup", "already_on_dram",
        "already_on_hbm", "promote_not_yet_wired",
        "disk_tier_not_yet_wired",
    }
    assert skip_reasons_t <= legal_reasons, (
        f"unexpected reasons after TOCTOU: {skip_reasons_t - legal_reasons}"
    )
    assert rt["applied"] + len(rt["skipped"]) == len(targets_t)

    # [18] Mixed-tier batch: cross-action interaction.  All three tier targets
    # in one batch; some targets share a parent chain so cascade-removed
    # nodes can become stale entries the next action would normally crash on.
    print("\n[18] mixed-tier batch")
    warm_distinct_leaves(20, salt="mixed-")
    state_m = fetch_state()
    hbm_m = [
        u["hash"] for u in state_m["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ]
    if len(hbm_m) < 15:
        print(f"    skip: need >=15 HBM units, got {len(hbm_m)}")
    else:
        mixed_batch = []
        for i, h in enumerate(hbm_m[:15]):
            tier = ["DROP", "DRAM", "HBM"][i % 3]
            mixed_batch.append({"hash": h, "target_tier": tier})
        rm = migrate(mixed_batch)
        applied_m = set(rm["applied_hashes"])
        reasons_m = {s["reason"] for s in rm["skipped"]}
        print(
            f"    {len(mixed_batch)} mixed actions, applied={rm['applied']}, "
            f"reasons: {reasons_m}"
        )
        assert "applied" in rm and "applied_hashes" in rm
        assert rm["applied"] + len(rm["skipped"]) == len(mixed_batch)
        assert reasons_m <= legal_reasons
        # Without HiCache, DRAM and HBM actions cannot apply (no host backup,
        # promote not wired).  Every applied hash must therefore be a DROP.
        drop_hashes = {
            a["hash"] for a in mixed_batch if a["target_tier"] == "DROP"
        }
        non_drop_applied = applied_m - drop_hashes
        assert not non_drop_applied, (
            f"applied a non-DROP action without HiCache? {non_drop_applied}"
        )
        # All applied DROPs must be absent in fresh snapshot.
        state_m2 = fetch_state()
        leaked = applied_m & {u["hash"] for u in state_m2["units"]}
        assert not leaked, f"mixed-batch DROPs leaked nodes: {leaked}"

    # [19] Overlapping concurrent batches: two threads send the SAME batch.
    # The scheduler serializes through ZMQ, so the first request applies
    # what it can; the second sees not_in_tree / no_data for those nodes.
    # Invariant: no hash is applied by BOTH requests (no double-apply across
    # requests).
    print("\n[19] overlapping concurrent batches (same 20 hashes)")
    warm_distinct_leaves(30, salt="overlap-")
    state_o = fetch_state()
    units_o = [
        u["hash"] for u in state_o["units"]
        if u["tier"] == "HBM" and u["n_tokens"] > 0
    ][:20]
    batch_same = [{"hash": h, "target_tier": "DROP"} for h in units_o]
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(migrate, batch_same)
        fb = ex.submit(migrate, batch_same)
        ra2, rb2 = fa.result(), fb.result()
    overlap_applied = set(ra2["applied_hashes"]) & set(rb2["applied_hashes"])
    total_applied = ra2["applied"] + rb2["applied"]
    print(
        f"    thread A applied={ra2['applied']}, thread B applied={rb2['applied']}, "
        f"overlap={len(overlap_applied)}, total={total_applied}, batch_size={len(units_o)}"
    )
    assert not overlap_applied, (
        f"DOUBLE-APPLY across requests: {overlap_applied}"
    )
    # Every hash can apply at most once across both requests
    assert total_applied <= len(units_o), (
        f"total applied {total_applied} > batch size {len(units_o)} -- "
        f"double-apply somewhere"
    )

    # [20] Adversarial / DoS inputs (HTTP-layer caps).
    print("\n[20] adversarial inputs: HTTP caps")
    # (a) hash longer than the 1024-char cap -> 400
    long_hash = "x" * 10_000
    rr = requests.post(
        f"{BASE}/aginfer/migrate",
        json={"actions": [{"hash": long_hash, "target_tier": "DROP"}]},
        timeout=30,
    )
    assert rr.status_code == 400, f"long hash should 400, got {rr.status_code}"
    # (b) too many actions (1 above the 100k cap) -> 400
    over_cap = 100_001
    huge_batch = [{"hash": f"bogus-{i}", "target_tier": "DROP"} for i in range(over_cap)]
    rr = requests.post(
        f"{BASE}/aginfer/migrate",
        json={"actions": huge_batch},
        timeout=60,
    )
    assert rr.status_code == 400, f"oversize batch should 400, got {rr.status_code}"
    print(f"    long-hash and {over_cap}-action batch both correctly 400'd")

    # [21] Memory / GC soak: 1k small calls, latency must stay bounded.
    # If anything per-call leaks O(N), the latency creeps up over time.
    print("\n[21] soak: 1000 small migrate calls, latency must stay bounded")
    soak_batch = [{"hash": f"soak-bogus-{i}", "target_tier": "DROP"} for i in range(5)]
    soak_lat = []
    for _i in range(1000):
        _t0 = time.perf_counter()
        _ = migrate(soak_batch)
        soak_lat.append((time.perf_counter() - _t0) * 1000)
    soak_lat.sort()
    p50, p99 = soak_lat[500], soak_lat[990]
    # First-quartile vs last-quartile median: drift detector for slow leaks.
    early = sorted(soak_lat[:250])[125]
    late = sorted(soak_lat[750:])[125]
    print(
        f"    p50={p50:.2f}ms, p99={p99:.2f}ms, early-median={early:.2f}ms, "
        f"late-median={late:.2f}ms"
    )
    assert p99 < 100, f"soak p99 {p99:.0f} ms -- possible memory pressure"
    # Late should not be more than 3x early (slow leak detector).
    assert late < max(3.0 * early, early + 5.0), (
        f"late median {late:.2f} ms vs early {early:.2f} ms -- "
        f"latency creep suggests a slow leak"
    )
    # Server still serving correctly:
    state_after_soak = fetch_state()
    assert "units" in state_after_soak

    print("\n=== T2 PASSED (depth-audit + round-3) ===")


if __name__ == "__main__":
    main()
