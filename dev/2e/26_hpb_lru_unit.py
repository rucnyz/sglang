"""
Phase 3 (paper §4.2) — hits-per-byte LRU eviction unit test.

Verifies the load-bearing claim of paper §4.2:
  Recency LRU evicts a high-hit-count system-prompt big page when a
  cold burst floods the cache with fresh leaves. Hits-per-byte LRU
  preserves the high-hit page and evicts the cold leaves first.

Test setup (synthetic, no GPU/engine — direct TreeNode + selector):
  1. Create a "system prompt" big-mamba node H with many simulated hits.
  2. Create N "cold burst" leaf nodes L_i with zero hits, each more
     recent than H.
  3. Recency LRU's `min(last_access_time)` returns H (oldest).
  4. HPB LRU's `min(eviction_priority)` returns one of the L_i.

The test exercises the eviction-priority math and the cold-burst
resistance argument the paper makes.

Run:
  PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
    .venv/bin/python -u dev/2e/26_hpb_lru_unit.py
"""
from __future__ import annotations
import os
import sys
import time

# Force the HPB window to a tiny value so we don't have to wait 60 s
# in the test. The TreeNode reads this at class-init time via env var.
os.environ.setdefault("SGLANG_HPB_WINDOW_S", "5.0")

# Don't import torch unless we have to — just exercise the TreeNode
# arithmetic. But TreeNode depends on torch via the file's imports.
import torch  # noqa: F401

from sglang.srt.mem_cache.mamba_radix_cache import TreeNode


def make_node(*, mamba_size: int = 0, value_size: int = 0) -> TreeNode:
    n = TreeNode()
    if value_size > 0:
        n.value = torch.zeros(value_size, dtype=torch.int64)
    if mamba_size > 0:
        n.mamba_value = torch.zeros(mamba_size, dtype=torch.int64)
    return n


def test_priority_ranking():
    print("== Test 1: hits-per-byte priority ranking ==")
    # H = "system prompt" big-mamba page: heavy snapshot, high hit rate.
    H = make_node(mamba_size=1, value_size=2048)   # ~ 1*1024 + 2048 = 3072 weight
    # Simulate 50 hits on H within window.
    for _ in range(50):
        H.record_hit()
    print(f"  H hits_in_window = {H.hits_in_window()}, priority = {H.eviction_priority():.6f}")

    # L = cold-burst leaves: small KV-only page, zero hits.
    Ls = [make_node(value_size=512) for _ in range(20)]
    # Make L_i more recent in wall time than H (just construct after H).
    for L in Ls:
        # No record_hit calls; hits_in_window = 0.
        pass
    print(f"  L[0] hits = {Ls[0].hits_in_window()}, priority = {Ls[0].eviction_priority():.6f}")

    # Assert H has higher priority than every L.
    H_p = H.eviction_priority()
    L_ps = [L.eviction_priority() for L in Ls]
    for i, lp in enumerate(L_ps):
        assert H_p > lp, f"H priority {H_p} should beat L[{i}] {lp}"
    print(f"  H.priority ({H_p:.4f}) > max(L.priority) ({max(L_ps):.4f}). GOOD.")
    print("PASS Test 1\n")


def test_hpb_picks_cold_first():
    print("== Test 2: HPB selector picks cold leaves before hot system page ==")
    # Build a list of nodes mimicking the lru_list cache. Use a simple
    # min-priority scan, replicating MambaRadixCache._hpb_pick_mamba_eviction.
    nodes = []
    H = make_node(mamba_size=1, value_size=2048)
    for _ in range(50):
        H.record_hit()
    nodes.append(("H_system", H))
    for i in range(10):
        L = make_node(value_size=512)  # small KV-only leaf
        nodes.append((f"L_{i}", L))

    # Pick the lowest-priority node.
    best_name = None
    best_priority = float("inf")
    for name, n in nodes:
        p = n.eviction_priority()
        if p < best_priority:
            best_priority = p
            best_name = name
    print(f"  HPB-selected for eviction: {best_name} (priority={best_priority:.6f})")
    assert best_name.startswith("L_"), \
        f"HPB must pick a cold leaf, not {best_name}"

    # Now flip to recency: H was created first, so H is oldest.
    recency_pick = min(nodes, key=lambda kv: kv[1].last_access_time)[0]
    print(f"  Recency-LRU would evict: {recency_pick}")
    assert recency_pick == "H_system", \
        "recency-LRU should pick H first (oldest); test setup is wrong otherwise"

    print("  HPB and recency disagree exactly as paper §4.2 predicts. GOOD.")
    print("PASS Test 2\n")


def test_window_decay():
    print("== Test 3: hit window decays over time ==")
    # Set window very short so we can observe decay.
    TreeNode.hpb_window_s = 0.5  # half a second

    H = make_node(mamba_size=1, value_size=2048)
    for _ in range(20):
        H.record_hit()
    p_before = H.eviction_priority()
    print(f"  immediately after 20 hits: priority = {p_before:.6f}, hits = {H.hits_in_window()}")
    assert H.hits_in_window() == 20

    time.sleep(0.7)  # > window
    p_after = H.eviction_priority()
    print(f"  after {0.7}s sleep (> {TreeNode.hpb_window_s}s window): priority = {p_after:.6f}, hits = {H.hits_in_window()}")
    assert H.hits_in_window() == 0
    assert p_after < p_before, \
        f"priority should drop from {p_before} to ~0 after window expires (got {p_after})"
    print("  windowed counter decays correctly. GOOD.")
    print("PASS Test 3\n")


def test_zero_size_guard():
    print("== Test 4: zero-byte node guard ==")
    # A degenerate node with no value/mamba_value: eviction_priority
    # should not div-by-zero.
    N = TreeNode()
    assert N.eviction_priority() == 0.0, \
        "node with no bytes and no hits should have priority 0"
    N.record_hit()
    p = N.eviction_priority()
    assert p == float("inf"), \
        f"node with hits but zero bytes should have priority +inf (got {p})"
    print(f"  empty node + 1 hit: priority = {p}. GOOD.")
    print("PASS Test 4\n")


def test_estimate_v_prefix_marginal():
    print("== Test 5: V_prefix' marginal-value reporter (paper §4.2 Eq 4.4) ==")
    # Stubbed cache class with the bare minimum the estimator needs.
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, RadixKey

    # Patch enough of MambaRadixCache to skip __init__ heavy machinery.
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.disable = False
    # Need a mamba_lru_list whose `cache` dict the HPB selector iterates.
    class StubLRU:
        def __init__(self): self.cache = {}
    cache.mamba_lru_list = StubLRU()

    # Build a small tree: root -> A -> B (B is the boundary)
    # boundary has 5 hits, prefix length up to root = 100 + 200 = 300 tokens.
    root = TreeNode()
    A = TreeNode()
    A.parent = root
    A.key = RadixKey([0]*100, extra_key=None)
    B = make_node(mamba_size=1, value_size=512)
    B.parent = A
    B.key = RadixKey([0]*200, extra_key=None)
    for _ in range(5):
        B.record_hit()

    cache.mamba_lru_list.cache = {id(B): B}

    # Use a small window for predictable arithmetic.
    TreeNode.hpb_window_s = 60.0
    v = cache.estimate_v_prefix_marginal(c_prefill=1.0)
    # Expected: (5 / 60) * 300 * 1.0 = 25.0 tokens-prefill-saved/s
    expected = (5.0 / 60.0) * 300.0
    print(f"  V_prefix' = {v:.4f} (expected ~{expected:.4f})")
    assert abs(v - expected) < 1e-3, f"expected {expected}, got {v}"
    print("  V_prefix' arithmetic matches paper §4.2 Eq 4.4. GOOD.")

    # Empty tree → 0.
    cache.mamba_lru_list.cache = {}
    v0 = cache.estimate_v_prefix_marginal()
    assert v0 == 0.0, f"empty tree should report 0.0, got {v0}"
    print("  empty tree reports 0.0. GOOD.")
    print("PASS Test 5\n")


def main() -> int:
    test_priority_ranking()
    test_hpb_picks_cold_first()
    test_window_decay()
    test_zero_size_guard()
    test_estimate_v_prefix_marginal()
    print("== ALL PASS: hits-per-byte LRU primitives + V_prefix reporter ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
