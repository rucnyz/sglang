"""Correctness + conservation bug-net for the MambaPool hot-path fix.

Complements `test_mamba_free_fastpath.py` (which pins the `_no_cross_fire`
predicate flips). This file proves the FIX did not change behaviour and that
the whole cap machinery conserves slots, the class of bug (#319/#320) that
must be caught BEFORE the budgeter e2e integration:

  A  clear_slots zeroes (deferred-clear model): alloc hands out a slot dirty
     and `clear_slots(indices)` zeroes it on the forward stream (driven by
     `req.mamba_needs_clear`). Pins that clear_slots leaves every cleared slot
     all-zero in both temporal layouts (one scalar `torch.zeros(1)` per dtype,
     broadcast per layer).
  B  copy_from fidelity: the per-layer indexed copy writes dst byte-identical
     to src (conv + temporal), both layouts. Guards the flagged-but-unchanged
     copy loop against any future batching change.
  C  free fast-path == slow-path: on a no-cross-fire input the fast path
     (`torch.cat`) and the full capped-aware path produce the IDENTICAL
     free_slots. The fast path is a pure optimization, not a behaviour change.
  D  fast path takes no capped detour: the no-cross-fire free never calls
     `torch.isin` (the membership test that gates the `.item()` device sync),
     while the capped path does. Deterministic structural pin of "no sync",
     corroborated by the wall-time free-delta in test_mamba_pool_perf_targets.
  E  conservation under randomized churn: a seeded alloc/free/shrink/grow/
     migrate/unmark sequence keeps {free, capped, allocated} a partition of
     [1, max_size] after EVERY op (pairwise disjoint, full cover, no dup,
     capped <= max_size, live_size == size - |capped within cap|, free <= cap).

Uses the REAL `MambaPool` constructor on CPU at a tiny geometry (production
state shape, num_layers=2, size=8) so every assertion runs against the
shipped code paths, not a hand-rolled imitation. No GPU needed.

Run: .venv/bin/python dev/interlayer/3_budgeter/mamba_pool_perf/test_mamba_pool_invariants.py
"""
import os
import random
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from bench_mamba_pool_ops import _cache_params  # noqa: E402  reuse prod shape

DEVICE = "cpu"
NUM_LAYERS = 2
SIZE = 8


def _build_pool(per_layer: bool, size: int = SIZE, max_size: int = SIZE):
    """Real `MambaPool` on CPU. `per_layer` toggles the arena/budgeter
    per-layer-list temporal layout vs the baseline stacked single tensor;
    arena (VMM) is force-OFF so this stays CPU-only. The production state
    shape is reused via `_cache_params` (tests reuse prod components)."""
    os.environ.pop("SGLANG_MAMBA_ARENA", None)
    os.environ.pop("SGLANG_ARENA_SHARED", None)
    if per_layer:
        os.environ["SGLANG_MAMBA_PERLAYER"] = "1"
    else:
        os.environ.pop("SGLANG_MAMBA_PERLAYER", None)

    from sglang.srt.mem_cache.memory_pool import MambaPool

    return MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=_cache_params(NUM_LAYERS),
        mamba_layer_ids=list(range(NUM_LAYERS)),
        device=DEVICE,
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=max_size,
    )


def _conv_list(pool):
    return list(pool.mamba_cache.conv)


def _temporal_list(pool):
    t = pool.mamba_cache.temporal
    return list(t) if isinstance(t, list) else [t]


def _fill_garbage(pool, value: float = 7.0):
    """Write a non-zero sentinel into every conv + temporal slot so a stale
    (non-zeroed) alloc would be caught."""
    for t in _conv_list(pool):
        t.fill_(value)
    for t in _temporal_list(pool):
        t.fill_(value)


# ------------------------------------------------------------ A: clear_slots zero
def test_A_clear_slots_zeroes(per_layer: bool):
    """Deferred-clear model: alloc hands out a slot dirty; clear_slots(indices)
    zeroes it on the forward stream. Pins that clear_slots leaves every cleared
    slot all-zero in both temporal layouts."""
    pool = _build_pool(per_layer)
    _fill_garbage(pool)
    idx = pool.alloc(3)
    assert idx is not None and idx.numel() == 3
    pool.clear_slots(idx)
    for t in _conv_list(pool):
        # conv slot axis is dim 1: t[:, idx]
        sub = t[:, idx]
        assert torch.count_nonzero(sub) == 0, "conv clear_slots left stale state"
    for t in _temporal_list(pool):
        # per-layer temporal slot axis is dim 0; stacked is indexed on dim 1.
        if pool._mamba_perlayer:
            sub = t[idx]
        else:
            sub = t[:, idx]
        assert torch.count_nonzero(sub) == 0, "temporal clear_slots left stale state"
    print(f"  PASS  A  clear_slots zeroes (per_layer={per_layer})")


# ---------------------------------------------------------------- B: copy_from
def test_B_copy_from_fidelity(per_layer: bool):
    pool = _build_pool(per_layer)
    _fill_garbage(pool, value=0.0)  # clean baseline
    src = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)
    dst = torch.tensor([5, 6], dtype=torch.int64, device=DEVICE)
    # Write a distinct per-slot pattern into src.
    for t in _conv_list(pool):
        t[:, src[0]] = 3.0
        t[:, src[1]] = 4.0
    for t in _temporal_list(pool):
        if pool._mamba_perlayer:
            t[src[0]] = 3.0
            t[src[1]] = 4.0
        else:
            t[:, src[0]] = 3.0
            t[:, src[1]] = 4.0
    pool.copy_from(src, dst)
    for t in _conv_list(pool):
        assert torch.equal(t[:, dst[0]], t[:, src[0]]), "conv copy_from not byte-identical"
        assert torch.equal(t[:, dst[1]], t[:, src[1]])
    for t in _temporal_list(pool):
        if pool._mamba_perlayer:
            assert torch.equal(t[dst[0]], t[src[0]]), "temporal copy_from not byte-identical"
            assert torch.equal(t[dst[1]], t[src[1]])
        else:
            assert torch.equal(t[:, dst[0]], t[:, src[0]])
            assert torch.equal(t[:, dst[1]], t[:, src[1]])
    print(f"  PASS  B  copy_from fidelity (per_layer={per_layer})")


# ------------------------------------------------- C: fast-path == slow-path
class _ForceSlowFree(Exception):
    pass


def test_C_free_fast_eq_slow():
    """On a no-cross-fire input with no capped / no above-cap ids, the fast
    path and the full capped-aware slow path must yield identical free_slots.
    We run the slow path on an otherwise-identical pool by overriding
    `_no_cross_fire` to False (capped still empty, size == max_size, so the
    slow path simply routes every freed id <= cap back to free_slots)."""
    freed = torch.tensor([2, 4, 7], dtype=torch.int64, device=DEVICE)

    # Fast path.
    fast = _build_pool(per_layer=False)
    fast.free_slots = fast.free_slots[~torch.isin(fast.free_slots, freed)]
    assert fast._no_cross_fire is True
    fast.free(freed)

    # Slow path on identical state: force the predicate False.
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _SlowPool(MambaPool):
        @property
        def _no_cross_fire(self):  # type: ignore[override]
            return False

    slow = _build_pool(per_layer=False)
    slow.__class__ = _SlowPool
    slow.free_slots = slow.free_slots[~torch.isin(slow.free_slots, freed)]
    assert slow._no_cross_fire is False
    slow.free(freed)

    assert sorted(fast.free_slots.tolist()) == sorted(slow.free_slots.tolist()), (
        f"fast {sorted(fast.free_slots.tolist())} != slow "
        f"{sorted(slow.free_slots.tolist())}"
    )
    assert fast._capped_slots.numel() == 0 and slow._capped_slots.numel() == 0
    print("  PASS  C  free fast-path == slow-path (identical free_slots)")


# --------------------------------------- D: fast path triggers no sync primitive
def test_D_fast_path_skips_sync_primitives():
    """The no-cross-fire free must trigger NONE of the device-sync primitives the
    fix removes: not `torch.isin` (capped membership) and not `Tensor.item` (the
    `.any().item()` host pull). Counting BOTH (not just isin) pins the actual
    property: a refactor that dropped isin but kept an `.item()`/`.cpu()` in the
    fast path would still reintroduce the per-free sync. Patch both, assert the
    fast path calls neither and the capped slow path calls them.

    (On CPU `.item()` is not a real device sync, so this pins the exact
    primitives the fix targets, not wall-time; the GPU wall-time is pinned by
    test_mamba_pool_perf_targets.)"""
    real_isin = torch.isin
    real_item = torch.Tensor.item
    counts = {"isin": 0, "item": 0}

    def counting_isin(*a, **k):
        counts["isin"] += 1
        return real_isin(*a, **k)

    def counting_item(self, *a, **k):
        counts["item"] += 1
        return real_item(self, *a, **k)

    torch.isin = counting_isin
    torch.Tensor.item = counting_item
    try:
        # Fast path: capped empty, size == max_size.
        fast = _build_pool(per_layer=False)
        counts["isin"] = 0
        counts["item"] = 0
        fast.free(torch.tensor([3], dtype=torch.int64, device=DEVICE))
        assert counts["isin"] == 0 and counts["item"] == 0, \
            f"fast-path free triggered a sync primitive: {counts}"

        # Slow path: a populated _capped_slots forces the capped detour.
        slow = _build_pool(per_layer=False)
        slow.free_slots = slow.free_slots[slow.free_slots != 5]
        slow._capped_slots = torch.tensor([5], dtype=torch.int64, device=DEVICE)
        assert slow._no_cross_fire is False
        counts["isin"] = 0
        counts["item"] = 0
        slow.free(torch.tensor([3], dtype=torch.int64, device=DEVICE))
        assert counts["isin"] >= 1 and counts["item"] >= 1, \
            f"capped-path free must use isin + item (the sync the fast path skips): {counts}"
    finally:
        torch.isin = real_isin
        torch.Tensor.item = real_item
    print("  PASS  D  fast path triggers zero sync primitives (isin/item); capped path uses both")


# ------------------------------------------------- E: conservation under churn
def _assert_partition(pool, allocated: set, max_size: int):
    free = pool.free_slots.tolist()
    capped = pool._capped_slots.tolist()
    s_free, s_capped = set(free), set(capped)
    # No duplicates within free_slots.
    assert len(free) == len(s_free), f"dup in free_slots: {sorted(free)}"
    assert len(capped) == len(s_capped), f"dup in _capped_slots: {sorted(capped)}"
    # Pairwise disjoint.
    assert s_free.isdisjoint(s_capped), f"free ∩ capped = {s_free & s_capped}"
    assert s_free.isdisjoint(allocated), f"free ∩ alloc = {s_free & allocated}"
    assert s_capped.isdisjoint(allocated), f"capped ∩ alloc = {s_capped & allocated}"
    # Full cover of [1, max_size].
    union = s_free | s_capped | allocated
    assert union == set(range(1, max_size + 1)), (
        f"partition gap/extra: missing={set(range(1, max_size + 1)) - union}, "
        f"extra={union - set(range(1, max_size + 1))}"
    )
    # Free ids never exceed the live cap.
    assert all(i <= pool.size for i in s_free), f"free id > size={pool.size}: {sorted(s_free)}"
    # Capped invariant + non-negative live_size consistent with accounting.
    assert pool._capped_slots.numel() <= max_size
    n_capped_within = len(s_capped & set(range(1, pool.size + 1)))
    assert pool.live_size == pool.size - n_capped_within, (
        f"live_size={pool.live_size} != size({pool.size}) - capped_within({n_capped_within})"
    )
    assert pool.live_size >= 0


def test_E_conservation_under_churn(per_layer: bool, max_size: int = 8, n_ops: int = 400):
    rng = random.Random(1234 if not per_layer else 5678)
    pool = _build_pool(per_layer, size=max_size, max_size=max_size)
    allocated: set = set()
    _assert_partition(pool, allocated, max_size)

    for step in range(n_ops):
        op = rng.choice(["alloc", "free", "shrink", "grow", "migrate", "unmark"])
        if op == "alloc":
            k = rng.randint(1, 3)
            if k <= pool.available_size():
                idx = pool.alloc(k)
                if idx is not None:
                    allocated |= set(idx.tolist())
        elif op == "free" and allocated:
            k = rng.randint(1, max(1, len(allocated)))
            chosen = rng.sample(sorted(allocated), min(k, len(allocated)))
            allocated -= set(chosen)
            pool.free(torch.tensor(chosen, dtype=torch.int64, device=DEVICE))
        elif op == "shrink":
            pool.set_capacity_slots(rng.randint(1, max(1, pool.size)))
        elif op == "grow":
            pool.set_capacity_slots(rng.randint(pool.size, max_size))
        elif op == "migrate" and allocated:
            free_now = pool.free_slots.tolist()
            if free_now:
                src = rng.choice(sorted(allocated))
                dst = rng.choice(free_now)
                if src != dst and pool.migrate_slot(src, dst):
                    allocated.discard(src)   # src -> capped (chunk to be unmapped)
                    allocated.add(dst)       # dst now carries src's live state
        elif op == "unmark":
            capped_now = pool._capped_slots.tolist()
            if capped_now:
                k = rng.randint(1, len(capped_now))
                ids = rng.sample(capped_now, k)
                pool.unmark_slots(torch.tensor(ids, dtype=torch.int64, device=DEVICE))
        _assert_partition(pool, allocated, max_size)
    print(f"  PASS  E  conservation under {n_ops} churn ops (per_layer={per_layer})")


def main():
    tests = [
        lambda: test_A_clear_slots_zeroes(False),
        lambda: test_A_clear_slots_zeroes(True),
        lambda: test_B_copy_from_fidelity(False),
        lambda: test_B_copy_from_fidelity(True),
        test_C_free_fast_eq_slow,
        test_D_fast_path_skips_sync_primitives,
        lambda: test_E_conservation_under_churn(False),
        lambda: test_E_conservation_under_churn(True),
    ]
    print(f"\nMambaPool correctness + conservation (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {getattr(t, '__name__', t)}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\ninvariants: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
