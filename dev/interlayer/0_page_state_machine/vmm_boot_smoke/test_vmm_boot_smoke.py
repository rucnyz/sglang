"""vmm_boot_smoke — mechanism boots (transfer cycle completes).

Diagnostic edition v5:

Sub-tests, each designed to catch a SPECIFIC failure mode that no
other sub-test would catch:

  1.  basic transfer + handle-set diff
  2.  cross_arena handle identity
  3.  ping-pong handle multiset stable (10× iters)
  4.  re-grow: no double-pop (A ∩ B == ∅)
  5.  edges + validation guards (specific exceptions)
  6.  full-chunk byte integrity (every byte)
  7.  tail-eviction explicit
  8.  VA layout: pools disjoint, tensor.data_ptr aliases base
  9.  shrink_explicit honours slot list (list and torch.Tensor inputs)
 10.  owned-handle path WITH handle-identity check (regression-proofed)
 11.  lazy SharedHandlePool growth: handles created on demand
 12.  cleanup semantics: arena.cleanup leaves pool.free untouched;
      pool.cleanup releases; second call no-op
 13.  over-provisioned VA (Σ caps > n_handles)
 14.  cross_arena_transfer with full dst pool → handles stranded in pool.free
 15.  SharedHandlePool.allocate_subpool_range

Out of scope: CUDA graph capture (see ../cuda_graph_safety/); decode-stream
wall under captured graph + concurrent unmap (see ../decode_wall/).
"""
import sys
import time
import torch

from sglang.srt.arena.chunk_arena import (
    SharedHandlePool, ChunkArena, cross_arena_transfer,
)
from sglang.srt.arena.from_blob_ext import tensor_from_va


CHUNK_SIZE = 2 * 1024 * 1024
DEVICE     = 0


# ---------- helpers ----------

def _fpc():
    return CHUNK_SIZE // 4


def _make_pool(n_handles: int) -> SharedHandlePool:
    return SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=n_handles)


def _make_arena(pool, pool_capacities):
    return ChunkArena(DEVICE, CHUNK_SIZE,
                      n_handles=sum(c for _, c in pool_capacities),
                      pool_capacities=pool_capacities,
                      external_handle_pool=pool)


def _handles_mapped_in(arena, pool_name):
    return {h for h in arena.pools[pool_name].mapped if h is not None}


def _tensor_over(arena, pool_name, cap):
    return tensor_from_va(arena.pool_va_base(pool_name),
                          [cap * _fpc()], torch.float32, DEVICE)


def _verify_chunk_all(t, slot, value):
    s = slot * _fpc()
    actual = t[s:s + _fpc()]
    if not torch.all(actual == value):
        first_bad = (actual != value).nonzero(as_tuple=True)[0][0].item()
        raise AssertionError(
            f"slot {slot} byte offset {first_bad * 4}: "
            f"got {actual[first_bad].item()}, expected {value}")


def _fill_slot(t, slot, value):
    s = slot * _fpc()
    t[s:s + _fpc()].fill_(value)


# ---------- sub-tests ----------

def test_1_basic_transfer_handle_set():
    """Specific handles move (not just counter changed)."""
    N, CAP, INIT, MOVE = 16, 8, 4, 2
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            A_before = _handles_mapped_in(arena, "A")
            B_before = _handles_mapped_in(arena, "B")
            free_before = pool.free_count()

            arena.transfer_chunks("A", "B", MOVE)
            torch.cuda.synchronize()

            A_after = _handles_mapped_in(arena, "A")
            B_after = _handles_mapped_in(arena, "B")
            assert pool.free_count() == free_before
            moved = A_before - A_after
            assert len(moved) == MOVE
            assert moved.issubset(B_after)
            assert B_after == B_before | moved
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_2_cross_arena_handle_identity():
    """Same physical handles flow KV → mamba; no re-create."""
    N, CAP, INIT, MOVE = 16, 8, 4, 2
    pool = _make_pool(N)
    try:
        arena_KV = _make_arena(pool, [("kv", CAP)])
        arena_M  = _make_arena(pool, [("mamba", CAP)])
        try:
            arena_KV.grow("kv", INIT); arena_M.grow("mamba", INIT)
            kv_before = _handles_mapped_in(arena_KV, "kv")
            cross_arena_transfer(arena_KV, "kv", arena_M, "mamba", MOVE)
            torch.cuda.synchronize()
            moved = kv_before - _handles_mapped_in(arena_KV, "kv")
            assert len(moved) == MOVE
            assert moved.issubset(_handles_mapped_in(arena_M, "mamba"))
        finally:
            arena_M.cleanup(); arena_KV.cleanup()
    finally:
        pool.cleanup()


def test_3_pingpong_handle_multiset_stable():
    """A↔B ping-pong: handle universe invariant (no leak, no re-create)."""
    N, CAP, INIT, MOVE, ITERS = 16, 8, 4, 2, 10
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            universe = (_handles_mapped_in(arena, "A")
                        | _handles_mapped_in(arena, "B"))
            for i in range(ITERS):
                arena.transfer_chunks("A", "B", MOVE)
                arena.transfer_chunks("B", "A", MOVE)
                torch.cuda.synchronize()
                now = (_handles_mapped_in(arena, "A")
                       | _handles_mapped_in(arena, "B"))
                assert now == universe, f"iter {i}: handle universe drifted"
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_4_regrow_no_double_pop():
    """A ∩ B = ∅ after transfer + regrow (no double-pop)."""
    N, CAP, INIT, MOVE = 16, 8, 4, 2
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            arena.transfer_chunks("A", "B", MOVE)
            arena.grow("A", MOVE)
            torch.cuda.synchronize()
            shared = _handles_mapped_in(arena, "A") & _handles_mapped_in(arena, "B")
            assert not shared, f"double-pop bug: {shared}"
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_5_edges_and_validation_guards():
    """Edges + validation guards with specific exception types."""
    # (5a) n=0 / partial / from==to (ValueError, specific)
    N, CAP, INIT = 16, 8, 4
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            assert arena.transfer_chunks("A", "B", 0) == 0
            granted = arena.transfer_chunks("A", "B", 10)
            assert granted == INIT
            try:
                arena.transfer_chunks("B", "B", 1)
                assert False, "from==to should raise ValueError"
            except ValueError:
                pass
            # bad evict_policy
            try:
                arena.shrink("B", 1, evict_policy="random")
                assert False, "unknown evict_policy should raise ValueError"
            except ValueError:
                pass
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()

    # (5b) cross_arena across different shared pools → specific ValueError
    pool1 = _make_pool(8); pool2 = _make_pool(8)
    try:
        a1 = _make_arena(pool1, [("x", 4)])
        a2 = _make_arena(pool2, [("y", 4)])
        try:
            a1.grow("x", 2); a2.grow("y", 2)
            try:
                cross_arena_transfer(a1, "x", a2, "y", 1)
                assert False, "cross-arena across different pools should raise"
            except ValueError:
                pass
        finally:
            a2.cleanup(); a1.cleanup()
    finally:
        pool2.cleanup(); pool1.cleanup()

    # (5c) cross_arena where one arena has external_pool=None
    pool3 = _make_pool(8)
    try:
        owned = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=4,
                           pool_capacities=[("x", 4)],
                           external_handle_pool=None)
        shared = _make_arena(pool3, [("y", 4)])
        try:
            owned.grow("x", 2); shared.grow("y", 2)
            try:
                cross_arena_transfer(owned, "x", shared, "y", 1)
                assert False, "cross-arena with one owned-pool side should raise"
            except (ValueError, AttributeError):
                # Implementation may raise either; both indicate the
                # guard rejected the call.
                pass
        finally:
            shared.cleanup(); owned.cleanup()
    finally:
        pool3.cleanup()

    # (5d) cross_arena with from_arena IS to_arena
    pool4 = _make_pool(8)
    try:
        arena = _make_arena(pool4, [("p", 4), ("q", 4)])
        try:
            arena.grow("p", 2)
            try:
                cross_arena_transfer(arena, "p", arena, "q", 1)
                assert False, "from_arena IS to_arena should raise"
            except ValueError:
                pass
        finally:
            arena.cleanup()
    finally:
        pool4.cleanup()


def test_6_full_chunk_byte_integrity():
    """Every byte of every chunk; asymmetric values across pools."""
    N, CAP, INIT, MOVE = 16, 8, 4, 2
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            t_A = _tensor_over(arena, "A", CAP)
            t_B = _tensor_over(arena, "B", CAP)
            for slot in range(INIT):
                _fill_slot(t_A, slot, float(100 + slot))
                _fill_slot(t_B, slot, float(200 + slot))
            torch.cuda.synchronize()
            for slot in range(INIT):
                _verify_chunk_all(t_A, slot, float(100 + slot))
                _verify_chunk_all(t_B, slot, float(200 + slot))
            arena.transfer_chunks("A", "B", MOVE)
            torch.cuda.synchronize()
            for slot in range(INIT - MOVE):
                _verify_chunk_all(t_A, slot, float(100 + slot))
            for slot in range(INIT, INIT + MOVE):
                _fill_slot(t_B, slot, float(300 + slot))
            torch.cuda.synchronize()
            for slot in range(INIT, INIT + MOVE):
                _verify_chunk_all(t_B, slot, float(300 + slot))
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_7_tail_eviction_explicit():
    """After transfer, src head is mapped, tail is None."""
    N, CAP, INIT, MOVE = 16, 8, 4, 2
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
        try:
            arena.grow("A", INIT); arena.grow("B", INIT)
            arena.transfer_chunks("A", "B", MOVE)
            mapped = arena.pools["A"].mapped
            for i in range(INIT - MOVE):
                assert mapped[i] is not None
            for i in range(INIT - MOVE, INIT):
                assert mapped[i] is None
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_8_va_layout_disjoint_and_data_ptr_aliasing():
    """Pools occupy disjoint VA; tensor.data_ptr aliases pool base."""
    N, CAP_A, CAP_B = 16, 6, 8
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP_A), ("B", CAP_B)])
        try:
            # disjoint, non-overlapping (substantive — would catch a
            # bug where two pools end up sharing VA)
            assert (arena.pool_va_base("A") + arena.pool_va_size("A")
                    <= arena.pool_va_base("B")), \
                f"A range overlaps B"
            # pin the per-pool size formula (cap × chunk_size)
            assert arena.pool_va_size("A") == CAP_A * CHUNK_SIZE
            assert arena.pool_va_size("B") == CAP_B * CHUNK_SIZE

            # Tensor-from-VA aliasing (would catch wrong pointer in
            # tensor_from_va helper)
            arena.grow("A", 1); arena.grow("B", 1)
            t_A = _tensor_over(arena, "A", CAP_A)
            t_B = _tensor_over(arena, "B", CAP_B)
            assert t_A.data_ptr() == arena.pool_va_base("A")
            assert t_B.data_ptr() == arena.pool_va_base("B")
            # data_ptr is not the same as the other pool's
            assert t_A.data_ptr() != t_B.data_ptr()
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_9_shrink_explicit_list_and_tensor():
    """shrink_explicit honours caller list; accepts torch.Tensor (planner path)."""
    N, CAP, INIT = 16, 8, 6
    pool = _make_pool(N)
    try:
        arena = _make_arena(pool, [("A", CAP)])
        try:
            arena.grow("A", INIT)
            # (a) Python list with out-of-range + duplicate
            n = arena.shrink_explicit("A", [1, 3, 5, 99, -1, 2])
            assert n == 4
            mapped = arena.pools["A"].mapped
            for s in (1, 2, 3, 5):
                assert mapped[s] is None
            for s in (0, 4):
                assert mapped[s] is not None
            # Already-unmapped: skip silently
            assert arena.shrink_explicit("A", [1, 3]) == 0
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()

    # (b) torch.Tensor argument — production planner passes a Tensor
    pool2 = _make_pool(N)
    try:
        arena = _make_arena(pool2, [("A", CAP)])
        try:
            arena.grow("A", INIT)
            slots = torch.tensor([0, 2, 4], dtype=torch.int64)
            n = arena.shrink_explicit("A", slots)
            assert n == 3
            for s in (0, 2, 4):
                assert arena.pools["A"].mapped[s] is None
        finally:
            arena.cleanup()
    finally:
        pool2.cleanup()


def test_10_owned_handle_path_with_handle_identity():
    """Owned-handle path: handle-identity diff, not counter-only.

    (REGRESSION-FIX from prior review — previous version was
    counter-only, which would pass a stub that re-creates handles
    on every transfer.)
    """
    N, CAP, INIT, MOVE = 8, 4, 2, 1
    arena = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N,
                       pool_capacities=[("A", CAP), ("B", CAP)],
                       external_handle_pool=None)
    try:
        assert len(arena._owned_handles) == N
        assert arena._external_pool is None
        # Aliasing: in owned mode, _handles IS _owned_handles
        assert arena._handles is arena._owned_handles
        arena.grow("A", INIT); arena.grow("B", INIT)

        # Same handle-identity check as test_1, but for owned path
        A_before = _handles_mapped_in(arena, "A")
        B_before = _handles_mapped_in(arena, "B")
        arena.transfer_chunks("A", "B", MOVE)
        torch.cuda.synchronize()
        A_after = _handles_mapped_in(arena, "A")
        B_after = _handles_mapped_in(arena, "B")
        moved = A_before - A_after
        assert len(moved) == MOVE
        assert moved.issubset(B_after), \
            f"owned-mode handle identity broken: " \
            f"A lost {moved}, B did not gain it ({B_after - B_before})"
    finally:
        arena.cleanup()


def test_11_lazy_pool_growth():
    """SharedHandlePool(n=0); arena init triggers grow; mapped indices = {0..N-1}."""
    pool = _make_pool(0)
    try:
        assert pool.total_count() == 0
        arena = _make_arena(pool, [("A", 4), ("B", 4)])
        try:
            assert pool.total_count() == 8, "pool should have grown on demand"
            arena.grow("A", 4); arena.grow("B", 4)
            # The 8 mapped indices are exactly the lazily-created ones {0..7}
            all_mapped = (_handles_mapped_in(arena, "A")
                          | _handles_mapped_in(arena, "B"))
            assert all_mapped == set(range(8)), \
                f"mapped indices should be {{0..7}}, got {sorted(all_mapped)}"
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_12_cleanup_semantics():
    """Pin down arena.cleanup vs pool.cleanup semantics, including idempotency."""
    N, CAP, INIT = 8, 4, 2
    pool = _make_pool(N)
    arena = _make_arena(pool, [("A", CAP), ("B", CAP)])
    arena.grow("A", INIT); arena.grow("B", INIT)

    # Before arena.cleanup: pool has N - 2*INIT free
    assert pool.free_count() == N - 2 * INIT

    arena.cleanup()
    # SEMANTIC PIN: arena.cleanup unmaps but does NOT push handles back
    # to pool.free (read chunk_arena.py lines 452-459). So pool.free_count
    # is unchanged by arena.cleanup. This is the documented behavior;
    # pin it so a regression that changes this would surface here.
    assert pool.free_count() == N - 2 * INIT, \
        "arena.cleanup should NOT return handles to pool.free " \
        "(handles' lifetime is owned by pool)"
    # pool.handles list is unchanged too
    assert len(pool.handles) == N

    # pool.cleanup releases all handles
    pool.cleanup()
    assert pool.handles == []
    assert pool.free == []

    # Idempotency: second call should not raise (no assertion on the
    # state — already verified above; this is purely a no-throw check)
    try:
        pool.cleanup()
    except Exception as e:
        raise AssertionError(f"second pool.cleanup raised: {e}")


def test_13_over_provisioned_va():
    """Σ pool_capacities > n_handles: grow() should run out of free handles.

    Use OWNED-handle path here (external_pool=None) to bypass the lazy
    growth tested in test_11. With external_handle_pool, the arena
    would lazily grow the pool to satisfy n_handles, defeating the
    test's premise.
    """
    N, CAP_A, CAP_B = 4, 8, 8  # 16 VA slots, only 4 owned handles
    arena = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N,
                       pool_capacities=[("A", CAP_A), ("B", CAP_B)],
                       external_handle_pool=None)
    try:
        assert arena.total_va_size == (CAP_A + CAP_B) * CHUNK_SIZE
        assert arena.free_handle_count() == N

        # Try to grow A by 6: only N=4 handles available; saturates.
        # arena.grow returns list[int] of mapped slot IDs (post-#213).
        granted = arena.grow("A", 6)
        assert len(granted) == N, \
            f"over-provisioned: grow should saturate at {N}, got {granted}"
        assert arena.free_handle_count() == 0
        # Now grow B by 2: should grant 0 (no free handles)
        assert len(arena.grow("B", 2)) == 0
        # After shrinking A, B can grow
        arena.shrink("A", 2)
        assert len(arena.grow("B", 2)) == 2
    finally:
        arena.cleanup()


def test_14_cross_arena_full_dst_strands_handles():
    """cross_arena_transfer to a full-dst arena: handles stranded in pool.free.

    Docstring promises they can be retried (chunk_arena.py:441-443);
    pin that semantic.
    """
    N, CAP = 8, 2
    pool = _make_pool(N)
    try:
        arena_a = _make_arena(pool, [("a", CAP)])
        arena_b = _make_arena(pool, [("b", CAP)])
        try:
            arena_a.grow("a", CAP)  # 2 mapped on A
            arena_b.grow("b", CAP)  # 2 mapped on B (B is full)
            free_before = pool.free_count()

            # Transfer 2 from a→b; dst is full, so grow on dst returns 0
            granted = cross_arena_transfer(arena_a, "a", arena_b, "b", 2)
            torch.cuda.synchronize()

            assert granted == 0, \
                f"cross-arena to full dst should grant 0, got {granted}"
            assert arena_a.pool_mapped_chunks("a") == 0, \
                "src should still have been shrunk"
            assert arena_b.pool_mapped_chunks("b") == CAP, "dst unchanged"
            # The 2 unmapped handles are stranded in pool.free for retry
            assert pool.free_count() == free_before + CAP, \
                f"stranded handles missing: expected {free_before + CAP} " \
                f"free, got {pool.free_count()}"

            # Retry: shrink b, then grow a — those stranded handles
            # should be reusable
            arena_b.shrink("b", CAP)
            n = arena_a.grow("a", CAP)
            assert len(n) == CAP, "stranded handles should be re-grant-able"
        finally:
            arena_b.cleanup(); arena_a.cleanup()
    finally:
        pool.cleanup()


def test_15_allocate_subpool_range():
    """SharedHandlePool.allocate_subpool_range: watermark + n=0 raises."""
    pool = _make_pool(4)
    try:
        a = pool.allocate_subpool_range(5)
        assert a == 0
        b = pool.allocate_subpool_range(3)
        assert b == 5, f"watermark should be 5, got {b}"
        c = pool.allocate_subpool_range(1)
        assert c == 8
        try:
            pool.allocate_subpool_range(0)
            assert False, "n=0 should raise ValueError"
        except ValueError:
            pass
        try:
            pool.allocate_subpool_range(-1)
            assert False, "n<0 should raise ValueError"
        except ValueError:
            pass
    finally:
        pool.cleanup()


# ---------- runner ----------

def main():
    tests = [
        ("1  basic transfer + handle-set diff",       test_1_basic_transfer_handle_set),
        ("2  cross_arena handle identity",             test_2_cross_arena_handle_identity),
        ("3  ping-pong handle multiset stable",        test_3_pingpong_handle_multiset_stable),
        ("4  re-grow no double-pop",                   test_4_regrow_no_double_pop),
        ("5  edges + validation guards (specific)",    test_5_edges_and_validation_guards),
        ("6  full-chunk byte integrity",               test_6_full_chunk_byte_integrity),
        ("7  tail-eviction explicit",                  test_7_tail_eviction_explicit),
        ("8  VA disjoint + data_ptr aliasing",         test_8_va_layout_disjoint_and_data_ptr_aliasing),
        ("9  shrink_explicit list + torch.Tensor",     test_9_shrink_explicit_list_and_tensor),
        ("10 owned-handle path + identity check",      test_10_owned_handle_path_with_handle_identity),
        ("11 lazy SharedHandlePool growth",            test_11_lazy_pool_growth),
        ("12 cleanup semantics (arena vs pool)",       test_12_cleanup_semantics),
        ("13 over-provisioned VA",                     test_13_over_provisioned_va),
        ("14 cross_arena full dst: stranded handles",  test_14_cross_arena_full_dst_strands_handles),
        ("15 allocate_subpool_range",                  test_15_allocate_subpool_range),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\nvmm_boot_smoke: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
