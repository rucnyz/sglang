"""Unit test for the LCM-balanced cross-pool transfer math in
xpool_actuator. Verifies both invariants hold under various
n_src/n_dst/target combinations.

The math:
  target_src_total = n_src * len(pages_to_unmap)
  target_dst_total = n_dst * pages_to_map_dst
  target_total = min(...)
  total = floor(target_total / lcm(n_src, n_dst)) * lcm(n_src, n_dst)
  per_src = total // n_src
  per_dst = total // n_dst

Invariants:
  cross-pool atomic:  per_src * n_src == per_dst * n_dst == total
  cross-layer uniform: each src shrinks by per_src; each dst grows per_dst
  bounded:            per_src ≤ len(pages_to_unmap), per_dst ≤ pages_to_map_dst
"""
import math
import sys


def _compute(n_src, n_dst, len_pages_to_unmap, pages_to_map_dst):
    target_src_total = n_src * len_pages_to_unmap
    target_dst_total = n_dst * pages_to_map_dst
    target_total = min(target_src_total, target_dst_total)
    lcm_n = math.lcm(n_src, n_dst) if (n_src > 0 and n_dst > 0) else 0
    total = (target_total // lcm_n) * lcm_n if lcm_n else 0
    per_src = total // n_src if n_src else 0
    per_dst = total // n_dst if n_dst else 0
    return per_src, per_dst, total


def _check(n_src, n_dst, len_unmap, pages_dst, expected):
    per_src, per_dst, total = _compute(n_src, n_dst, len_unmap, pages_dst)
    assert (per_src, per_dst, total) == expected, \
        f"n_src={n_src} n_dst={n_dst} unmap={len_unmap} dst={pages_dst}: " \
        f"got ({per_src}, {per_dst}, {total}), expected {expected}"
    # Invariants
    assert per_src * n_src == total
    assert per_dst * n_dst == total
    assert per_src <= len_unmap
    assert per_dst <= pages_dst


def test_1_qwen3_5_9b_hybrid_target_4():
    """Real config from D8: KV 16 sub-pools, mamba 24 sub-pools, target=4."""
    # lcm(16, 24) = 48
    # target_src_total = 16*4 = 64, target_dst_total = 24*4 = 96, target = 64
    # total = floor(64/48)*48 = 48
    # per_src = 48/16 = 3, per_dst = 48/24 = 2
    _check(n_src=16, n_dst=24, len_unmap=4, pages_dst=4,
           expected=(3, 2, 48))
    print("  PASS  1  Qwen3.5-9B (16 KV / 24 mamba), target 4 → per_src=3 per_dst=2 total=48")


def test_2_equal_subpools():
    """n_src == n_dst: per_src == per_dst == target (trivial case)."""
    _check(n_src=8, n_dst=8, len_unmap=4, pages_dst=4,
           expected=(4, 4, 32))
    print("  PASS  2  equal sub-pools: per_src == per_dst == target")


def test_3_target_too_small():
    """If target × n_src < lcm, total rounds to 0."""
    # lcm(16, 24) = 48
    # target_src_total = 16*2 = 32 < 48
    # total = 0; per_src=0, per_dst=0
    _check(n_src=16, n_dst=24, len_unmap=2, pages_dst=2,
           expected=(0, 0, 0))
    print("  PASS  3  target × n_src < lcm → zero transfer (rounded down)")


def test_4_asymmetric_targets():
    """Different per-sub-pool targets on src vs dst sides."""
    # n_src=4, n_dst=6, src wants to unmap 6 per pool, dst can take 3 per pool
    # target_src_total = 4*6 = 24, target_dst_total = 6*3 = 18, target = 18
    # lcm(4, 6) = 12; total = (18//12)*12 = 12
    # per_src = 12/4 = 3, per_dst = 12/6 = 2
    _check(n_src=4, n_dst=6, len_unmap=6, pages_dst=3,
           expected=(3, 2, 12))
    print("  PASS  4  asymmetric targets: respects min, LCM-aligned")


def test_5_one_subpool_per_side():
    """Degenerate case: 1 src sub-pool, 1 dst sub-pool."""
    _check(n_src=1, n_dst=1, len_unmap=10, pages_dst=10,
           expected=(10, 10, 10))
    print("  PASS  5  1 src / 1 dst: trivial pass-through")


def test_6_invariants_smoke():
    """Random-style smoke: try many combos, verify invariants always hold."""
    for n_src in (1, 4, 8, 12, 16, 24):
        for n_dst in (1, 4, 8, 12, 16, 24, 32):
            for tgt in (1, 2, 3, 4, 8):
                per_src, per_dst, total = _compute(n_src, n_dst, tgt, tgt)
                assert per_src * n_src == total
                assert per_dst * n_dst == total
                assert per_src <= tgt
                assert per_dst <= tgt
    print("  PASS  6  invariants hold across many n_src × n_dst × target combos")


def test_7_integration_with_real_chunk_arena():
    """P0 from audit: the math test alone doesn't validate that
    actuator's call to chunk_arena.shrink_explicit + chunk_arena.grow
    uses the correct per_src / per_dst values. Build two real arenas
    over a shared handle pool, mimic the actuator's exact call sequence,
    and verify cross-pool atomicity AND cross-layer uniformity hold.

    Setup mirrors xpool_actuator.execute_async lines 218-294 but in
    isolation (no FirePlanResult / FireToken plumbing)."""
    import sys
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    import torch
    torch.cuda.set_device(0)

    from sglang.srt.arena.chunk_arena import (
        ChunkArena, SharedHandlePool, cross_arena_transfer,
    )

    CHUNK = 2 * 1024 * 1024
    # n_src=4 src sub-pools, n_dst=6 dst sub-pools (small but mismatched)
    # Each sub-pool capped at 10 chunks; init mapped to 4 chunks.
    n_src, n_dst = 4, 6
    n_init = 4
    n_cap = 10
    shared = SharedHandlePool(device_id=0, chunk_size=CHUNK,
                              n_handles=(n_src + n_dst) * n_cap)
    src_arena = ChunkArena(
        device_id=0, chunk_size=CHUNK, n_handles=0,
        pool_capacities=[(f"src{i}", n_cap) for i in range(n_src)],
        external_handle_pool=shared,
    )
    dst_arena = ChunkArena(
        device_id=0, chunk_size=CHUNK, n_handles=0,
        pool_capacities=[(f"dst{j}", n_cap) for j in range(n_dst)],
        external_handle_pool=shared,
    )
    # Map init chunks into each pool.
    for i in range(n_src):
        src_arena.grow(f"src{i}", n_init)
    for j in range(n_dst):
        dst_arena.grow(f"dst{j}", n_init)
    src_init_total = sum(src_arena.pool_mapped_chunks(f"src{i}") for i in range(n_src))
    dst_init_total = sum(dst_arena.pool_mapped_chunks(f"dst{j}") for j in range(n_dst))
    shared_init = shared.free_count()
    assert src_init_total == n_src * n_init
    assert dst_init_total == n_dst * n_init

    # Mimic actuator's balanced-atomic math (target=4 per sub-pool both sides)
    target_src = n_src * 4  # 16
    target_dst = n_dst * 4  # 24
    target_total = min(target_src, target_dst)  # 16
    lcm_n = math.lcm(n_src, n_dst)  # 12
    total = (target_total // lcm_n) * lcm_n  # 12
    per_src = total // n_src  # 3
    per_dst = total // n_dst  # 2

    # Sanity-check the math one more time.
    assert per_src == 3 and per_dst == 2 and total == 12

    # Actuator-equivalent: shrink each src sub-pool by per_src.
    unmapped_total = 0
    for i in range(n_src):
        # Use the *tail* slot ids (matches actuator's
        # tail-evict shrink_explicit pattern).
        pool_name = f"src{i}"
        pool_state = src_arena.pools[pool_name]
        # pool.mapped is a list[Optional[int]] indexed by slot
        mapped = [slot for slot, h in enumerate(pool_state.mapped) if h is not None]
        tail_pages = mapped[-per_src:]
        unmapped_total += src_arena.shrink_explicit(pool_name, tail_pages)
    assert unmapped_total == total, f"src unmap {unmapped_total} != {total}"

    # Grow each dst sub-pool by per_dst.
    granted_per_subpool = []
    for j in range(n_dst):
        granted_per_subpool.append(len(dst_arena.grow(f"dst{j}", per_dst)))  # #213
    granted_total = sum(granted_per_subpool)

    # ===== Invariants =====
    # (1) cross-pool atomic: total handles unmapped == total mapped
    assert unmapped_total == granted_total, \
        f"unmap {unmapped_total} != grant {granted_total}"
    # (2) cross-layer uniform: every dst sub-pool grew by EXACTLY per_dst
    assert all(g == per_dst for g in granted_per_subpool), \
        f"non-uniform dst grow: {granted_per_subpool}"
    # (3) shared free count unchanged across the round-trip
    assert shared.free_count() == shared_init, \
        f"shared free count drifted: {shared_init} → {shared.free_count()}"
    # (4) per-pool state mathematics
    for i in range(n_src):
        m = src_arena.pool_mapped_chunks(f"src{i}")
        assert m == n_init - per_src, f"src{i} mapped={m}, expected {n_init - per_src}"
    for j in range(n_dst):
        m = dst_arena.pool_mapped_chunks(f"dst{j}")
        assert m == n_init + per_dst, f"dst{j} mapped={m}, expected {n_init + per_dst}"

    src_arena.cleanup()
    dst_arena.cleanup()
    shared.cleanup()
    print("  PASS  7  integration: math → real chunk_arena calls, all 4 invariants")


def main():
    tests = [test_1_qwen3_5_9b_hybrid_target_4, test_2_equal_subpools,
             test_3_target_too_small, test_4_asymmetric_targets,
             test_5_one_subpool_per_side, test_6_invariants_smoke,
             test_7_integration_with_real_chunk_arena]
    print(f"\nxpool balanced-atomic math tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nBalanced-atomic: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
