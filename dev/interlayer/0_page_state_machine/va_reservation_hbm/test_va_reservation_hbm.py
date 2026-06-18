"""va_reservation_hbm — VA reservation is free (mechanism-level HBM accounting, v2).

SCOPE: mechanism-only (no sglang server). The full design.md §va_reservation_hbm
conjecture ("interlayer boot HBM ≤ baseline + 1 GB") is deferred to
the engine-level pair (`../pristine_saturation/`), which requires the
actuator wired into the scheduler. This file verifies the underlying
claim — cuMemAddressReserve is HBM-free — at the chunk_arena.py layer.

Measurement: torch.cuda.mem_get_info()[0] (free bytes on device).
Tolerance is calibrated in test_0 against measured idle noise (σ × 4),
not hardcoded.

Sub-tests:
  0. Noise calibration: 20 samples back-to-back, compute σ, set tol = 4σ
  1. Handle creation (construction + incremental grow + cleanup return)
  2. cuMemAddressReserve is free across sweep of sizes incl. huge
  3. Single arena oversized VA, AND two arenas sharing one pool
  4. Map/unmap leak detection — loop 100× to amplify per-call leak signal

(test_5 deleted; subsumed by test_1's paired owned-vs-external check.)
"""
import ctypes
import statistics
import sys
import torch

from sglang.srt.arena.chunk_arena import (
    SharedHandlePool, ChunkArena,
    CUDA, _DPTR, _check,
)


CHUNK_SIZE = 2 * 1024 * 1024
DEVICE     = 0
# Calibrated by test_0; set as module global so other tests can use.
TOL_BYTES  = 4 * 1024 * 1024   # initial default; replaced after test_0


def _free_bytes():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info(DEVICE)[0]


def _mib(b):
    return b / (1024 ** 2)


def _within(measured, expected, tol=None):
    if tol is None:
        tol = TOL_BYTES
    return abs(measured - expected) <= tol


def _warm_cuda():
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()


# ---------- sub-tests ----------

def test_0_calibrate_noise_floor():
    """Measure mem_get_info noise across 20 idle samples; set TOL = 4σ.

    Without this, the chosen tolerance is arbitrary and we'd either
    accept real leaks (too loose) or fail on background driver
    activity (too tight).
    """
    global TOL_BYTES
    samples = []
    for _ in range(20):
        samples.append(_free_bytes())
    sigma = statistics.stdev(samples)
    span = max(samples) - min(samples)
    new_tol = max(int(4 * sigma), 64 * 1024)   # at least 64 KiB
    print(f"    20 idle samples: σ = {_mib(sigma):.3f} MiB, "
          f"span = {_mib(span):.3f} MiB")
    print(f"    setting TOL = 4σ = {_mib(new_tol):.3f} MiB")
    TOL_BYTES = new_tol
    # Sanity: noise should be well under 1 MiB on idle GPU
    assert sigma < 1024 * 1024, (
        f"GPU mem_get_info noise σ = {_mib(sigma):.2f} MiB > 1 MiB. "
        f"Something else is using the GPU; rerun on idle device.")


def test_1_handle_lifecycle_hbm_accounting():
    """Handle creation, incremental grow, AND cleanup HBM return —
    all in one paired test (covers production lazy-grow path)."""
    N_INIT = 64                            # large enough that signal >> tol
    N_GROW = 32
    expected_init = N_INIT * CHUNK_SIZE
    expected_grow = N_GROW * CHUNK_SIZE
    expected_total = expected_init + expected_grow

    # (a) Construction
    free_before = _free_bytes()
    pool = SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=N_INIT)
    free_after_init = _free_bytes()
    consumed_init = free_before - free_after_init
    print(f"    construct({N_INIT} handles): "
          f"consumed {_mib(consumed_init):.2f} MiB "
          f"(expected {_mib(expected_init):.2f})")
    assert _within(consumed_init, expected_init), \
        f"init: {_mib(consumed_init):.2f} vs {_mib(expected_init):.2f} MiB"

    # (b) Incremental grow — the production path used by
    # ChunkArena.__init__ shortfall check (chunk_arena.py:294-305).
    # Snapshot existing handle ids first; after grow, assert the
    # ORIGINAL handles are still there (no release+recreate identity bug).
    handles_before_grow = list(pool.handles)
    pool.grow(N_GROW)
    free_after_grow = _free_bytes()
    consumed_grow = free_after_init - free_after_grow
    print(f"    pool.grow({N_GROW}) incremental: "
          f"consumed {_mib(consumed_grow):.2f} MiB "
          f"(expected {_mib(expected_grow):.2f})")
    assert _within(consumed_grow, expected_grow), \
        f"grow: {_mib(consumed_grow):.2f} vs {_mib(expected_grow):.2f} MiB"
    # Identity: original handles unchanged; new handles appended
    assert pool.total_count() == N_INIT + N_GROW
    assert pool.handles[:N_INIT] == handles_before_grow, \
        "pool.grow released existing handles instead of just appending — " \
        "HBM net is correct but handle IDs would have changed silently"

    # (c) Cleanup must return all HBM. Without this assertion,
    # the "swap reduces HBM" claim has no test backing.
    pool.cleanup()
    free_after_cleanup = _free_bytes()
    leaked = free_before - free_after_cleanup
    print(f"    after cleanup: leaked {_mib(leaked):.3f} MiB (expected ~0)")
    assert _within(leaked, 0), \
        f"cleanup leaked {_mib(leaked):.2f} MiB out of " \
        f"{_mib(expected_total):.2f} MiB allocated"


def test_2_address_reserve_sweep_is_free():
    """cuMemAddressReserve is HBM-free across many sizes including
    near-total-free (= the actual production use case)."""
    total_free = _free_bytes()
    # IMPORTANT: cuMemAddressReserve requires size aligned to chunk_size
    # (granularity). Earlier version used `int(total_free * 0.5)`
    # unaligned, got CUDA_ERROR_INVALID_VALUE, and was mis-diagnosed
    # as a max-size issue.

    def align_down(x):
        return (x // CHUNK_SIZE) * CHUNK_SIZE

    # Sweep: tiny → small → moderate → large → near-max free.
    # All sizes aligned to CHUNK_SIZE.
    sizes = [
        CHUNK_SIZE,                              # 2 MiB
        16 * 1024 * 1024,                        # 16 MiB
        1024 * 1024 * 1024,                      # 1 GiB
        16 * 1024 * 1024 * 1024,                 # 16 GiB
        align_down(total_free // 2),             # half of free (aligned)
    ]
    for size in sizes:
        free_before = _free_bytes()
        ptr = _DPTR(0)
        _check(CUDA.cuMemAddressReserve(ctypes.byref(ptr), size, 0, 0, 0),
               f"cuMemAddressReserve({size})")
        free_after = _free_bytes()
        consumed = free_before - free_after
        print(f"    cuMemAddressReserve({_mib(size):>9.0f} MiB): "
              f"consumed {_mib(consumed):.3f} MiB")
        try:
            assert _within(consumed, 0), \
                f"reserve {_mib(size):.0f} MiB consumed " \
                f"{_mib(consumed):.2f} MiB (should be ~0)"
        finally:
            _check(CUDA.cuMemAddressFree(ptr.value, size),
                   "cuMemAddressFree")


def test_3_arenas_share_pool_hbm_not_doubled():
    """Two ChunkArenas sharing one SharedHandlePool — production
    headline path (KV arena + mamba arena, one shared pool). Combined
    HBM must equal pool's handle count, NOT double."""
    N = 32                              # 64 MiB total handles
    expected = N * CHUNK_SIZE

    free_before = _free_bytes()
    pool = SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=N)
    # Two arenas, each oversized (32× the handle count)
    arena_KV = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N // 2,
                          pool_capacities=[("kv", 32 * N)],
                          external_handle_pool=pool)
    arena_M  = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N // 2,
                          pool_capacities=[("mamba", 32 * N)],
                          external_handle_pool=pool)
    free_after = _free_bytes()
    consumed = free_before - free_after
    total_va = arena_KV.total_va_size + arena_M.total_va_size
    print(f"    2 arenas share {N} handles ({_mib(N*CHUNK_SIZE):.0f} MiB), "
          f"combined VA {_mib(total_va):.0f} MiB ({32*2}× handles)")
    print(f"    HBM consumed: {_mib(consumed):.2f} MiB "
          f"(expected ~{_mib(expected):.2f}; not 2× that)")
    try:
        assert _within(consumed, expected), \
            f"two arenas with shared pool consumed {_mib(consumed):.2f} MiB, " \
            f"expected ~{_mib(expected):.2f} MiB. If 2×, arenas are " \
            f"double-counting handles."
        # Aliasing IDENTITY: arenas' _handles must be the SAME list object
        # as pool.handles (chunk_arena.py:307-308). HBM aggregate alone
        # doesn't catch a bug where arenas alias an EMPTY list — they'd
        # break functionally but show correct HBM.
        assert pool.total_count() == N
        assert arena_KV._handles is pool.handles, \
            "arena_KV._handles not aliased to pool.handles"
        assert arena_M._handles is pool.handles, \
            "arena_M._handles not aliased to pool.handles"
        assert arena_KV._free_handles is pool.free, \
            "arena_KV._free_handles not aliased to pool.free"
        assert arena_M._free_handles is pool.free, \
            "arena_M._free_handles not aliased to pool.free"
    finally:
        arena_M.cleanup()
        arena_KV.cleanup()
        pool.cleanup()


def test_4_map_unmap_loop_no_leak():
    """Loop map/unmap 100× to amplify per-call leak signal.

    Old test had loop = 1, so any per-call leak < tol/calls (= 4 MiB / 4
    = 1 MiB/call) would silently pass. With 100 iters and tol = 4σ
    (typically << 1 MiB), the detection floor drops to ~tol/100 ≈ 10 KiB/call.
    """
    N = 8
    LOOPS = 100
    pool = SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=N)
    try:
        arena = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N,
                           pool_capacities=[("A", 8), ("B", 8)],
                           external_handle_pool=pool)
        try:
            # Warm: do one map+unmap so any first-call setup is amortized
            arena.grow("A", 4)
            arena.shrink("A", 4)
            torch.cuda.synchronize()

            free_before_loop = _free_bytes()
            for _ in range(LOOPS):
                arena.grow("A", 4)
                arena.shrink("A", 4)
            torch.cuda.synchronize()
            free_after_loop = _free_bytes()

            drift = free_before_loop - free_after_loop
            per_call = drift / (2 * LOOPS)   # 2 ops per loop
            print(f"    {LOOPS} map+unmap cycles ({2*LOOPS} ops): "
                  f"drift = {_mib(drift):.3f} MiB; "
                  f"per-op floor = {_mib(per_call):.1f} KiB")
            # Pass: total drift within calibrated noise floor
            assert _within(drift, 0), (
                f"after {2*LOOPS} ops, HBM drifted by {_mib(drift):.2f} MiB "
                f"(per-call ≈ {_mib(per_call):.2f} MiB). Likely leak in "
                f"cuMemMap or cuMemUnmap.")
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_6_lazy_init_via_arena_shortfall():
    """Pool constructed with n_handles=0, arena drives the grow via
    its shortfall check (chunk_arena.py:294-305). This is the
    DOCUMENTED lazy-init path the design depends on; without this
    test the path is unverified."""
    N_REQUEST = 32
    expected = N_REQUEST * CHUNK_SIZE

    # Pool starts empty
    free_before = _free_bytes()
    pool = SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=0)
    free_after_empty = _free_bytes()
    assert _within(free_before - free_after_empty, 0), \
        "empty pool construction should consume ~0 HBM"
    assert pool.total_count() == 0

    # Arena requests N handles → triggers pool.grow(shortfall) inline
    arena = ChunkArena(DEVICE, CHUNK_SIZE,
                       n_handles=N_REQUEST,
                       pool_capacities=[("A", 16), ("B", 16)],
                       external_handle_pool=pool)
    free_after_arena = _free_bytes()
    consumed = free_after_empty - free_after_arena
    print(f"    empty pool → arena requests {N_REQUEST}: "
          f"pool grew to {pool.total_count()} handles, "
          f"consumed {_mib(consumed):.2f} MiB")
    try:
        assert pool.total_count() == N_REQUEST, \
            f"arena init should have lazy-grown pool to {N_REQUEST}, " \
            f"got {pool.total_count()}"
        assert _within(consumed, expected)
    finally:
        arena.cleanup()
        pool.cleanup()


def test_5_owned_vs_external_paired():
    """Owned-handle path and external-pool path consume the SAME HBM
    for the same N — paired comparison (the original test_5 just
    asserted owned ≈ N×chunk, redundant with test_1)."""
    N = 32
    expected = N * CHUNK_SIZE

    # External-pool path
    free_before_ext = _free_bytes()
    pool = SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=N)
    arena_ext = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N,
                            pool_capacities=[("A", 16), ("B", 16)],
                            external_handle_pool=pool)
    free_after_ext = _free_bytes()
    consumed_ext = free_before_ext - free_after_ext
    arena_ext.cleanup()
    pool.cleanup()

    torch.cuda.synchronize()

    # Owned-handle path
    free_before_own = _free_bytes()
    arena_own = ChunkArena(DEVICE, CHUNK_SIZE, n_handles=N,
                            pool_capacities=[("A", 16), ("B", 16)],
                            external_handle_pool=None)
    free_after_own = _free_bytes()
    consumed_own = free_before_own - free_after_own
    arena_own.cleanup()

    print(f"    external-pool path:  consumed {_mib(consumed_ext):.2f} MiB")
    print(f"    owned-handle path:   consumed {_mib(consumed_own):.2f} MiB")
    print(f"    expected (both):     {_mib(expected):.2f} MiB")

    # Both paths individually
    assert _within(consumed_ext, expected), \
        f"external: {_mib(consumed_ext):.2f} vs {_mib(expected):.2f} MiB"
    assert _within(consumed_own, expected), \
        f"owned: {_mib(consumed_own):.2f} vs {_mib(expected):.2f} MiB"
    # AND paired equality
    delta = abs(consumed_ext - consumed_own)
    print(f"    paired delta:        {_mib(delta):.3f} MiB")
    assert delta <= TOL_BYTES, \
        f"owned and external differ by {_mib(delta):.2f} MiB"


# ---------- runner ----------

def main():
    _warm_cuda()
    tests = [
        ("0 noise calibration (sets TOL = 4σ)",
         test_0_calibrate_noise_floor),
        ("1 handle lifecycle (construct + grow + cleanup return)",
         test_1_handle_lifecycle_hbm_accounting),
        ("2 cuMemAddressReserve sweep is HBM-free",
         test_2_address_reserve_sweep_is_free),
        ("3 two arenas share one pool (no double-counting)",
         test_3_arenas_share_pool_hbm_not_doubled),
        ("4 map/unmap loop ×100 no leak",
         test_4_map_unmap_loop_no_leak),
        ("5 owned vs external paths: paired equal HBM",
         test_5_owned_vs_external_paired),
        ("6 lazy init: empty pool + arena shortfall triggers grow",
         test_6_lazy_init_via_arena_shortfall),
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
    print(f"\nva_reservation_hbm: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
