"""m2k fire SRC-surplus leak: the XPoolActuator success path must conserve
the mamba pool's `_capped_slots` / `live_size`.

Builds a REAL m2k fire over a real `MambaPool` (src) + real `MHATokenToKVPool`
+ `TokenToKVPoolAllocator` (dst) sharing one `SharedHandlePool`, wired through
the full `XPoolActuator`. No stubs: `cap_barrier` marks real
`MambaPool._capped_slots`, `_execute_async_locked` calls real
`MultiTensorArena.shrink_explicit` / `grow`.

The bug: on an m2k fire the LCM-floor transfers only `per_src` of the
`len(pages_to_unmap)` capped pages. `cap_barrier` marked the FULL set into
`_capped_slots` (dropping them from `free_slots`), but the success path's
only src-side touch is `shrink_explicit` of the first `per_src` pages, which
is chunk-level only (it never touches `_capped_slots` / `free_slots`). So the
un-transferred surplus pages stay capped forever: their chunks are STILL
mapped (allocatable VA) yet their slots are stranded out of `free_slots`.
Repeated misaligned m2k fires erode `live_size` until the working set starves.

INVARIANT (what GREEN must satisfy):
  After an m2k fire that transfers `per_src` pages, the only residual drop in
  `live_size` attributable to the fire equals exactly the transferred slot
  count (`per_src * slots_per_page`). Every surplus capped slot
  (`pages_to_unmap[per_src:]`, whose chunks are still mapped) is restored to
  `free_slots`. The transferred slots stay capped (their chunks were
  unmapped).
"""
import math
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

# Arena flags MUST be set before importing the pools so the MultiTensorArena
# branch + shared handle pool are built.
os.environ["SGLANG_ARENA_SHARED"] = "1"
os.environ.setdefault("SGLANG_ARENA_CHUNK_BYTES", str(2 * 1024 * 1024))

import torch  # noqa: E402

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def _build_real_m2k_actuator(*, n_mamba_layers, n_kv_layers, mamba_init_slots):
    """Real MambaPool (src) + MHATokenToKVPool/allocator (dst) on one shared
    pool, wired through the full XPoolActuator. Returns
    (actuator, mamba_pool, kv_allocator)."""
    from sglang.srt.arena.shared_pool import reset_shared_handle_pool_for_test
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    reset_shared_handle_pool_for_test()

    # --- SRC: real MambaPool over the shared arena. n_kinds=1 → n_src =
    # n_mamba_layers sub-pools. headroom small so the test stays cheap. ---
    os.environ["SGLANG_ARENA_MAMBA_HEADROOM_BYTES"] = str(256 * 1024 * 1024)
    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=128, n_groups=1,
        num_heads=4, head_dim=64, state_size=16, conv_kernel=4,
    )
    layer_ids = list(range(n_mamba_layers))
    cache_params = Mamba2CacheParams(shape=shape, layers=layer_ids)
    mamba_pool = MambaPool(
        size=mamba_init_slots,
        spec_state_size=mamba_init_slots,
        cache_params=cache_params,
        mamba_layer_ids=layer_ids,
        device=DEVICE,
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=mamba_init_slots,
    )
    mamba_arena = mamba_pool._mamba_temporal_arena
    assert mamba_arena is not None, "mamba arena not built (SGLANG_ARENA_SHARED?)"
    mamba_pool._allocator = MambaSlotAllocator(mamba_init_slots, DEVICE)

    # --- DST: real MHATokenToKVPool over the SAME shared pool. n_kinds=2
    # (k, v) → n_dst = 2 * n_kv_layers sub-pools. ---
    kv_pool = MHATokenToKVPool(
        size=2048,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=4,
        head_dim=64,
        layer_num=n_kv_layers,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=n_kv_layers,
        enable_alt_stream=False,
    )
    kv_arena = kv_pool._kv_arena
    assert kv_arena is not None, "kv arena not built (SGLANG_ARENA_SHARED?)"

    kv_boot_tokens = kv_arena.current_capacity_tokens()
    kv_alloc = TokenToKVPoolAllocator(
        size=kv_boot_tokens,
        dtype=torch.bfloat16,
        device=DEVICE,
        kvcache=kv_pool,
        need_sort=True,
        max_size=kv_arena.max_tokens,
    )

    shared_pool = mamba_arena._arena._external_pool
    assert kv_arena._arena._external_pool is shared_pool, (
        "kv/mamba arenas landed on different shared pools"
    )

    kv_act = KVArenaActuator(pool=kv_pool, allocator=kv_alloc)
    mamba_act = MambaArenaActuator(pool=mamba_pool)
    actuator = XPoolActuator(
        kv_arena=kv_arena, mamba_arena=mamba_arena, shared_pool=shared_pool,
        kv_actuator=kv_act, mamba_actuator=mamba_act,
    )
    return actuator, mamba_pool, kv_alloc


def _n_mamba_pages(mamba_pool) -> int:
    """Highest-addressable mamba page id + 1 = boot-mapped chunks per
    sub-pool. Page p backs slots [p*tps, (p+1)*tps); the top page is
    `init_chunks_per_pool - 1`."""
    return int(mamba_pool._mamba_temporal_arena.init_chunks_per_pool)


def _capped_count_le_size(pool) -> int:
    """Capped slots within the live range (the quantity that subtracts from
    live_size)."""
    capped = pool._capped_slots
    if capped.numel() == 0:
        return 0
    return int((capped <= pool.size).sum().item())


def _free_mamba_tail_pages(actuator, mamba_pool, page_ids):
    """The actuator requires every page in `pages_to_unmap` to be FREE
    (cap_barrier's `count_referenced` guard). Boot leaves all live slots in
    `free_slots`, so the chosen pages are free already; this just asserts it
    and returns the slot ids the pages cover."""
    mamba_act = actuator.mamba_actuator
    slots = mamba_act.expand_pages_to_token_slots(page_ids)
    free_set = set(mamba_pool.free_slots.tolist())
    for s in slots:
        assert s in free_set, f"slot {s} (page set {page_ids}) not free at boot"
    return slots


def _run_one_m2k_fire(actuator, mamba_pool, pages_to_unmap, plan_seq):
    from sglang.srt.arena.fire_plan import FirePlan
    _free_mamba_tail_pages(actuator, mamba_pool, pages_to_unmap)
    plan = FirePlan(
        direction="mamba_to_kv",
        pages_to_unmap=list(pages_to_unmap),
        pages_to_map_dst=len(pages_to_unmap),
        plan_seq=plan_seq,
    )
    result = actuator.execute(plan)
    assert not result.aborted, f"fire aborted: {result.abort_reason}"
    return result


def test_m2k_misaligned_fire_conserves_capped_slots():
    """Single misaligned m2k fire: live_size must drop by EXACTLY the
    transferred slot count, and no surplus slot may linger capped."""
    # n_src=24 (mamba layers, n_kinds=1), n_dst=16 (8 kv layers * 2 kinds).
    # lcm(24,16)=48. Pick the highest mamba init that fits in the small VA
    # headroom; L=5 pages → per_src=2, surplus=3 pages (misaligned).
    actuator, mamba_pool, _ = _build_real_m2k_actuator(
        n_mamba_layers=24, n_kv_layers=8, mamba_init_slots=4095,
    )
    n_src = actuator.n_mamba_subpools
    n_dst = actuator.n_kv_subpools
    lcm = actuator.lcm_pages
    assert (n_src, n_dst, lcm) == (24, 16, 48), (
        f"unexpected subpool geometry: n_src={n_src} n_dst={n_dst} lcm={lcm}"
    )
    slots_per_page = actuator.mamba_actuator._tokens_per_page()
    L = 5
    target_total = (min(n_src, n_dst) * L // lcm) * lcm
    per_src = target_total // n_src
    surplus_pages = L - per_src
    assert per_src < L and surplus_pages > 0, (
        f"chose an aligned L: per_src={per_src} L={L}"
    )

    # Use tail pages (highest mapped page ids) — matches the planner's
    # tail-evict order and keeps page 0 (padded slot) out.
    n_pages_total = _n_mamba_pages(mamba_pool)
    pages_to_unmap = list(range(n_pages_total - L, n_pages_total))
    assert 0 not in pages_to_unmap

    live_before = mamba_pool.live_size
    capped_before = _capped_count_le_size(mamba_pool)
    size_before = mamba_pool.size
    free_before = int(mamba_pool.free_slots.numel())

    _run_one_m2k_fire(actuator, mamba_pool, pages_to_unmap, plan_seq=1)

    live_after = mamba_pool.live_size
    capped_after = _capped_count_le_size(mamba_pool)
    size_after = mamba_pool.size
    free_after = int(mamba_pool.free_slots.numel())

    transferred_slots = per_src * slots_per_page
    surplus_slots = surplus_pages * slots_per_page
    live_drop = live_before - live_after

    print(
        f"\n  m2k misaligned fire: n_src={n_src} n_dst={n_dst} lcm={lcm} "
        f"slots/page={slots_per_page}\n"
        f"  L={L} pages → per_src={per_src} (transferred), "
        f"surplus_pages={surplus_pages}\n"
        f"  transferred_slots={transferred_slots} "
        f"surplus_slots={surplus_slots}\n"
        f"  live_size  {live_before} -> {live_after}  (drop={live_drop}, "
        f"want={transferred_slots})\n"
        f"  capped<=sz {capped_before} -> {capped_after}  "
        f"(want={capped_before + transferred_slots})\n"
        f"  size       {size_before} -> {size_after}\n"
        f"  free_slots {free_before} -> {free_after}"
    )

    assert size_after == size_before, (
        "m2k must not change mamba self.size (only the cap-barrier marks)"
    )
    # THE invariant: the fire's net live_size drop == only the transferred
    # slots. A larger drop means the surplus pages (still mapped, allocatable)
    # were stranded in _capped_slots.
    assert live_drop == transferred_slots, (
        f"LEAK: live_size dropped by {live_drop}, expected only "
        f"{transferred_slots} (the transferred slots). The extra "
        f"{live_drop - transferred_slots} slots are the un-transferred "
        f"surplus ({surplus_pages} pages * {slots_per_page} slots/page = "
        f"{surplus_slots}) stranded in _capped_slots — their chunks are "
        f"STILL mapped yet they can never be allocated again."
    )
    assert capped_after == capped_before + transferred_slots, (
        f"_capped_slots within live range = {capped_after}, expected "
        f"{capped_before + transferred_slots} (only the transferred slots)."
    )
    print("  PASS  single misaligned m2k fire conserves _capped_slots/live_size")


def test_repeated_misaligned_fires_no_cumulative_leak():
    """Repeated misaligned m2k fires on disjoint tail pages: cumulative
    live_size drop == sum of transferred slots only. RED: each fire strands
    its surplus, so the drop grows by L*slots/page per fire, not per_src."""
    actuator, mamba_pool, _ = _build_real_m2k_actuator(
        n_mamba_layers=24, n_kv_layers=8, mamba_init_slots=4095,
    )
    n_src = actuator.n_mamba_subpools
    n_dst = actuator.n_kv_subpools
    lcm = actuator.lcm_pages
    slots_per_page = actuator.mamba_actuator._tokens_per_page()
    L = 5
    target_total = (min(n_src, n_dst) * L // lcm) * lcm
    per_src = target_total // n_src
    transferred_per_fire = per_src * slots_per_page

    n_pages_total = _n_mamba_pages(mamba_pool)
    n_fires = 3
    live_before = mamba_pool.live_size

    # Disjoint tail page windows, one per fire (top of the pool downward),
    # avoiding page 0.
    top = n_pages_total
    for k in range(n_fires):
        hi = top - k * L
        lo = hi - L
        assert lo > 0, "ran out of non-zero tail pages for the test geometry"
        pages = list(range(lo, hi))
        _run_one_m2k_fire(actuator, mamba_pool, pages, plan_seq=10 + k)

    live_after = mamba_pool.live_size
    live_drop = live_before - live_after
    want = n_fires * transferred_per_fire
    print(
        f"\n  {n_fires} misaligned m2k fires (L={L}, per_src={per_src}): "
        f"live_size {live_before} -> {live_after} "
        f"(drop={live_drop}, want={want})"
    )
    assert live_drop == want, (
        f"CUMULATIVE LEAK: after {n_fires} fires live_size dropped {live_drop}, "
        f"expected {want} (only transferred slots). Each fire stranded its "
        f"surplus, eroding the working set."
    )
    print("  PASS  repeated misaligned m2k fires leave no cumulative leak")


def main():
    tests = [
        test_m2k_misaligned_fire_conserves_capped_slots,
        test_repeated_misaligned_fires_no_cumulative_leak,
    ]
    print(f"\nm2k SRC-surplus leak tests (n={len(tests)}, real pools):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nm2k SRC-surplus: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
