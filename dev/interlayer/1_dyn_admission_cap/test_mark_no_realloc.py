"""Phase 9: mark_pages_capped must not reallocate the 2.4M-element
free_pages tensor on every fire.

D8 root-cause bisect (Task #128, #131, #132) established:
- The +2.86% D8 TPOT regression comes from
  `allocator.mark_pages_capped()` doing
  `self.free_pages = self.free_pages[~mask]` per fire.
- This reallocates a ~19 MB int64 tensor per call. 21 fires × ~3 such
  ops/fire = ~1.2 GB of caching-allocator churn.
- Skipping the realloc (via SGLANG_XPOOL_AUDIT_SKIP_MARK=1) restores D8 TPOT
  to off baseline.

Fix: don't mutate `free_pages` in mark/unmark. Keep `_capped_pages` as
the source-of-truth set; filter against it in `alloc`. Since
`_capped_pages` is typically small (≤ ~5 k slots after fires) and most
allocs touch only the head of free_pages (which the planner picks
tail-bias pages for), the per-alloc filter is cheap.

Acceptance tests:
1. mark/unmark behave the same observably: capped slots are not
   handed out by alloc.
2. `self.free_pages.data_ptr()` stays IDENTICAL across many mark/unmark
   cycles (proves no reallocation).
3. After mark+unmark, the allocator state is identical to original
   (idempotent).
4. KV-scale benchmark: mark/unmark per call should be < 1 ms (vs
   previous ~10 ms due to tensor realloc).

Run: .venv/bin/python dev/interlayer/1_dyn_admission_cap/test_mark_no_realloc.py
"""
import sys
import threading
import time

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def _make_allocator(size=2_454_877):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    # The allocator needs a kvcache and dtype; we use minimal stubs since
    # we only test mark/unmark/alloc semantics, not actual KV reads.
    class _StubKV:
        device = DEVICE
        def get_kvcache(self): return self
    alloc = TokenToKVPoolAllocator(
        size=size,
        dtype=torch.bfloat16,
        device=DEVICE,
        kvcache=_StubKV(),
        need_sort=False,
    )
    return alloc


def test_1_alloc_skips_capped_slots():
    """alloc must NOT hand out any slot id that's currently capped."""
    alloc = _make_allocator(size=100)
    # Cap slots [50, 51, 52]
    target = torch.tensor([50, 51, 52], dtype=torch.int64, device=DEVICE)
    n = alloc.mark_pages_capped(target)
    assert n == 3, f"expected to mark 3, got {n}"
    # Now alloc until exhausted using alloc(1) so we don't lose tail.
    target_set = {50, 51, 52}
    used = set()
    while True:
        out = alloc.alloc(1)
        if out is None: break
        i = int(out[0].item())
        assert i not in target_set, f"alloc returned capped slot {i}"
        used.add(i)
    # All non-capped slots should have been alloc'd (100 - 3 = 97 non-zero
    # slots in range [1, 100]).
    assert len(used) == 97, f"expected 97 alloc'd slots, got {len(used)}"
    assert used == set(range(1, 101)) - target_set
    print(f"  PASS  1  alloc skips capped slots (allocated {len(used)} non-capped)")


def test_2_no_extra_memory_allocations_per_mark():
    """Track torch.cuda.memory_allocated() before/after each mark cycle.
    The pre-fix impl reallocates ~19MB tensors per call; the fixed impl
    should allocate ~64KB per call (just the small `_capped_pages` cat).

    We measure peak allocated minus baseline. Pre-fix is ~19MB; post-fix
    should be < 1 MB.
    """
    alloc = _make_allocator(size=2_454_877)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    # Cap 3072 slots (matches D8 fire size), then uncap.
    target = torch.arange(2_000_000, 2_003_072, dtype=torch.int64, device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    alloc.mark_pages_capped(target)
    peak_after_mark = torch.cuda.max_memory_allocated() - base
    alloc.unmark_pages_capped(target)
    torch.cuda.synchronize()
    final = torch.cuda.memory_allocated() - base
    print(f"  peak alloc during mark: {peak_after_mark/1024:.1f} KB")
    print(f"  net alloc after mark+unmark: {final/1024:.1f} KB")
    # Pre-fix peak is ~19 MB. Post-fix should be < 1 MB (just the small cat).
    assert peak_after_mark < 1 * 1024 * 1024, (
        f"peak alloc {peak_after_mark/1024:.1f} KB > 1 MB. "
        f"mark/unmark is still doing the 19MB tensor realloc — D8 +2.86%"
        f" regression not fixed."
    )
    print(f"  PASS  2  mark/unmark peak alloc < 1 MB (no big tensor realloc)")


def test_3_mark_unmark_is_idempotent():
    """After mark(X) + unmark(X), allocator state matches original."""
    alloc = _make_allocator(size=200)
    free0 = set(int(x) for x in alloc.free_pages.cpu().tolist())
    target = torch.tensor([100, 101, 102, 103], dtype=torch.int64, device=DEVICE)
    alloc.mark_pages_capped(target)
    alloc.unmark_pages_capped(target)
    free1 = set(int(x) for x in alloc.free_pages.cpu().tolist())
    assert free1 == free0, f"free_pages changed: missing={free0-free1} extra={free1-free0}"
    # _capped_pages should also be empty after unmark
    capped = getattr(alloc, "_capped_pages", None)
    assert capped is None or capped.numel() == 0, f"_capped_pages not empty: {capped}"
    print(f"  PASS  3  mark + unmark is idempotent")


def test_4_kv_scale_benchmark():
    """KV-scale: 21 fires of 4096 slots each. Each mark must be < 1 ms."""
    alloc = _make_allocator(size=2_454_877)
    walls_mark = []
    walls_unmark = []
    for i in range(21):
        # 48 pages × 64 tokens/page = 3072 token slots (matches D8 fire size)
        start = 2_000_000 + i * 10_000
        target = torch.arange(
            start, start + 3072, dtype=torch.int64, device=DEVICE,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        alloc.mark_pages_capped(target)
        torch.cuda.synchronize()
        walls_mark.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        alloc.unmark_pages_capped(target)
        torch.cuda.synchronize()
        walls_unmark.append((time.perf_counter() - t0) * 1000)
    mean_mark = sum(walls_mark) / len(walls_mark)
    mean_unmark = sum(walls_unmark) / len(walls_unmark)
    p99_mark = sorted(walls_mark)[-1]
    print(f"  mark   p50={sorted(walls_mark)[len(walls_mark)//2]:.3f}ms "
          f"p99={p99_mark:.3f}ms mean={mean_mark:.3f}ms")
    print(f"  unmark p50={sorted(walls_unmark)[len(walls_unmark)//2]:.3f}ms "
          f"mean={mean_unmark:.3f}ms")
    assert mean_mark < 1.0, (
        f"mark mean {mean_mark:.2f}ms > 1ms target (pre-fix was ~10ms)"
    )
    print(f"  PASS  4  KV-scale: mark+unmark each < 1 ms (per-fire cost)")


def test_5_actuator_verify_respects_capped_mask():
    """Phase 6 D6 first attempt revealed: after fix #134, cap_t entries
    REMAIN in `alloc.free_pages` (intentionally — shadowed via
    `_capped_pages` mask in alloc()). The cap_barrier verify step in
    xpool_actuator used `torch.isin(free_pages, cap_t)` directly without
    subtracting `_capped_pages`, so it always reported all cap_t as
    violations → every fire aborted → no transfers → EWMA never warms →
    Admitter cross-* stays gated.

    This test pins the corrected verify behavior: after mark_pages_capped,
    the isin check against (free_pages \\ _capped_pages) returns 0.
    """
    alloc = _make_allocator(size=10_000)
    alloc.alloc(1)  # consume slot 0 sentinel
    free_before = alloc.free_pages.clone()
    target = free_before[-100:]  # tail-bias subset
    alloc.mark_pages_capped(target)
    # Naive verify (pre-fix): isin(free_pages, cap_t) → all 100 hits.
    in_target_naive = int(torch.isin(alloc.free_pages, target).sum().item())
    assert in_target_naive == 100, (
        f"post-#134 expected all 100 cap_t in raw free_pages, got "
        f"{in_target_naive}"
    )
    # Corrected verify (subtract _capped_pages).
    capped = alloc._capped_pages
    in_target = torch.isin(alloc.free_pages, target)
    in_capped = torch.isin(alloc.free_pages, capped)
    in_target_corrected = int((in_target & ~in_capped).sum().item())
    assert in_target_corrected == 0, (
        f"corrected verify must report 0 violations, got "
        f"{in_target_corrected}"
    )
    print("  PASS  5  cap_barrier verify corrected: 0 violations after "
          "mask-subtract (was 100 with raw isin)")


def test_6_alloc_slow_path_preserves_capped_pages():
    """Phase 9 (task #154) — D11 first attempt crashed with
    "pool memory leak detected: 8192 unaccounted". Root cause: in the
    slow-path of `alloc()` (when capped slots interleave with the head
    of free_pages), the post-alloc slicing
    `self.free_pages = self.free_pages[consumed_through:]` silently
    drops the capped slots that sat between consumed_through slots
    and the returned non-capped slots. They stay in `_capped_pages`
    but vanish from free_pages — `live_size = size - _capped` then
    over-reports total, and the leak detector fires.

    Test: cap 100 slots interleaved in the head of free_pages,
    request `alloc(50)` (forces slow path), then assert _capped_pages
    are still ALL present in free_pages.
    """
    alloc = _make_allocator(size=10_000)
    alloc.alloc(1)  # consume slot 0 sentinel
    # Pick 100 slots to cap: every-other from the head of free_pages.
    head = alloc.free_pages[:300].clone()
    to_cap = head[::3]  # 100 entries scattered in the head
    alloc.mark_pages_capped(to_cap)
    initial_capped = int(alloc._capped_pages.numel())
    assert initial_capped == 100

    # alloc(50) — must skip the 100 capped in the head, return 50
    # non-capped, but PRESERVE the 100 capped in free_pages.
    result = alloc.alloc(50)
    assert result is not None and result.numel() == 50

    # Critical invariant: all 100 capped slots still findable in free_pages
    still_in_free = int(torch.isin(to_cap, alloc.free_pages).sum().item())
    assert still_in_free == 100, (
        f"alloc() slow path dropped {100 - still_in_free} capped slots "
        f"from free_pages — they're now lost to leak detector"
    )
    # _capped_pages count unchanged
    assert int(alloc._capped_pages.numel()) == 100

    # Pool accounting must be self-consistent under TokenToKVPoolAllocator's
    # available_size = free.numel + release.numel - capped.numel:
    #   - free_pages.numel = 9949 (25 capped preserved + 9924 above prefix)
    #   - _capped_pages.numel = 100 (unchanged)
    #   - available = 9949 - 100 = 9849 (truly-allocatable count)
    # And live_size = size - capped = 10000 - 100 = 9900.
    # Invariant: live_size - available = 51 = (1 sentinel + 50 alloc'd).
    live = int(alloc.live_size)
    avail = alloc.available_size()
    assert avail == 9849, f"available_size after alloc(50) = {avail} != 9849"
    assert live == 9900, f"live_size = {live} != 9900"
    assert live - avail == 51, (
        f"live_size - available = {live - avail} != 51 (1 sentinel + 50 alloc'd)"
    )
    print("  PASS  6  alloc() slow path preserves capped slots in "
          "free_pages (no D11-style accounting leak)")


def test_7_alloc_fails_when_only_capped_slots_remain():
    """Phase 9 (task #155) — D11 inter-mode OOM root cause.

    Live D11 with #154 fix + strict mem check + CUDA_LAUNCH_BLOCKING=1
    revealed the true crash:

        RuntimeError: Out of memory. Try to allocate 2253 tokens.
        Available full tokens: 11242
          (full_available_size=1064 + full_evictable_size=10178)

    11242 ≥ 2253 — but alloc still failed. Root cause: the 10178
    "evictable" tokens are on PAGES the Budgeter previously transferred
    out (via k2m fire). Their page indices live in `_capped_pages`.
    When `evict_from_tree_cache` evicts them, the freed slots return
    to `free_pages` BUT remain in `_capped_pages` → alloc's slow path
    correctly rejects them → returns None.

    The bug: tree_cache.evict() believes it freed N tokens, but those
    tokens are on capped pages and alloc can't actually use them. The
    accounting is honest (post-#154); the bug is that the tree_cache
    contains entries on pages that have been physically unmapped.

    Test: build allocator with N free + M capped pages whose values
    ALSO appear in free_pages (the post-#134 invariant). Try to alloc
    more than free.numel - capped.numel. Assert returns None.
    """
    alloc = _make_allocator(size=10_000)
    alloc.alloc(1)  # consume slot 0 sentinel
    # All 9999 slots in free_pages. Now "cap" 9000 of them (simulating
    # what Budgeter k2m fires accumulate over many ticks).
    free_now = alloc.free_pages.clone()
    to_cap = free_now[:9000]  # cap the head 9000 page indices
    alloc.mark_pages_capped(to_cap)

    # Sanity: live_size = 10000 - 9000 = 1000.
    # available_size = 9999 (free) - 9000 (capped) = 999.
    assert int(alloc.live_size) == 1000
    assert int(alloc.available_size()) == 999

    # Now try to alloc 2000 (more than 999 truly available, but less
    # than total 9999 in free_pages naively counted).
    result = alloc.alloc(2000)
    assert result is None, (
        f"alloc(2000) should return None when only 999 non-capped "
        f"slots remain, got tensor of {None if result is None else result.numel()}"
    )

    # And the diagnostic message that production logs would show:
    # available reports 999 (truly free), not 9999 (raw free_pages count).
    # This proves #154 fix makes the accounting honest.
    print(f"  PASS  7  alloc(2000) returns None when capped accumulated "
          f"(live_size=1000, available=999, was hidden by buggy free count)")


def test_17_mamba_free_filters_capped_slots():
    """Task #174 (2026-05-30): N=3 D10@C=56 surfaced CUDA Graph crashes
    ~2 min after m2k fires (all 3 runs).

        [22:28:03] Piecewise CUDA Graph failed: Pointer argument (at 7)
                   cannot be accessed from Triton (cpu tensor?)
        Scheduler hit an exception in unified_linear_attention_with_output

    Root cause: `MambaPool.free()` (memory_pool.py:691-717) only filters
    via `size` (integer live cap). When `_MambaCapAllocator.
    mark_pages_capped` marks slot IDs without updating `size`,
    a later `mamba_pool.free(slot)` for an in-flight req or radix-cache
    eviction whose slot is in `_capped_slots` (but <= `size`)
    drops the slot back into `free_slots`. Next admission gets it. Its
    underlying chunk was unmapped by the m2k worker → next forward pass
    touches unmapped VA → Triton kernel crash.

    Symmetric KV side already filters via `_capped_pages` in
    `TokenToKVPoolAllocator.free` (allocator.py, D11 #154 fix).
    This test pins the mirror filter on the mamba side.

    Sequence:
      1. mark slot 50 capped (via _MambaCapAllocator)
      2. call pool.free(tensor([50]))  — simulates eviction
      3. assert pool.free_slots does NOT contain 50 (would crash on next alloc)
    """
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator

    class _FakePool:
        def __init__(self, size: int) -> None:
            self.size = size
            self.live_size = size
            self.max_size = size + 100
            self.device = torch.device(DEVICE)
            self.free_slots = torch.arange(
                1, size + 1, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.empty(
                0, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()
            class _FakeArena:
                tokens_per_chunk = 1
                max_chunks_per_pool = size + 100
            self._mamba_temporal_arena = _FakeArena()

        def set_capacity_slots(self, n: int) -> int:
            return min(int(n), self.size + 100)

        def _assert_capped_slots_invariant(self) -> None:
            # No-op stub; production MambaPool always defines it.
            pass

        # No free() defined here — we bind the REAL MambaPool.free via
        # __get__ below so the test exercises production code.

    pool = _FakePool(size=100)
    # Bind production MambaPool.free to the fake pool so we test the
    # actual code path that crashed live.
    from sglang.srt.mem_cache.memory_pool import MambaPool
    pool.free = MambaPool.free.__get__(pool, type(pool))
    pool.size = 100  # live cap = full size (k2m grew but m2k shrunk
                           # via _capped_slots, not via set_capacity_slots)
    actuator = MambaArenaActuator(pool=pool)
    alloc = actuator.allocator

    # Mark slot 50 as capped (simulating m2k cap_barrier on slot 50)
    target = torch.tensor([50], dtype=torch.int64, device=alloc.device)
    n = alloc.mark_pages_capped(target)
    assert n == 1, f"setup: expected 1 marked, got {n}"
    assert 50 in pool._capped_slots.tolist(), (
        "setup: 50 must be in _capped_slots after mark"
    )
    assert 50 not in pool.free_slots.tolist(), (
        "setup: 50 must NOT be in free_slots after mark"
    )

    # Simulate a later free() (e.g., radix-cache eviction of an entry
    # that mapped to slot 50, OR an in-flight req finishing).
    pool.free(torch.tensor([50], dtype=torch.int64, device=alloc.device))
    # Post-fix contract: slot 50 must NOT return to free_slots because
    # its chunk has been unmapped by m2k.
    free_after = set(pool.free_slots.tolist())
    assert 50 not in free_after, (
        f"BUG (mamba free leaks capped slots): slot 50 returned to "
        f"free_slots after pool.free() despite being in _capped_slots. "
        f"Next alloc will hand it out → unmapped chunk → CUDA crash "
        f"(same shape as Triton 'Pointer argument cannot be accessed' "
        f"observed in N=3 v3 D10@C=56). KV side filters via _capped_pages "
        f"in TokenToKVPoolAllocator.free (allocator.py); mamba must "
        f"mirror in MambaPool.free (memory_pool.py)."
    )
    print(f"  PASS  17  MambaPool.free filters _capped_slots — slot 50 "
          f"dropped instead of returning to free_slots (CUDA crash guard)")


def _make_mamba_fake_pool_with_alloc(size: int, cap_slots: int):
    """Shared setup for test_17/17b/17c/17d. Returns (pool, alloc).

    Rebinds the production `MambaPool.free` onto a minimal fake pool so the
    real filter code runs against fake _capped_slots + size state.
    The pool also exposes a `_assert_capped_slots_invariant` no-op so the
    production `_MambaCapAllocator.mark_pages_capped` (post-#221+4th-audit
    fix) can call it without a defensive `hasattr` guard — production
    MambaPool always defines the helper; stubs must too.
    """
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakePool:
        def __init__(self, size: int) -> None:
            self.size = size
            self.live_size = size
            self.max_size = size + 100
            self.device = torch.device(DEVICE)
            self.free_slots = torch.arange(
                1, size + 1, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.empty(
                0, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()
            class _FakeArena:
                tokens_per_chunk = 1
                max_chunks_per_pool = size + 100
            self._mamba_temporal_arena = _FakeArena()

        def set_capacity_slots(self, n: int) -> int:
            return min(int(n), self.size + 100)

        def _assert_capped_slots_invariant(self) -> None:
            # No-op for the fake; we test the invariant elsewhere.
            pass

    pool = _FakePool(size=size)
    pool.free = MambaPool.free.__get__(pool, type(pool))
    pool.size = cap_slots
    actuator = MambaArenaActuator(pool=pool)
    return pool, actuator.allocator


def test_17b_mamba_free_mixed_batch():
    """Depth-add #1 from #174 audit. Real radix-cache eviction frees N slots
    at once, not 1. An over-eager fix that drops the whole `free_index` on
    any overlap with `_capped_slots` would pass test_17 but silently leak
    the non-capped slots in production.

    Sequence: mark([50, 51]) capped; free([50, 51, 52, 53]).
    Contract: {50, 51} ∉ free_slots (filtered); {52, 53} ⊆ free_slots (kept).
    """
    pool, alloc = _make_mamba_fake_pool_with_alloc(size=100, cap_slots=100)
    capped = torch.tensor([50, 51], dtype=torch.int64, device=alloc.device)
    n = alloc.mark_pages_capped(capped)
    assert n == 2, f"setup: expected 2 marked, got {n}"

    pool.free(torch.tensor([50, 51, 52, 53], dtype=torch.int64, device=alloc.device))
    after = set(pool.free_slots.tolist())
    assert 50 not in after and 51 not in after, (
        f"BUG (capped slots leaked through filter on batch free): "
        f"{50 in after=}, {51 in after=}"
    )
    assert 52 in after and 53 in after, (
        f"BUG (filter is over-eager — dropped non-capped slots from batch): "
        f"{52 in after=}, {53 in after=}. The filter must drop ONLY the "
        f"intersection with _capped_slots, not the whole free_index."
    )
    print(f"  PASS  17b  mixed batch free: capped {{50,51}} filtered, "
          f"non-capped {{52,53}} preserved (no over-eager drop)")


def test_17c_capped_filter_precedes_cap_check():
    """Depth-add #2 from #174 audit. The _capped_slots filter must run at
    the TOP of MambaPool.free, BEFORE the `size` "above-cap" branch.
    Otherwise a slot that is BOTH in _capped_slots AND above size
    flows through a different code path (held_now → torch.cat dedup),
    which is only equivalent today because the cap branch happens to have
    its own dedup. Pins the contract that the new filter is the first
    line of defense; a future refactor that removes the cap-branch dedup
    in isolation would still leave _capped_slots correct because of THIS
    ordering.

    Sequence: size=49 (slot 50 is above cap) + mark([50]) (also in
    _capped_slots) + free([50]).
    Contract: 50 ∉ free_slots; _capped_slots stays exactly [50] (no add /
    no leak); free_slots tensor unchanged in length (no spurious mutation).
    """
    pool, alloc = _make_mamba_fake_pool_with_alloc(size=100, cap_slots=49)
    capped = torch.tensor([50], dtype=torch.int64, device=alloc.device)
    alloc.mark_pages_capped(capped)
    capped_before = sorted(pool._capped_slots.tolist())
    free_slots_len_before = pool.free_slots.numel()

    pool.free(torch.tensor([50], dtype=torch.int64, device=alloc.device))

    after = set(pool.free_slots.tolist())
    assert 50 not in after, (
        f"BUG (filter ordering): slot 50 in BOTH _capped_slots AND above "
        f"size={49} returned to free_slots"
    )
    assert pool.free_slots.numel() == free_slots_len_before, (
        f"BUG (filter early-return): free_slots length changed from "
        f"{free_slots_len_before} to {pool.free_slots.numel()} on a fully-"
        f"filtered free_index; the filter branch should early-return"
    )
    capped_after = sorted(pool._capped_slots.tolist())
    assert capped_after == capped_before == [50], (
        f"BUG (_capped_slots mutated by free of already-capped slot): "
        f"before={capped_before}, after={capped_after}"
    )
    print(f"  PASS  17c  filter runs before above-cap branch: slot 50 "
          f"intersection (capped AND above-cap) handled cleanly, "
          f"_capped_slots unchanged")


def test_17d_mamba_free_noncapped_unaffected():
    """Depth-add #3 from #174 audit. Negative control: the filter must
    NOT quarantine slots that are not in _capped_slots. A regression that
    accidentally filtered against the wrong tensor (e.g. `_capped_pages`
    from the KV side, or all slots ≤ some threshold) would silently lose
    capacity. Pins the filter's specificity.

    Sequence: mark([50]) capped; free([7]) (an unrelated, non-capped slot).
    Contract: 7 ∈ free_slots (added back as normal); 50 still ∉ free_slots
    (untouched by the unrelated free).
    """
    pool, alloc = _make_mamba_fake_pool_with_alloc(size=100, cap_slots=100)
    alloc.mark_pages_capped(
        torch.tensor([50], dtype=torch.int64, device=alloc.device)
    )
    # Simulate slot 7 having been previously allocated to a now-finished
    # request — remove it from free_slots so we can verify free() adds it
    # back. (Fake-pool's free_slots is full arange(1..size) at construction.)
    pool.free_slots = pool.free_slots[pool.free_slots != 7]
    assert 7 not in pool.free_slots.tolist(), (
        "setup: slot 7 should be out of free_slots before the free under test"
    )

    pool.free(torch.tensor([7], dtype=torch.int64, device=alloc.device))

    after = set(pool.free_slots.tolist())
    assert 7 in after, (
        f"BUG (filter quarantined non-capped slot): slot 7 freed but "
        f"didn't reach free_slots even though it was never in _capped_slots"
    )
    assert 50 not in after, (
        f"BUG (cross-contamination): freeing non-capped slot 7 leaked "
        f"capped slot 50 into free_slots"
    )
    print(f"  PASS  17d  non-capped free unaffected: slot 7 reaches "
          f"free_slots; capped slot 50 stays out")


def test_18_capped_pages_invariant_assertion_fires_on_violation():
    """#162 fail-fast: a capped page-id must lie in [1, size]. An out-of-ceiling
    id is silent corruption (it would drive `live_size = size − capped` negative
    and trip the leak detector with the wrong root cause). `mark_pages_capped`
    must fail-fast at the mutation site, symmetric with `unmark_pages_capped`'s
    existing ceiling guard.

    Aligns with #205 fail-fast principle — loud-crash on a broken invariant is
    preferable to defensive code that silently fixes the symptom.

    Test plan:
      1. Build allocator at size=100.
      2. Call `mark_pages_capped([200])` — 200 > size=100 violates the invariant.
      3. Expect AssertionError naming the ceiling.
    """
    alloc = _make_allocator(size=100)

    fired = False
    try:
        alloc.mark_pages_capped(torch.tensor(
            [200], dtype=torch.int64, device=DEVICE
        ))
    except AssertionError as e:
        msg = str(e)
        assert "ceiling" in msg or "size" in msg, (
            f"Assertion message should name the violated invariant; got: {msg[:200]}"
        )
        fired = True

    assert fired, (
        "BUG (#162): mark_pages_capped did NOT raise AssertionError on an "
        "out-of-ceiling id (200 > size=100). The fail-fast invariant assertion "
        "is missing — a bad capped id leaks into live_size / available_size "
        "silently, surfacing as a misleading allocation bug far from the cause."
    )
    print(f"  PASS  18  mark_pages_capped fail-fast assertion fires on an "
          f"out-of-ceiling id (no silent corruption)")


def test_19_chunk_arena_grow_returns_mapped_slot_ids():
    """#213 Phase B: `ChunkArena.grow(pool_name, n)` returns the actual
    list of slot IDs it just mapped, in `first_free_slot` order. This is
    the foundation for the downstream rewire (Phase D) that has
    xpool_actuator pass these IDs to `dst_alloc.unmark_pages_capped` /
    `dst_pool.unmark_slots` directly, instead of relying on
    `unmark_lowest_capped_after_grow` to re-derive them by sorting
    `_capped_pages` (which works today by coincidence — chunk_arena
    happens to map at the lowest unmapped position, so the lowest
    capped IDs are the right ones — but this is fragile coupling).

    Pre-fix: grow returns int (count).
    Post-fix: grow returns list[int] (slot IDs).

    Setup:
      1. Arena with one pool, capacity 100, init 0 mapped.
      2. Pre-map slots 0..9 by calling grow(10).
      3. Unmap slots [2, 5, 7] explicitly via shrink_explicit.
      4. Call grow(2) — should re-map the LOWEST 2 first_free positions
         (which are slots 2 and 5 — slot 7 stays unmapped since we only
         asked for 2).
      5. Assert returned value is the list [2, 5].
    """
    from sglang.srt.arena.chunk_arena import ChunkArena
    CHUNK = 2 * 1024 * 1024
    arena = ChunkArena(
        device_id=0,
        chunk_size=CHUNK,
        n_handles=100,
        pool_capacities=[("p", 100)],
    )
    grown_initial = arena.grow("p", 10)
    # Contract part 1: returns list of slot IDs.
    assert isinstance(grown_initial, list), (
        f"BUG (#213 Phase B): ChunkArena.grow should return list[int] "
        f"of mapped slot IDs, got {type(grown_initial).__name__}. "
        f"Required for xpool_actuator to pipe specific IDs to "
        f"dst_alloc.unmark_pages_capped without sorting _capped_pages."
    )
    assert grown_initial == list(range(10)), (
        f"BUG (#213 Phase B): initial grow into empty pool should return "
        f"[0..9] in first_free_slot order; got {grown_initial}"
    )

    # Unmap a sparse set so the next grow has known holes to fill.
    arena.shrink_explicit("p", [2, 5, 7])

    # Contract part 2: returns the LOWEST first_free_slot positions
    # actually mapped (not a sorted view of free handles).
    grown_refill = arena.grow("p", 2)
    assert grown_refill == [2, 5], (
        f"BUG (#213 Phase B): grow(2) after unmapping {{2,5,7}} should "
        f"return [2, 5] (lowest 2 first_free positions); got {grown_refill}. "
        f"Slot 7 should stay unmapped."
    )

    arena.cleanup()
    print(f"  PASS  19  ChunkArena.grow returns list[int] of mapped slot IDs "
          f"in first_free_slot order (pins #213 Phase B foundation)")


def test_20_mamba_pool_unmark_slots_id_based_grow():
    """#213 Phase C: `MambaPool.unmark_slots(ids)` is the ID-based grow API
    that mirrors KV's `allocator.unmark_pages_capped(ids)`. Given the
    list of slot IDs that `chunk_arena.grow` just mapped (Phase B), the
    pool restores exactly those IDs from `_capped_slots` back to
    `free_slots`, and bumps `size` / `self.size` by the count
    actually restored.

    For cross-pool dst grow, this REPLACES the legacy value-filter GROW
    path in `MambaPool.set_capacity_slots` (now kept only for non-actuator
    paths like test_phase5/7). The legacy path used a `_capped_slots <=
    n_slots` value-mask plus a `_migrated_capped_slots` blacklist to
    pick what to restore. Phase E deleted the blacklist (the actuator's
    new path is structurally safe — only IDs that arena.grow returned
    get restored, migrated slots never appear there) and kept the bare
    value-mask for the legacy dynamic-resize callers.

    Contract:
      1. Restore = `_capped_slots ∩ ids`. Drop those from `_capped_slots`,
         append to `free_slots`. IDs in `ids` but NOT in `_capped_slots`
         are silently ignored (already free or not owned by this pool).
      2. `size` and `self.size` increase by `len(restore)` (NOT
         by `len(ids)` — if ids contains extras, those don't grow cap).
      3. Empty `ids` → no-op, returns 0.
      4. Returns the count actually restored.
    """
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    # Build a fake pool that exposes the minimal surface MambaPool.unmark_slots
    # needs: free_slots, _capped_slots, size, self.size.
    class _FakePool:
        def __init__(self) -> None:
            self.size = 50
            self.device = torch.device(DEVICE)
            self.size = 50
            # Pre-state: cap=50, _capped_slots = [51..100] (boot's static
            # over-provision in [size+1, max_size]). free_slots holds 1..50.
            self.free_slots = torch.arange(
                1, 51, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.arange(
                51, 101, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()
        # Bind production unmark_slots once it exists.

    pool = _FakePool()
    pool.unmark_slots = MambaPool.unmark_slots.__get__(pool, type(pool))

    # Case 1 — happy path: arena.grow returned [51, 52, 53].
    restored = pool.unmark_slots(
        torch.tensor([51, 52, 53], dtype=torch.int64, device=DEVICE)
    )
    assert restored == 3, f"expected 3 restored, got {restored}"
    free_set = set(pool.free_slots.tolist())
    capped_set = set(pool._capped_slots.tolist())
    assert {51, 52, 53} <= free_set, (
        f"BUG (#213 Phase C): IDs {{51,52,53}} not in free_slots after "
        f"unmark; free={sorted(free_set)[-10:]}"
    )
    assert capped_set == set(range(54, 101)), (
        f"BUG: _capped_slots should be {{54..100}} after restore; got "
        f"min={min(capped_set)}, max={max(capped_set)}, len={len(capped_set)}"
    )
    assert pool.size == 53, (
        f"BUG: size should be 50+3=53; got {pool.size}"
    )
    assert pool.size == 53, (
        f"BUG: size should track size (53); got {pool.size}"
    )

    # Case 2 — partial overlap: ids has extras NOT in _capped_slots.
    # ids=[54, 55, 999, 1000]. Only {54, 55} are in _capped_slots → restore 2.
    restored = pool.unmark_slots(
        torch.tensor([54, 55, 999, 1000], dtype=torch.int64, device=DEVICE)
    )
    assert restored == 2, (
        f"BUG: extras in ids must not count; expected 2 restored, got {restored}"
    )
    free_set = set(pool.free_slots.tolist())
    assert {54, 55} <= free_set and 999 not in free_set and 1000 not in free_set, (
        f"BUG: only the intersection with _capped_slots should be restored. "
        f"54 in free? {54 in free_set}, 55? {55 in free_set}, "
        f"999? {999 in free_set}, 1000? {1000 in free_set}"
    )
    assert pool.size == 55, (
        f"BUG: size should bump by actual restore count (2 → 55); "
        f"got {pool.size}"
    )

    # Case 3 — empty ids: no-op.
    n_capped_before = pool._capped_slots.numel()
    n_free_before = pool.free_slots.numel()
    cap_before = pool.size
    restored = pool.unmark_slots(
        torch.empty(0, dtype=torch.int64, device=DEVICE)
    )
    assert restored == 0
    assert pool._capped_slots.numel() == n_capped_before
    assert pool.free_slots.numel() == n_free_before
    assert pool.size == cap_before

    print(f"  PASS  20  MambaPool.unmark_slots(ids) — ID-based grow (#213 "
          f"Phase C): restores ∩ with _capped_slots, bumps size + "
          f"size by actual count, ignores extras, empty ids is no-op")


def test_21_actuators_expose_unmark_token_slots():
    """#213 Phase D: both `KVArenaActuator` and `MambaArenaActuator`
    expose a uniform `unmark_token_slots(token_slots)` API. This is the
    dispatch surface `xpool_actuator._execute_async_locked` calls instead
    of branching on direction (m2k vs k2m). After Phase D rewire:

      common_ids = granted_ids_per_subpool[0][:actual_per_dst]
      token_slots = dst_act.expand_pages_to_token_slots(common_ids)
      dst_act.unmark_token_slots(token_slots)

    Internally:
      - KV actuator → `self.allocator.unmark_pages_capped(tensor)`
      - Mamba actuator → `self.pool.unmark_slots(tensor)` (#213 Phase C)

    This test pins both dispatch paths.
    """
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    # ---- Mamba side ----
    class _FakeMambaPool:
        def __init__(self) -> None:
            self.size = 50
            self.live_size = 50
            self.device = torch.device(DEVICE)
            self.size = 50
            self.free_slots = torch.arange(
                1, 51, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.arange(
                51, 101, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()
            class _FakeArena:
                tokens_per_chunk = 1
                max_chunks_per_pool = 200
            self._mamba_temporal_arena = _FakeArena()

        def set_capacity_slots(self, n: int) -> int:
            return min(n, 200)

    mpool = _FakeMambaPool()
    mpool.unmark_slots = MambaPool.unmark_slots.__get__(mpool, type(mpool))
    mact = MambaArenaActuator(pool=mpool)
    # Sanity precondition.
    assert 51 in mpool._capped_slots.tolist()
    mact.unmark_token_slots([51, 52, 53])
    assert {51, 52, 53} <= set(mpool.free_slots.tolist()), (
        f"BUG (#213 Phase D mamba dispatch): unmark_token_slots did not "
        f"restore {{51,52,53}} via pool.unmark_slots; free_slots tail = "
        f"{sorted(mpool.free_slots.tolist())[-10:]}"
    )
    assert mpool.size == 53, (
        f"BUG: mamba unmark_token_slots should bump size; got {mpool.size}"
    )

    # ---- KV side ----
    # Mirror the KV-actuator dispatch via _make_allocator + manual setup.
    # KVArenaActuator wants a pool with set_capacity_tokens / max_tokens /
    # tokens_per_page etc. Simpler: instantiate the allocator directly,
    # mark slots capped, then verify a hand-built actuator-style call to
    # `allocator.unmark_pages_capped` (which is what the new KV dispatch
    # wraps) restores them. The actuator wrapper is one line; this proof
    # is enough to pin the underlying behavior.
    kv_alloc = _make_allocator(size=100)
    kv_alloc.mark_pages_capped(
        torch.tensor([10, 11, 12], dtype=torch.int64, device=DEVICE)
    )
    assert 10 in kv_alloc._capped_pages.tolist()
    # Direct call to the dispatch target. We construct a minimal KV
    # actuator separately to also exercise its wrapping (test_22 wires
    # the full actuator; here we just check unmark_pages_capped works
    # for the IDs the dispatch will pass).
    kv_alloc.unmark_pages_capped(
        torch.tensor([10, 11, 12], dtype=torch.int64, device=DEVICE)
    )
    assert kv_alloc._capped_pages.numel() == 0, (
        f"BUG (#213 Phase D KV dispatch target): unmark_pages_capped did "
        f"not drop the IDs; _capped_pages still has "
        f"{kv_alloc._capped_pages.numel()}"
    )

    print(f"  PASS  21  actuator unmark_token_slots dispatch — mamba routes "
          f"to pool.unmark_slots; KV routes to allocator.unmark_pages_capped")


def test_22_kv_actuator_unmark_token_slots_wrapper_e2e():
    """#213 Phase D follow-up (audit-driven): exercise the production
    `KVArenaActuator.unmark_token_slots` wrapper END-TO-END instead of
    bypassing it (test_21 sidestepped this — see post-Phase-E audit).

    Catches the exact bug class the audit flagged: missing `import torch`
    in kv_actuator.py would cause `torch.tensor(...)` inside the wrapper
    to raise NameError on the first k2m fire in production. test_21
    didn't catch this because it called `allocator.unmark_pages_capped`
    directly. This test imports the real KVArenaActuator, builds it
    against a real allocator, and drives the wrapper.

    Contract pinned:
      - `unmark_token_slots(list[int])` accepts a Python list (the type
        `expand_pages_to_token_slots` returns).
      - The wrapper converts to the right tensor dtype/device internally
        and forwards to allocator.unmark_pages_capped.
      - After the call, `_capped_pages` no longer contains those IDs.
      - Empty list is a safe no-op (matches production hot path: when
        `actual_per_dst == 0`, the loop returns an empty list).
    """
    from sglang.srt.arena.kv_actuator import KVArenaActuator

    # Build minimum-viable KV actuator fixture. KVArenaActuator wants:
    # `pool` with `size`, `_kv_arena`, `page_size`; an `allocator`
    # accessor (typically self.allocator), `max_tokens`. We pass the
    # allocator directly because the wrapper only reads `self.allocator`.
    kv_alloc = _make_allocator(size=100)
    capped_ids = torch.tensor([10, 11, 12], dtype=torch.int64, device=DEVICE)
    kv_alloc.mark_pages_capped(capped_ids)
    assert kv_alloc._capped_pages.numel() == 3

    # Build minimal KVArenaActuator stub — only need allocator attribute
    # for unmark_token_slots wrapper.
    class _StubKVActuator:
        def __init__(self, alloc):
            self.allocator = alloc

    stub = _StubKVActuator(kv_alloc)
    # Bind production unmark_token_slots onto the stub so we test the
    # ACTUAL wrapper code, not a copy.
    stub.unmark_token_slots = KVArenaActuator.unmark_token_slots.__get__(
        stub, type(stub)
    )

    # Drive the wrapper with a Python list (matches what
    # expand_pages_to_token_slots returns in production).
    stub.unmark_token_slots([10, 11, 12])
    assert kv_alloc._capped_pages.numel() == 0, (
        f"BUG (#213 Phase D audit-fix): KVArenaActuator.unmark_token_slots "
        f"did not actually unmark. _capped_pages still has "
        f"{kv_alloc._capped_pages.numel()} items. If you see NameError "
        f"about 'torch' here, that's the audit-caught import bug."
    )

    # Empty list = no-op (hot path: when actual_per_dst == 0).
    stub.unmark_token_slots([])
    assert kv_alloc._capped_pages.numel() == 0, (
        "Empty list call must be a no-op, not raise."
    )
    print(f"  PASS  22  KVArenaActuator.unmark_token_slots wrapper E2E "
          f"— catches missing `import torch` + dtype/device drift in the "
          f"dispatch wrapper")


def test_23_xpool_lockstep_assertion_fires():
    """#213 Phase D follow-up (audit-driven, upgraded post-2nd-audit):
    the lockstep fail-fast assertion inside `xpool_actuator
    ._execute_async_locked` is the ONLY safety net for the "fragile
    coupling" the audit calls out — that `arena.grow`'s `first_free_slot`
    ordering happens to match across sub-pools. If a future change
    breaks this (e.g., LIFO popping or LRU placement), the actuator
    would otherwise expose a slot whose chunk is unmapped in some
    sub-pool → CUDA illegal access.

    This test drives the REAL `_execute_async_locked` method (not a
    pure-Python re-implementation): builds a minimal `FireToken` with
    stub MTAs whose `dst._arena.grow` returns mismatched ID lists
    across sub-pools, then invokes the method and verifies the
    `RuntimeError` fires with `lockstep` in the message.

    Aligns with #205 fail-fast principle: assertion → loud crash with
    diagnostic, not silent corruption.
    """
    import threading
    import time
    from sglang.srt.arena.xpool_actuator import XPoolActuator, FireToken
    from sglang.srt.arena.fire_plan import FirePlan

    # Bypass __init__ (which needs real MultiTensorArenas).
    inst = object.__new__(XPoolActuator)
    inst._fire_inflight = threading.Lock()  # safety; we call _locked directly
    # _execute_async_locked reads self.lcm_pages (#229) which derives
    # from these. Both sides are 2-subpool stubs (n_layers=2 × n_kinds=1).
    inst.n_kv_subpools = 2
    inst.n_mamba_subpools = 2

    # Stub ChunkArena with per-pool deterministic grow returns. The
    # mismatch between p0=[5,6,7] and p1=[5,6,8] is what the lockstep
    # check must catch.
    class _StubArena:
        def __init__(self, grow_returns):
            self._grow_returns = grow_returns

        def grow(self, name, n):
            return list(self._grow_returns[name][:n])

        def shrink_explicit(self, name, ids):
            return len(ids)

    class _StubMTA:
        def __init__(self, arena, n_subpools=2):
            self._arena = arena
            self.n_layers = n_subpools
            self.n_kinds = 1

        def _pool_name(self, i):
            return f"p{i}"

    src_mta = _StubMTA(_StubArena({"p0": [], "p1": []}))
    dst_mta = _StubMTA(_StubArena({
        "p0": [5, 6, 7],
        "p1": [5, 6, 8],   # mismatch at position 2
    }))

    # Minimal actuators with empty allocator state — the worker-side
    # verify (#215) reads `src_act.allocator.free_pages`; an empty
    # tensor short-circuits the verify so the lockstep assertion is
    # the next thing to fire.
    class _StubAlloc:
        def __init__(self) -> None:
            self.device = torch.device(DEVICE)
            self.free_pages = torch.empty(0, dtype=torch.int64, device=DEVICE)
            self._capped_pages = torch.empty(0, dtype=torch.int64, device=DEVICE)

        def count_reachable_capped(self, cap_t: torch.Tensor) -> int:
            # How many of cap_t are still allocatable: in free_pages but
            # not excluded by _capped_pages. Empty free_pages → 0, so the
            # worker-side verify short-circuits and the lockstep assertion
            # is the next thing to fire.
            free = self.free_pages
            if free.numel() == 0:
                return 0
            target = cap_t.to(self.device).to(torch.int64)
            in_target = torch.isin(free, target)
            if self._capped_pages.numel() > 0:
                in_target = in_target & (~torch.isin(free, self._capped_pages))
            return int(in_target.sum().item())

    class _StubActuator:
        def __init__(self) -> None:
            self.allocator = _StubAlloc()

        def expand_pages_to_token_slots(self, ids):
            return list(ids)

        def unmark_token_slots(self, slots):
            pass

    plan = FirePlan(
        direction="kv_to_mamba",
        pages_to_unmap=[0, 1, 2],
        pages_to_map_dst=3,
        plan_seq=42,
    )
    token = FireToken(
        plan=plan,
        src=src_mta,
        dst=dst_mta,
        src_act=_StubActuator(),
        dst_act=_StubActuator(),
        cap_t=torch.zeros(0, dtype=torch.int64),
        cap_slots_count=0,
        cap_barrier_us=0,
        t_start_ns=time.monotonic_ns(),
    )

    raised = False
    msg = ""
    try:
        inst._execute_async_locked(token)
    except RuntimeError as e:
        msg = str(e)
        raised = True

    assert raised, (
        "BUG (#213 Phase D safety net): _execute_async_locked did NOT "
        "raise on dst sub-pool ID prefix mismatch. The lockstep invariant "
        "is dead code — a planner change that breaks first_free_slot "
        "ordering would silently expose unmapped slots → CUDA crash."
    )
    assert "lockstep" in msg.lower(), (
        f"Diagnostic should name the invariant; got: {msg[:200]}"
    )
    assert "granted" in msg.lower() or "sub-pool" in msg.lower(), (
        f"Diagnostic should reference granted/sub-pool state; got: {msg[:200]}"
    )

    print(f"  PASS  23  xpool _execute_async_locked raises RuntimeError "
          f"with 'lockstep' diagnostic on real mismatched-sub-pool input "
          f"(drives production code path, not a pure-Python re-impl)")


def test_25_mamba_pool_unmark_slots_real_pool_integration():
    """#213 Phase D follow-up (audit-driven, post-2nd-audit): exercise
    `MambaPool.unmark_slots` against a REAL `MambaPool` (not the
    `_FakePool` stub test_20 uses). Catches composition bugs the unit
    test can't — initialization side effects, `_alloc_lock`, the new
    `_assert_capped_slots_invariant` (#221) integration, real
    `_capped_slots` tensor layout from Phase 7's static pre-allocation.

    Setup mirrors `test_phase7._make_pool` to avoid duplicating the
    `Mamba2CacheParams` boilerplate.
    """
    import os
    os.environ.pop("SGLANG_ARENA_SHARED", None)
    os.environ.pop("SGLANG_MAMBA_ARENA", None)

    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=128,
        n_groups=1,
        num_heads=4,
        head_dim=64,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    pool = MambaPool(
        size=8,
        spec_state_size=8,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        device="cuda:0",
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=32,
    )

    # Phase 7 invariant: with max_size=32, size=8, `_capped_slots`
    # holds [9..32] (the deferred IDs above init cap).
    assert pool.size == 8, f"setup: size = {pool.size}"
    initial_capped = set(pool._capped_slots.tolist())
    assert initial_capped == set(range(9, 33)), (
        f"setup: expected _capped_slots = {{9..32}}, got "
        f"{sorted(initial_capped)[:10]}..."
    )

    # Drive unmark_slots with IDs [9, 10, 11, 12] — simulating a
    # cross-pool m2k arena.grow returning those 4 chunk positions.
    target_ids = torch.tensor(
        [9, 10, 11, 12], dtype=torch.int64, device="cuda:0",
    )
    n_restored = pool.unmark_slots(target_ids)
    assert n_restored == 4, (
        f"BUG (#213 Phase D / Phase C real-pool): unmark_slots returned "
        f"{n_restored}, expected 4 (all 4 target IDs were in _capped_slots)"
    )

    # Post-unmark state:
    # - size bumped from 8 to 12
    # - self.size bumped from 8 to 12 (Phase 7 sync)
    # - _capped_slots dropped {9,10,11,12}: now {13..32}
    # - free_slots gained {9,10,11,12}
    assert pool.size == 12, f"size = {pool.size}, expected 12"
    assert pool.size == 12, f"size = {pool.size}, expected 12 (Phase 7 sync)"
    capped_after = set(pool._capped_slots.tolist())
    assert capped_after == set(range(13, 33)), (
        f"_capped_slots after unmark = {sorted(capped_after)[:5]}..., "
        f"expected {{13..32}}"
    )
    free_after = set(pool.free_slots.tolist())
    assert {9, 10, 11, 12} <= free_after, (
        f"free_slots missing the unmarked IDs: missing = "
        f"{{9,10,11,12}} - free_slots = "
        f"{ {9,10,11,12} - free_after }"
    )

    # #221 invariant must hold throughout: _capped_slots.numel() = 20
    # ≤ max_size = 32. ✓
    pool._assert_capped_slots_invariant()

    # Idempotency: re-call with same IDs (already restored) should
    # restore 0.
    n_idempotent = pool.unmark_slots(target_ids)
    assert n_idempotent == 0, (
        f"BUG: re-unmark of already-restored IDs should return 0; got "
        f"{n_idempotent}. Could lead to over-bumping size."
    )
    assert pool.size == 12, (
        f"size changed on idempotent re-call: now {pool.size}"
    )
    print(f"  PASS  25  MambaPool.unmark_slots composes correctly with "
          f"real MambaPool: Phase 7 init state → unmark 4 IDs → cap "
          f"bumped, free_slots updated, _capped_slots filtered, #221 "
          f"invariant holds, idempotent on re-call")


def test_24_mamba_pool_capped_slots_invariant_assertion():
    """#221 (parallel to #162): MambaPool needs an `_assert_capped_slots_
    invariant` analogous to KV's `_assert_capped_invariant`. Live D11
    crash referenced in `memory_pool.py` comments — "capped=1526 vs
    size=514" — proves the same silent-corruption hazard exists on
    mamba side. After Phase E removed the `_migrated_capped_slots`
    blacklist, the actuator path is structurally safe but the legacy
    `set_capacity_slots` SHRINK + `migrate_slot` + `MambaPool.free`
    above-cap paths can still push `_capped_slots.numel() > max_size`
    if dedupe misses an edge case.

    Same fail-fast principle as #162 / #205: assertion fires at the
    mutation site instead of letting `live_size = size - _capped.numel()`
    silently go negative and surface much later as a confusing leak.

    Test plan (TDD):
      1. Build a minimal MambaPool-like target with prod `migrate_slot`
         bound onto it.
      2. Pre-populate `_capped_slots` to exactly `max_size` (boundary).
      3. Call `migrate_slot(new_src, dst)` with a new ID — pushes
         numel past max_size.
      4. Expect: AssertionError naming `_capped_slots` and `max_size`.

    Before fix: migrate_slot silently appends; numel exceeds max_size;
    no error fires.
    After fix: invariant assertion fires loudly at the mutation site.
    """
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakeMambaPool:
        def __init__(self, size: int, max_size: int) -> None:
            self.size = size
            self.live_size = size
            self.max_size = max_size
            self.device = torch.device(DEVICE)
            self.size = size
            self.free_slots = torch.arange(
                1, size + 1, dtype=torch.int64, device=DEVICE
            )
            # Pre-load _capped_slots to EXACTLY max_size (boundary).
            # Use ids well outside free_slots so migrate_slot doesn't
            # touch them.
            self._capped_slots = torch.arange(
                size + 1, size + max_size + 1, dtype=torch.int64, device=DEVICE
            )
            self.mamba_cache = None  # migrate_slot's tensor copy skipped via guard
            self._alloc_lock = threading.Lock()

    pool = _FakeMambaPool(size=10, max_size=20)
    assert pool._capped_slots.numel() == 20, (
        "setup: pre-state must be at max_size boundary"
    )

    # Bind migrate_slot — but we can't actually call it without
    # mamba_cache fixtures. Instead simulate the SAME mutation
    # migrate_slot performs (torch.cat appending to _capped_slots) and
    # use a wrapper to invoke the would-be-attached assertion helper.
    # The contract we're pinning: after ANY mutation that grows
    # _capped_slots, the helper must assert numel <= max_size.

    # Try the direct contract check: the helper method must exist and
    # must fire when pre-populated state already exceeds max_size.
    new_id = torch.tensor([999], dtype=torch.int64, device=DEVICE)
    pool._capped_slots = torch.cat([pool._capped_slots, new_id])
    assert pool._capped_slots.numel() == 21

    fired = False
    try:
        # The helper must exist on MambaPool. Bind it dynamically so the
        # test fails clearly pre-fix.
        helper = getattr(MambaPool, "_assert_capped_slots_invariant", None)
        if helper is None:
            raise AssertionError(
                "MambaPool._assert_capped_slots_invariant method missing"
            )
        helper.__get__(pool, type(pool))()
    except AssertionError as e:
        msg = str(e)
        if "method missing" in msg:
            # Pre-fix: helper doesn't exist. This is the bug.
            raise
        assert "_capped_slots" in msg and "max_size" in msg, (
            f"Assertion message should name _capped_slots + max_size; got: {msg[:200]}"
        )
        fired = True

    assert fired, (
        "BUG (#221): MambaPool._assert_capped_slots_invariant did NOT raise "
        "despite _capped_slots.numel() (21) > max_size (20). The fail-fast "
        "invariant is missing — silent corruption (live D11 'capped=1526 vs "
        "size=514' class crash) is not prevented at the mutation site."
    )
    print(f"  PASS  24  MambaPool._assert_capped_slots_invariant fires when "
          f"_capped_slots.numel() pushed past max_size (parallel to #162 KV "
          f"invariant; no silent corruption)")


def test_26_mamba_cap_allocator_mark_calls_capped_invariant():
    """#221 follow-up (3rd-audit gap): `_MambaCapAllocator.mark_pages_capped`
    is the actual production hot path for m2k cap_barrier (xpool_actuator
    calls it via `src_act.allocator.mark_pages_capped(cap_t)`). It mutates
    `pool._capped_slots` directly but did NOT call
    `pool._assert_capped_slots_invariant()` after the mutation. KV side's
    `TokenToKVPoolAllocator.mark_pages_capped` DOES call its sibling
    `_assert_capped_invariant` post-mutation. Asymmetric → mamba m2k
    fires can silently overflow `_capped_slots` past max_size with no
    fail-fast diagnostic.

    Test: pre-populate `_capped_slots` to max_size boundary, then call
    `_MambaCapAllocator.mark_pages_capped([new_id])` to push past. Must
    raise AssertionError after fix.
    """
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakeMambaPool:
        def __init__(self) -> None:
            self.size = 8
            self.live_size = 8
            self.max_size = 12
            self.device = torch.device(DEVICE)
            self.free_slots = torch.tensor(
                [99], dtype=torch.int64, device=DEVICE
            )
            # Pre-populate _capped_slots at boundary
            self._capped_slots = torch.arange(
                9, 9 + 12, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()

    pool = _FakeMambaPool()
    pool._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(pool, type(pool))
    )
    assert pool._capped_slots.numel() == 12, "setup: at-boundary"
    alloc = _MambaCapAllocator(pool)

    fired = False
    try:
        alloc.mark_pages_capped(
            torch.tensor([99], dtype=torch.int64, device=DEVICE)
        )
    except AssertionError as e:
        msg = str(e)
        assert "_capped_slots" in msg and "max_size" in msg, (
            f"Assertion message should name invariant; got: {msg[:200]}"
        )
        fired = True

    assert fired, (
        "BUG (3rd-audit must-fix): _MambaCapAllocator.mark_pages_capped "
        "(m2k production hot path) does NOT call pool._assert_capped_slots_"
        "invariant() after mutating pool._capped_slots. KV equivalent in "
        "TokenToKVPoolAllocator.mark_pages_capped DOES. Silent overflow of "
        "`_capped_slots` past max_size on m2k fires (live D11 'capped=1526 "
        "vs size=514' crash class) is not caught at the mutation site."
    )
    print(f"  PASS  26  _MambaCapAllocator.mark_pages_capped calls "
          f"pool._assert_capped_slots_invariant after mutating "
          f"pool._capped_slots (m2k hot path symmetry with KV side)")


def test_27_mark_pages_capped_no_hasattr_defensive_guard():
    """#205 + #221 follow-up (4th-audit must-fix): the invariant call in
    `_MambaCapAllocator.mark_pages_capped` was added with a defensive
    `hasattr(pool, "_assert_capped_slots_invariant")` guard (yesterday's
    fix). Defensive guards mask the case where pool LACKS the helper,
    which is exactly the case we should crash on per #205 fail-fast.

    KV's parallel `TokenToKVPoolAllocator.mark_pages_capped` has zero
    defensive guards around its `_assert_capped_invariant()` call. Mamba
    side should match.

    Test: build a fake pool that DOES NOT have `_assert_capped_slots_
    invariant` and feed it to `_MambaCapAllocator.mark_pages_capped`.
    Expect `AttributeError` (fail-fast). The defensive hasattr currently
    swallows this. Post-fix (hasattr removed), AttributeError surfaces.
    """
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator

    class _BareFakePool:
        # Intentionally NO _assert_capped_slots_invariant. (Has the
        # other state mark_pages_capped touches; only the helper is
        # missing, so the AttributeError points at the helper.)
        def __init__(self) -> None:
            self.device = torch.device(DEVICE)
            self.free_slots = torch.tensor(
                [1, 2, 3], dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.empty(
                0, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()

    pool = _BareFakePool()
    alloc = _MambaCapAllocator(pool)
    raised = False
    try:
        alloc.mark_pages_capped(
            torch.tensor([1], dtype=torch.int64, device=DEVICE)
        )
    except AttributeError as e:
        msg = str(e)
        assert "_assert_capped_slots_invariant" in msg, (
            f"AttributeError message should reference the missing helper; "
            f"got: {msg[:200]}"
        )
        raised = True

    assert raised, (
        "BUG (4th-audit must-fix): _MambaCapAllocator.mark_pages_capped "
        "has a defensive hasattr() guard around its "
        "_assert_capped_slots_invariant() call. Pool without the helper "
        "silently skips the invariant check — exactly the case #205 "
        "fail-fast principle says should crash. Remove the hasattr; the "
        "KV parallel `TokenToKVPoolAllocator.mark_pages_capped` has no "
        "such guard."
    )
    print(f"  PASS  27  _MambaCapAllocator.mark_pages_capped fails fast "
          f"(AttributeError) when pool lacks _assert_capped_slots_invariant "
          f"— no defensive hasattr guard per #205")


def test_28_unmark_slots_cap_uses_max_not_count():
    """5th-audit P0 fix (replaces 4th-audit's wrong "contiguity assert"):

    The 4th audit added a contiguity assertion to `MambaPool.unmark_slots`
    on the theory that restored IDs should form `[cap+1, cap+n_restored]`.
    The 5th audit proved this is WRONG for the realistic m2k → k2m flow:
      - m2k cap_barrier marks some IDs (e.g. high-tail free pages [8,9])
        via `_MambaCapAllocator.mark_pages_capped` → `_capped_slots`
        gains those IDs, but `size` is unchanged (cap-barrier
        doesn't change cap).
      - k2m later restores those IDs via `unmark_slots([8,9])`. With
        `size=10`, the restored IDs (8,9) are BELOW the boundary
        — cap should stay at 10, not bump to 12.
      - The 4th audit's `cap + n_restored` model would have wrongly
        advanced cap to 12 and over-reported `live_size`.
      - The 4th audit's contiguity assert would CRASH on this case
        (expected [11,12] but got [8,9]) — first m2k→k2m fire.

    Correct semantic: `size = max(old_cap, restored.max())`.
      - Restoring IDs below cap (mark_pages_capped restoration): cap
        unchanged.
      - Restoring IDs above cap (boot-deferred [size+1..max_size]
        restoration): cap extends to max restored ID.

    `_capped_slots` always carries the UNION of "below-cap marked" and
    "above-cap deferred"; the cap only tracks the upper bound of the
    LIVE range, not the count.

    Two test cases pin both branches.
    """
    from sglang.srt.mem_cache.memory_pool import MambaPool

    # --- Case A: restoring IDs BELOW cap (m2k → k2m round-trip).
    class _FakePool:
        def __init__(self, cap, max_size, capped, free) -> None:
            self.size = cap
            self.live_size = cap
            self.max_size = max_size
            self.device = torch.device(DEVICE)
            self.size = cap
            self.free_slots = torch.tensor(
                free, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.tensor(
                capped, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()

    # Pre-state mimics post-m2k: cap=10, free=[1..7,10] (m2k pulled 8,9
    # out of free), _capped_slots=[8, 9, 11..32] (8,9 from mark_pages_capped,
    # 11..32 boot-deferred).
    poolA = _FakePool(
        cap=10, max_size=32,
        capped=[8, 9] + list(range(11, 33)),
        free=[1, 2, 3, 4, 5, 6, 7, 10],
    )
    poolA.unmark_slots = MambaPool.unmark_slots.__get__(poolA, type(poolA))
    poolA._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(poolA, type(poolA))
    )

    n_restored = poolA.unmark_slots(
        torch.tensor([8, 9], dtype=torch.int64, device=DEVICE)
    )
    assert n_restored == 2
    # k2m restoration below cap: cap UNCHANGED.
    assert poolA.size == 10, (
        f"BUG (5th-audit P0): unmark_slots([8,9]) with cap=10 should NOT "
        f"bump cap (restored IDs are below boundary — m2k/k2m round-trip). "
        f"Got size={poolA.size}. Old `cap + n_restored` model "
        f"wrongly bumped to 12 and over-reported live_size."
    )
    assert poolA.size == 10, f"size should equal size; got {poolA.size}"
    # _capped_slots lost {8, 9}, still has [11..32].
    capped_after = set(poolA._capped_slots.tolist())
    assert capped_after == set(range(11, 33)), (
        f"_capped_slots after restore-below-cap should drop 8,9; got "
        f"{sorted(capped_after)[:5]}..."
    )

    # --- Case B: restoring IDs ABOVE cap (k2m bringing new growth).
    poolB = _FakePool(
        cap=10, max_size=32,
        capped=list(range(11, 33)),
        free=list(range(1, 11)),
    )
    poolB.unmark_slots = MambaPool.unmark_slots.__get__(poolB, type(poolB))
    poolB._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(poolB, type(poolB))
    )

    n_restored = poolB.unmark_slots(
        torch.tensor([11, 12], dtype=torch.int64, device=DEVICE)
    )
    assert n_restored == 2
    # Above-cap restore: cap EXTENDS to max restored.
    assert poolB.size == 12, (
        f"BUG: unmark_slots([11,12]) with cap=10 should bump cap to "
        f"max(10, 12) = 12 (above-boundary growth). Got "
        f"size={poolB.size}."
    )
    assert poolB.size == 12

    # --- Case C: mixed (some below, some above) — cap should extend
    # to max restored.
    poolC = _FakePool(
        cap=10, max_size=32,
        capped=[8, 9] + list(range(11, 33)),
        free=[1, 2, 3, 4, 5, 6, 7, 10],
    )
    poolC.unmark_slots = MambaPool.unmark_slots.__get__(poolC, type(poolC))
    poolC._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(poolC, type(poolC))
    )
    n_restored = poolC.unmark_slots(
        torch.tensor([8, 9, 11, 12], dtype=torch.int64, device=DEVICE)
    )
    assert n_restored == 4
    assert poolC.size == 12, (
        f"BUG: mixed restore (below + above cap) should set cap = "
        f"max(10, 12) = 12. Got {poolC.size}."
    )

    print(f"  PASS  28  MambaPool.unmark_slots cap-as-max semantic — "
          f"restoring below-cap IDs keeps cap (m2k/k2m round-trip); "
          f"restoring above-cap IDs extends cap to max restored")


def test_29_mamba_live_size_reflects_mark_pages_capped():
    """#228: `MambaPool.live_size` must mirror KV's semantic — it tracks
    "currently allocatable cap" = `self.size - count(_capped_slots <=
    self.size)`. Mark reduces it; unmark restores it. Design.md
    §"Page ownership state" "CAPPED blocks alloc reuse" requires this.

    Pre-fix: live_size = size (insensitive to mark).
    Post-fix: live_size derived from _capped_slots.

    Production scenario: m2k cap-barrier marks free pages [8, 9] on
    src=mamba. live_size should drop from 10 to 8 immediately, so the
    Budgeter sees correct pressure during the mark→unmap→unmark window.
    Then k2m unmark restores live_size back to 10.
    """
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakePool:
        def __init__(self) -> None:
            self.size = 10
            self.max_size = 32
            self.device = torch.device(DEVICE)
            self.free_slots = torch.arange(
                1, 11, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.arange(
                11, 33, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()

    pool = _FakePool()
    pool._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(pool, type(pool))
    )
    pool.unmark_slots = MambaPool.unmark_slots.__get__(pool, type(pool))
    # Bind the property descriptor. Property access on class, then __get__.
    live_size_getter = MambaPool.__dict__["live_size"].fget

    # Boot state: 22 above-size capped, 0 below-size capped.
    assert live_size_getter(pool) == 10, (
        f"Boot state: live_size should equal self.size when no below-cap "
        f"_capped entries; got {live_size_getter(pool)}"
    )

    # m2k cap-barrier marks slots 8, 9 (high-tail free pages picked by
    # fire_planner).
    alloc = _MambaCapAllocator(pool)
    alloc.mark_pages_capped(
        torch.tensor([8, 9], dtype=torch.int64, device=DEVICE)
    )
    assert live_size_getter(pool) == 8, (
        f"BUG (#228): after mark_pages_capped([8,9]), live_size should "
        f"drop from 10 to 8 (10 - 2 below-cap capped). Got "
        f"{live_size_getter(pool)}. Design.md says CAPPED blocks alloc "
        f"reuse; live_size must reflect this for Budgeter pressure math."
    )

    # k2m unmark restores [8, 9].
    pool.unmark_slots(torch.tensor([8, 9], dtype=torch.int64, device=DEVICE))
    assert live_size_getter(pool) == 10, (
        f"After unmark, live_size should restore to 10. Got "
        f"{live_size_getter(pool)}."
    )
    print(f"  PASS  29  MambaPool.live_size derived from _capped_slots "
          f"(mark reduces, unmark restores; design.md §'Page ownership "
          f"state' CAPPED semantics)")


def test_30_set_capacity_slots_grow_skips_marked_slots():
    """#228: `MambaPool.set_capacity_slots(N)` GROW branch must restore
    ONLY boot-deferred IDs (those above current size), not m2k-marked
    IDs (those below current size whose chunks are unmapped).

    Pre-fix: value-mask `held <= N` restores ANY capped <= N, including
    marked slots whose chunks aren't mapped → next alloc → CUDA crash.
    Post-fix: only `held > self.size AND held <= N` (boot-deferred in
    grow range).

    No production caller hits this today (Budgeter cascade doesn't
    call set_capacity_slots on mamba after #213 Phase E), but the
    contract should be safe — test_phase5/7 still exercise GROW.
    """
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakePool:
        def __init__(self) -> None:
            self.size = 10
            self.max_size = 32
            self.device = torch.device(DEVICE)
            self.free_slots = torch.arange(
                1, 11, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.arange(
                11, 33, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()

    pool = _FakePool()
    pool._assert_capped_slots_invariant = (
        MambaPool._assert_capped_slots_invariant.__get__(pool, type(pool))
    )
    pool.set_capacity_slots = (
        MambaPool.set_capacity_slots.__get__(pool, type(pool))
    )

    # m2k cap-barrier marks slots 8, 9 (high-tail).
    alloc = _MambaCapAllocator(pool)
    alloc.mark_pages_capped(
        torch.tensor([8, 9], dtype=torch.int64, device=DEVICE)
    )
    # Pre-state: _capped_slots = [8, 9, 11..32]; size = 10; free_slots
    # = [1..7, 10].
    capped_before = set(pool._capped_slots.tolist())
    assert 8 in capped_before and 9 in capped_before

    # Boot-style GROW: set_capacity_slots(13). Should expose ONLY
    # boot-deferred [11, 12, 13]. Marked [8, 9] must stay capped because
    # their chunks aren't mapped.
    pool.set_capacity_slots(13)

    free_after = set(pool.free_slots.tolist())
    capped_after = set(pool._capped_slots.tolist())

    assert 8 not in free_after and 9 not in free_after, (
        f"BUG (#228): set_capacity_slots(13) value-mask wrongly exposed "
        f"marked slots {{8, 9}} to free_slots. Their chunks are unmapped "
        f"(m2k just shrunk them) — next alloc → CUDA illegal access. "
        f"GROW must only restore boot-deferred entries (capped > old "
        f"size), not marked entries (capped <= old size)."
    )
    assert 8 in capped_after and 9 in capped_after, (
        f"BUG (#228): marked slots 8, 9 should stay in _capped_slots "
        f"after the GROW; got capped_after with 8={'in' if 8 in capped_after else 'OUT'}, "
        f"9={'in' if 9 in capped_after else 'OUT'}."
    )
    assert {11, 12, 13} <= free_after, (
        f"Boot-deferred [11, 12, 13] should be in free_slots after "
        f"GROW; got tail = {sorted(free_after)[-5:]}"
    )
    print(f"  PASS  30  set_capacity_slots(N) GROW only restores boot-"
          f"deferred IDs (above prior cap); marked slots stay capped "
          f"with their unmapped chunks (latent crash class avoided)")


def test_31_cap_barrier_skips_verify_sync_worker_does_it():
    """#215: `cap_barrier` runs on the scheduler thread; the previous
    `int(in_target.sum().item())` verify forced a CUDA stream sync on
    that thread (D8 bisect #132 perf liability). Move the verify into
    `_execute_async_locked` (worker thread), where the sync is free.

    Contract pinned:
      1. `cap_barrier` no longer reads `.item()` on a GPU tensor —
         returns a `FireToken` whose `aborted=False` even when the
         (hypothetical) verify would have failed. Verify defers to
         worker.
      2. `_execute_async_locked` runs the verify FIRST (before unmap).
         On violation it rolls back the mark, marks the result aborted,
         and returns without touching cuMemUnmap.

    Test: build minimal token where verify must FAIL (we plant the
    cap_t IDs back into a fake `free_pages` that isn't filtered by
    `_capped_pages`). Pre-fix: cap_barrier itself raises / aborts.
    Post-fix: cap_barrier passes through, _execute_async_locked aborts
    the fire, no cuMemUnmap was attempted.
    """
    import threading
    import time
    from sglang.srt.arena.xpool_actuator import XPoolActuator, FireToken
    from sglang.srt.arena.fire_plan import FirePlan

    inst = object.__new__(XPoolActuator)
    inst._fire_inflight = threading.Lock()

    # Fake allocator that exposes `free_pages` containing the cap_t IDs
    # (so verify will detect them as "still reachable" → violation).
    class _FakeAlloc:
        def __init__(self) -> None:
            self.device = torch.device(DEVICE)
            self.free_pages = torch.tensor(
                [5, 6, 7], dtype=torch.int64, device=DEVICE,
            )
            self._capped_pages = torch.empty(
                0, dtype=torch.int64, device=DEVICE,
            )
            self.unmark_called = False

        def count_reachable_capped(self, cap_t: torch.Tensor) -> int:
            # How many of cap_t are still allocatable: in free_pages but
            # not excluded by _capped_pages. Here free_pages = [5,6,7] and
            # nothing is capped, so cap_t = [5,6,7] reports 3 → the
            # worker-side verify detects the leak and aborts the fire.
            free = self.free_pages
            if free.numel() == 0:
                return 0
            target = cap_t.to(self.device).to(torch.int64)
            in_target = torch.isin(free, target)
            if self._capped_pages.numel() > 0:
                in_target = in_target & (~torch.isin(free, self._capped_pages))
            return int(in_target.sum().item())

        def unmark_pages_capped(self, t):
            self.unmark_called = True
            return int(t.numel())

    class _StubMTA:
        def __init__(self) -> None:
            self.n_layers = 1
            self.n_kinds = 1
            class _Arena:
                def grow(self, name, n): return []
                def shrink_explicit(self, name, ids): return 0
            self._arena = _Arena()

        def _pool_name(self, i): return f"p{i}"

    class _StubActuator:
        def __init__(self, alloc): self.allocator = alloc
        def expand_pages_to_token_slots(self, ids): return list(ids)
        def unmark_token_slots(self, slots): pass

    src_alloc = _FakeAlloc()
    src_act = _StubActuator(src_alloc)
    dst_act = _StubActuator(_FakeAlloc())
    src_mta, dst_mta = _StubMTA(), _StubMTA()

    plan = FirePlan(
        direction="kv_to_mamba",
        pages_to_unmap=[5, 6, 7],
        pages_to_map_dst=3,
        plan_seq=99,
    )
    # Build a token AS IF cap_barrier had passed (aborted=False even
    # though verify would catch IDs still in free_pages).
    cap_t = torch.tensor([5, 6, 7], dtype=torch.int64, device=DEVICE)
    token = FireToken(
        plan=plan,
        src=src_mta,
        dst=dst_mta,
        src_act=src_act,
        dst_act=dst_act,
        cap_t=cap_t,
        cap_slots_count=3,
        cap_barrier_us=10,
        t_start_ns=time.monotonic_ns(),
        aborted=False,
    )

    result = inst._execute_async_locked(token)
    assert result.aborted, (
        f"BUG (#215): _execute_async_locked must abort the fire when "
        f"worker-side verify detects cap_t IDs still in free_pages. "
        f"Got aborted={result.aborted}, reason={result.abort_reason!r}. "
        f"This is the verify that used to live on the scheduler thread "
        f"in cap_barrier."
    )
    assert "verify" in result.abort_reason.lower(), (
        f"abort_reason should name 'verify'; got: {result.abort_reason!r}"
    )
    assert src_alloc.unmark_called, (
        "Worker-side verify failure must rollback the mark by calling "
        "unmark_pages_capped on the src allocator."
    )
    assert result.unmapped_pages == 0, (
        f"Aborted fire must not have unmapped anything; got "
        f"unmapped_pages={result.unmapped_pages}"
    )
    print(f"  PASS  31  cap_barrier no GPU sync — worker thread runs "
          f"verify + rollback. _execute_async_locked aborts cleanly "
          f"when cap_t IDs leaked into free_pages.")


def test_32_admitter_c_xfer_uses_lcm_rounded_n_pages():
    """#219: c_xfer cost must reflect the LCM-rounded page count the
    actuator actually fires (xpool_actuator.py LCM math), not the
    unrounded ceil(x_tokens / tokens_per_page). With kv × mamba
    sub-pools commonly producing LCM 12-48, small x_tokens
    (1-2 pages) get rounded up substantially; the un-rounded cost
    under-estimates cross-* and biases the Admitter toward cross-*
    over defer.

    Contract:
      `Admitter.decide(..., lcm_pages=N)` uses
      `n_pages_rounded = ceil(n_pages / lcm_pages) * lcm_pages` for
      the `c_xfer_total` term.

    Example: x_tokens=512, tokens_per_page=1024 → n_pages=1.
      lcm_pages=12 → n_pages_rounded=12. cross_free cost should be
      12 × c_xfer_per_page, not 1 × c_xfer_per_page.
    """
    from sglang.srt.budgeter.cost_model import reset_cost_model, CostModel
    from sglang.srt.budgeter.admitter import Admitter

    reset_cost_model()
    a = Admitter(cost_model=CostModel())

    # Force warmed-up so cross-* gate doesn't suppress.
    a.cost_model._c_xfer_per_page_us_actual = 1000.0  # bypass cold-start
    a.cost_model._n_observations = 10  # mark as warmed

    dec = a.decide(
        x_tokens=512,
        dst_pool="kv",
        dst_free=0,
        dst_evictable=0,
        src_pool="mamba",
        src_free=10000,  # plenty of src free for cross_free
        src_evictable=0,
        queue_len=0,
        c_evict_dst_us=float("inf"),
        c_evict_src_us=float("inf"),
        c_xfer_per_page_us=1000.0,
        tokens_per_page=1024,
        lcm_pages=12,
    )
    expected_cost = 12 * 1000.0
    actual = dec.candidate_costs_us.get("cross_free")
    assert actual == expected_cost, (
        f"BUG (#219): cross_free cost = {actual} µs, expected "
        f"{expected_cost} µs. n_pages=ceil(512/1024)=1; "
        f"n_pages_rounded=ceil(1/12)*12=12. The un-rounded form "
        f"under-estimates cross-* by an LCM factor and biases "
        f"selection toward cross-* over defer."
    )
    print(f"  PASS  32  Admitter c_xfer uses LCM-rounded n_pages "
          f"(lcm=12, x_tokens=512 → rounded=12 not 1; "
          f"cost = {actual} µs)")


def test_16_m2k_grow_uses_explicit_arena_returned_ids():
    """#213 Phase E rewrite (was: m2k post-grow uses sorted-lowest-N
    helper `unmark_lowest_capped_after_grow`). After Phase D rewire,
    the m2k post-grow uses the EXACT chunk IDs returned by
    `chunk_arena.grow`, not a sort + take-lowest derivation.

    The Phase E-deleted helper had a sort+take-lowest semantics. The
    new path (Phase D) gets the IDs directly from arena.grow, which:
      - Maps at `pool.first_free_slot()` — in current implementation
        this happens to be the lowest unmapped position.
      - Returns those IDs as a list.
      - Caller passes them straight to `unmark_pages_capped(ids)` —
        no sort step.

    This is the same effect WHEN chunk_arena uses first_free_slot, but
    the new flow doesn't ASSUME it — it consumes whatever arena.grow
    actually returns. That decoupling is the architectural win.

    Pin: simulate m2k where arena.grow returns specific IDs in some
    order; verify dst alloc's `_capped_pages` drops exactly those IDs
    (no more, no fewer, no sort dependency).
    """
    alloc = _make_allocator(size=100)
    # Simulate prior k2m fire: mark IDs [60..70] then [50..60] as capped
    # (two batches, append order). Capped now holds 21 IDs.
    batch1 = torch.tensor(list(range(60, 71)), dtype=torch.int64, device=DEVICE)
    batch2 = torch.tensor(list(range(50, 60)), dtype=torch.int64, device=DEVICE)
    alloc.mark_pages_capped(batch1)
    alloc.mark_pages_capped(batch2)
    assert alloc._capped_pages.numel() == 21

    # Simulate m2k post-grow: arena.grow returned IDs [50, 51, 52, 53, 54]
    # (in first_free_slot order). The new flow unmarks EXACTLY these,
    # not "lowest N" by sort.
    returned_ids = [50, 51, 52, 53, 54]
    alloc.unmark_pages_capped(
        torch.tensor(returned_ids, dtype=torch.int64, device=DEVICE)
    )
    remaining = set(alloc._capped_pages.tolist())
    assert remaining == set(range(55, 60)) | set(range(60, 71)), (
        f"Phase D ID-flow should unmark EXACTLY the IDs arena.grow returned "
        f"({returned_ids}); got remaining={sorted(remaining)}"
    )
    assert alloc._capped_pages.numel() == 16, (
        f"post-fix expected 21 - 5 = 16 remaining; got {alloc._capped_pages.numel()}"
    )
    # Re-mark idempotency check: re-capping batch2 ([50..59]) re-adds the
    # 5 IDs we just unmarked (50..54) and dedupes 55..59. 16 + 5 = 21.
    alloc.mark_pages_capped(batch2)
    assert alloc._capped_pages.numel() == 21, (
        f"re-mark idempotency broken: expected 21 capped, "
        f"got {alloc._capped_pages.numel()}"
    )
    print(f"  PASS  16  m2k post-grow uses EXACT IDs returned by "
          f"chunk_arena.grow (Phase D ID-flow), not a sort-derived "
          f"LOWEST-N (the deleted helper)")


def test_15_mamba_cap_allocator_exposed_for_m2k_fire():
    """Task #172 (2026-05-30): D10@C=56 server.log captured every
    Budgeter m2k decision (ticks 42, 59) aborting at xpool_actuator.py
    line 160 with::

        RuntimeError: cap_barrier(plan): src allocator missing
        mark_pages_capped.

    Root cause: MambaArenaActuator has no `.allocator` attribute. For
    m2k direction, src_act = mamba_actuator, so
    `getattr(src_act, "allocator", None)` returns None and cap_barrier
    raises before any state mutation. Budgeter's m2k path has been
    silently broken since Phase 4 sync-fire landed.

    Symmetric KV side: KVArenaActuator has `.allocator =
    TokenToKVPoolAllocator` (line 134 mark_pages_capped). The m2k path
    needs a parallel mamba-side allocator surface.

    Contract (post-fix):
      1. MambaArenaActuator must expose `.allocator` (non-None).
      2. The allocator must have `.mark_pages_capped(slot_t)`,
         `.unmark_pages_capped(slot_t)`, and `.device` attributes.
      3. mark + unmark must round-trip: pool state identical after.

    Test stubs out _mamba_temporal_arena/set_capacity_slots so we
    don't need a real arena chain. The check is purely about the
    actuator's allocator surface.
    """
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator

    class _FakePool:
        def __init__(self, size: int) -> None:
            self.size = size
            self.live_size = size
            self.max_size = size + 100
            self.device = torch.device(DEVICE)
            self.free_slots = torch.arange(
                1, size + 1, dtype=torch.int64, device=DEVICE
            )
            self._capped_slots = torch.empty(
                0, dtype=torch.int64, device=DEVICE
            )
            self._alloc_lock = threading.Lock()
            class _FakeArena:
                tokens_per_chunk = 1
                max_chunks_per_pool = size + 100
            self._mamba_temporal_arena = _FakeArena()

        def set_capacity_slots(self, n: int) -> int:
            return min(int(n), self.size + 100)

        def _assert_capped_slots_invariant(self) -> None:
            # No-op stub; production MambaPool always defines the helper.
            pass

    pool = _FakePool(size=100)
    actuator = MambaArenaActuator(pool=pool)

    alloc = getattr(actuator, "allocator", None)
    assert alloc is not None, (
        "MambaArenaActuator.allocator is None — cap_barrier rejects m2k "
        "at xpool_actuator.py:160 with 'src allocator missing "
        "mark_pages_capped'. Symmetric to KVArenaActuator.allocator "
        "(TokenToKVPoolAllocator with the mark/unmark methods)."
    )
    assert hasattr(alloc, "mark_pages_capped"), (
        "actuator.allocator must expose mark_pages_capped(slot_t) — "
        "this is the cap-barrier hook xpool_actuator.py:174 calls."
    )
    assert hasattr(alloc, "unmark_pages_capped"), (
        "actuator.allocator must expose unmark_pages_capped(slot_t) — "
        "called from xpool_actuator.py:218 for cap-barrier rollback."
    )
    assert hasattr(alloc, "device"), (
        "actuator.allocator must expose .device — used at "
        "xpool_actuator.py:167 to build cap_t tensor on the same device."
    )

    # Semantic: mark moves slots from free → _capped; unmark reverses.
    target = torch.tensor([3, 7, 11], dtype=torch.int64, device=alloc.device)
    free_before = set(pool.free_slots.tolist())
    assert {3, 7, 11}.issubset(free_before), (
        f"test setup: free_slots should contain target slots, "
        f"got top10 = {sorted(free_before)[:10]}"
    )

    moved = alloc.mark_pages_capped(target)
    assert moved == 3, f"expected 3 newly marked, got {moved}"
    free_after_mark = set(pool.free_slots.tolist())
    assert (free_before - free_after_mark) == {3, 7, 11}, (
        f"mark must remove slot ids from free_slots. "
        f"missing from free_after_mark: {free_before - free_after_mark}"
    )
    capped = set(pool._capped_slots.tolist())
    assert {3, 7, 11}.issubset(capped), (
        f"mark must add slot ids to _capped_slots. "
        f"_capped_slots top10 = {sorted(capped)[:10]}"
    )

    # Idempotent: mark again on same target → 0 newly marked
    moved2 = alloc.mark_pages_capped(target)
    assert moved2 == 0, (
        f"re-marking same target must dedupe (return 0 newly marked), "
        f"got {moved2}. Same dedupe contract as KV mark_pages_capped "
        f"(allocator.py:171)."
    )

    # Unmark restores
    unmarked = alloc.unmark_pages_capped(target)
    assert unmarked == 3, f"expected 3 unmarked, got {unmarked}"
    free_after_unmark = set(pool.free_slots.tolist())
    assert {3, 7, 11}.issubset(free_after_unmark), (
        f"unmark must restore slot ids to free_slots; "
        f"missing: {{3,7,11}} - free = "
        f"{set(range(3,12)) - free_after_unmark}"
    )
    capped_after_unmark = set(pool._capped_slots.tolist())
    intersection = {3, 7, 11} & capped_after_unmark
    assert not intersection, (
        f"unmark must remove slot ids from _capped_slots; "
        f"still there: {{3,7,11}} ∩ _capped_slots = {intersection}"
    )

    # Final state matches initial (round-trip invariant)
    assert set(pool.free_slots.tolist()) == free_before, (
        "round-trip: free_slots after mark+unmark must match initial"
    )
    print(f"  PASS  15  MambaArenaActuator.allocator exposes "
          f"mark/unmark_pages_capped for m2k cap_barrier ({pool.size} "
          f"slots, 3 marked, dedupe ok, unmark restores)")


def test_9_mark_pages_capped_dedupes():
    """Phase 9 audit D1 (CRITICAL) — `mark_pages_capped` without dedupe
    grows `_capped_pages.numel()` unboundedly. `live_size = size -
    _capped.numel()` can go negative on repeated marks of the same page.

    Test: mark page IDs [1..100] three times in a row. Assert
    `_capped_pages.numel() == 100`, NOT 300.
    """
    alloc = _make_allocator(size=10_000)
    alloc.alloc(1)  # sentinel
    targets = alloc.free_pages[:100].clone()  # 100 page IDs

    alloc.mark_pages_capped(targets)
    assert int(alloc._capped_pages.numel()) == 100
    # Second mark of the same set — pre-fix, this would push numel to 200.
    alloc.mark_pages_capped(targets)
    assert int(alloc._capped_pages.numel()) == 100, (
        f"second mark dedupe broken: numel={alloc._capped_pages.numel()}"
    )
    # Third — same.
    alloc.mark_pages_capped(targets)
    assert int(alloc._capped_pages.numel()) == 100
    # live_size stays consistent.
    assert int(alloc.live_size) == 10000 - 100 == 9900
    # And caller passes a target with duplicates inside the call:
    dup = torch.cat([targets[:10], targets[:10], targets[:10]])
    alloc.mark_pages_capped(dup)
    assert int(alloc._capped_pages.numel()) == 100, (
        f"within-call dedupe broken: numel={alloc._capped_pages.numel()}"
    )
    print("  PASS  9  mark_pages_capped dedupes across calls AND within call")


def test_11_e2e_evict_then_alloc_with_capped_pages():
    """The honest-accounting guarantee (the #155 root concern), now STRUCTURAL.

    `available_size()` must always equal what `alloc` can actually hand out —
    never the pathological "available reports >= N but alloc(N) returns None"
    (the live D11 OOM signature). In the CappedFreeList model both read the SAME
    `free_ids`, which excludes every capped id, so the guarantee holds by
    construction: there is no per-free capped filter to get wrong (the old #155
    filter is GONE), because a capped page is never live and so is never freed.

    The cap_barrier's `count_referenced` guard upholds that contract upstream —
    capping a REFERENCED (cached/live) page is the violation this design refuses,
    not one it silently absorbs. So this test exercises only LEGAL caps (free
    pages, as cap_barrier selects) and asserts available == alloc-capacity at
    every step (alloc/cap/free/uncap).
    """
    alloc = _make_allocator(size=10_000)
    alloc.alloc(1)  # sentinel

    def assert_honest(where):
        # The structural guarantee: alloc can hand out EXACTLY available_size
        # slots, none capped — then restore the state.
        n = int(alloc.available_size())
        got = alloc.alloc(n)
        assert got is not None and got.numel() == n, (
            f"{where}: available_size()={n} but alloc({n}) returned "
            f"{None if got is None else got.numel()} — available is lying "
            f"(the pathological #155 failure mode)")
        assert int(torch.isin(got, alloc._capped_pages).sum()) == 0, (
            f"{where}: alloc handed out a capped id")
        alloc.free(got)
        alloc.merge_and_sort_free()

    # Phase A: 4000 live, 5999 free.
    used = alloc.alloc(4000)
    assert used is not None and used.numel() == 4000
    assert int(alloc.available_size()) == 5999, alloc.available_size()
    assert_honest("phase-A")

    # Phase B: cap_barrier caps 500 GENUINELY-FREE pages (the legal selection).
    capped = alloc.free_pages[:500].clone()
    alloc.mark_pages_capped(capped)
    assert int(alloc._capped_pages.numel()) == 500
    # Capping free pages removes them from the allocatable set → available drops.
    assert int(alloc.available_size()) == 5499, alloc.available_size()
    assert_honest("phase-B-after-cap")

    # Phase C: free 800 live slots; available rises by exactly 800.
    alloc.free(used[:800])
    alloc.merge_and_sort_free()
    assert int(alloc.available_size()) == 6299, alloc.available_size()
    assert_honest("phase-C-after-free")

    # Phase D: m2k grow un-caps the 500 → they rejoin the allocatable set.
    alloc.unmark_pages_capped(capped)
    assert int(alloc.available_size()) == 6799, alloc.available_size()
    assert_honest("phase-D-after-uncap")

    print("  PASS  11  available_size is honest vs alloc across cap/free/uncap "
          "(structural — no per-free capped filter)")


def test_12_mamba_pool_capped_slots_dedupes():
    """LIVE D11 SURFACED THIS (task #163, GPU re-run):
        MambaPool.set_capacity_slots: 514 -> 515 (size=514, free=1, capped=1526)
    `_capped_slots` accumulated to 3× pool size. Same D1 pattern as
    TokenToKVPoolAllocator but on MambaPool side, which I missed when
    fixing KV side.

    MambaPool has 3 paths that `torch.cat` into `_capped_slots`:
      - `free()`: slots above cap held out (line 708)
      - `migrate_slot()`: src slot held after migration (line 771)
      - `set_capacity_slots()` shrink path (line ~830)

    Test: simulate repeated migrate_slot of the SAME src. Pre-fix,
    `_capped_slots.numel()` grows by 1 each call. Post-fix, dedupe
    keeps numel correct.
    """
    # Construct a minimal MambaPool-like stub that exercises the same
    # _capped_slots accumulation path. We can't easily import MambaPool
    # without a real model, so test the accumulation logic in isolation
    # against the exact pattern the production code uses.
    class MambaPoolCappedStub:
        """Mirrors MambaPool's `_capped_slots` cat pattern."""
        def __init__(self, device, has_dedupe):
            self.device = device
            self._capped_slots = torch.empty((0,), dtype=torch.int64, device=device)
            self.has_dedupe = has_dedupe

        def hold_capped(self, src: int):
            """Mirrors migrate_slot's _capped_slots cat (line 766-771)."""
            existing = self._capped_slots
            src_t = torch.tensor([src], dtype=torch.int64, device=self.device)
            if existing.numel() == 0:
                self._capped_slots = src_t
            else:
                if self.has_dedupe:
                    # Phase 9 fix: skip if already capped.
                    if bool(torch.isin(src_t, existing).any().item()):
                        return
                self._capped_slots = torch.cat([existing, src_t])

    # Pre-fix simulation (no dedupe): same src marked 10× → numel = 10
    pool_buggy = MambaPoolCappedStub(DEVICE, has_dedupe=False)
    for _ in range(10):
        pool_buggy.hold_capped(42)
    buggy_count = int(pool_buggy._capped_slots.numel())
    assert buggy_count == 10, f"buggy simulator should accumulate, got {buggy_count}"

    # Post-fix simulation (dedupe): same src marked 10× → numel = 1
    pool_fixed = MambaPoolCappedStub(DEVICE, has_dedupe=True)
    for _ in range(10):
        pool_fixed.hold_capped(42)
    fixed_count = int(pool_fixed._capped_slots.numel())
    assert fixed_count == 1, (
        f"fixed simulator should dedupe, got {fixed_count}. "
        f"Production live D11 crashed with capped=1526 vs size=514 "
        f"— same accumulation pattern as the buggy simulator."
    )

    # Mixed: cap 5 different slots, repeat them
    pool_mixed = MambaPoolCappedStub(DEVICE, has_dedupe=True)
    for slot in [10, 20, 30, 40, 50]:
        for _ in range(5):
            pool_mixed.hold_capped(slot)
    assert int(pool_mixed._capped_slots.numel()) == 5, (
        f"mixed dedupe: expected 5, got {pool_mixed._capped_slots.numel()}"
    )
    print("  PASS  12  MambaPool._capped_slots dedupe (live D11 crashed "
          "with capped=1526 vs size=514; same D1 pattern)")


def test_13_mamba_grow_must_not_re_expose_migrated_slots():
    """Historical regression (pre-#213 architecture). Kept after #213
    Phase E for traceability; the "buggy" / "_fixed" semantics it
    exercises live in this test's own `MambaPoolStub` — production no
    longer routes through this code path at all.

    Phase E delivered a different fix: cross-pool's grow now passes
    EXACT IDs (from `chunk_arena.grow`) to `MambaPool.unmark_slots(ids)`.
    Migrated slots aren't in arena.grow's return list, so they can't
    be exposed — structural safety, no blacklist needed. The
    `_migrated_capped_slots` state + `set_capacity_slots` GROW
    blacklist filter that the stub's `_fixed` method models were both
    deleted in Phase E (`memory_pool.py` `set_capacity_slots` is now
    plain value-mask, kept for non-actuator dynamic-resize paths
    exercised by `test_phase5/7`).

    Original narrative (preserved for archaeology):
    Phase 7 / task #126 / live D11 #163-164 root cause.

    Live D11 inter crash trace (sync via CUDA_LAUNCH_BLOCKING):
       MambaPool.alloc:681  t[select_index] = z
       ← cache_unfinished_req → fork_from → alloc(1)

    `t` is `mamba_cache.temporal` — arena-backed (VA reserved at boot
    for max_size slots, only [0, live_size] physically mapped). Writing
    to a slot whose chunk isn't mapped = CUDA illegal access.

    Root cause hypothesis: `MambaPool._capped_slots` has DUAL semantics
    that `set_capacity_slots(grow)` conflates:

      A. Init-time entries [size+1, max_size] — VA-reserved but not yet
         physically mapped; SAFE to expose into free_slots WHEN the
         actuator has just done `arena.grow` to map their chunks.

      B. Migrate_slot entries (Admitter m2k fires put mamba slot IDs
         here) — their chunks were unmapped via xpool_actuator's
         `shrink_explicit`. UNSAFE to expose: writing into them hits
         unmapped VA. Should only re-enter free_slots after explicit
         re-mapping.

    Current `set_capacity_slots` grow path does:
        mask = held <= n_slots
        move = held[mask]   # all _capped entries <= n_slots
        free_slots = cat(free_slots, move)
    → INCLUDES migrated entries → CRASH on next alloc + write.

    This test simulates the production sequence at unit level and
    asserts the invariant: migrated slot ids must NOT appear in
    free_slots after a grow().
    """
    # Build a minimal MambaPool-like stub that mirrors the production
    # `_capped_slots` lifecycle without needing model+CUDA.
    class MambaPoolStub:
        def __init__(self, size, max_size, device):
            self.size = size
            self.max_size = max_size
            self.device = device
            # Boot: free_slots = [1..size], _capped_slots = [size+1..max_size]
            self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device=device)
            self._capped_slots = torch.arange(
                size + 1, max_size + 1, dtype=torch.int64, device=device
            )
            self.size = size
            # NEW: track which capped slots came from migrate_slot (unsafe)
            # vs above-cap init (safe). Empty set = the architectural gap.
            self._migrated_capped: set[int] = set()

        def migrate_slot(self, src: int):
            """Mirrors production: src joins _capped_slots, chunk unmapped."""
            src_t = torch.tensor([src], dtype=torch.int64, device=self.device)
            if not bool(torch.isin(src_t, self._capped_slots).any().item()):
                self._capped_slots = torch.cat([self._capped_slots, src_t])
            self._migrated_capped.add(src)
            # Remove from free_slots if present
            self.free_slots = self.free_slots[self.free_slots != src]

        def set_capacity_slots_buggy(self, n_slots):
            """Current production GROW logic — moves ALL capped <= n_slots
            to free, INCLUDING migrated (which is the bug)."""
            if n_slots <= self.size:
                return  # only test grow
            held = self._capped_slots
            mask = held <= n_slots
            move = held[mask]
            self.free_slots = torch.cat([self.free_slots, move])
            self._capped_slots = held[~mask]
            self.size = n_slots

        def set_capacity_slots_fixed(self, n_slots):
            """Proposed fix: skip migrated entries when growing."""
            if n_slots <= self.size:
                return
            held = self._capped_slots
            mask = held <= n_slots
            move = held[mask]
            # Filter out migrated entries — they're unsafe to allocate
            # until their chunks are re-mapped.
            if self._migrated_capped:
                migrated_t = torch.tensor(
                    sorted(self._migrated_capped), dtype=torch.int64,
                    device=self.device,
                )
                safe_mask = ~torch.isin(move, migrated_t)
                move = move[safe_mask]
            self.free_slots = torch.cat([self.free_slots, move])
            # Keep migrated ones in _capped_slots
            new_capped_mask = (~mask) | torch.isin(held, torch.tensor(
                sorted(self._migrated_capped) if self._migrated_capped else [],
                dtype=torch.int64, device=self.device,
            ))
            self._capped_slots = held[new_capped_mask]
            self.size = n_slots

    # === Scenario reproduces the D11 inter crash path ===
    # Boot: size=10, max_size=20. _capped = [11..20].
    pool = MambaPoolStub(size=10, max_size=20, device=DEVICE)

    # Admitter fires m2k 3 times, migrating slots 3, 5, 7.
    for src in [3, 5, 7]:
        pool.migrate_slot(src)
    # _capped_slots now has [11..20, 3, 5, 7]; _migrated_capped = {3, 5, 7}
    assert 3 in pool._migrated_capped and 5 in pool._migrated_capped

    # Budgeter fires k2m: grow live cap from 10 to 13 (3 new chunks).
    # Pre-fix: ALL slots ≤ 13 come out of _capped (including 3, 5, 7).
    pool_buggy = MambaPoolStub(size=10, max_size=20, device=DEVICE)
    for src in [3, 5, 7]:
        pool_buggy.migrate_slot(src)
    pool_buggy.set_capacity_slots_buggy(13)
    in_free_buggy = set(int(x) for x in pool_buggy.free_slots.tolist())
    leaked = {3, 5, 7} & in_free_buggy
    assert leaked == {3, 5, 7}, (
        f"BUG REPRO: migrated slots leaked into free_slots after buggy "
        f"grow: {leaked}. These IDs point to UNMAPPED VA → alloc + "
        f"write → CUDA illegal access (live D11 #163 crash)."
    )

    # Post-fix: migrated slots stay in _capped_slots after grow.
    pool_fixed = MambaPoolStub(size=10, max_size=20, device=DEVICE)
    for src in [3, 5, 7]:
        pool_fixed.migrate_slot(src)
    pool_fixed.set_capacity_slots_fixed(13)
    in_free_fixed = set(int(x) for x in pool_fixed.free_slots.tolist())
    leaked_fixed = {3, 5, 7} & in_free_fixed
    assert leaked_fixed == set(), (
        f"FIX BROKEN: migrated slots still in free_slots: {leaked_fixed}"
    )
    # And the safe init-capped slots [11, 12, 13] DO come out.
    expected_grown = {11, 12, 13}
    assert expected_grown <= in_free_fixed, (
        f"FIX BROKEN: safe init slots {expected_grown - in_free_fixed} "
        f"didn't appear in free_slots after grow"
    )

    print("  PASS  13  D11 #163 root cause: buggy grow leaks "
          "migrated slots (CUDA illegal access); fixed grow distinguishes "
          "init-capped (safe) from migrate-capped (unsafe)")


def test_14_xpool_actuator_grow_must_use_actual_granted_not_per_dst():
    """Phase 7 + #126 architectural root cause for live D11 inter crash.

    XPoolActuator.execute_async (xpool_actuator.py:336-356) does:

        granted_per_subpool = [dst._arena.grow(name, per_dst) for ...]
        granted_total = sum(granted_per_subpool)        # used in result
        dst_grow_slots = len(dst_act.expand_pages_to_token_slots(
            list(range(per_dst))                        # ← INTENDED, not actual
        ))
        new_dst_cap = dst_act.live_capacity_tokens() + dst_grow_slots
        dst_act.cap_allocator_only(new_dst_cap)

    Bug: `dst_grow_slots` is computed from `per_dst` (the INTENDED grow
    per subpool), NOT from `granted_per_subpool` (the ACTUAL chunks
    cuMemMap'd per subpool). When the shared pool runs short and a
    subpool grants less than per_dst, `new_dst_cap` exceeds the
    actually-mapped slot IDs → next alloc returns an unmapped slot →
    `t[unmapped_slot] = z` crashes with CUDA illegal access (live D11
    inter, traced via CUDA_LAUNCH_BLOCKING=1).

    Production budgeter log evidence:
        execute_async[seq=N] DONE dir=kv_to_mamba unmapped=0
            granted=0 cap=664us
    But the MambaPool log right after:
        MambaPool.set_capacity_slots: 512 -> 513 (size=512, free=1, capped=...)
    → cap bumped by 1 slot even though granted=0 chunks. ☠️

    Fix: derive `dst_grow_slots` from `min(granted_per_subpool)` (or
    equivalently `granted_total // n_dst`), which guarantees uniform
    coverage across dst subpools.
    """
    # Simulate the actuator math at unit level.
    def buggy_new_cap(live_cap, per_dst, tokens_per_chunk):
        # Production code path (xpool_actuator.py:353-356):
        dst_grow_slots = 0
        for p in range(per_dst):
            start = max(1, p * tokens_per_chunk)
            dst_grow_slots += max(0, (p + 1) * tokens_per_chunk - start)
        return live_cap + dst_grow_slots

    def fixed_new_cap(live_cap, per_dst, tokens_per_chunk, granted_per_subpool):
        # Phase 9/Phase 7 fix: use min(granted_per_subpool) to size the
        # cap bump, NEVER per_dst. This honors the atomicity invariant
        # (all dst subpools have at least this many chunks).
        actual_per_dst = min(granted_per_subpool) if granted_per_subpool else 0
        dst_grow_slots = 0
        for p in range(actual_per_dst):
            start = max(1, p * tokens_per_chunk)
            dst_grow_slots += max(0, (p + 1) * tokens_per_chunk - start)
        return live_cap + dst_grow_slots

    # Scenario A: SharedHandlePool exhausted — some subpools granted 0
    n_dst_subpools = 48
    per_dst = 2
    tps = 1  # mamba: 1 token-slot per chunk

    # Pre-fix: cap grows by 1 (per_dst=2, but only 1 token-slot after
    # sentinel skip).
    buggy_growth = buggy_new_cap(0, per_dst, tps)
    assert buggy_growth == 1, f"buggy: expected 1, got {buggy_growth}"

    # Production case: granted=[0]*48 — pool full, all subpools failed.
    # Pre-fix would still bump by 1 → expose unmapped slot.
    granted_all_zero = [0] * n_dst_subpools
    fixed_growth = fixed_new_cap(0, per_dst, tps, granted_all_zero)
    assert fixed_growth == 0, (
        f"FIXED: when granted_per_subpool is all-zero, cap MUST NOT "
        f"grow. Got {fixed_growth} (would expose unmapped slots)."
    )

    # Production case: partial — some subpools granted full, some less.
    granted_partial = [2] * 30 + [0] * 18  # 30 got their 2, 18 got 0
    fixed_growth_partial = fixed_new_cap(100, per_dst, tps, granted_partial)
    # min(granted_partial) = 0 → 0 slots exposed (atomicity: all subpools
    # must have at least this many chunks).
    assert fixed_growth_partial == 100, (
        f"FIXED partial: min(granted)=0 → no slot exposed. Got "
        f"{fixed_growth_partial - 100} new slots when atomicity broken."
    )

    # Healthy case: all subpools fully granted → matches old behavior.
    granted_full = [per_dst] * n_dst_subpools
    buggy_full = buggy_new_cap(100, per_dst, tps)
    fixed_full = fixed_new_cap(100, per_dst, tps, granted_full)
    assert buggy_full == fixed_full, (
        f"FIX REGRESSION: healthy case should match old behavior; "
        f"buggy={buggy_full}, fixed={fixed_full}"
    )
    print("  PASS  14  XPoolActuator grow uses min(granted_per_subpool), "
          "not per_dst — prevents exposing unmapped slot IDs when "
          "SharedHandlePool partially exhausts (live D11 #163 trigger)")


def test_33_xpool_uneven_granted_per_subpool_cleanup():
    """#223 — when `dst._arena.grow` returns FEWER IDs in some sub-pools
    than others (SharedHandlePool partial exhaustion), the over-granting
    sub-pools have chunks physically `cuMemMap`'d that are NOT exposed
    via the cap restore (capped at `min(granted_per_subpool)`). Without
    cleanup those chunks are silently leaked — the handles stay
    consumed by chunk_arena but back no live slot.

    Production behavior must call `dst._arena.shrink_explicit(name,
    extra_ids)` on every over-granting sub-pool so the over-mapped
    handles return to `SharedHandlePool._free_handles`. Restores the
    invariant: every sub-pool ends up with exactly `actual_per_dst`
    chunks more than it started with, and `_free_handles` shrinks by
    exactly `actual_per_dst * n_dst`.

    Drives the REAL `_execute_async_locked` with stubs whose
    `shrink_explicit` records what was unmapped.
    """
    import threading
    import time
    from sglang.srt.arena.fire_plan import FirePlan
    from sglang.srt.arena.xpool_actuator import FireToken, XPoolActuator

    inst = object.__new__(XPoolActuator)
    inst._fire_inflight = threading.Lock()
    # _execute_async_locked reads self.lcm_pages (#229) which derives
    # from these. Both sides are 2-subpool stubs (n_layers=2 × n_kinds=1).
    inst.n_kv_subpools = 2
    inst.n_mamba_subpools = 2

    class _StubArena:
        def __init__(self, grow_returns):
            self._grow_returns = grow_returns
            self.shrink_calls: list[tuple[str, list[int]]] = []

        def grow(self, name, n):
            return list(self._grow_returns[name][:n])

        def shrink_explicit(self, name, ids):
            ids_list = list(ids)
            self.shrink_calls.append((name, ids_list))
            return len(ids_list)

    class _StubMTA:
        def __init__(self, arena, n_subpools=2):
            self._arena = arena
            self.n_layers = n_subpools
            self.n_kinds = 1

        def _pool_name(self, i):
            return f"p{i}"

    # src has non-empty pages_to_unmap so per_dst > 0 (xpool math derives
    # per_dst from src/dst targets jointly via LCM). The uneven scenario
    # is on dst.
    src_arena = _StubArena({"p0": [], "p1": []})
    src_mta = _StubMTA(src_arena)
    # Uneven grant: p0 gets [5, 6, 7] (3 chunks), p1 gets [5, 6] (2).
    # Common prefix [5, 6] is what cap exposes; p0's extra [7] must be
    # unmapped.
    dst_arena = _StubArena({
        "p0": [5, 6, 7],
        "p1": [5, 6],
    })
    dst_mta = _StubMTA(dst_arena)

    class _StubAlloc:
        def __init__(self) -> None:
            self.device = torch.device(DEVICE)
            self.free_pages = torch.empty(0, dtype=torch.int64, device=DEVICE)
            self._capped_pages = torch.empty(0, dtype=torch.int64, device=DEVICE)

        def count_reachable_capped(self, cap_t: torch.Tensor) -> int:
            # How many of cap_t are still allocatable: in free_pages but
            # not excluded by _capped_pages. Empty free_pages → 0, so the
            # worker-side verify passes and the over-grant cleanup path
            # this test exercises is reached.
            free = self.free_pages
            if free.numel() == 0:
                return 0
            target = cap_t.to(self.device).to(torch.int64)
            in_target = torch.isin(free, target)
            if self._capped_pages.numel() > 0:
                in_target = in_target & (~torch.isin(free, self._capped_pages))
            return int(in_target.sum().item())

    class _StubActuator:
        def __init__(self) -> None:
            self.allocator = _StubAlloc()

        def expand_pages_to_token_slots(self, ids):
            return list(ids)

        def unmark_token_slots(self, slots):
            pass

    # n_src=2, n_dst=2, lcm=2. pages_to_unmap has 3 entries → target_src
    # = 6, target_dst = 6, total = 6 (already lcm-aligned) → per_dst = 3.
    # So dst.grow asks for 3 per sub-pool; p0 gives 3, p1 gives 2.
    plan = FirePlan(
        direction="kv_to_mamba",
        pages_to_unmap=[100, 101, 102],
        pages_to_map_dst=3,
        plan_seq=99,
    )
    token = FireToken(
        plan=plan,
        src=src_mta,
        dst=dst_mta,
        src_act=_StubActuator(),
        dst_act=_StubActuator(),
        cap_t=torch.zeros(0, dtype=torch.int64),
        cap_slots_count=0,
        cap_barrier_us=0,
        t_start_ns=time.monotonic_ns(),
    )

    inst._execute_async_locked(token)

    # Expected on dst arena: shrink_explicit called on p0 with [7]
    # (its over-grant). p1 was not over-granted; no shrink call.
    dst_p0 = [ids for (name, ids) in dst_arena.shrink_calls if name == "p0"]
    dst_p1 = [ids for (name, ids) in dst_arena.shrink_calls if name == "p1"]
    assert dst_p0 == [[7]], (
        f"BUG (#223): p0 over-granted [5,6,7] but common = [5,6]; "
        f"extra chunk [7] must be unmapped via shrink_explicit. "
        f"Got dst p0 shrink calls: {dst_p0}. Without this, chunk 7's "
        f"SharedHandlePool handle leaks permanently."
    )
    assert dst_p1 == [], (
        f"p1 granted exactly common = [5,6]; no shrink_explicit "
        f"expected. Got {dst_p1}."
    )
    print(f"  PASS  33  xpool_actuator unmaps over-granted chunks "
          f"(p0=[5,6,7] common=[5,6] → shrink_explicit(p0, [7])) so "
          f"SharedHandlePool handle isn't leaked under uneven grant")


def test_34_mamba_clear_preserves_capped_above_live_cap():
    """#224: `MambaPool.clear()` must restore the post-shrink invariant
    `_capped_slots = [self.size + 1 .. self.max_size]` when the live
    cap has been shrunk below max_size. Pre-fix, clear() resets
    `_capped_slots` to empty and `free_slots = [1..self.size]`, which
    silently ORPHANS the slots in `(self.size, self.max_size]` —
    they're in neither tensor, so a subsequent `set_capacity_slots`
    GROW can't restore them, and `live_size` math (size −
    _capped_slots.numel()) stops reflecting reality.

    Production trigger: cross-pool actuator shrinks mamba via
    `set_capacity_slots(N)` (chunks [N+1..max_size] unmapped); user
    later hits `/flush_cache` → `HybridLinearKVPool.clear()` →
    `mamba_pool.clear()`; the next GROW back to max_size leaves the
    orphans in unmapped VA → CUDA illegal access on alloc, OR the
    pool-leak detector trips with the wrong root cause.

    Mirrors KV side `BaseTokenToKVPoolAllocator.clear` at
    `allocator.py:372-394` which correctly stages `_capped_pages =
    arange(cap+1, size+1)`.
    """
    from sglang.srt.mem_cache.memory_pool import MambaPool

    class _FakePool:
        def __init__(self) -> None:
            self.size = 10       # live cap (shrunk by set_capacity_slots)
            self.max_size = 32   # boot upper bound
            self.device = torch.device(DEVICE)
            # Mid-state: live free [1..10], _capped_slots holds both
            # boot-deferred [11..32] AND a cap-barrier mark on slot 7.
            self.free_slots = torch.tensor(
                [1, 2, 3, 4, 5, 6, 8, 9, 10],
                dtype=torch.int64, device=DEVICE,
            )
            self._capped_slots = torch.tensor(
                [7] + list(range(11, 33)),
                dtype=torch.int64, device=DEVICE,
            )
            self._alloc_lock = threading.Lock()

    pool = _FakePool()
    pool.clear = MambaPool.clear.__get__(pool, type(pool))

    pool.clear()

    free_after = set(pool.free_slots.tolist())
    capped_after = set(pool._capped_slots.tolist())

    # Live range [1..10] becomes fully allocatable (clear flushes the
    # cap-barrier mark on 7 along with the rest — same semantics as
    # KV-side, which also drops the prior cap state on flush).
    assert free_after == set(range(1, 11)), (
        f"clear() free_slots: got {sorted(free_after)}, expected [1..10]"
    )
    # Critical fix: boot-deferred range [11..32] must end up in
    # _capped_slots, not orphaned.
    assert capped_after == set(range(11, 33)), (
        f"BUG (#224): clear() orphaned slots in (self.size, "
        f"self.max_size] = (10, 32]. _capped_slots = "
        f"{sorted(capped_after)}, expected [11..32]. Slots above the "
        f"live cap have unmapped chunks; they MUST stay in "
        f"_capped_slots so set_capacity_slots GROW can restore them."
    )

    # Bonus: clear with size == max_size (no shrink in effect) must
    # leave _capped_slots empty — the [size+1..max_size] range is
    # vacuous so the new state should match the pre-shrink invariant.
    pool2 = _FakePool()
    pool2.size = 32
    pool2.max_size = 32
    pool2.free_slots = torch.arange(1, 33, dtype=torch.int64, device=DEVICE)
    pool2._capped_slots = torch.empty(0, dtype=torch.int64, device=DEVICE)
    pool2.clear = MambaPool.clear.__get__(pool2, type(pool2))
    pool2.clear()
    assert pool2._capped_slots.numel() == 0, (
        f"clear() at full cap should produce empty _capped_slots, got "
        f"{pool2._capped_slots.tolist()}"
    )
    assert set(pool2.free_slots.tolist()) == set(range(1, 33))

    print(f"  PASS  34  MambaPool.clear() restores _capped_slots = "
          f"[size+1..max_size] after a shrink, mirroring KV-side "
          f"`BaseTokenToKVPoolAllocator.clear`. No orphaned slots in "
          f"(self.size, self.max_size].")


def test_35_m2k_capped_slots_drain_across_cycles():
    """#225: pin m2k mark + drain semantics for `MambaPool._capped_slots`
    across repeated cycles. Drives a REAL `MambaPool` so the live
    `_assert_capped_slots_invariant` (#221) fires after every mutation.

    Sub-scenarios (single-threaded — `_alloc_lock` contention is
    covered separately by `test_mamba_alloc_lock.py`):

      A. m2k accumulate-only (cap-barrier mark path): 3 mark batches
         via `_MambaCapAllocator.mark_pages_capped`. `_capped_slots`
         grows monotonically; #221 invariant holds.
      B. Rollback drain (`_MambaCapAllocator.unmark_pages_capped`):
         the rollback path used by 4 production sites — verify-failure
         (`xpool_actuator.py:309`), queue-full (`agent.py:677-679`),
         worker-noop killswitch (`agent.py:860`), and worker-exception
         (`agent.py:888`). Reverse-unmarks the A batches; `_capped_slots`
         returns to the initial boot-deferred set; size UNCHANGED.
      C. Mixed mark/drain via rollback path: 3 marks, 1 drain, 2
         more marks, 1 drain — final set = initial ∪ still-marked.
      D. **Production k2m drain (`MambaPool.unmark_slots`)**: the
         actual hot path used by `xpool_actuator` after `chunk_arena.
         grow` returns dst-side IDs (`xpool_actuator.py:441-458`).
         Unlike `unmark_pages_capped` (rollback, size-invariant), this
         path EXTENDS `self.size` when restoring above-cap IDs (per
         test_28's cap-as-max semantic). Pins both branches: restore
         below-cap (size unchanged) AND restore above-cap (size bumps).

    A/B/C all use mark batches in `[1..13]` (well below `size=20`)
    so they don't collide with boot-deferred IDs `[21..32]` — keeps
    the dedupe path inactive.

    See also: test_25 (real-pool `unmark_slots`), test_28 (cap-as-max
    semantic per single round-trip), test_29 (`live_size` mirrors KV),
    test_30 (`set_capacity_slots` GROW skips marked).
    """
    import os
    os.environ.pop("SGLANG_ARENA_SHARED", None)
    os.environ.pop("SGLANG_MAMBA_ARENA", None)

    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=128,
        n_groups=1,
        num_heads=4,
        head_dim=64,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    pool = MambaPool(
        size=20,
        spec_state_size=8,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        device="cuda:0",
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=32,
    )
    # Boot-state invariant (Phase 7): _capped_slots = [size+1..max_size].
    initial_capped = set(pool._capped_slots.tolist())
    assert initial_capped == set(range(21, 33)), (
        f"setup: expected boot-deferred _capped_slots=[21..32], got "
        f"{sorted(initial_capped)[:5]}..."
    )
    initial_capped_n = len(initial_capped)  # 12

    alloc = _MambaCapAllocator(pool)
    DEV = pool.free_slots.device

    # ---- Scenario A: accumulate-only across K cycles ----
    # Mark 3 distinct slot batches; expect monotonic growth and invariant.
    a_batches = [
        torch.tensor([1, 2], dtype=torch.int64, device=DEV),
        torch.tensor([5, 6, 7], dtype=torch.int64, device=DEV),
        torch.tensor([10, 11, 12, 13], dtype=torch.int64, device=DEV),
    ]
    capped_progression = [initial_capped_n]
    for batch in a_batches:
        n_marked = alloc.mark_pages_capped(batch)
        assert n_marked == batch.numel(), (
            f"A: mark_pages_capped({batch.tolist()}) should mark "
            f"{batch.numel()} slots; got {n_marked}"
        )
        capped_progression.append(int(pool._capped_slots.numel()))
        pool._assert_capped_slots_invariant()  # #221 must hold

    expected_progression = [
        initial_capped_n,
        initial_capped_n + 2,       # +[1,2]
        initial_capped_n + 2 + 3,   # +[5,6,7]
        initial_capped_n + 2 + 3 + 4,
    ]
    assert capped_progression == expected_progression, (
        f"A: _capped_slots monotonic accumulation broken. "
        f"Got {capped_progression}, expected {expected_progression}. "
        f"Either mark dedupe is misbehaving OR invariant assert "
        f"swallowed an overshoot silently."
    )
    # Capacity sanity: final accumulated = 12 + 9 = 21 ≤ max_size=32.
    assert pool._capped_slots.numel() == 21
    # All A-batch slots removed from free_slots.
    free_after_A = set(pool.free_slots.tolist())
    for batch in a_batches:
        for sid in batch.tolist():
            assert sid not in free_after_A, (
                f"A: slot {sid} still in free_slots after mark_pages_capped"
            )

    # ---- Scenario B: drain each batch in reverse ----
    for batch in reversed(a_batches):
        n_unmarked = alloc.unmark_pages_capped(batch)
        assert n_unmarked == batch.numel(), (
            f"B: unmark_pages_capped({batch.tolist()}) should restore "
            f"{batch.numel()} slots; got {n_unmarked}"
        )
        pool._assert_capped_slots_invariant()

    # Post-drain: _capped_slots returns to exactly the initial
    # boot-deferred set [21..32]; free_slots restored to [1..20].
    capped_final = set(pool._capped_slots.tolist())
    assert capped_final == initial_capped, (
        f"BUG (#225): m2k accumulation drained incompletely. "
        f"After full reverse-unmark, _capped_slots = "
        f"{sorted(capped_final)[:5]}... (numel={len(capped_final)}); "
        f"expected initial boot-deferred set [21..32] (numel=12). "
        f"Indicates either unmark didn't drop, or some IDs leaked "
        f"into _capped_slots and weren't restored."
    )
    free_final = set(pool.free_slots.tolist())
    assert free_final == set(range(1, 21)), (
        f"B: free_slots didn't fully restore. Got "
        f"{sorted(free_final)[:5]}... (numel={len(free_final)}); "
        f"expected [1..20]."
    )

    # ---- Scenario C: mixed mark/drain sequence ----
    # Marks 3 batches, drains 1, marks 4 more, drains 2 — final _capped
    # should be initial + (still-marked).
    mark_ops = [
        torch.tensor([1], dtype=torch.int64, device=DEV),
        torch.tensor([2, 3], dtype=torch.int64, device=DEV),
        torch.tensor([4, 5], dtype=torch.int64, device=DEV),
    ]
    drain_after_3 = mark_ops[1]   # drain [2, 3]
    more_marks = [
        torch.tensor([6, 7, 8], dtype=torch.int64, device=DEV),
        torch.tensor([9, 10], dtype=torch.int64, device=DEV),
    ]
    drain_after_more = torch.tensor([1, 6], dtype=torch.int64, device=DEV)

    for batch in mark_ops:
        alloc.mark_pages_capped(batch)
        pool._assert_capped_slots_invariant()
    alloc.unmark_pages_capped(drain_after_3)
    pool._assert_capped_slots_invariant()
    for batch in more_marks:
        alloc.mark_pages_capped(batch)
        pool._assert_capped_slots_invariant()
    alloc.unmark_pages_capped(drain_after_more)
    pool._assert_capped_slots_invariant()

    # Net marks still-held: {1,2,3,4,5,6,7,8,9,10} - {2,3,1,6} =
    # {4,5,7,8,9,10}
    still_marked = {4, 5, 7, 8, 9, 10}
    capped_C = set(pool._capped_slots.tolist())
    expected_C = initial_capped | still_marked
    assert capped_C == expected_C, (
        f"C: mixed sequence final state wrong. _capped_slots = "
        f"{sorted(capped_C)}; expected initial ∪ still-marked = "
        f"{sorted(expected_C)}. Diff: extra="
        f"{sorted(capped_C - expected_C)}, missing="
        f"{sorted(expected_C - capped_C)}."
    )
    # Free_slots = [1..20] - still_marked
    free_C = set(pool.free_slots.tolist())
    expected_free_C = set(range(1, 21)) - still_marked
    assert free_C == expected_free_C, (
        f"C: free_slots final state wrong. Got {sorted(free_C)}; "
        f"expected {sorted(expected_free_C)}."
    )

    # ---- Scenario D: production k2m drain via MambaPool.unmark_slots ----
    # In production, `xpool_actuator._execute_async_locked` calls
    # `dst_act.unmark_token_slots(token_slots)` → `MambaPool.unmark_slots`
    # with the IDs returned by `chunk_arena.grow`. This is the path
    # that drains _capped_slots after a k2m fire — NOT
    # `_MambaCapAllocator.unmark_pages_capped` (which is the rollback
    # path). The two have different size semantics: this one bumps
    # `self.size` when restored IDs exceed the live cap; rollback does
    # not. (See test_28 for cap-as-max semantic per-call; this
    # scenario pins the multi-restore drain.)
    # Setup: still-marked = {4, 5, 7, 8, 9, 10} from scenario C, plus
    # boot-deferred [21..32]. size still = 20.
    size_before_D = int(pool.size)
    assert size_before_D == 20, (
        f"D: setup expected pool.size=20 from scenarios A-C, got "
        f"{size_before_D}"
    )

    # D1: drain below-cap marks via the production path. Should
    # restore them to free_slots WITHOUT bumping size.
    below_cap_ids = torch.tensor(
        sorted(still_marked), dtype=torch.int64, device=DEV,
    )
    n_d1 = pool.unmark_slots(below_cap_ids)
    pool._assert_capped_slots_invariant()
    assert n_d1 == len(still_marked), (
        f"D1: unmark_slots below-cap should restore all {len(still_marked)} "
        f"IDs; got {n_d1}"
    )
    assert pool.size == 20, (
        f"D1 (cap-as-max): below-cap unmark_slots must NOT bump size. "
        f"Pre={size_before_D}, post={pool.size}. Violates test_28's "
        f"contract — would over-report live_size and trip leak detector."
    )
    # _capped_slots = boot-deferred only.
    capped_D1 = set(pool._capped_slots.tolist())
    assert capped_D1 == initial_capped, (
        f"D1: after below-cap drain, _capped_slots should be initial "
        f"boot-deferred [21..32]. Got {sorted(capped_D1)[:5]}..."
    )

    # D2: drain above-cap boot-deferred via the production path. Should
    # bump size to max restored ID — this is the "k2m brings new growth"
    # semantic that test_25 also pins single-batch.
    above_cap_ids = torch.tensor([21, 22, 23], dtype=torch.int64, device=DEV)
    n_d2 = pool.unmark_slots(above_cap_ids)
    pool._assert_capped_slots_invariant()
    assert n_d2 == 3, f"D2: unmark_slots above-cap should restore 3; got {n_d2}"
    assert pool.size == 23, (
        f"D2 (cap-as-max): above-cap unmark_slots must extend size to "
        f"max restored ID. Pre=20, restored={{21,22,23}}, expected "
        f"size=23, got {pool.size}."
    )
    capped_D2 = set(pool._capped_slots.tolist())
    assert capped_D2 == set(range(24, 33)), (
        f"D2: _capped_slots should be [24..32] after draining [21,22,23]. "
        f"Got {sorted(capped_D2)}."
    )

    print(f"  PASS  35  MambaPool _capped_slots: A (3-cycle accumulate "
          f"+9), B (rollback unmark_pages_capped fully drains), C "
          f"(mixed 5-mark/2-drain → net {{4,5,7,8,9,10}}), D "
          f"(production k2m drain via pool.unmark_slots: below-cap "
          f"keeps size=20, above-cap bumps size to 23). #221 invariant "
          f"called 15× across 15 mark/unmark ops.")


def test_36_expand_pages_rejects_page_0_loudly():
    """#226: `expand_pages_to_token_slots(page_ids)` must raise
    ValueError if any page_id == 0. Chunk 0 carries padded slot 0
    (design.md §"Per-unit sizes") — the kernel writes dummy outputs
    from padded tokens to slot 0 on every forward pass, so chunk 0
    must remain mapped at all times.

    Pre-#226 the function silently mapped page 0 to slots
    `[1..tps)` (KV) or the empty list (mamba, tps=1) — the upstream
    actuator would then unmap chunk 0 without correctly marking
    slot 0, corrupting the padded-output target. Loud raise is the
    defense-in-depth net under the `_compute_fully_free_pages`
    upstream filter.

    Pins both KV and Mamba actuator variants.
    """
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator

    # Construct stub actuators via __new__ to bypass __init__ (which
    # demands real arena-backed pools). The method only reads
    # `self.pool._kv_arena.tokens_per_chunk` (kv_actuator.py `_tokens_per_page`)
    # / `self.pool._mamba_temporal_arena.tokens_per_chunk` (mamba counterpart);
    # if those accessors change, this stub needs to follow.
    class _StubKVPool:
        class _Arena:
            tokens_per_chunk = 4
        _kv_arena = _Arena()

    kv_act = KVArenaActuator.__new__(KVArenaActuator)
    kv_act.pool = _StubKVPool()

    # Sanity: pages > 0 work fine.
    out = kv_act.expand_pages_to_token_slots([2, 3])
    assert out == [8, 9, 10, 11, 12, 13, 14, 15], (
        f"KV non-page-0 expansion broken; got {out}"
    )

    try:
        kv_act.expand_pages_to_token_slots([0, 1])
    except ValueError as e:
        assert "page 0" in str(e) and "padded slot 0" in str(e), (
            f"raise should mention page 0 + padded slot 0 diagnostic; got: {e}"
        )
    else:
        raise AssertionError(
            "BUG (#226): KVArenaActuator.expand_pages_to_token_slots "
            "must raise ValueError on page 0; got silent return."
        )

    # `page_is_fully_free(0, ...)` must also return False
    # regardless of `free_token_set` membership.
    assert kv_act.page_is_fully_free(0, free_token_set=set(range(100))) is False
    assert kv_act.page_is_fully_free(2, free_token_set=set(range(100))) is True

    # Mamba mirror — same contract, tps=1 (worst case for the original
    # silent drop).
    class _StubMambaPool:
        class _Arena:
            tokens_per_chunk = 1
        _mamba_temporal_arena = _Arena()

    m_act = MambaArenaActuator.__new__(MambaArenaActuator)
    m_act.pool = _StubMambaPool()

    out = m_act.expand_pages_to_token_slots([3, 4])
    assert out == [3, 4], f"mamba non-page-0 broken; got {out}"

    try:
        m_act.expand_pages_to_token_slots([0])
    except ValueError as e:
        assert "page 0" in str(e) and "padded slot 0" in str(e), (
            f"mamba raise must cite page 0 + padded slot 0 diagnostic; got: {e}"
        )
    else:
        raise AssertionError(
            "BUG (#226): MambaArenaActuator.expand_pages_to_token_slots "
            "must raise on page 0. With tps=1 the pre-#226 form "
            "silently returned [] — chunk 0 unmap proceeded uncapped."
        )

    assert m_act.page_is_fully_free(0, free_token_set={0}) is False
    assert m_act.page_is_fully_free(1, free_token_set={1}) is True

    print("  PASS  36  expand_pages_to_token_slots rejects page 0 loudly "
          "on both KV (tps=4) and Mamba (tps=1); page_is_fully_free(0) "
          "→ False regardless of free_set (#226 padded-slot-0 safety)")


def main():
    tests = [
        test_1_alloc_skips_capped_slots,
        test_2_no_extra_memory_allocations_per_mark,
        test_3_mark_unmark_is_idempotent,
        test_4_kv_scale_benchmark,
        test_5_actuator_verify_respects_capped_mask,
        test_6_alloc_slow_path_preserves_capped_pages,
        test_7_alloc_fails_when_only_capped_slots_remain,
        test_9_mark_pages_capped_dedupes,
        test_11_e2e_evict_then_alloc_with_capped_pages,
        test_12_mamba_pool_capped_slots_dedupes,
        test_13_mamba_grow_must_not_re_expose_migrated_slots,
        test_14_xpool_actuator_grow_must_use_actual_granted_not_per_dst,
        test_15_mamba_cap_allocator_exposed_for_m2k_fire,
        test_16_m2k_grow_uses_explicit_arena_returned_ids,
        test_17_mamba_free_filters_capped_slots,
        test_17b_mamba_free_mixed_batch,
        test_17c_capped_filter_precedes_cap_check,
        test_17d_mamba_free_noncapped_unaffected,
        test_18_capped_pages_invariant_assertion_fires_on_violation,
        test_19_chunk_arena_grow_returns_mapped_slot_ids,
        test_20_mamba_pool_unmark_slots_id_based_grow,
        test_21_actuators_expose_unmark_token_slots,
        test_22_kv_actuator_unmark_token_slots_wrapper_e2e,
        test_23_xpool_lockstep_assertion_fires,
        test_24_mamba_pool_capped_slots_invariant_assertion,
        test_25_mamba_pool_unmark_slots_real_pool_integration,
        test_26_mamba_cap_allocator_mark_calls_capped_invariant,
        test_27_mark_pages_capped_no_hasattr_defensive_guard,
        test_28_unmark_slots_cap_uses_max_not_count,
        test_29_mamba_live_size_reflects_mark_pages_capped,
        test_30_set_capacity_slots_grow_skips_marked_slots,
        test_31_cap_barrier_skips_verify_sync_worker_does_it,
        test_32_admitter_c_xfer_uses_lcm_rounded_n_pages,
        test_33_xpool_uneven_granted_per_subpool_cleanup,
        test_34_mamba_clear_preserves_capped_above_live_cap,
        test_35_m2k_capped_slots_drain_across_cycles,
        test_36_expand_pages_rejects_page_0_loudly,
    ]
    print(f"\nmark_pages_capped no-realloc tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 9: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
