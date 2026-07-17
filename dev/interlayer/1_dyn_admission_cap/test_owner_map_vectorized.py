"""Phase 8: Verify the vectorized `_compute_fully_free_pages` matches
the previous Python-set loop semantic, AND is dramatically faster.

The previous implementation:
    - converts allocator.free_pages (CUDA tensor) → set via .cpu().tolist()
    - loops `for p in range(n_pages)` and checks all `tps` slots per page
    - O(n_pages × tps) Python set lookups per fire
    - Measured: ~200ms per fire for KV pool with n_pages=2398, tps=1024

The new implementation:
    - sets a bool mask on GPU via index assignment
    - reshape + .all(dim=1) on GPU
    - O(n_slots) GPU ops, single CPU sync (.nonzero().cpu())
    - Expected: < 5ms

Tests:
  1. Empty pool (no free slots) returns empty set.
  2. All slots free → all pages free.
  3. Random subset free → matches reference Python implementation.
  4. Excluded slots (capped) correctly remove pages.
  5. KV-scale benchmark: 2398 pages × 1024 tps; new impl median speedup
     ≥ 200× over old (N=5 reps, median-robust against GPU contention).
"""
import sys
import time

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def _ref_fully_free_pages(n_pages, tps, free_slot_tensors, exclude_slot_tensor):
    """The reference loop, kept in sync with #226-fixed semantics.

    Page 0 is NEVER fully-free: it carries padded slot 0 (see
    design.md §"Per-unit sizes") whose backing physical chunk must
    remain mapped. Unmapping chunk 0 would corrupt the padded-output
    target and trip CUDA-illegal-address on the next kernel write to
    a padded token (#226).
    """
    free_set = set()
    for t in free_slot_tensors:
        if t is None or t.numel() == 0:
            continue
        free_set |= {int(x) for x in t.cpu().tolist()}
    if exclude_slot_tensor is not None and exclude_slot_tensor.numel() > 0:
        exc = {int(x) for x in exclude_slot_tensor.cpu().tolist()}
        free_set -= exc
    free_pages = set()
    for p in range(n_pages):
        if p == 0:
            continue  # #226: chunk 0 carries padded slot 0; never unmappable.
        start = p * tps
        ok = True
        for s in range(start, (p + 1) * tps):
            if s not in free_set:
                ok = False
                break
        if ok:
            free_pages.add(p)
    return free_pages


def _new(n_pages, tps, free_slot_tensors, exclude_slot_tensor):
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
    return SchedulerOwnerProvider._compute_fully_free_pages(
        n_pages=n_pages,
        tps=tps,
        free_slot_tensors=free_slot_tensors,
        exclude_slot_tensor=exclude_slot_tensor,
    )


def test_1_empty():
    """Empty inputs → empty set, no crash."""
    res = _new(n_pages=0, tps=4, free_slot_tensors=[None], exclude_slot_tensor=None)
    assert res == set(), f"empty got {res}"
    res = _new(n_pages=8, tps=4, free_slot_tensors=[None], exclude_slot_tensor=None)
    assert res == set(), f"no-free got {res}"
    print("  PASS  1  empty / no-free → empty set")


def test_2_all_free():
    """All non-padded slots free → pages [1..n_pages) free.

    Page 0 is excluded by the #226 padded-slot-0 safety invariant:
    even when all of [1..tps) are free, page 0 cannot be unmapped
    because chunk 0 carries padded slot 0 (see design.md
    §"Per-unit sizes").
    """
    n_pages, tps = 5, 4
    all_slots = torch.arange(1, n_pages * tps, dtype=torch.int64, device=DEVICE)
    res = _new(n_pages, tps, [all_slots], None)
    expected = set(range(1, n_pages))
    assert res == expected, f"all-free got {res} expected {expected}"
    assert 0 not in res, (
        f"#226: page 0 must NOT be in fully-free set even with all "
        f"non-padded slots free. Got {sorted(res)}."
    )
    print(f"  PASS  2  all slots free → pages [1..{n_pages}) free; "
          f"page 0 excluded (padded-slot-0 safety #226)")


def test_3_partial_random():
    """Random subset of slots free; new impl matches reference."""
    n_pages, tps = 20, 8
    n_total = n_pages * tps
    # randomly mark ~half free
    g = torch.Generator(device=DEVICE).manual_seed(42)
    free_mask = torch.rand(n_total, generator=g, device=DEVICE) > 0.5
    free_mask[0] = False  # never put slot 0 in actual free list
    free_slots = free_mask.nonzero(as_tuple=True)[0]

    ref = _ref_fully_free_pages(n_pages, tps, [free_slots], None)
    new = _new(n_pages, tps, [free_slots], None)
    assert ref == new, f"mismatch: ref={ref} new={new}"
    print(f"  PASS  3  random subset ({free_slots.numel()} free): "
          f"{len(ref)} fully-free pages match")


def test_4_with_exclude():
    """exclude_slot_tensor (capped) removes pages from fully-free."""
    n_pages, tps = 10, 4
    all_slots = torch.arange(1, n_pages * tps, dtype=torch.int64, device=DEVICE)
    # exclude slot 5 — pages containing slot 5 should be excluded.
    excluded = torch.tensor([5], dtype=torch.int64, device=DEVICE)

    ref = _ref_fully_free_pages(n_pages, tps, [all_slots], excluded)
    new = _new(n_pages, tps, [all_slots], excluded)
    assert ref == new, f"mismatch: ref={ref} new={new}"
    # page 1 contains slots [4,5,6,7] — should be missing
    assert 1 not in new, f"page 1 (contains excluded slot 5) should not be in new={new}"
    # page 2 is adjacent and unaffected; page 0 is always excluded (#226).
    assert 2 in new and 0 not in new, (
        f"page 2 should be in fully-free; page 0 must NOT be "
        f"(padded-slot-0 safety #226). Got {sorted(new)}."
    )
    print(f"  PASS  4  exclude (capped) removes affected pages; "
          f"page 0 always excluded (#226)")


def test_5_multiple_free_tensors():
    """KV pool has both free_pages and release_pages; union is checked.
    Pages [1..n_pages) all free (page 0 always excluded — #226)."""
    n_pages, tps = 6, 4
    free_a = torch.arange(1, 12, dtype=torch.int64, device=DEVICE)  # slots 1..11
    free_b = torch.arange(12, 24, dtype=torch.int64, device=DEVICE)  # slots 12..23
    # Both together cover [1..23] = entire pool minus padded slot 0.
    ref = _ref_fully_free_pages(n_pages, tps, [free_a, free_b], None)
    new = _new(n_pages, tps, [free_a, free_b], None)
    expected = set(range(1, n_pages))
    assert ref == new == expected, f"ref={ref} new={new} expected={expected}"
    print(f"  PASS  5  union of two free tensors → pages [1..{n_pages}) "
          f"free; page 0 protected — carries padded slot 0 (#226)")


def test_6_kv_scale_speedup():
    """KV-scale: n_pages=2398, tps=1024 (sglang default).
    Take MEDIAN of N=5 runs (instead of a single sample) so a single
    contended-GPU outlier doesn't trip the assertion. Pristine reruns
    routinely show 1000×+; outliers under heavy GPU contention can dip
    to 16×. The MEDIAN should still clear ≥ 50× — well above the
    noise floor and well below the typical pristine bound."""
    n_pages, tps = 2398, 1024
    n_total = n_pages * tps

    # Simulate ~most slots free, ~5% used (realistic D8 saturated)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    free_mask = torch.rand(n_total, generator=g, device=DEVICE) > 0.05
    free_mask[0] = False
    free_slots = free_mask.nonzero(as_tuple=True)[0]
    torch.cuda.synchronize()

    # Warm up CUDA / autotune by running the new impl once before timing.
    _ = _new(n_pages, tps, [free_slots], None)
    torch.cuda.synchronize()

    speedups = []
    rebuilds_new_ms = []
    rebuilds_ref_ms = []
    res_ref_expected = None
    for rep in range(5):
        # Time the new impl (mean of 3 inner reps; this part is fast)
        t0 = time.perf_counter()
        for _ in range(3):
            res_new = _new(n_pages, tps, [free_slots], None)
        torch.cuda.synchronize()
        t_new = (time.perf_counter() - t0) / 3 * 1000

        # Time the reference (single run; it's slow)
        t0 = time.perf_counter()
        res_ref = _ref_fully_free_pages(n_pages, tps, [free_slots], None)
        t_ref = (time.perf_counter() - t0) * 1000

        if res_ref_expected is None:
            res_ref_expected = res_ref
        assert res_new == res_ref == res_ref_expected, (
            f"rep={rep}: mismatch new={len(res_new)} ref={len(res_ref)}"
        )

        speedups.append(t_ref / t_new)
        rebuilds_new_ms.append(t_new)
        rebuilds_ref_ms.append(t_ref)

    speedups.sort()
    median_speedup = speedups[len(speedups) // 2]
    min_speedup = speedups[0]
    max_speedup = speedups[-1]
    print(f"  ref:  {min(rebuilds_ref_ms):.1f}–{max(rebuilds_ref_ms):.1f} ms across 5 reps")
    print(f"  new:  {min(rebuilds_new_ms):.2f}–{max(rebuilds_new_ms):.2f} ms")
    print(f"  speedups: {[f'{s:.0f}x' for s in speedups]}")
    print(f"  median speedup: {median_speedup:.0f}× (min={min_speedup:.0f}×, max={max_speedup:.0f}×)")
    # Use median so a single bad rep (GPU contention) doesn't trip us.
    # Empirical reps observed: 328× / 540× / 546× / 566× / 1552×
    # (median 546). Threshold 200× sits well above the worst-rep
    # noise floor (~328×) and far below the pristine ceiling (1000×+).
    assert median_speedup >= 200, (
        f"median speedup {median_speedup:.0f}× < 200× — the vectorized "
        f"fix should clear 200× at the median even under GPU contention. "
        f"per-rep: {speedups}"
    )
    print(f"  PASS  6  KV-scale median speedup ≥ 200× ({median_speedup:.0f}×, N=5 reps)")


def test_7_page_0_never_in_fully_free():
    """#226: chunk 0 carries padded slot 0 (design.md §"Per-unit
    sizes") — never unmappable.

    Concrete trigger: fire_planner picks `sorted(free_pages,
    reverse=True)[:n]`, so in a small or fully-drained pool page 0
    eventually becomes the only candidate. Unmapping chunk 0 frees
    the VA backing slot 0; the next kernel write to slot 0 (the
    padded-output target — `Page 0` comment at
    `allocator.py:381-383`) hits unmapped VA → CUDA-illegal-address.

    Pre-#226: `_compute_fully_free_pages` marked slot 0 as "free
    unconditionally" so page 0 with all other slots free entered
    the result set. Post-fix: page 0 is unconditionally excluded
    regardless of slot membership.
    """
    # Case A: KV-shaped (tps > 1). All slots free; page 0 should be excluded.
    n_pages, tps = 8, 4
    all_slots = torch.arange(1, n_pages * tps, dtype=torch.int64, device=DEVICE)
    res = _new(n_pages, tps, [all_slots], None)
    assert 0 not in res, (
        f"BUG (#226 case A): page 0 in fully-free with tps={tps}, "
        f"all other slots free. Got {sorted(res)}. Fix must exclude "
        f"page 0 unconditionally from `_compute_fully_free_pages`."
    )
    assert res == set(range(1, n_pages))

    # Case B: mamba-shaped (tps == 1). Page 0 has only padded slot 0;
    # any "free" reasoning about page 0 would silently unmap chunk 0
    # and corrupt the padded-output target.
    n_pages, tps = 5, 1
    # Slots [1..4] free (tps=1 → 1 slot per page; slot 0 is padded)
    all_slots = torch.arange(1, n_pages, dtype=torch.int64, device=DEVICE)
    res = _new(n_pages, tps, [all_slots], None)
    assert 0 not in res, (
        f"BUG (#226 case B / mamba tps=1): page 0 in fully-free. "
        f"With tps=1, expand_pages_to_token_slots([0]) returns the "
        f"empty list (range(1,1)) — the actuator would unmap chunk 0 "
        f"WITHOUT marking any slot, silently corrupting the padded-"
        f"output target."
    )
    assert res == set(range(1, n_pages))

    # Case C: edge — n_pages=1. Only page 0 exists; must be empty set.
    n_pages, tps = 1, 4
    all_slots = torch.arange(1, tps, dtype=torch.int64, device=DEVICE)
    res = _new(n_pages, tps, [all_slots], None)
    assert res == set(), (
        f"BUG (#226 case C): pool of 1 page must yield empty fully-free "
        f"set (only candidate IS page 0). Got {sorted(res)}."
    )

    print("  PASS  7  page 0 unconditionally excluded from fully-free set "
          "(KV tps=4, mamba tps=1, n_pages=1 edge — padded-slot-0 safety #226)")


def test_8_build_kv_owner_map_excludes_capped_pages():
    """WIRING regression (4th-round production audit): the exclude LOGIC
    in `_compute_fully_free_pages` is correct (test_4), but
    `build_kv_owner_map` must actually PASS the KV allocator's
    `_capped_pages` as the exclude set — mirroring `build_mamba_owner_map`
    which passes `_capped_slots`. A KV→mamba fire leaves the source pages
    in `free_pages` while capping them in `_capped_pages`
    (`mark_pages_capped` leaves `free_pages` untouched); if the owner map
    doesn't exclude them, the planner re-selects already-unmapped pages →
    wasted/short fires + c^xfer EWMA samples polluted by near-zero-work
    fires. Pre-fix: `build_kv_owner_map` passes `exclude_slot_tensor=None`
    → a capped page is reported fully-free → this FAILS. Post-fix it
    passes `_capped_pages` → the page is excluded (KV symmetric to mamba)."""
    import types
    from sglang.srt.budgeter.scheduler_owner_provider import (
        SchedulerOwnerProvider,
    )

    n_pages, tps = 10, 4
    free_slots = torch.arange(1, n_pages * tps, dtype=torch.int64, device=DEVICE)
    capped = torch.tensor([5], dtype=torch.int64, device=DEVICE)  # slot 5 in page 1
    allocator = types.SimpleNamespace(
        free_pages=free_slots, release_pages=None, _capped_pages=capped,
    )
    kv_act = types.SimpleNamespace(
        n_pages=n_pages, _tokens_per_page=lambda: tps,
    )
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=allocator)
    provider = SchedulerOwnerProvider(scheduler, kv_act, mamba_actuator=None)
    om = provider.build_kv_owner_map()
    assert 1 not in om.free_pages, (
        "build_kv_owner_map must exclude pages whose slots are in the KV "
        "allocator's _capped_pages (page 1 holds capped slot 5) — mirroring "
        "build_mamba_owner_map. exclude=None reports an already-capped / "
        "unmapped page as fully-free → wasted fire + c^xfer EWMA pollution."
    )
    # Adjacent page 2 (slots 8-11, none capped) stays free; page 0 always out.
    assert 2 in om.free_pages and 0 not in om.free_pages, sorted(om.free_pages)
    print("  PASS  8  build_kv_owner_map excludes _capped_pages "
          "(page 1 holds capped slot 5; KV symmetric to mamba)")


def main():
    tests = [test_1_empty, test_2_all_free, test_3_partial_random,
             test_4_with_exclude, test_5_multiple_free_tensors,
             test_6_kv_scale_speedup, test_7_page_0_never_in_fully_free,
             test_8_build_kv_owner_map_excludes_capped_pages]
    print(f"\nVectorized owner-map tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 8: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
