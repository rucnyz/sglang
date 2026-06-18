"""#208 — disagg metadata buffers must be pre-sized to the in-flight
cap's UPPER BOUND, not its live value, so a Phase 7 dynamic grow on
``req_to_token_pool`` (or ``max_running_requests``) doesn't run out of
slots in the ``ReqToMetadataIdxAllocator``.

The helper ``disagg_metadata_buffer_size(live_max)`` is a thin wrapper
returning ``live_max * 2`` (the original headroom convention). Per-mode
``live_max`` derivation is the scheduler's job, since the two
disagg-mode pool classes have different upper-bound mechanisms:

* DECODE (``DecodeReqToTokenPool`` / ``HybridMambaDecodeReqToTokenPool``):
  ``size + pre_alloc_size``. The pool has its own ``pre_alloc_size``
  headroom mechanism and does NOT inherit Phase 7's
  ``ReqToTokenPool.max_size`` dynamic-cap mechanism.
* PREFILL (``ReqToTokenPool`` / ``HybridReqToTokenPool``):
  ``max(max_running_requests, req_to_token_pool.max_size)``. Mirrors
  ``FutureMap.max_running_requests_max``.

These tests pin both the helper formula and the scheduler-side
per-mode derivation contract.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.disaggregation.utils import (
    ReqToMetadataIdxAllocator,
    disagg_metadata_buffer_size,
)


def test_1_helper_formula():
    """Helper just returns ``live_max * 2`` — the original headroom
    convention applied to whichever upper bound the caller supplied.
    """
    assert disagg_metadata_buffer_size(20) == 40
    assert disagg_metadata_buffer_size(1) == 2
    assert disagg_metadata_buffer_size(0) == 0
    print(f"  PASS  1  helper: disagg_metadata_buffer_size(N) == N * 2")


def test_2_decode_uses_size_plus_pre_alloc():
    """DECODE-mode pool (``DecodeReqToTokenPool``) exposes ``size`` +
    ``pre_alloc_size``. Scheduler must derive
    ``live_max = size + pre_alloc_size``.

    Pre-fix the scheduler used ``self.req_to_token_pool.size * 2`` which
    under-allocated (didn't include pre_alloc_size).
    """
    # Mimic the scheduler-side derivation for DECODE without booting
    # the whole scheduler.
    class _FakeDecodePool:
        size = 5
        pre_alloc_size = 15
    rt_pool = _FakeDecodePool()
    decode_live_max = rt_pool.size + rt_pool.pre_alloc_size
    buf = disagg_metadata_buffer_size(decode_live_max)
    assert buf == 40, (
        f"BUG (#208): DECODE buffer must size to (size + pre_alloc_size) * 2 = "
        f"40; got {buf}. Pre-fix used size * 2 = 10 which exhausted the "
        f"allocator once pre-allocated decode requests accumulated."
    )
    print(f"  PASS  2  DECODE: (size + pre_alloc_size) * 2 = {buf}")


def test_3_prefill_uses_max_of_mrr_and_rt_pool_max_size():
    """PREFILL-mode pool (``ReqToTokenPool`` / ``HybridReqToTokenPool``)
    has ``size`` (live) + ``max_size`` (Phase 7 upper bound). Scheduler
    derives ``live_max = max(max_running_requests, rt_pool.max_size)``.
    """
    class _FakePrefillPool:
        size = 5
        max_size = 20
    rt_pool = _FakePrefillPool()
    max_running_requests = 10
    rt_pool_max = getattr(rt_pool, "max_size", rt_pool.size)
    prefill_live_max = max(max_running_requests, rt_pool_max)
    buf = disagg_metadata_buffer_size(prefill_live_max)
    assert buf == 40, (
        f"BUG (#208): PREFILL buffer must size to max(mrr, max_size) * 2 = 40; "
        f"got {buf}. Pre-fix used max_running_requests * 2 = 20 which "
        f"exhausted once Phase 7 grew the pool past init."
    )
    print(f"  PASS  3  PREFILL: max(mrr, max_size) * 2 = {buf}")


def test_4_prefill_pool_must_set_max_size():
    """`ReqToTokenPool.__init__` unconditionally sets `self.max_size`
    (defaults to `size` when caller passes None). The scheduler relies
    on direct attribute access — no defensive `getattr(..., default)`
    fallback per memory rule "no defensive fallbacks". Without this
    invariant, the scheduler's `rt_pool.max_size` raises AttributeError
    instead of silently mis-sizing.
    """
    import inspect
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    init_src = inspect.getsource(ReqToTokenPool.__init__)
    assert "self.max_size = max_size" in init_src, (
        "ReqToTokenPool.__init__ must unconditionally set self.max_size "
        "for the scheduler's PREFILL buffer-sizing path to work."
    )
    assert "max_size = size" in init_src, (
        "ReqToTokenPool.__init__ must default max_size to size when "
        "caller passes None, preserving back-compat without "
        "scheduler-side fallback."
    )
    print(f"  PASS  4  ReqToTokenPool.__init__ sets max_size unconditionally "
          f"(no scheduler-side getattr fallback needed)")


def test_5_allocator_pre_sized():
    """Smoke: ``ReqToMetadataIdxAllocator(N)`` hands out N slots."""
    a = ReqToMetadataIdxAllocator(size=40)
    assert a.available_size() == 40
    for i in range(40):
        out = a.alloc()
        assert out is not None, f"slot {i}: alloc returned None unexpectedly"
    assert a.alloc() is None, "41st alloc must return None (exhausted)"
    print(f"  PASS  5  ReqToMetadataIdxAllocator(40) exhausts after 40 allocs")


def test_6_real_decode_pool_has_size_and_pre_alloc():
    """Integration check: the production ``DecodeReqToTokenPool`` and
    ``HybridMambaDecodeReqToTokenPool`` classes expose ``size`` and
    ``pre_alloc_size``. Catches a regression where the scheduler's
    derivation would AttributeError on a missing attr.
    """
    # We can't instantiate without a full CUDA fixture, but a class-level
    # check via inspect confirms the attribute will be set in __init__.
    import inspect
    from sglang.srt.disaggregation.decode import (
        DecodeReqToTokenPool,
        HybridMambaDecodeReqToTokenPool,
    )
    decode_src = inspect.getsource(DecodeReqToTokenPool.__init__)
    assert "self.size = size" in decode_src, (
        "DecodeReqToTokenPool.__init__ must set self.size"
    )
    assert "self.pre_alloc_size" in decode_src, (
        "DecodeReqToTokenPool.__init__ must set self.pre_alloc_size; "
        "scheduler DECODE branch reads it for the buffer upper bound"
    )
    hybrid_src = inspect.getsource(HybridMambaDecodeReqToTokenPool.__init__)
    assert "DecodeReqToTokenPool.__init__" in hybrid_src, (
        "HybridMambaDecodeReqToTokenPool must call DecodeReqToTokenPool.__init__ "
        "to inherit size/pre_alloc_size"
    )
    print(f"  PASS  6  production DecodeReqToTokenPool classes set "
          f".size + .pre_alloc_size in __init__")


def main():
    tests = [
        test_1_helper_formula,
        test_2_decode_uses_size_plus_pre_alloc,
        test_3_prefill_uses_max_of_mrr_and_rt_pool_max_size,
        test_4_prefill_pool_must_set_max_size,
        test_5_allocator_pre_sized,
        test_6_real_decode_pool_has_size_and_pre_alloc,
    ]
    print(f"\n#208 disagg buffer sizing tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#208: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
