"""1-based + dynamic-cap invariants for ReqToTokenPool (task #335 rebase).

Upstream made ReqToTokenPool 1-based: row 0 is a reserved pad row (cuda-graph
padded batches default req_pool_idx to 0, writing there harmlessly), so valid
slot ids are 1..size and admission must NEVER hand out id 0. The HiMA rebase
merges that 1-based scheme into the dynamic-cap VA-arena path (`max_size > size`,
where `grow()`/`shrink()` map/unmap rows on a stable data_ptr). This test pins
the merge: the pad row, the 1-based free_slots, and the grow/shrink/clear math
in BOTH the static and dynamic modes, against the REAL constructor.

Run: CUDA_VISIBLE_DEVICES=7 .venv/bin/python <thisfile>   (dynamic mode needs CUDA)
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch  # noqa: E402

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402

MCL = 16  # max_context_len, tiny


def _assert_one_based(pool, size):
    assert 0 not in pool.free_slots, "pad row id 0 must never be in free_slots"
    assert pool.free_slots == list(range(1, size + 1)), pool.free_slots
    assert pool._alloc_size == size + 1
    assert pool.size == size


def test_static_mode_one_based():
    """max_size == size: plain torch.zeros backing, still 1-based with pad row."""
    pool = ReqToTokenPool(size=8, max_context_len=MCL, device="cpu",
                          enable_memory_saver=False)
    assert pool._va_arena is None
    assert pool.req_to_token.shape[0] == 9  # size+1 rows (pad + 8)
    _assert_one_based(pool, 8)
    pool.clear()
    _assert_one_based(pool, 8)
    print("  PASS  static: pad row 0, free_slots 1..size, shape size+1")


def test_dynamic_grow_shrink():
    """max_size > size: VA arena sized max_size+1; grow/shrink offset by +1."""
    pool = ReqToTokenPool(size=8, max_context_len=MCL, device="cuda",
                          enable_memory_saver=False, max_size=16)
    assert pool._va_arena is not None
    # Tensor aliases the FULL VA range: max_size+1 rows, stable data_ptr.
    assert pool.req_to_token.shape[0] == 17, pool.req_to_token.shape
    ptr0 = pool.req_to_token.data_ptr()
    _assert_one_based(pool, 8)

    # grow to 12: new ids 9..12 exposed, data_ptr unchanged.
    pool.grow(12)
    assert pool.req_to_token.data_ptr() == ptr0, "grow moved data_ptr (CUDA-graph unsafe)"
    _assert_one_based(pool, 12)
    # New rows must be zeroed (clean admission state).
    assert torch.count_nonzero(pool.req_to_token[9:13]) == 0

    # shrink back to 8: ids 9..12 dropped (all were free), data_ptr unchanged.
    pool.shrink(8)
    assert pool.req_to_token.data_ptr() == ptr0
    _assert_one_based(pool, 8)
    print("  PASS  dynamic: shape max_size+1, grow/shrink 1-based, data_ptr stable")


def test_dynamic_clear_after_grow():
    """clear() rebuilds free_slots to 1.._alloc_size-1 at the current size."""
    pool = ReqToTokenPool(size=8, max_context_len=MCL, device="cuda",
                          enable_memory_saver=False, max_size=16)
    pool.grow(14)
    _assert_one_based(pool, 14)
    pool.clear()
    _assert_one_based(pool, 14)  # clear keeps the grown size, not the init size
    print("  PASS  dynamic: clear() rebuilds 1-based free_slots at grown size")


def main():
    tests = [test_static_mode_one_based]
    if torch.cuda.is_available():
        tests += [test_dynamic_grow_shrink, test_dynamic_clear_after_grow]
    else:
        print("  SKIP  dynamic-mode tests (no CUDA)")
    print(f"\nReqToTokenPool 1-based + dynamic (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nreq_token 1-based: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
