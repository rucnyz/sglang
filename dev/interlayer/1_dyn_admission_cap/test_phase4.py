"""Phase 4 — FutureMap dynamic grow tests.

Acceptance per `plan.md` Phase 4:
  1. back-compat default (max_running_requests_max omitted): identical
     to pre-refactor behavior
  2. dynamic mode: grow updates future_limit; alloc with bs > old live
     max still wraps correctly
  3. token_ids_buf data_ptr stable across grow (pre-allocated at max)
  4. grow rejects when new > max_running_requests_max
"""
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
from sglang.srt.managers.overlap_utils import FutureMap  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(0)


def _make(max_running, max_running_max=None):
    return FutureMap(
        max_running_requests=max_running,
        chunked_prefill_size=8192,
        context_len=16384,
        device=DEVICE,
        spec_algo=SpeculativeAlgorithm.NONE,
        max_running_requests_max=max_running_max,
    )


def test_1_back_compat():
    """Default max_running_requests_max → behavior unchanged."""
    fm = _make(max_running=33)
    assert fm.max_running_requests == 33
    assert fm.max_running_requests_max == 33
    # slots_per_req = 3 + max_num_chunks; max_num_chunks for 16384 / 8192 = 2
    expected_slots_per_req = 3 + 2
    assert fm._slots_per_req == expected_slots_per_req
    assert fm.future_limit == 33 * expected_slots_per_req
    assert fm.future_buffer_len == 33 * expected_slots_per_req + 2 * 33
    assert fm.token_ids_buf.shape == (fm.future_buffer_len,)
    print("  PASS  1  back-compat default (max_max == live, identical sizing)")


def test_2_dynamic_grow_updates_future_limit():
    """grow(N) updates live max + future_limit; buffer was sized for max."""
    fm = _make(max_running=33, max_running_max=128)
    assert fm.max_running_requests == 33
    assert fm.max_running_requests_max == 128
    # Buffer sized for 128
    expected_buf_len = 128 * (3 + 2) + 2 * 128
    assert fm.future_buffer_len == expected_buf_len
    assert fm.future_limit == 33 * (3 + 2)

    new = fm.grow(100)
    assert new == 100
    assert fm.max_running_requests == 100
    assert fm.future_limit == 100 * (3 + 2)
    # Buffer untouched
    assert fm.future_buffer_len == expected_buf_len
    print("  PASS  2  grow updates max + future_limit; buffer pre-sized at max")


def test_3_token_ids_buf_ptr_stable():
    """data_ptr unchanged across multiple grow calls."""
    fm = _make(max_running=33, max_running_max=128)
    ptr0 = fm.token_ids_buf.data_ptr()
    for n in (50, 80, 100, 128):
        fm.grow(n)
        assert fm.token_ids_buf.data_ptr() == ptr0, \
            f"data_ptr changed at grow({n})"
    print("  PASS  3  token_ids_buf data_ptr stable across grow")


def test_4_grow_rejects_overshoot():
    """grow above max_running_requests_max raises ValueError."""
    fm = _make(max_running=33, max_running_max=64)
    try:
        fm.grow(100)
    except ValueError as e:
        assert "max_running_requests_max" in str(e)
        print("  PASS  4  grow rejects new > max_running_requests_max")
        return
    raise AssertionError("grow should have raised")


def main():
    tests = [test_1_back_compat, test_2_dynamic_grow_updates_future_limit,
             test_3_token_ids_buf_ptr_stable, test_4_grow_rejects_overshoot]
    print(f"\nFutureMap Phase 4 tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 4: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
