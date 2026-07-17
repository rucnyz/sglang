"""Phase 2 — ReqToTokenPool grow/shrink unit tests.

Acceptance gate per `plan.md` Phase 2:
  1. back-compat default (max_size==size) matches today's behavior
  2. dynamic-cap mode: grow(N) extends free_slots, data_ptr stable
  3. post-grow alloc() returns ids in grown range
  4. free() of a grown slot returns it to free_slots
  5. shrink rejects when slot is held (assertion / RuntimeError)
  6. shrink succeeds when slots in range are free; subsequent grow re-maps

Pure-Python, runs with sglang import. Uses cuda directly.
Run: .venv/bin/python dev/interlayer/1_dyn_admission_cap/test_phase2.py
"""
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402

DEVICE = "cuda:0"
DEVICE_IDX = 0
torch.cuda.set_device(DEVICE_IDX)

# Use 1 MiB rows × small N for fast tests.
MAX_CONTEXT_LEN = 262144  # int32 × 262144 = 1 MiB per row


@dataclass
class _StubReq:
    """Minimal Req shim matching what ReqToTokenPool.alloc reads."""
    req_pool_idx: int = None
    is_chunked: int = 0
    kv_committed_len: int = 0


def test_1_back_compat_default():
    """max_size omitted → torch.zeros allocation, no arena. Behavior matches
    pre-2026-05-26 exactly."""
    p = ReqToTokenPool(
        size=4, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
    )
    assert p._va_arena is None, "default mode must NOT use arena"
    assert p.size == 4
    assert p.max_size == 4
    assert p.free_slots == [0, 1, 2, 3]
    # Alloc returns 2 slot ids
    reqs = [_StubReq(), _StubReq()]
    ids = p.alloc(reqs)
    assert ids == [0, 1]
    assert p.free_slots == [2, 3]
    # Free returns to pool
    p.free(reqs[0])
    assert 0 in p.free_slots
    # Writes work
    p.write(torch.tensor([2], device=DEVICE),
            torch.zeros(1, MAX_CONTEXT_LEN, dtype=torch.int32, device=DEVICE))
    print("  PASS  1  back-compat default (no arena)")


def test_2_dynamic_grow():
    """max_size=8, size=2; grow to 4. data_ptr unchanged. free_slots extended."""
    p = ReqToTokenPool(
        size=2, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    assert p._va_arena is not None, "dynamic-cap mode must use arena"
    assert p.size == 2
    assert p.max_size == 8
    assert p.free_slots == [0, 1]
    ptr0 = p.req_to_token.data_ptr()

    # Write known pattern to rows 0:2
    pat = torch.full((2, MAX_CONTEXT_LEN), 777, dtype=torch.int32, device=DEVICE)
    p.req_to_token[:2] = pat
    torch.cuda.synchronize()

    # Grow to 4 rows
    new_size = p.grow(4)
    assert new_size == 4
    assert p.size == 4
    assert p.free_slots == [0, 1, 2, 3]
    assert p.req_to_token.data_ptr() == ptr0, \
        f"data_ptr changed across grow: {ptr0:#x} → {p.req_to_token.data_ptr():#x}"

    # Old rows preserved
    assert torch.equal(p.req_to_token[:2].cpu(), pat.cpu()), \
        "rows 0:2 data lost across grow"
    # New rows writable + zero-initialized
    assert (p.req_to_token[2:4] == 0).all(), "new rows must be zero-initialized"
    p.req_to_token[2:4] = 555
    torch.cuda.synchronize()
    assert (p.req_to_token[2:4] == 555).all()
    print("  PASS  2  dynamic-cap grow + preserve old + zero-init new + ptr stable")


def test_3_post_grow_alloc():
    """After grow, alloc() returns slot ids in the grown range."""
    p = ReqToTokenPool(
        size=2, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    # Exhaust the initial 2 slots
    reqs = [_StubReq(), _StubReq()]
    ids = p.alloc(reqs)
    assert ids == [0, 1]
    assert p.alloc([_StubReq()]) is None, "should fail before grow"

    # Grow to 4
    p.grow(4)
    # Now alloc 2 more
    reqs2 = [_StubReq(), _StubReq()]
    ids2 = p.alloc(reqs2)
    assert ids2 == [2, 3]
    assert p.available_size() == 0
    print("  PASS  3  post-grow alloc returns ids in grown range")


def test_4_free_grown_slot():
    """free() of a grown slot returns it to free_slots."""
    p = ReqToTokenPool(
        size=2, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    p.grow(4)
    reqs = [_StubReq() for _ in range(4)]
    ids = p.alloc(reqs)
    assert ids == [0, 1, 2, 3]
    p.free(reqs[2])  # free slot id 2 (was a grown id)
    assert 2 in p.free_slots
    # Re-alloc gets slot 2 back
    new_req = _StubReq()
    ids2 = p.alloc([new_req])
    assert ids2 == [2]
    print("  PASS  4  free of grown slot recycles via free_slots")


def test_5_shrink_rejects_held_slot():
    """Shrink that would drop a held slot raises RuntimeError."""
    p = ReqToTokenPool(
        size=2, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    p.grow(4)
    # Hold slot 3
    req = _StubReq()
    req.req_pool_idx = 3
    # Mark slot 3 as held by removing from free_slots
    p.free_slots = [s for s in p.free_slots if s != 3]
    # Attempt to shrink to 2 (would drop slot 3) — should fail
    try:
        p.shrink(2)
    except RuntimeError as e:
        assert "still held" in str(e) or "slot" in str(e).lower()
        print("  PASS  5  shrink rejects when slot id in shrunk range is held")
        return
    raise AssertionError("shrink should have raised RuntimeError")


def test_6_shrink_then_regrow():
    """Shrink succeeds when range is free; subsequent grow re-maps + recycles."""
    p = ReqToTokenPool(
        size=2, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    p.grow(4)
    ptr0 = p.req_to_token.data_ptr()
    # All 4 slots free, shrink to 2 should succeed
    new_size = p.shrink(2)
    assert new_size == 2
    assert p.size == 2
    assert p.free_slots == [0, 1]
    assert p.req_to_token.data_ptr() == ptr0, \
        "data_ptr should stay stable across shrink too"

    # Subsequent grow re-maps + appends new slot ids
    p.grow(6)
    assert p.size == 6
    assert p.free_slots == [0, 1, 2, 3, 4, 5]
    assert p.req_to_token.data_ptr() == ptr0
    print("  PASS  6  shrink then re-grow + ptr stable + slot ids extend")


def test_7_shrink_to_zero_boundary():
    """P0: boundary case — shrink(0). All slots must be free; pool ends
    with size=0, free_slots=[], no held slots. Subsequent grow re-maps
    starting from slot 0."""
    p = ReqToTokenPool(
        size=4, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=8,
    )
    ptr0 = p.req_to_token.data_ptr()
    # All 4 slots free at boot
    assert p.free_slots == [0, 1, 2, 3]
    new = p.shrink(0)
    assert new == 0
    assert p.size == 0
    assert p.free_slots == []
    # data_ptr stable even at 0
    # (can't index the tensor, but VA arena should still hold the VA)
    # Re-grow back to 4
    p.grow(4)
    assert p.size == 4
    assert p.free_slots == [0, 1, 2, 3]
    assert p.req_to_token.data_ptr() == ptr0
    print("  PASS  7  shrink to 0 + re-grow restores; data_ptr stable")


def main():
    tests = [test_1_back_compat_default, test_2_dynamic_grow,
             test_3_post_grow_alloc, test_4_free_grown_slot,
             test_5_shrink_rejects_held_slot, test_6_shrink_then_regrow,
             test_7_shrink_to_zero_boundary]
    print(f"\nReqToTokenPool Phase 2 tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 2: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
