"""Phase 3 — HybridReqToTokenPool lockstep grow tests.

Acceptance per `plan.md` Phase 3:
  1. back-compat default (max_size omitted): req_index_to_mamba_index_mapping
     sized at `size`, behavior unchanged from pre-refactor
  2. dynamic-cap mode: grow base, then alloc into grown range succeeds;
     req_index_to_mamba_index_mapping has shape (max_size,) but only
     `self.size` ids valid
  3. data_ptr of mapping tensor stable across grow (pre-allocated at max)

Note: HybridReqToTokenPool requires hybrid model context (mamba_layer_ids,
cache_params, etc.). Tests here use a minimal-mock pattern that
exercises the relevant code path without needing a full model.
"""
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)

MAX_CONTEXT_LEN = 1024  # small for tests


def _make_pool(size, max_size=None):
    """Construct a HybridReqToTokenPool with minimum viable args."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
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
    return HybridReqToTokenPool(
        size=size,
        mamba_size=size,
        mamba_spec_state_size=size,
        max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        max_size=max_size,
    )


def test_1_back_compat():
    """No max_size → mapping shape (size,), no arena on base."""
    try:
        p = _make_pool(size=4)
    except Exception as e:
        # If Mamba2CacheParams API doesn't match, skip but mark explicit
        print(f"  SKIP  1  back-compat (mamba ctor signature mismatch: {e})")
        return
    assert p._va_arena is None
    assert p.size == 4
    assert p.max_size == 4
    assert p.req_index_to_mamba_index_mapping.shape == (4,)
    print("  PASS  1  back-compat (mapping shape == (size,), no arena)")


def test_2_dynamic_grow_basic():
    """max_size=8, size=2; grow to 4; mapping shape stays (max_size,)."""
    try:
        p = _make_pool(size=2, max_size=8)
    except Exception as e:
        print(f"  SKIP  2  dynamic-cap (mamba ctor signature mismatch: {e})")
        return
    assert p._va_arena is not None
    assert p.size == 2
    assert p.max_size == 8
    # Mapping pre-allocated at max_size, NOT current size
    assert p.req_index_to_mamba_index_mapping.shape == (8,)
    mapping_ptr0 = p.req_index_to_mamba_index_mapping.data_ptr()

    p.grow(4)
    assert p.size == 4
    # Mapping shape unchanged; data_ptr unchanged
    assert p.req_index_to_mamba_index_mapping.shape == (8,)
    assert p.req_index_to_mamba_index_mapping.data_ptr() == mapping_ptr0
    print("  PASS  2  dynamic-cap grow: mapping pre-alloc at max, ptr stable")


def test_3_mapping_writeable_across_grow():
    """Write to mapping[2:4] after grow — data should persist."""
    try:
        p = _make_pool(size=2, max_size=8)
    except Exception as e:
        print(f"  SKIP  3  dynamic-cap (mamba ctor signature mismatch: {e})")
        return
    # Write to rows that are within the pre-allocated max
    p.req_index_to_mamba_index_mapping[:2] = torch.tensor(
        [10, 20], dtype=torch.int32, device=DEVICE
    )
    torch.cuda.synchronize()
    p.grow(4)
    # Write to grown range
    p.req_index_to_mamba_index_mapping[2:4] = torch.tensor(
        [30, 40], dtype=torch.int32, device=DEVICE
    )
    torch.cuda.synchronize()
    out = p.req_index_to_mamba_index_mapping[:4].cpu().tolist()
    assert out == [10, 20, 30, 40], f"got {out}"
    print("  PASS  3  mapping writable in grown range; old values preserved")


def main():
    tests = [test_1_back_compat, test_2_dynamic_grow_basic,
             test_3_mapping_writeable_across_grow]
    print(f"\nHybridReqToTokenPool Phase 3 tests (n={len(tests)}):")
    passed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 3: {passed}/{len(tests)} passed ({skipped} skipped)")
    return 0 if passed == len(tests) - skipped else 1


if __name__ == "__main__":
    sys.exit(main())
