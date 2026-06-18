"""Phase 7 — MambaPool dynamic resize unit tests.

Acceptance: MambaPool with `max_size > size` pre-allocates conv_state +
temporal_state at max_size; `_capped_slots` holds deferred ids;
`set_capacity_slots(N)` for N > init_size pulls them back into
free_slots. data_ptr of conv_state stable across the grow.

Also exercises the lower-level path that D8 v4 crashed on: write to
mamba conv_state at a slot id > init_size after grow.
"""
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def _make_pool(size, max_size=None, shared_arena=False):
    """Construct a MambaPool with minimal viable args.

    Note: full MambaPool init requires a Mamba2CacheParams + several
    flags. Most tests use the non-arena path (size + 1 contiguous) for
    simplicity. Set shared_arena=True to test the arena-backed
    temporal_state branch.
    """
    import os
    if shared_arena:
        os.environ["SGLANG_ARENA_SHARED"] = "1"
        os.environ.setdefault("SGLANG_MAMBA_ARENA", "1")
    else:
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
    return MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        device=DEVICE,
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=max_size,
    )


def test_1_back_compat():
    """max_size omitted → self.size==size, self.max_size==size, no _capped."""
    p = _make_pool(size=8)
    assert p.size == 8, f"self.size = {p.size}"
    assert p.max_size == 8
    conv = p.mamba_cache.conv[0]
    assert conv.shape[1] == 8 + 1, f"expected size+1 shape, got {conv.shape}"
    # free_slots covers [1..8]
    fs = p.free_slots.cpu().tolist()
    assert fs == list(range(1, 9)), f"free_slots={fs}"
    # No _capped_slots in back-compat
    assert (getattr(p, "_capped_slots", None) is None
            or p._capped_slots.numel() == 0)
    print("  PASS  1  back-compat: self.size == size (stock semantic preserved)")


def test_2_dynamic_pre_allocation():
    """max_size > size → conv_state at max_size, but self.size stays at init.

    Critical: stock sglang code (snapshot collectors, runtime checker,
    scheduler stats) reads `mamba_pool.size` expecting LIVE cap. Phase
    7 must preserve that semantic; the pre-allocated upper bound lives
    in `max_size` (new attr)."""
    p = _make_pool(size=8, max_size=32)
    assert p.size == 8, f"self.size MUST be init (8), got {p.size}"
    assert p.max_size == 32
    assert p.live_size == 8
    conv = p.mamba_cache.conv[0]
    # Tensor IS allocated at max_size for grow capability
    assert conv.shape[1] == 32 + 1, \
        f"expected max_size+1 shape (32+1), got {conv.shape}"
    # free_slots covers ONLY [1..8] (init range)
    fs = sorted(p.free_slots.cpu().tolist())
    assert fs == list(range(1, 9)), f"free_slots={fs}"
    # Deferred ids in _capped_slots: [9..32]
    cs = sorted(p._capped_slots.cpu().tolist())
    assert cs == list(range(9, 33)), f"_capped_slots={cs}"
    assert p.size == 8
    print("  PASS  2  self.size=init (stock-compat), max_size=32 (Phase 7), "
          "conv tensor at max_size+1")


def test_3_set_capacity_grow_past_init():
    """set_capacity_slots(N>init) pulls ids from _capped back to free."""
    p = _make_pool(size=8, max_size=32)
    conv = p.mamba_cache.conv[0]
    conv_ptr0 = conv.data_ptr()
    # Grow to 12 (4 new slots from _capped)
    n = p.set_capacity_slots(12)
    assert n == 12, f"expected 12, got {n}"
    assert p.live_size == 12
    fs = sorted(p.free_slots.cpu().tolist())
    # Should now include [1..12] (init + 4 grown)
    assert set(fs) == set(range(1, 13)), f"free_slots={fs}"
    cs = sorted(p._capped_slots.cpu().tolist())
    assert cs == list(range(13, 33)), f"_capped_slots={cs}"
    # conv tensor data_ptr unchanged
    assert p.mamba_cache.conv[0].data_ptr() == conv_ptr0
    # Write to slot 12 (newly exposed) — should not fault
    p.mamba_cache.conv[0][0, 12] = 7  # layer 0, slot 12, all zeros becomes 7
    torch.cuda.synchronize()
    assert (p.mamba_cache.conv[0][0, 12] == 7).all()
    print("  PASS  3  set_capacity_slots grow past init: conv writable")


def test_4_grow_then_shrink_then_grow():
    """Grow to max, shrink to half, grow back. State consistent."""
    p = _make_pool(size=4, max_size=16)
    # Grow to 12
    p.set_capacity_slots(12)
    assert p.live_size == 12
    assert sorted(p.free_slots.cpu().tolist()) == list(range(1, 13))
    # Shrink to 6
    p.set_capacity_slots(6)
    assert p.live_size == 6
    fs = sorted(p.free_slots.cpu().tolist())
    assert fs == list(range(1, 7)), f"free_slots={fs}"
    # _capped_slots holds [7..16]
    cs = sorted(p._capped_slots.cpu().tolist())
    assert cs == list(range(7, 17)), f"_capped_slots={cs}"
    # Grow back to 16
    p.set_capacity_slots(16)
    assert p.live_size == 16
    fs = sorted(p.free_slots.cpu().tolist())
    assert fs == list(range(1, 17)), f"free_slots after re-grow={fs}"
    print("  PASS  4  grow → shrink → grow restores all slots")


def test_5_cuda_graph_capture_then_grow():
    """Critical: capture a graph reading conv_state at slot 5; grow pool;
    write to slot 12; replay; check slot 5 unchanged + can write new
    pattern. This is the analog of test_phase5b for MambaPool."""
    p = _make_pool(size=8, max_size=32)
    # Use only 1 layer's conv tensor for simplicity
    conv = p.mamba_cache.conv[0]  # shape (num_layers=2, max_size+1=33, ...)
    # Write known value to slot 5 (layer 0)
    conv[0, 5] = 42
    torch.cuda.synchronize()

    output_buf = torch.empty_like(conv[0, 5])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            output_buf.copy_(conv[0, 5])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        output_buf.copy_(conv[0, 5])

    # Replay #1: pre-grow
    output_buf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert (output_buf == 42).all()

    # Grow + write to a slot beyond init
    p.set_capacity_slots(20)
    conv[0, 12] = 99
    torch.cuda.synchronize()

    # Replay #2: post-grow. Should still read 42 from slot 5
    # (conv data_ptr unchanged, slot 5 unchanged).
    output_buf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert (output_buf == 42).all(), \
        f"post-grow replay corrupted: got {output_buf.cpu().tolist()}"
    print("  PASS  5  CUDA graph reads conv_state correctly across grow")


def test_6_set_capacity_above_max_clamps():
    """set_capacity_slots(N>max_size) clamps to max_size."""
    p = _make_pool(size=4, max_size=8)
    n = p.set_capacity_slots(20)
    assert n == 8, f"expected clamped to max_size=8, got {n}"
    assert p.live_size == 8
    print("  PASS  6  set_capacity_slots above max_size clamps to max_size")


def main():
    tests = [test_1_back_compat, test_2_dynamic_pre_allocation,
             test_3_set_capacity_grow_past_init,
             test_4_grow_then_shrink_then_grow,
             test_5_cuda_graph_capture_then_grow,
             test_6_set_capacity_above_max_clamps]
    print(f"\nMambaPool Phase 7 tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 7: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
