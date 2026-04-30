"""
Phase 2e.5.2 — MambaPool per-layer split (SGLANG_MAMBA_PERLAYER) unit test.

Constructs two MambaPool instances on the same dummy config:
  - pool_stacked: SGLANG_MAMBA_PERLAYER unset (default, temporal is a single
    stacked tensor of shape (num_layers, size+1, *)).
  - pool_perlayer: SGLANG_MAMBA_PERLAYER=1 (temporal is a List[Tensor], each
    of shape (size+1, *), one per layer).

Verifies that for every operation the engine exercises, the two pools
produce bit-identical observable behavior:
  1. alloc returns the same indices, available_size matches.
  2. mamba2_layer_cache(layer) returns equivalent views (same shape, dtype,
     same content under identical writes).
  3. copy_from semantics: write to src slots, copy to dst slots, both pools
     produce identical content at dst.
  4. get_cpu_copy / load_cpu_copy roundtrip: dump from one, load into both,
     both pools end up with identical content.
  5. get_state_dim_per_tensor returns the same list.

If all five hold, the SGLANG_MAMBA_PERLAYER feature is logically
equivalent to the default and safe to A/B at higher levels.

Run: CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
        .venv/bin/python dev/2e/09_mamba_perlayer_unit.py
"""

from __future__ import annotations
import os
import sys
import types

import torch


def _make_cache_params(num_layers: int):
    """Mock the minimum surface MambaPool reads from cache_params."""
    # Conv shape is per-conv-shape (typically a single shape).
    # Temporal shape is fixed per pool.
    shape = types.SimpleNamespace(
        conv=[(16, 4)],         # one conv shape
        temporal=(2, 8, 4),     # (heads, head_dim, state_size)-ish
    )
    dtype = types.SimpleNamespace(
        conv=torch.bfloat16,
        temporal=torch.float32,
    )
    return types.SimpleNamespace(
        shape=shape,
        dtype=dtype,
        layers=list(range(num_layers)),
    )


def _make_pool(perlayer: bool, *, num_layers: int = 4, size: int = 8):
    if perlayer:
        os.environ["SGLANG_MAMBA_PERLAYER"] = "1"
    else:
        os.environ.pop("SGLANG_MAMBA_PERLAYER", None)
    from sglang.srt.mem_cache.memory_pool import MambaPool

    cache_params = _make_cache_params(num_layers)
    pool = MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=cache_params,
        mamba_layer_ids=list(range(num_layers)),
        device="cuda",
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
    )
    return pool


def assert_temporal_equal(p0, p1, label: str) -> None:
    # Read all layers via mamba2_layer_cache so the comparison is layout-blind.
    for layer in range(p0.num_mamba_layers):
        v0 = p0.mamba2_layer_cache(layer).temporal
        v1 = p1.mamba2_layer_cache(layer).temporal
        assert v0.shape == v1.shape, f"[{label}] layer={layer}: shape {v0.shape} vs {v1.shape}"
        assert v0.dtype == v1.dtype, f"[{label}] layer={layer}: dtype mismatch"
        if not torch.equal(v0, v1):
            diff = (v0 - v1).abs().max().item()
            raise AssertionError(
                f"[{label}] layer={layer} temporal differs (max abs diff={diff})"
            )


def assert_conv_equal(p0, p1, label: str) -> None:
    for layer in range(p0.num_mamba_layers):
        c0 = p0.mamba2_layer_cache(layer).conv
        c1 = p1.mamba2_layer_cache(layer).conv
        assert len(c0) == len(c1), f"[{label}] layer={layer}: conv list length mismatch"
        for i, (a, b) in enumerate(zip(c0, c1)):
            assert a.shape == b.shape, f"[{label}] layer={layer} conv[{i}]: shape mismatch"
            if not torch.equal(a, b):
                raise AssertionError(f"[{label}] layer={layer} conv[{i}] differs")


def main() -> int:
    print("== Phase 2e.5.2: MambaPool perlayer-split unit test ==")
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    # ---- Test 1: pool construction + initial state.
    p0 = _make_pool(False)
    p1 = _make_pool(True)
    print(f"layouts: stacked={type(p0.mamba_cache.temporal).__name__}, "
          f"perlayer={type(p1.mamba_cache.temporal).__name__}")
    assert isinstance(p0.mamba_cache.temporal, torch.Tensor)
    assert isinstance(p1.mamba_cache.temporal, list)
    assert len(p1.mamba_cache.temporal) == p1.num_mamba_layers

    # Both initialize to all-zero.
    assert_temporal_equal(p0, p1, "initial-zeros")
    assert_conv_equal(p0, p1, "initial-zeros")
    assert p0.available_size() == p1.available_size()
    print(f"initial: available_size={p0.available_size()} (both)")

    # ---- Test 2: alloc returns same indices.
    idx0 = p0.alloc(3)
    idx1 = p1.alloc(3)
    assert torch.equal(idx0, idx1), f"alloc indices differ: {idx0} vs {idx1}"
    assert p0.available_size() == p1.available_size()
    # Post-alloc state should have been zeroed at the alloc'd slots.
    assert_temporal_equal(p0, p1, "post-alloc")
    print(f"alloc(3) indices: {idx0.tolist()}, post-state equal")

    # ---- Test 3: deterministic write through layer view, then read back.
    for layer in range(p0.num_mamba_layers):
        view0 = p0.mamba2_layer_cache(layer)
        view1 = p1.mamba2_layer_cache(layer)
        # Write a layer-distinguishable pattern at slot=2 of every layer.
        val = float(10 + layer)
        view0.temporal[2].fill_(val)
        view1.temporal[2].fill_(val)
    torch.cuda.synchronize()
    assert_temporal_equal(p0, p1, "post-write")

    # Cross-layer leak check: layer 0's write should not be visible in layer 1.
    for layer in range(p0.num_mamba_layers):
        v0 = p0.mamba2_layer_cache(layer).temporal[2]
        v1 = p1.mamba2_layer_cache(layer).temporal[2]
        expected = float(10 + layer)
        actual0 = v0[0, 0, 0].item()
        actual1 = v1[0, 0, 0].item()
        assert actual0 == expected, f"stacked layer {layer} slot 2: {actual0} != {expected}"
        assert actual1 == expected, f"perlayer layer {layer} slot 2: {actual1} != {expected}"
    print("layer isolation: each layer's write stays in its layer (both)")

    # ---- Test 4: copy_from semantics.
    # Copy slot 2 (which has the layer-pattern) to slot 5 in each pool.
    # Both should produce the same end state.
    src = torch.tensor([2], dtype=torch.int64, device="cuda")
    dst = torch.tensor([5], dtype=torch.int64, device="cuda")
    p0.copy_from(src, dst)
    p1.copy_from(src, dst)
    torch.cuda.synchronize()
    assert_temporal_equal(p0, p1, "post-copy_from")
    # Slot 5 should now match slot 2's pattern.
    for layer in range(p0.num_mamba_layers):
        s0 = p0.mamba2_layer_cache(layer).temporal[5, 0, 0, 0].item()
        s1 = p1.mamba2_layer_cache(layer).temporal[5, 0, 0, 0].item()
        expected = float(10 + layer)
        assert s0 == expected and s1 == expected
    print("copy_from(2->5) carries content correctly (both)")

    # ---- Test 5: get_cpu_copy / load_cpu_copy.
    indices = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
    cpu0 = p0.get_cpu_copy(indices)
    cpu1 = p1.get_cpu_copy(indices)
    # The CPU copies have different layouts (stacked tensor vs list-of-tensors),
    # but their *content* must be byte-identical for what the engine cares about.
    conv_cpu0, temporal_cpu0 = cpu0
    conv_cpu1, temporal_cpu1 = cpu1
    # Conv: list of tensors in both cases.
    assert len(conv_cpu0) == len(conv_cpu1)
    for c0, c1 in zip(conv_cpu0, conv_cpu1):
        assert torch.equal(c0, c1), "conv cpu copy differs"
    # Temporal: stacked tensor vs list-of-tensors. Compare per-layer.
    if isinstance(temporal_cpu0, torch.Tensor) and isinstance(temporal_cpu1, list):
        for layer in range(p0.num_mamba_layers):
            assert torch.equal(temporal_cpu0[layer], temporal_cpu1[layer]), \
                f"temporal cpu copy differs at layer {layer}"
    elif isinstance(temporal_cpu0, list) and isinstance(temporal_cpu1, list):
        for a, b in zip(temporal_cpu0, temporal_cpu1):
            assert torch.equal(a, b)
    else:
        raise AssertionError(f"unexpected types: {type(temporal_cpu0)}, {type(temporal_cpu1)}")
    print("get_cpu_copy: layouts differ but contents match per-layer")

    # Load back into a different slot.
    new_dst = torch.tensor([6, 7], dtype=torch.int64, device="cuda")
    p0.load_cpu_copy(cpu0, new_dst)
    p1.load_cpu_copy(cpu1, new_dst)
    torch.cuda.synchronize()
    assert_temporal_equal(p0, p1, "post-load_cpu_copy")
    print("load_cpu_copy round-trip: both pools end in the same state")

    # ---- Test 6: get_state_dim_per_tensor returns the same list.
    dims0 = p0.get_state_dim_per_tensor()
    dims1 = p1.get_state_dim_per_tensor()
    assert dims0 == dims1, f"state_dim_per_tensor differ: {dims0} vs {dims1}"
    print(f"get_state_dim_per_tensor: {dims0} (both)")

    # ---- Test 7: free returns slots to pool.
    p0.free(idx0)
    p1.free(idx1)
    assert p0.available_size() == p1.available_size()
    print(f"after free: available_size={p0.available_size()} (both)")

    print("\n== PASSED: SGLANG_MAMBA_PERLAYER=1 is bit-equivalent to default ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
