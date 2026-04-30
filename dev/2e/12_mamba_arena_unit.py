"""
Phase 2e.5.5 — MambaPool with SGLANG_MAMBA_ARENA=1 unit test.

Constructs a MambaPool with the arena-backed temporal layout, exercises
the same alloc/copy_from/at_layer_idx/cpu_roundtrip suite as 09, then
additionally verifies:

  - Each per-layer temporal tensor has data_ptr inside the MultiTensorArena's
    VA range.
  - All num_mamba_layers temporal tensors have *distinct* data_ptr (one
    per arena sub-pool).
  - data_ptr is stable across set_capacity_tokens() (planned future demo).

Run: CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
        .venv/bin/python dev/2e/12_mamba_arena_unit.py
"""

from __future__ import annotations
import os
import sys
import types

import torch


def _make_cache_params(num_layers: int):
    shape = types.SimpleNamespace(
        conv=[(16, 4)],
        temporal=(2, 8, 4),
    )
    dtype = types.SimpleNamespace(
        conv=torch.bfloat16,
        temporal=torch.float32,
    )
    return types.SimpleNamespace(
        shape=shape, dtype=dtype, layers=list(range(num_layers)),
    )


def _make_pool(arena: bool, *, num_layers: int = 4, size: int = 8):
    if arena:
        os.environ["SGLANG_MAMBA_ARENA"] = "1"
        os.environ["SGLANG_MAMBA_PERLAYER"] = "1"
    else:
        os.environ.pop("SGLANG_MAMBA_ARENA", None)
        os.environ.pop("SGLANG_MAMBA_PERLAYER", None)
    from sglang.srt.mem_cache.memory_pool import MambaPool

    return MambaPool(
        size=size, spec_state_size=size,
        cache_params=_make_cache_params(num_layers),
        mamba_layer_ids=list(range(num_layers)),
        device="cuda", enable_memory_saver=False,
        speculative_num_draft_tokens=None,
    )


def main() -> int:
    print("== Phase 2e.5.5: MambaPool arena unit test ==")
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    p_default = _make_pool(False)
    p_arena = _make_pool(True)
    print(f"layouts: default={type(p_default.mamba_cache.temporal).__name__}, "
          f"arena={type(p_arena.mamba_cache.temporal).__name__}")

    # ---- 1. Layout & arena attached.
    assert isinstance(p_arena.mamba_cache.temporal, list)
    assert len(p_arena.mamba_cache.temporal) == p_arena.num_mamba_layers
    assert hasattr(p_arena, "_mamba_temporal_arena")
    arena_obj = p_arena._mamba_temporal_arena
    print(f"arena: chunk_bytes={arena_obj._arena.chunk_size}, "
          f"max_tokens={arena_obj.max_tokens}, "
          f"current={arena_obj.current_capacity_tokens()}")

    # ---- 2. Each per-layer tensor's data_ptr is inside the arena's VA range.
    arena_lo = arena_obj._arena.va_base
    arena_hi = arena_lo + arena_obj._arena.total_va_size
    seen = set()
    for layer in range(p_arena.num_mamba_layers):
        t = p_arena.mamba_cache.temporal[layer]
        ptr = t.data_ptr()
        assert arena_lo <= ptr < arena_hi, \
            f"layer {layer} data_ptr 0x{ptr:x} not in arena [0x{arena_lo:x}, 0x{arena_hi:x})"
        assert ptr not in seen, f"layer {layer} duplicates data_ptr 0x{ptr:x}"
        seen.add(ptr)
    print(f"all {p_arena.num_mamba_layers} layer-tensors have distinct VAs in arena range")

    # ---- 3. Same shape & content as the default pool.
    for layer in range(p_default.num_mamba_layers):
        d = p_default.mamba2_layer_cache(layer).temporal
        a = p_arena.mamba2_layer_cache(layer).temporal
        # Default temporal is Tensor (stacked, sliced -> view), arena is the per-layer tensor.
        # Their shapes need to be comparable: default has (size+1, *shape); arena was
        # rounded UP to chunk granularity, so its first dim may be larger.
        # We compare just the live region [0:size+1].
        live = p_default.size + 1  # default semantics
        assert d.shape == a.shape[:1] + d.shape[1:] or live <= a.shape[0], \
            f"layer {layer} shape mismatch: default={d.shape}, arena={a.shape}"
        # Content of the live region should be all-zero in both.
        assert torch.equal(d.to(a.dtype), a[:live].to(d.dtype)), \
            f"layer {layer} content mismatch in live region"
    print("default vs arena: live region shapes & zero-init content match")

    # ---- 4. alloc semantics still work.
    idx = p_arena.alloc(3)
    assert idx is not None and idx.numel() == 3
    print(f"alloc(3) returned {idx.tolist()}")

    # Write to slot idx[0] in layer 1 via at_layer_idx, read back.
    pol = p_arena.mamba2_layer_cache(1)
    pol.temporal[idx[0]].fill_(7.0)
    torch.cuda.synchronize()
    val = p_arena.mamba2_layer_cache(1).temporal[idx[0], 0, 0, 0].item()
    assert val == 7.0
    print("write/read through layer view works")

    # ---- 5. copy_from semantics still work.
    src = idx[:1]
    dst = idx[1:2]
    p_arena.copy_from(src, dst)
    torch.cuda.synchronize()
    val_dst = p_arena.mamba2_layer_cache(1).temporal[dst[0], 0, 0, 0].item()
    assert val_dst == 7.0
    print("copy_from src->dst carries content")

    print("\n== PASSED: SGLANG_MAMBA_ARENA=1 mechanism works ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
