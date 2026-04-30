"""
Phase 2e.4 — MultiTensorArena smoke test.

Mimic the KV pool layout: 4 layers, 2 kinds (k, v), per-token shape (8, 128),
bfloat16. 8 sub-tensors total. Verify:
  1. Each sub-tensor lives at its own VA, distinct across layers and kinds.
  2. Writes through PyTorch are visible at the corresponding VAs.
  3. Initial state is bounded by init_tokens.

This pilot proves the multi-tensor mechanism that the actual SGLang KV
pool migration will plug into.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/07_multi_tensor_arena.py
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunk_arena import CUDA, _check  # noqa: E402
from multi_tensor_arena import MultiTensorArena  # noqa: E402


def main() -> int:
    print("== Phase 2e.4: MultiTensorArena smoke ==")

    _check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    _check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    import torch
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    # KV-pool-like configuration: 4 layers, k+v.
    # per_token = 8*128*2 = 2 KiB; chunk = 32 MiB -> 16384 tokens/chunk/tensor.
    # 32 MiB chunk size avoids PyTorch's segment-fetch (~20 MiB) overrunning a chunk.
    arena = MultiTensorArena(
        device_id=dev_id,
        n_layers=4,
        n_kinds=2,
        per_token_shape=(8, 128),
        dtype=torch.bfloat16,
        max_tokens=32768,
        init_tokens=16384,
        chunk_bytes=32 * 1024 * 1024,
    )
    print(f"per_token_bytes = {arena.per_token_bytes}")
    print(f"tokens_per_chunk = {arena.tokens_per_chunk}")
    print(f"current capacity = {arena.current_capacity_tokens()} tokens")

    # ---- Step 1: each sub-tensor at its own VA, content writeable.
    seen_ptrs = set()
    # Soft-cap design: tensor shape is full max_tokens; only first
    # current_capacity_tokens rows are physically backed.
    expected_shape = (32768, 8, 128)
    init_tokens = 16384
    for layer in range(4):
        for kind in range(2):
            t = arena.tensor(layer, kind)
            assert t.shape == expected_shape, f"tensor[{layer},{kind}].shape = {t.shape}"
            assert t.data_ptr() not in seen_ptrs, "duplicate data_ptr!"
            seen_ptrs.add(t.data_ptr())
    print(f"all 8 sub-tensors at distinct VAs, shape={expected_shape}")

    # ---- Step 2: write distinguishable patterns within the backed region.
    for layer in range(4):
        for kind in range(2):
            arena.tensor(layer, kind)[:init_tokens].fill_(float(layer * 10 + kind))
    torch.cuda.synchronize()

    for layer in range(4):
        for kind in range(2):
            v = arena.tensor(layer, kind)[0, 0, 0].item()
            expected = float(layer * 10 + kind)
            assert v == expected, f"tensor[{layer},{kind}][0,0,0] = {v}, expected {expected}"
    print(f"backed region [0:{init_tokens}) writes/reads correctly")

    # ---- Step 3: capture data_ptrs so we can prove they're stable.
    ptrs_before = [arena.tensor(L, K).data_ptr() for L in range(4) for K in range(2)]

    # ---- Step 4: grow capacity by 1 chunk per sub-pool (16384 -> 32768 tokens).
    arena.set_capacity_tokens(32768)
    torch.cuda.synchronize()

    # data_ptrs unchanged.
    ptrs_after = [arena.tensor(L, K).data_ptr() for L in range(4) for K in range(2)]
    for p1, p2 in zip(ptrs_before, ptrs_after):
        assert p1 == p2, f"data_ptr drifted: 0x{p1:x} -> 0x{p2:x}"
    print(f"after grow to {arena.current_capacity_tokens()} tokens, all data_ptrs stable")

    # The newly-mapped region is now writable.
    for layer in range(4):
        for kind in range(2):
            arena.tensor(layer, kind)[init_tokens:].fill_(float(100 + layer * 10 + kind))
    torch.cuda.synchronize()

    # Verify the just-grown region has its new pattern AND the old region kept its.
    for layer in range(4):
        for kind in range(2):
            t = arena.tensor(layer, kind)
            assert t[0, 0, 0].item() == float(layer * 10 + kind)
            assert t[init_tokens, 0, 0].item() == float(100 + layer * 10 + kind)
    print(f"newly-grown region [{init_tokens}:) is writable, old region intact")

    # ---- Step 5: shrink back. data_ptr stable; old data on the still-mapped
    # region preserved.
    arena.set_capacity_tokens(16384)
    ptrs_after2 = [arena.tensor(L, K).data_ptr() for L in range(4) for K in range(2)]
    for p1, p2 in zip(ptrs_before, ptrs_after2):
        assert p1 == p2, f"data_ptr drifted on shrink: 0x{p1:x} -> 0x{p2:x}"
    print(f"after shrink back to {arena.current_capacity_tokens()} tokens, data_ptrs still stable")

    for layer in range(4):
        for kind in range(2):
            assert arena.tensor(layer, kind)[0, 0, 0].item() == float(layer * 10 + kind)
    print("backed region's data preserved across grow+shrink cycle")

    # ---- Step 4: cleanup.
    arena.cleanup()
    print("\n== PASSED: MultiTensorArena smoke complete ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
