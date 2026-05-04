"""T4 unit test: MambaPool.migrate_slot copies bytes from src to dst,
moves src to _capped_slots, removes dst from free_slots.

Constructs a tiny MambaPool stand-in (no engine boot) with synthetic
conv + temporal tensors, populates slot 5 with known bytes, runs
migrate_slot(5, 7), asserts slot 7 contains those bytes and allocator
state is correctly updated.
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


class _FakeMambaCache:
    """Minimal stand-in: just `conv` (list[Tensor]) and `temporal` (Tensor)."""
    def __init__(self, n_layers, size, device):
        # Each conv tensor: (n_layers, size+1, 16, 4) for a synthetic shape.
        self.conv = [
            torch.zeros(n_layers, size + 1, 16, 4, dtype=torch.bfloat16, device=device)
        ]
        # Temporal: (n_layers, size+1, 8, 32)
        self.temporal = torch.zeros(
            n_layers, size + 1, 8, 32, dtype=torch.bfloat16, device=device
        )


class _FakeSpeculativeMambaCache:
    """Speculative-decoding stand-in: adds intermediate_ssm and
    intermediate_conv_window. Real SpeculativeState dataclass — match
    the field shapes the migrate_slot copies.
    """
    def __init__(self, n_layers, size, device):
        self.conv = [
            torch.zeros(n_layers, size + 1, 16, 4, dtype=torch.bfloat16, device=device)
        ]
        self.temporal = torch.zeros(
            n_layers, size + 1, 8, 32, dtype=torch.bfloat16, device=device
        )
        # Speculative-only fields; shapes are arbitrary stand-ins.
        self.intermediate_ssm = torch.zeros(
            n_layers, size + 1, 4, 16, dtype=torch.bfloat16, device=device
        )
        self.intermediate_conv_window = [
            torch.zeros(n_layers, size + 1, 8, 8, dtype=torch.bfloat16, device=device)
        ]


class _FakeMambaPool:
    """Simulates just enough MambaPool surface for migrate_slot.

    Reuses the real MambaPool.migrate_slot bound as an unbound method —
    avoids constructing the full engine-side MambaPool (which requires
    cache_params, layer ids, etc.).
    """
    def __init__(self, size, device, n_layers=4):
        self.size = size
        self.device = device
        self.mamba_cache = _FakeMambaCache(n_layers, size, device)
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device=device)
        # _capped_slots starts empty.

    # Bind the real method.
    from sglang.srt.mem_cache.memory_pool import MambaPool
    migrate_slot = MambaPool.migrate_slot


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pool = _FakeMambaPool(size=10, device=device)

    # Mark slot 5 as live by removing from free_slots.
    pool.free_slots = pool.free_slots[pool.free_slots != 5]
    assert (pool.free_slots == 5).sum().item() == 0
    assert pool.free_slots.numel() == 9

    # Write known bytes into slot 5.
    pool.mamba_cache.conv[0][:, 5, :, :] = 1.5
    pool.mamba_cache.temporal[:, 5, :, :] = -2.25
    print("[setup] slot 5 marked live, conv[*,5]=1.5, temporal[*,5]=-2.25")

    # Migrate 5 → 7.
    ok = pool.migrate_slot(5, 7)
    assert ok, "migrate_slot should return True for free dst"
    print("[migrate 5 → 7] returned True")

    # Verify dst (7) has src's bytes.
    assert (pool.mamba_cache.conv[0][:, 7, :, :] == 1.5).all()
    assert (pool.mamba_cache.temporal[:, 7, :, :] == -2.25).all()
    print("[verify dst] slot 7 has slot 5's original bytes")

    # Verify allocator side: dst removed from free_slots, src in _capped_slots.
    assert (pool.free_slots == 7).sum().item() == 0, "dst still in free_slots!"
    assert (pool.free_slots == 5).sum().item() == 0, "src must NOT be in free_slots"
    assert hasattr(pool, "_capped_slots") and (pool._capped_slots == 5).any().item(), \
        f"src not in _capped_slots: {getattr(pool, '_capped_slots', None)}"
    assert pool.free_slots.numel() == 8, f"free count wrong: {pool.free_slots.numel()}"
    print(f"[verify alloc state] free count={pool.free_slots.numel()}, "
          f"_capped_slots={pool._capped_slots.tolist()}")

    # Edge: migrate_slot to a dst that's already live → False.
    ok2 = pool.migrate_slot(7, 3)  # 7 is now live; 3 is free
    # Wait — 7 is now live (just received). 3 is free. So this is "live src → free dst" again.
    # Test is migrate to a non-free dst:
    pool.free_slots = pool.free_slots[pool.free_slots != 3]  # mark 3 live
    ok3 = pool.migrate_slot(7, 3)
    assert not ok3, f"migrate to non-free dst should return False, got {ok3}"
    print("[edge: dst not free] migrate_slot returned False as expected")

    # Edge: src == dst → False.
    ok4 = pool.migrate_slot(8, 8)
    assert not ok4, f"src == dst should return False, got {ok4}"
    print("[edge: src == dst] migrate_slot returned False as expected")

    # SpeculativeState path: ensure intermediate_ssm and
    # intermediate_conv_window also get copied. Audit-found gap:
    # earlier migrate_slot only copied conv + temporal, silently
    # dropping speculative state.
    spec_pool = _FakeMambaPool(size=10, device=device)
    spec_pool.mamba_cache = _FakeSpeculativeMambaCache(4, 10, device)
    # isinstance(self.mamba_cache, self.SpeculativeState) needs the
    # FakeSpeculativeMambaCache to BE a SpeculativeState. Simplest: add
    # to the real SpeculativeState class hierarchy.
    from sglang.srt.mem_cache.memory_pool import MambaPool
    # Patch isinstance check by setting __class__ to SpeculativeState
    # OR by passing a real SpeculativeState. Simpler: subclass it.
    SpecState = MambaPool.SpeculativeState
    # Build a real SpeculativeState instance with our tensors.
    spec_pool.mamba_cache = SpecState(
        conv=[torch.zeros(4, 11, 16, 4, dtype=torch.bfloat16, device=device)],
        temporal=torch.zeros(4, 11, 8, 32, dtype=torch.bfloat16, device=device),
        intermediate_ssm=torch.zeros(4, 11, 4, 16, dtype=torch.bfloat16, device=device),
        intermediate_conv_window=[
            torch.zeros(4, 11, 8, 8, dtype=torch.bfloat16, device=device)
        ],
    )
    # Mark slot 5 live, populate ALL state fields with known values.
    spec_pool.free_slots = spec_pool.free_slots[spec_pool.free_slots != 5]
    spec_pool.mamba_cache.conv[0][:, 5, :, :] = 1.5
    spec_pool.mamba_cache.temporal[:, 5, :, :] = -2.25
    spec_pool.mamba_cache.intermediate_ssm[:, 5, :, :] = 7.0
    spec_pool.mamba_cache.intermediate_conv_window[0][:, 5, :, :] = -3.5

    ok_spec = spec_pool.migrate_slot(5, 7)
    assert ok_spec
    # Verify all four fields migrated.
    assert (spec_pool.mamba_cache.conv[0][:, 7, :, :] == 1.5).all()
    assert (spec_pool.mamba_cache.temporal[:, 7, :, :] == -2.25).all()
    assert (spec_pool.mamba_cache.intermediate_ssm[:, 7, :, :] == 7.0).all(), \
        "intermediate_ssm not migrated! Speculative state corruption."
    assert (spec_pool.mamba_cache.intermediate_conv_window[0][:, 7, :, :] == -3.5).all(), \
        "intermediate_conv_window not migrated! Speculative state corruption."
    print("[spec state] all 4 fields (conv, temporal, intermediate_ssm, "
          "intermediate_conv_window) migrated correctly")

    print("\nT4 migrate_slot unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
