"""m2k cap floor: after m2k fires shrink the mamba pool, max_running
must stay at boot value as long as mamba has enough physical slots.

Uses the real MambaSlotAllocator to verify that the cap floor
correctly sizes the allocator and that all expected slots are
allocatable after m2k-induced shrink.
"""
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BOOT_MAMBA = 977
BOOT_MAX_RUNNING = 195
RATIO = BOOT_MAMBA // BOOT_MAX_RUNNING  # 5


def _compute_cap(mamba_live):
    """New formula: floor at min(boot_cap, mamba_live // 2)."""
    raw = mamba_live // RATIO
    floor = min(BOOT_MAX_RUNNING, mamba_live // 2)
    return max(raw, floor)


def test_after_m2k_shrink_boot_cap_preserved():
    """mamba 977→461 via m2k fires. Floor keeps cap at 195.
    Real allocator can serve all 195 concurrent requests."""
    alloc = MambaSlotAllocator(size=BOOT_MAMBA, device=DEVICE, max_size=BOOT_MAMBA)
    alloc.set_capacity(461)  # simulate m2k shrink

    cap = _compute_cap(461)
    assert cap == 195, f"cap should be boot value 195, got {cap}"

    alloc.set_capacity(cap)
    slots = alloc.alloc(195)
    assert slots is not None and slots.numel() == 195, "all 195 slots allocatable"
    alloc.free(slots)


def test_boundary_390_still_full_cap():
    """mamba=390 (=195×2). Cap stays 195. All 195 allocatable from the
    real CappedFreeList."""
    alloc = MambaSlotAllocator(size=BOOT_MAMBA, device=DEVICE, max_size=BOOT_MAMBA)
    alloc.set_capacity(390)
    cap = _compute_cap(390)
    assert cap == 195

    alloc.set_capacity(cap)
    slots = alloc.alloc(195)
    assert slots is not None and slots.numel() == 195
    alloc.free(slots)


def test_small_mamba_graceful_cap():
    """mamba=300. Cap = 150 (300//2). 150 slots allocatable."""
    alloc = MambaSlotAllocator(size=BOOT_MAMBA, device=DEVICE, max_size=BOOT_MAMBA)
    alloc.set_capacity(300)
    cap = _compute_cap(300)
    assert cap == 150

    alloc.set_capacity(cap)
    slots = alloc.alloc(150)
    assert slots is not None and slots.numel() == 150
    alloc.free(slots)


def test_k2m_grow_increases_cap():
    """k2m: mamba grows to 1200. Cap = 240 (1200//5). All 240 allocatable."""
    alloc = MambaSlotAllocator(size=1200, device=DEVICE, max_size=1200)
    cap = _compute_cap(1200)
    assert cap == 240

    alloc.set_capacity(cap)
    slots = alloc.alloc(240)
    assert slots is not None and slots.numel() == 240
    alloc.free(slots)


def test_progressive_shrink_always_allocatable():
    """At every shrink level, cap slots are actually allocatable."""
    for mamba_live in [900, 700, 500, 390, 300, 200, 100]:
        alloc = MambaSlotAllocator(size=BOOT_MAMBA, device=DEVICE, max_size=BOOT_MAMBA)
        alloc.set_capacity(mamba_live)

        cap = _compute_cap(mamba_live)
        alloc.set_capacity(cap)

        slots = alloc.alloc(cap)
        assert slots is not None and slots.numel() == cap, (
            f"mamba={mamba_live}: expected {cap} allocatable, got {slots.numel() if slots is not None else 0}"
        )
        alloc.free(slots)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
