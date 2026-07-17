"""MambaPool.State.bytes_per_slot — LPB per-slot denominator correctness.

The LPB eviction priority divides a node's recompute cost by the bytes its
eviction frees; for a mamba snapshot that is `mamba_value.numel() *
bytes_per_mamba_slot`. Two prior implementations got the constant wrong:

  1. `mem_usage_bytes() // (max_size+1)` — arena-backed temporal tensors have
     VA-sized leading dims, inflating the result ~150x (observed 6.4 GB/slot
     vs the true 42.6 MB on Nemotron-3-120B TP4).
  2. `sum(prod(shape[1:]))` per field — assumed dim0 is the slot dim, but
     conv tensors and the stacked temporal layout are (num_layers, slots+1,
     ...), so this landed ~13x high in non-arena mode (observed 572 MB) and
     mildly high in arena mode (50.2 MB: temporal right, conv wrong).

The fixed form normalizes each tensor by ITS OWN slot-dim size
(conv/stacked-temporal: dim 1; per-layer temporal list: dim 0), which is
exact even when the leading dim is VA-inflated (numerator and denominator
scale together). Speculative draft caches are excluded (evicting a slot does
not free them). These tests pin all three layouts against the hand-computed
ground truth: layers * (temporal_shape + sum(conv_shapes)) * itemsize.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/yuzhou/projects/sglang/python")

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import MambaPool

LAYERS = 4
SLOTS = 16          # physical slots (alloc_size); tensors get SLOTS+1 rows
VA_ROWS = 257       # arena-style VA-inflated row count (>> SLOTS+1)
TEMPORAL_SHAPE = (8, 32)          # per-slot per-layer temporal state
CONV_SHAPES = ((24, 3), (12, 3))  # two conv kinds, per-slot per-layer
DTYPE = torch.bfloat16
ITEM = 2

TRUTH = (
    LAYERS
    * (int(np.prod(TEMPORAL_SHAPE)) + sum(int(np.prod(s)) for s in CONV_SHAPES))
    * ITEM
)


def _conv():
    """conv: list over kinds, each (layers, slots+1, *conv_shape)."""
    return [
        torch.zeros((LAYERS, SLOTS + 1) + s, dtype=DTYPE) for s in CONV_SHAPES
    ]


def test_1_stacked_temporal():
    """Non-arena default: temporal is ONE (layers, slots+1, ...) tensor."""
    st = MambaPool.State(
        conv=_conv(),
        temporal=torch.zeros((LAYERS, SLOTS + 1) + TEMPORAL_SHAPE, dtype=DTYPE),
    )
    got = st.bytes_per_slot()
    assert got == TRUTH, f"stacked: {got} != {TRUTH}"
    print(f"test_1 OK  (stacked temporal: {got} == truth {TRUTH})")


def test_2_perlayer_temporal():
    """SGLANG_MAMBA_PERLAYER: temporal is a list of (slots+1, ...) tensors."""
    st = MambaPool.State(
        conv=_conv(),
        temporal=[
            torch.zeros((SLOTS + 1,) + TEMPORAL_SHAPE, dtype=DTYPE)
            for _ in range(LAYERS)
        ],
    )
    got = st.bytes_per_slot()
    assert got == TRUTH, f"per-layer: {got} != {TRUTH}"
    print(f"test_2 OK  (per-layer temporal: {got} == truth {TRUTH})")


def test_3_arena_va_inflated_rows():
    """Arena mode: per-layer temporal tensors span the VA row count, far
    larger than the physical slot count. bytes_per_slot must be unchanged."""
    st = MambaPool.State(
        conv=_conv(),
        temporal=[
            torch.zeros((VA_ROWS,) + TEMPORAL_SHAPE, dtype=DTYPE)
            for _ in range(LAYERS)
        ],
    )
    got = st.bytes_per_slot()
    assert got == TRUTH, f"arena VA rows: {got} != {TRUTH}"
    print(f"test_3 OK  (VA-inflated rows {VA_ROWS}: {got} == truth {TRUTH}, "
          f"no inflation)")


def test_4_speculative_extras_excluded():
    """SpeculativeState draft caches must not leak into per-slot bytes."""
    st = MambaPool.SpeculativeState(
        conv=_conv(),
        temporal=torch.zeros((LAYERS, SLOTS + 1) + TEMPORAL_SHAPE, dtype=DTYPE),
        intermediate_ssm=torch.zeros((LAYERS, 999) + TEMPORAL_SHAPE, dtype=DTYPE),
        intermediate_conv_window=[
            torch.zeros((LAYERS, 999) + s, dtype=DTYPE) for s in CONV_SHAPES
        ],
    )
    got = st.bytes_per_slot()
    assert got == TRUTH, f"speculative extras leaked: {got} != {TRUTH}"
    print(f"test_4 OK  (speculative draft caches excluded: {got} == {TRUTH})")


def test_5_old_formulas_would_be_wrong():
    """Regression guard: document WHY the two prior formulas fail on the
    stacked layout (so a future refactor can't silently reintroduce them)."""
    conv = _conv()
    temporal = torch.zeros((LAYERS, SLOTS + 1) + TEMPORAL_SHAPE, dtype=DTYPE)
    st = MambaPool.State(conv=conv, temporal=temporal)

    # prior form 2: prod(shape[1:]) per field, dim0 assumed = slots
    wrong2 = sum(
        int(np.prod(t.shape[1:])) * ITEM for t in conv
    ) + int(np.prod(temporal.shape[1:])) * ITEM
    assert wrong2 != TRUTH and wrong2 == TRUTH * (SLOTS + 1) // LAYERS, \
        "expected the interim formula to be (slots+1)/layers too high"
    print(f"test_5 OK  (interim formula would give {wrong2} = "
          f"{(SLOTS+1)/LAYERS:.2f}x truth — pinned as wrong)")


if __name__ == "__main__":
    test_1_stacked_temporal()
    test_2_perlayer_temporal()
    test_3_arena_va_inflated_rows()
    test_4_speculative_extras_excluded()
    test_5_old_formulas_would_be_wrong()
    print("\nALL TESTS PASSED")
