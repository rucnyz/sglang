"""#182 — boot-time probe for c_m (per-slot mamba migration wall).

Pins the contract that design.md §"Shared cost model" specifies:
`c_m(X) ≈ X / side_stream_bw + per-slot constant` — measured at boot,
fixed for the engine's lifetime, consumed by Phase 3's Admitter
migrate candidate (#183).

Test scope:
- A: probe contract — runs on a real MambaPool, returns a positive
  value in a sane µs range (sentinel band, wide enough for hardware
  variance but narrow enough to catch a buggy probe that returns 0,
  inf, or a clearly-nonsense value).
- B: cost-model wire — `CostModel.c_migrate_us('mamba', N)` returns
  `N × per_slot_us` after the probe; +inf on cold-start; +inf for KV
  (no migrate primitive).
- C: cold-start fail-closed — without the probe, `c_migrate_us` is
  +inf so the Admitter's migrate candidates stay infeasible.

The probe touches a real CUDA stream + tensor copies, so the test
needs GPU. It's hermetic otherwise (no model, no scheduler).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _make_pool(size: int = 8, max_size: int = 16):
    """Real MambaPool — pattern mirrors test_25 in test_mark_no_realloc.py.

    `size >= 3` so the probe's default `src_slot=1, dst_slot=2` are in range.
    """
    import os
    os.environ.pop("SGLANG_ARENA_SHARED", None)
    os.environ.pop("SGLANG_MAMBA_ARENA", None)
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams, Mamba2StateShape,
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
        spec_state_size=8,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        device="cuda:0",
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=max_size,
    )


def test_1_probe_returns_positive_us_in_sane_band():
    """Scenario A: the probe runs on a real MambaPool and returns a
    per-slot wall in a wide-but-bounded range. The actual measured
    value depends on hardware (HBM bandwidth, slot size) but a
    healthy boot must produce *something* in the µs band — 0 means
    the timing broke; 100 ms means the probe is doing something
    catastrophically wrong (e.g. running on the default stream and
    serializing against an unrelated kernel).
    """
    from sglang.srt.budgeter.migrate_probe import measure_mamba_migrate
    pool = _make_pool(size=8)
    per_slot_us = measure_mamba_migrate(pool, n_iters=20, n_warmup=3)
    assert per_slot_us > 0.0, (
        f"BUG (#182): probe returned non-positive wall {per_slot_us!r}. "
        f"Either the CUDA event timing fell apart or the data-copy "
        f"block is a no-op (cache fields empty?)."
    )
    # Sentinel band: 5 µs (~10× CUDA event resolution on Hopper) to
    # 100 ms (~100× a worst-case probe). Below 5 µs the probe is
    # measuring noise — Hopper event resolution is ~0.5-1 µs so
    # anything in that band is almost certainly a timing bug (e.g.
    # event recorded on the wrong stream, or copy block elided).
    # Real values land in the 50-500 µs range on production H200;
    # this band catches order-of-magnitude bugs without locking in
    # hardware-specific numbers.
    assert 5.0 < per_slot_us < 100_000.0, (
        f"#182 probe wall {per_slot_us:.1f} µs outside the sentinel "
        f"band (5 µs, 100 ms). Either the timing is broken or the "
        f"probe is running on the wrong stream / wrong slot indices."
    )
    print(f"  PASS  1  probe returns per-slot wall = {per_slot_us:.1f} µs "
          f"(within sentinel band 5 µs–100 ms)")


def test_2_cost_model_c_migrate_us_scales_linearly():
    """Scenario B: post-probe, `CostModel.c_migrate_us('mamba', N)`
    returns `N × per_slot_us` exactly. Linear scaling is the
    Admitter's contract (each migrated slot pays one per-slot wall).
    """
    from sglang.srt.budgeter.cost_model import (
        CostModel, get_migrate_cost, reset_cost_model,
    )
    from sglang.srt.budgeter.migrate_probe import measure_mamba_migrate
    reset_cost_model()
    pool = _make_pool(size=8)
    per_slot_us = measure_mamba_migrate(pool, n_iters=20)
    get_migrate_cost().set_mamba(per_slot_us)

    cm = CostModel()
    assert cm.c_migrate_us("mamba", 1) == per_slot_us
    assert cm.c_migrate_us("mamba", 4) == 4 * per_slot_us
    assert cm.c_migrate_us("mamba", 16) == 16 * per_slot_us
    # Zero / negative is just 0, not a fault.
    assert cm.c_migrate_us("mamba", 0) == 0.0
    print(f"  PASS  2  c_migrate_us('mamba', N) = N × {per_slot_us:.1f} µs "
          f"(linear scaling pinned at N ∈ {{0, 1, 4, 16}})")


def test_3_cold_start_and_kv_return_inf():
    """Scenario C: fail-closed contract.

    - Cold-start: probe hasn't run → `c_migrate_us('mamba', N)` = +inf
      so the Admitter's migrate candidate is infeasible.
    - KV pool: there is no `migrate_slot` primitive on the KV side —
      `c_migrate_us('kv', N)` must be +inf regardless of N or probe.
    - Unknown pool name: raises ValueError (typo-as-loud-failure;
      mirrors `c_recompute_us`'s pool-string discipline).
    """
    from sglang.srt.budgeter.cost_model import CostModel, reset_cost_model
    reset_cost_model()
    cm = CostModel()
    # Cold-start mamba: probe not run yet.
    assert cm.c_migrate_us("mamba", 1) == float("inf"), (
        "BUG (#182): cold-start `c_migrate_us` must be +inf so "
        "Admitter migrate candidates stay infeasible until the boot "
        "probe lands. Got a finite value — pre-probe leakage."
    )
    # KV: no migrate primitive, ever.
    assert cm.c_migrate_us("kv", 1) == float("inf")
    assert cm.c_migrate_us("kv", 100) == float("inf")
    # Unknown pool: raise loudly (catches typos).
    try:
        cm.c_migrate_us("Mamba", 1)  # capitalization typo
    except ValueError as e:
        assert "unknown pool" in str(e).lower() or "mamba" in str(e).lower()
    else:
        raise AssertionError(
            "c_migrate_us must raise on unknown pool name (got silent "
            "+inf for typo 'Mamba'). Per feedback_no_fallbacks, "
            "unknown pool is a programming bug, not a runtime feature."
        )
    print("  PASS  3  cold-start mamba + KV both return +inf; unknown "
          "pool raises ValueError (fail-closed + fail-loud)")


def test_4_dst_slot_state_preserved():
    """Audit-driven (Lens 1/2): the probe MUST leave `pool.mamba_cache`
    tensors byte-identical to their pre-probe state on the dst slot.

    Pre-#182-audit, the probe copied src→dst on `dst_slot=2` without
    saving/restoring dst's contents. Currently safe only because
    `torch.zeros` boot-inits all slots to zero AND slot 1's contents
    at probe time were also zero — so a zero→zero copy was a no-op.
    The fix backs up + restores dst across the iteration loop so the
    probe is provably side-effect-free on slot tensor contents.
    """
    import torch
    from sglang.srt.budgeter.migrate_probe import measure_mamba_migrate
    pool = _make_pool(size=8)
    cache = pool.mamba_cache

    # Stamp distinctive non-zero values into slot 2 across every
    # tensor the probe's copy block touches.
    for t in cache.conv:
        t[:, 2, ...].fill_(7.0)
    if isinstance(cache.temporal, list):
        for t in cache.temporal:
            t[2, ...].fill_(13.0)
    else:
        cache.temporal[:, 2, ...].fill_(13.0)

    pre_conv = [t[:, 2, ...].clone() for t in cache.conv]
    if isinstance(cache.temporal, list):
        pre_temporal = [t[2, ...].clone() for t in cache.temporal]
    else:
        pre_temporal = cache.temporal[:, 2, ...].clone()

    # Slot 1 (src) holds whatever its boot value is — likely zero.
    # The probe should copy slot 1 → slot 2 inside the loop and
    # restore slot 2 to its pre-probe stamp afterwards.
    measure_mamba_migrate(pool, n_iters=5, n_warmup=2)

    for t, prev in zip(cache.conv, pre_conv):
        assert torch.equal(t[:, 2, ...], prev), (
            f"BUG (#182 audit lens 1/2): probe left slot 2 conv "
            f"tensor with non-original contents — dst restore failed"
        )
    if isinstance(cache.temporal, list):
        for t, prev in zip(cache.temporal, pre_temporal):
            assert torch.equal(t[2, ...], prev), (
                f"BUG (#182 audit lens 1/2): probe left slot 2 "
                f"temporal tensor with non-original contents"
            )
    else:
        assert torch.equal(cache.temporal[:, 2, ...], pre_temporal), (
            "BUG (#182 audit lens 1/2): probe left slot 2 temporal "
            "tensor with non-original contents"
        )
    print("  PASS  4  probe is byte-identical on slot[dst] across the "
          "probe window (save/restore round-trip preserves stamped state)")


def test_5_integration_budget_agent_runs_probe():
    """Audit-driven (Lens 7): exercise the full
    `BudgetAgent._run_migrate_probe` → singleton → `c_migrate_us`
    chain that production wires up. Verifies the WIRE between the
    three layers, not just each in isolation.

    Pre-fix the production wire-up could break (probe runs but
    singleton not updated, or singleton updated but `c_migrate_us`
    reads a different one) with no failing unit test.
    """
    from sglang.srt.budgeter.agent import BudgetAgent
    from sglang.srt.budgeter.cost_model import (
        CostModel, reset_cost_model,
    )
    reset_cost_model()
    pool = _make_pool(size=8)

    # Construct a minimal BudgetAgent: scheduler attribute is unused
    # by `_run_migrate_probe`, so a bare stub suffices. `__new__`
    # bypasses the heavy `__init__` (which needs a real scheduler).
    ba = BudgetAgent.__new__(BudgetAgent)
    ba._migrate_probe_warned = False

    ba._run_migrate_probe(pool)

    cm = CostModel()
    one_slot = cm.c_migrate_us("mamba", 1)
    four_slots = cm.c_migrate_us("mamba", 4)
    assert one_slot != float("inf"), (
        "BUG (#182 wire): probe ran but `c_migrate_us` still +inf — "
        "singleton populated by probe is NOT the one CostModel reads"
    )
    assert one_slot > 0.0
    assert four_slots == 4.0 * one_slot, (
        f"BUG (#182 wire): post-probe linearity broken; "
        f"1 slot = {one_slot}, 4 slots = {four_slots} (expected "
        f"{4*one_slot})"
    )
    print(f"  PASS  5  BudgetAgent._run_migrate_probe end-to-end: "
          f"probe → singleton → CostModel.c_migrate_us('mamba', 1) = "
          f"{one_slot:.1f} µs (was +inf pre-probe)")


def main() -> int:
    tests = [
        test_1_probe_returns_positive_us_in_sane_band,
        test_2_cost_model_c_migrate_us_scales_linearly,
        test_3_cold_start_and_kv_return_inf,
        test_4_dst_slot_state_preserved,
        test_5_integration_budget_agent_runs_probe,
    ]
    print(f"\n#182 c_m migrate probe tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#182: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
