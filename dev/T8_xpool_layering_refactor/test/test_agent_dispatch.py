"""BudgetAgent._maybe_t8_fire dispatch tests under page-only API."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _build_shell():
    from sglang.srt.arena.cross_pool_actuator import CrossPoolTransferActuator
    from sglang.srt.budgeter.agent import BudgetAgent

    a = BudgetAgent.__new__(BudgetAgent)

    n_token_slots = 32
    tps = 8  # tokens per page → 4 pages total
    free = list(range(1, n_token_slots + 1))
    allocator = SimpleNamespace(
        size=n_token_slots, page_size=1, device="cpu",
        free_pages=torch.tensor(free, dtype=torch.int64),
        release_pages=torch.tensor([], dtype=torch.int64),
        _capped_pages=torch.tensor([], dtype=torch.int64),
    )

    def mark_capped(t):
        target = t.to(torch.int64)
        n = 0
        if allocator.free_pages.numel() > 0:
            mask = torch.isin(allocator.free_pages, target)
            held = allocator.free_pages[mask]
            allocator.free_pages = allocator.free_pages[~mask]
            n += int(held.numel())
            allocator._capped_pages = torch.cat([allocator._capped_pages, held])
        return n
    def unmark_capped(t):
        return 0
    allocator.mark_pages_capped = mark_capped
    allocator.unmark_pages_capped = unmark_capped

    kv_act = MagicMock()
    kv_act.allocator = allocator
    kv_act.n_pages = n_token_slots // tps
    kv_act.expand_pages_to_token_slots = lambda pages: [
        s for p in pages for s in range(p * tps + 1, (p + 1) * tps + 1)
    ]
    kv_act.page_is_fully_free = lambda p, fset: all(
        s in fset for s in range(p * tps + 1, (p + 1) * tps + 1)
    )
    kv_act.live_capacity_tokens = MagicMock(return_value=n_token_slots)
    kv_act.cap_allocator_only = MagicMock()

    actuator = CrossPoolTransferActuator.__new__(CrossPoolTransferActuator)
    mamba_act = MagicMock()
    mamba_act.expand_pages_to_token_slots = lambda pages: list(pages)
    mamba_act.live_capacity_tokens = MagicMock(return_value=8)
    mamba_act.cap_allocator_only = MagicMock()
    mamba_act.n_pages = 4

    actuator.kv_actuator = kv_act
    actuator.mamba_actuator = mamba_act
    actuator.kv = MagicMock()
    actuator.kv.current_capacity_tokens = MagicMock(return_value=n_token_slots)
    actuator.kv._arena = MagicMock()
    actuator.kv._arena.shrink_explicit = MagicMock(return_value=1)
    actuator.kv.n_layers = 1
    actuator.kv.n_kinds = 1
    actuator.kv._pool_name = lambda i: f"k_{i}"
    actuator.mamba = MagicMock()
    actuator.mamba.current_capacity_tokens = MagicMock(return_value=8)
    actuator.mamba._arena = MagicMock()
    actuator.mamba._arena.grow = MagicMock(return_value=1)
    actuator.mamba.n_layers = 1
    actuator.mamba.n_kinds = 1
    actuator.mamba._pool_name = lambda i: f"m_{i}"
    actuator.shared = MagicMock()
    actuator.shared.free_count = MagicMock(return_value=4)

    a._xpool_actuator = actuator

    sched = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    a.scheduler = sched
    a._t8_state = None
    return a


def test_dispatch_emits_fire():
    a = _build_shell()
    snap = {}
    res = a._maybe_t8_fire("kv_to_mamba", unit=1, snapshot=snap)
    assert res is not None
    assert res["direction"] == "kv_to_mamba"
    assert res["unmapped_total"] > 0
    assert "fire_total_us" in res
    assert "xpool_t8_plan_seq" in snap
    print(f"[step6] dispatch: unmap={res['unmapped_total']} grant={res['granted_total']}")


def test_dispatch_refused_when_insufficient_free():
    a = _build_shell()
    # Drop free pages so no page is fully free → planner refuses.
    a._xpool_actuator.kv_actuator.allocator.free_pages = torch.tensor([1], dtype=torch.int64)
    snap = {}
    res = a._maybe_t8_fire("kv_to_mamba", unit=1, snapshot=snap)
    assert res is not None
    assert res["unmapped_total"] == 0
    assert res.get("skipped") == "t8_plan_refused"
    print(f"[step6] dispatch refused: skipped={res.get('skipped')}")


def main():
    test_dispatch_emits_fire()
    test_dispatch_refused_when_insufficient_free()
    print("\nT8 step6 agent dispatch test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
