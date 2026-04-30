"""
Phase 2e.3 — end-to-end smoke test of (StateSpec + LagrangePlanner + ArenaSpec).

Four pools sharing one ChunkArena. Each pool's `marginal_value()` is a
controllable mock. We drive a four-phase trace:

    Phase 0: KV is binding (high marginal). Should grow.
    Phase 1: mamba binding. KV should shrink, mamba should grow.
    Phase 2: LoRA binding.
    Phase 3: prefix binding.

After each phase the planner should have moved the bulk of the budget
to the binding pool while keeping each at >= min_bytes.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/06_planner_smoke.py
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunk_arena import CUDA, ChunkArena, _check  # noqa: E402
from arena_spec import ArenaSpec  # noqa: E402
from lagrange_planner import LagrangePlanner  # noqa: E402


def main() -> int:
    print("== Phase 2e.3: planner smoke test ==")

    _check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    _check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    import torch
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    chunk = 32 * 1024 * 1024
    arena = ChunkArena(
        device_id=dev_id,
        chunk_size=chunk,
        n_handles=8,  # 8 chunks total, ~256 MiB physical
        pool_capacities=[
            ("kv", 6),
            ("mamba", 6),
            ("lora", 6),
            ("prefix", 6),
        ],  # each pool reserves up to 6 chunks of VA, sum = 24 (over-provisioned)
    )
    print(f"arena: 4 pools, each up to 6 chunks of {chunk >> 20} MiB; "
          f"{arena.free_handle_count()} physical handles")

    # Mock marginal-value functions; we'll mutate them between phases.
    mvs = {"kv": [1.0], "mamba": [1.0], "lora": [1.0], "prefix": [1.0]}

    def mk_mv(name: str):
        return lambda: mvs[name][0]

    specs = [
        ArenaSpec(arena, "kv", min_chunks=1, marginal_value_fn=mk_mv("kv")),
        ArenaSpec(arena, "mamba", min_chunks=1, marginal_value_fn=mk_mv("mamba")),
        ArenaSpec(arena, "lora", min_chunks=1, marginal_value_fn=mk_mv("lora")),
        ArenaSpec(arena, "prefix", min_chunks=1, marginal_value_fn=mk_mv("prefix")),
    ]

    # Initial: each pool at its min. That's 4 chunks consumed, 4 free.
    for s in specs:
        s.resize(s.min_bytes())
    print(
        f"after init: "
        + ", ".join(f"{s.name}={arena.pool_mapped_chunks(s.name)}" for s in specs)
        + f", free={arena.free_handle_count()}"
    )

    planner = LagrangePlanner(delta_hyst=0.05)

    # We'll budget total = 8 chunks (= n_handles). Planner must fit
    # exactly within the physical handles available.
    total_budget = 8 * chunk

    def step(phase: str, mv_overrides: dict[str, float]) -> None:
        for k, v in mv_overrides.items():
            mvs[k][0] = v
        decisions = planner.plan(specs, total_budget)
        planner.apply(specs, decisions)
        sizes = ", ".join(
            f"{s.name}={arena.pool_mapped_chunks(s.name)}" for s in specs
        )
        print(f"[{phase:>10}] mvs={mv_overrides}, sizes: {sizes}, "
              f"free_handles={arena.free_handle_count()}")
        # Sanity: total mapped == total_budget / chunk
        total_mapped = sum(arena.pool_mapped_chunks(s.name) for s in specs)
        assert total_mapped == total_budget // chunk, (
            f"total_mapped {total_mapped} != target {total_budget // chunk}"
        )

    # Phase 0: KV binding.
    step("kv-bind", {"kv": 10.0, "mamba": 1.0, "lora": 1.0, "prefix": 1.0})
    assert arena.pool_mapped_chunks("kv") == 5, "KV should win 5 chunks (1 min + 4 from budget)"

    # Phase 1: mamba binding.
    step("mamba-bind", {"kv": 1.0, "mamba": 10.0, "lora": 1.0, "prefix": 1.0})
    assert arena.pool_mapped_chunks("mamba") == 5

    # Phase 2: LoRA binding.
    step("lora-bind", {"kv": 1.0, "mamba": 1.0, "lora": 10.0, "prefix": 1.0})
    assert arena.pool_mapped_chunks("lora") == 5

    # Phase 3: prefix binding.
    step("prefix-bind", {"kv": 1.0, "mamba": 1.0, "lora": 1.0, "prefix": 10.0})
    assert arena.pool_mapped_chunks("prefix") == 5

    # Phase 4: two pools tied; check tie-breaking is at least stable.
    step("kv+mamba-tie", {"kv": 5.0, "mamba": 5.0, "lora": 1.0, "prefix": 1.0})
    # KV and mamba both wanted to grow; with greedy fill and ranked-by-mv
    # iteration, the first-ranked spec hits its max (6), the second
    # takes the rest. Sum should be 6 + n + 1 + 1 = 8, n = 0. So
    # either kv=6 mamba=0... but min=1 prevents 0. Let me reason again:
    # mins consumed: 4. remaining = 4. Greedy: kv (mv=5) gets up to 5
    # (5 = 6-1 headroom), but only 4 to give; kv goes to 1+4=5; remaining=0;
    # mamba stays at min=1.
    assert arena.pool_mapped_chunks("kv") == 5 or arena.pool_mapped_chunks("mamba") == 5

    arena.cleanup()
    print("\n== PASSED: planner moves chunks to the binding pool each phase ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
