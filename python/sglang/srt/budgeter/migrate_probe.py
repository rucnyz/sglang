"""Boot-time probe for `c_m(X)` — the migration data-copy wall.

design.md §"Shared cost model" specifies `c_m(X) ≈ X / side_stream_bw +
per-slot constant` as a boot-time probe. The Admitter's `own_migrate` /
`cross_migrate` candidates (Phase 3) consume it; before that, the
probe is dormant cost-model state, deterministic at boot.

What is measured: the per-slot wall to copy mamba conv+temporal state
on a side CUDA stream. This is the dominant cost of
`MambaPool.migrate_slot` — the allocator-side bookkeeping (`free_slots`
/ `_capped_slots` mutations) is O(small-tensor) and negligible. We
probe the data copy directly with the same tensor slice pattern
`migrate_slot` uses, so the measured constant tracks the real
production path.

Probe contract:
- runs once, at the moment the actuator chain is built
  (`BudgetAgent._ensure_actuator_chain`)
- N iterations + warmup; reports median to absorb single-iter noise
- side stream (not the default decode stream) so the probe doesn't
  steal decode wall during warmup
- the dst slot's tensor contents are saved before the iteration loop
  and restored after, so any in-flight kernel reading slot[dst] sees
  byte-identical state pre/post probe (the copy IS otherwise mutative)
- returns wall in µs per slot

Cost-model exposure: `CostModel.c_migrate_us(pool, n_slots)` returns
`n_slots × measured_per_slot_us`. Until the probe runs (cold-start),
the cost is `+inf` — same fail-closed pattern as `c^xfer` cold-start.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def measure_mamba_migrate(
    pool: Any,
    *,
    n_iters: int = 20,
    n_warmup: int = 3,
    src_slot: int = 1,
    dst_slot: int = 2,
) -> float:
    """Probe per-slot mamba migration data-copy wall on a side stream.

    Returns median wall in microseconds.

    Args:
        pool: real `MambaPool` instance. `pool.size >= 3` so that
            `src=1, dst=2` are in range.
        n_iters: timed iterations. Median absorbs scheduler / driver
            noise on individual calls.
        n_warmup: untimed warmup runs to settle CUDA context + first-
            touch page-table commits.
        src_slot / dst_slot: any two distinct in-range slot indices.
            Defaults to 1 and 2.

    The copies mirror `MambaPool.migrate_slot`'s data-copy block —
    conv tensors + temporal tensor(s) + speculative-state intermediates
    when `mamba_cache` is a `SpeculativeState`. Allocator-side state
    (`free_slots`, `_capped_slots`) is never touched. The dst slot's
    pre-probe tensor contents are cloned and restored at the end of
    the iteration loop so that any in-flight kernel still reading
    slot[dst] sees byte-identical state across the probe window.
    """
    # Lazy import to avoid module-cycle (memory_pool imports nothing from
    # us, but this keeps the budgeter package free of mem_cache deps at
    # import time).
    from sglang.srt.mem_cache.memory_pool import MambaPool

    if pool is None:
        raise ValueError("measure_mamba_migrate: pool is None")
    if pool.size < 3:
        raise ValueError(
            f"measure_mamba_migrate: pool.size={pool.size} too small "
            f"(need ≥ 3 slots so src=1, dst=2 are addressable)"
        )

    cache = pool.mamba_cache
    device = pool.device
    has_spec = isinstance(cache, MambaPool.SpeculativeState)

    # Side stream so the probe doesn't queue behind the default stream's
    # in-flight work (matches production's migrate_slot which is also a
    # side-stream copy per design.md §"Page ownership state" property A2).
    probe_stream = torch.cuda.Stream(device=device)

    def _copy_block(src: int, dst: int) -> None:
        # Mirrors `MambaPool.migrate_slot`'s data-copy body — gating
        # the speculative branch on the same `isinstance` check
        # `migrate_slot` uses so the probe doesn't drift if a future
        # subclass of `State` exposes `intermediate_ssm` for a different
        # reason.
        for t in cache.conv:
            t[:, dst, ...].copy_(t[:, src, ...])
        if isinstance(cache.temporal, list):
            for t in cache.temporal:
                t[dst, ...].copy_(t[src, ...])
        else:
            t = cache.temporal
            t[:, dst, ...].copy_(t[:, src, ...])
        if has_spec:
            t = cache.intermediate_ssm
            t[:, dst, ...].copy_(t[:, src, ...])
            for t in cache.intermediate_conv_window:
                t[:, dst, ...].copy_(t[:, src, ...])

    # Snapshot slot[dst] state across all tensors the copy touches so
    # we can restore byte-identical contents after probing.
    dst_backup_conv = [t[:, dst_slot, ...].clone() for t in cache.conv]
    if isinstance(cache.temporal, list):
        dst_backup_temporal = [t[dst_slot, ...].clone() for t in cache.temporal]
    else:
        dst_backup_temporal = cache.temporal[:, dst_slot, ...].clone()
    if has_spec:
        dst_backup_ssm = cache.intermediate_ssm[:, dst_slot, ...].clone()
        dst_backup_iconv = [
            t[:, dst_slot, ...].clone() for t in cache.intermediate_conv_window
        ]

    def _restore_dst() -> None:
        for t, backup in zip(cache.conv, dst_backup_conv):
            t[:, dst_slot, ...].copy_(backup)
        if isinstance(cache.temporal, list):
            for t, backup in zip(cache.temporal, dst_backup_temporal):
                t[dst_slot, ...].copy_(backup)
        else:
            cache.temporal[:, dst_slot, ...].copy_(dst_backup_temporal)
        if has_spec:
            cache.intermediate_ssm[:, dst_slot, ...].copy_(dst_backup_ssm)
            for t, backup in zip(cache.intermediate_conv_window, dst_backup_iconv):
                t[:, dst_slot, ...].copy_(backup)

    # CUDA events bracket each iteration on the probe stream.
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    try:
        with torch.cuda.stream(probe_stream):
            for _ in range(n_warmup):
                _copy_block(src_slot, dst_slot)
            torch.cuda.synchronize(device)

            for i in range(n_iters):
                start_events[i].record(probe_stream)
                _copy_block(src_slot, dst_slot)
                end_events[i].record(probe_stream)

        torch.cuda.synchronize(device)
    finally:
        _restore_dst()
        torch.cuda.synchronize(device)

    walls_us = [
        start_events[i].elapsed_time(end_events[i]) * 1000.0
        for i in range(n_iters)
    ]
    walls_us.sort()
    median_us = walls_us[len(walls_us) // 2]

    logger.info(
        "MigrateProbe[mamba]: per-slot wall = %.1f µs (median over "
        "%d iters; p99 %.1f µs; min %.1f µs)",
        median_us, n_iters, walls_us[-1], walls_us[0],
    )
    return float(median_us)
