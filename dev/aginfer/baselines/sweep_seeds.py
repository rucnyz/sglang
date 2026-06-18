"""
Sensitivity sweep: run compare.py over multiple workload seeds; report the
mean and std of per-policy reward so a single-seed result can't game the
table. Writes a CSV-style summary to stdout.
"""
from __future__ import annotations

import random
import statistics
from typing import Dict, List

from .compare import (
    PolicyScore,
    WorldConfig,
    _build_event_stream,
    _build_units,
    _simulate_policy,
)
from .base import Tier, TierUsage
from .continuum import ContinuumPolicy
from .costs import default_costs
from .infercept import InferCeptPolicy
from .kvflow import KVFlowPolicy
from .lru import LRUPolicy
from .ours_greedy import OursGreedyPolicy
from .thunder_agent import ThunderAgentPolicy


def main(n_seeds: int = 8) -> None:
    cfg = WorldConfig()
    costs = default_costs()
    policy_factories = [
        ("lru", lambda: LRUPolicy()),
        ("thunder_agent", lambda: ThunderAgentPolicy()),
        ("infercept", lambda: InferCeptPolicy()),
        ("continuum", lambda: ContinuumPolicy(ttl_seconds=90.0, pin_threshold=0.4)),
        ("kvflow", lambda: KVFlowPolicy()),
        ("ours_greedy", lambda: OursGreedyPolicy(costs, prefill_cost_per_token=cfg.prefill_cost_per_token)),
    ]
    rewards: Dict[str, List[float]] = {n: [] for n, _ in policy_factories}
    hits:    Dict[str, List[float]] = {n: [] for n, _ in policy_factories}
    runtimes: Dict[str, List[float]] = {n: [] for n, _ in policy_factories}
    throughputs: Dict[str, List[float]] = {n: [] for n, _ in policy_factories}

    for seed in range(20260000, 20260000 + n_seeds):
        rng = random.Random(seed)
        units = _build_units(cfg, rng)
        cap = {Tier.HBM: cfg.hbm_cap, Tier.DRAM: cfg.dram_cap, Tier.DISK: cfg.disk_cap}
        used = {Tier.HBM: 0, Tier.DRAM: 0, Tier.DISK: 0}
        for u in units.values():
            used[u.tier] = used.get(u.tier, 0) + u.n_bytes
        bw_free = {pair: bw for pair, bw in costs.bw.items()}
        usage = TierUsage(used_bytes=used, capacity_bytes=cap, bw_free=bw_free)
        events = _build_event_stream(cfg, units, rng)

        for name, factory in policy_factories:
            policy = factory()
            s: PolicyScore = _simulate_policy(
                policy, units, usage, events, costs, cfg.prefill_cost_per_token
            )
            rewards[name].append(s.reward)
            total = s.n_hits + s.n_misses
            hits[name].append(100.0 * s.n_hits / total if total else 0.0)
            runtimes[name].append(s.total_runtime_s)
            throughputs[name].append(s.throughput_tok_per_s)

    print(f"# n_seeds={n_seeds}")
    hdr = (
        f"{'policy':<14} {'reward_mean':>12} {'reward_std':>11} "
        f"{'hit%_mean':>10} {'runtime_s_mean':>15} {'runtime_s_std':>14} "
        f"{'throughput_mean':>16} {'throughput_std':>15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, _ in policy_factories:
        rm = statistics.mean(rewards[name])
        rs = statistics.stdev(rewards[name]) if len(rewards[name]) > 1 else 0.0
        hm = statistics.mean(hits[name])
        rt_m = statistics.mean(runtimes[name])
        rt_s = statistics.stdev(runtimes[name]) if len(runtimes[name]) > 1 else 0.0
        tp_m = statistics.mean(throughputs[name])
        tp_s = statistics.stdev(throughputs[name]) if len(throughputs[name]) > 1 else 0.0
        print(
            f"{name:<14} {rm:>12.3e} {rs:>11.3e} {hm:>10.1f} "
            f"{rt_m:>15.3e} {rt_s:>14.3e} {tp_m:>16.3e} {tp_s:>15.3e}"
        )


if __name__ == "__main__":
    main()
