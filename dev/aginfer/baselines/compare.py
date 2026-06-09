"""
Comparison harness: replay a synthetic multi-session agent workload through
every paper §8 policy AND OursGreedy, score each by the paper's reward
decomposition (r1 saved prefill, r2 migration, r3 holding), and print the
table.

Each policy gets its own *copy* of the world state, so their tier assignments
don't bleed into each other. Events come from a single fixed sequence so the
comparison is apples-to-apples.

Run:
    cd /scratch/yuzhou/projects/sglang/dev/aginfer
    python -m baselines.compare
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .base import (
    Action,
    ReuseUnit,
    Scope,
    SchedulerState,
    Tier,
    TierUsage,
    UnitType,
)
from .continuum import ContinuumPolicy
from .costs import default_costs
from .infercept import InferCeptPolicy
from .kvflow import KVFlowPolicy
from .lru import LRUPolicy
from .ours_greedy import (
    OursGreedyPolicy,
    holding_unit_cost,
    migration_cost_effective,
    reload_cost,
)
from .thunder_agent import ThunderAgentPolicy


@dataclass
class WorldConfig:
    n_sessions: int = 24
    units_per_session: int = 8
    n_events: int = 500
    # Capacity in bytes -- sized so HBM is *under-provisioned* and policies
    # have to actively evict; otherwise every policy looks identical at 100%
    # hit rate. The numbers below give:
    #   total live unit bytes ~  24*8 * 2 KB/tok * 2 K tok = ~750 MB
    #   HBM cap                 200 MB  (~27% of working set)
    #   DRAM cap              1024 MB  (~140%)
    #   disk cap             unlimited
    # This mirrors the regime-3 ratio in our V4-Flash runs (cap 256K HiCache
    # ON, max swa usage ~97 %).
    hbm_cap: int = 200 * 2**20
    dram_cap: int = 1024 * 2**20
    disk_cap: int = 64 * 2**30
    # Unit-size distribution.
    bytes_per_token: int = 2048   # MLA + sidecar pools roughly 2 KB / token
    tokens_min: int = 512
    tokens_max: int = 4096
    # Workload mix -- bias toward llm_prefill (the "hit/miss" trigger) and
    # memory_pressure (the "now evict" trigger) so policies do something.
    p_tool_call: float = 0.15
    p_memory_pressure: float = 0.35
    p_session_arrival: float = 0.10
    p_llm_prefill: float = 0.30
    p_tool_call_end: float = 0.10
    # Reward weighting
    prefill_cost_per_token: float = 5e-5    # ~20K tok/s prefill on V4-Flash


def _build_units(cfg: WorldConfig, rng: random.Random) -> Dict[str, ReuseUnit]:
    units: Dict[str, ReuseUnit] = {}
    for s in range(cfg.n_sessions):
        sess = f"s{s:02d}"
        for k in range(cfg.units_per_session):
            uid = f"{sess}-u{k}"
            n_tok = rng.randint(cfg.tokens_min, cfg.tokens_max)
            u = ReuseUnit(
                id=uid,
                type=rng.choice(list(UnitType)),
                scope=rng.choice([Scope.SESSION, Scope.TENANT, Scope.GLOBAL]),
                n_tokens=n_tok,
                n_bytes=n_tok * cfg.bytes_per_token,
                tier=Tier.HBM if rng.random() < 0.5 else Tier.DRAM,
                age_seconds=rng.uniform(0, 120),
                p_hat=rng.random(),
                lambda_rate=rng.uniform(0.05, 1.0),
                holders=[sess],
            )
            units[uid] = u
    return units


def _build_event_stream(
    cfg: WorldConfig, units: Dict[str, ReuseUnit], rng: random.Random
) -> List[Tuple[float, str, str, List[str]]]:
    """Return list of (t, event_kind, session_id, decision_set)."""
    sessions = sorted({u.holders[0] for u in units.values() if u.holders})
    weights = [
        ("session_arrival", cfg.p_session_arrival),
        ("llm_prefill", cfg.p_llm_prefill),
        ("tool_call_start", cfg.p_tool_call),
        ("tool_call_end", cfg.p_tool_call_end),
        ("memory_pressure", cfg.p_memory_pressure),
    ]
    kinds, probs = zip(*weights)

    events: List[Tuple[float, str, str, List[str]]] = []
    t = 0.0
    for _ in range(cfg.n_events):
        t += rng.expovariate(2.0)  # ~2 events/sec on average
        k = rng.choices(kinds, probs)[0]
        sess = rng.choice(sessions)
        if k == "memory_pressure":
            d_set = list(units.keys())
        else:
            d_set = [uid for uid, u in units.items() if u.holders and u.holders[0] == sess]
        events.append((t, k, sess, d_set))
    return events


def _apply_action(
    units: Dict[str, ReuseUnit],
    usage: TierUsage,
    action: Action,
    costs,
) -> float:
    """Apply an action to (units, usage) in place. Return the migration cost
    paid (r2 contribution)."""
    r2 = 0.0
    for uid, target in action.assignments:
        u = units[uid]
        if u.tier == target:
            continue
        mig = migration_cost_effective(u, u.tier, target, usage.bw_free, costs)
        if mig == float("inf"):
            continue
        r2 += mig
        # Update occupancy.
        usage.used_bytes[u.tier] = max(0, usage.used_bytes[u.tier] - u.n_bytes)
        if target != Tier.DROP:
            usage.used_bytes[target] = usage.used_bytes.get(target, 0) + u.n_bytes
        u.tier = target
    return r2


def _saved_prefill(units, hit_uids, costs, pi_u) -> float:
    """r1 = sum_{u in H_t} R(u,0) - R(u, tier(u))."""
    r1 = 0.0
    for uid in hit_uids:
        u = units[uid]
        r1 += reload_cost(u, Tier.DROP, costs, pi_u) - reload_cost(u, u.tier, costs, pi_u)
    return r1


def _holding_step(units, usage, dt: float, costs) -> float:
    """r3 = sum over units of h_tier(used) * b_u * dt."""
    r3 = 0.0
    for u in units.values():
        if u.tier == Tier.DROP:
            continue
        h = holding_unit_cost(
            u.tier,
            usage.used_bytes.get(u.tier, 0),
            usage.capacity_bytes.get(u.tier, 0),
            costs,
        )
        r3 += h * u.n_bytes * dt
    return r3


@dataclass
class PolicyScore:
    name: str
    r1: float = 0.0   # saved prefill (higher = better)  -- units of seconds
    r2: float = 0.0   # migration paid (lower = better)  -- units of seconds
    r3: float = 0.0   # holding paid (lower = better)    -- bytes*sec*coef
    n_actions: int = 0
    n_hits: int = 0
    n_misses: int = 0

    # Wall-clock-flavored counters used to derive throughput / total runtime.
    # These are accumulated by _simulate_policy and read by main()/sweep_seeds.
    trace_duration_s: float = 0.0   # last_event.t -- "the workload's own length"
    total_workload_tokens: int = 0  # all unit n_tokens seen at hit/miss events
    prefill_paid_s: float = 0.0     # sum over miss units of pi_u * n_tokens

    @property
    def reward(self) -> float:
        return self.r1 - self.r2 - self.r3

    @property
    def total_runtime_s(self) -> float:
        """Wall-clock equivalent: workload's intrinsic duration + re-prefill
        time the policy failed to avoid + migration time the policy chose to
        pay. Holding cost (r3) is *not* added -- it is the sustainability
        penalty (memory hogging) that doesn't appear on wall-clock until
        someone OOMs."""
        return self.trace_duration_s + self.prefill_paid_s + self.r2

    @property
    def throughput_tok_per_s(self) -> float:
        if self.total_runtime_s <= 0:
            return 0.0
        return self.total_workload_tokens / self.total_runtime_s


def _simulate_policy(
    policy,
    units0: Dict[str, ReuseUnit],
    usage0: TierUsage,
    events: List[Tuple[float, str, str, List[str]]],
    costs,
    pi_u: float,
) -> PolicyScore:
    units = copy.deepcopy(units0)
    usage = copy.deepcopy(usage0)
    score = PolicyScore(name=policy.name)
    prev_t = 0.0

    for t, kind, sess, d_set in events:
        dt = max(0.0, t - prev_t)
        # Age all units by dt and integrate holding.
        for u in units.values():
            u.age_seconds += dt
        score.r3 += _holding_step(units, usage, dt, costs)

        state = SchedulerState(
            t=t,
            units=units,
            tier_usage=usage,
            event_kind=kind,
            event_session_id=sess,
            decision_set=d_set,
        )

        # On llm_prefill / tool_call_end, count an implicit hit/miss for the
        # session's units: tier HBM/DRAM = hit, DROP = miss.
        if kind in ("llm_prefill", "tool_call_end"):
            hit_uids = [
                uid for uid in d_set
                if units[uid].tier in (Tier.HBM, Tier.DRAM)
            ]
            score.n_hits += len(hit_uids)
            score.n_misses += len(d_set) - len(hit_uids)
            score.r1 += _saved_prefill(units, hit_uids, costs, pi_u)
            # Accumulate the workload's total token volume (for throughput)
            # and the wall-clock prefill seconds the policy actually had to
            # pay (= units missed at HBM/DRAM, served from DROP).
            for uid in d_set:
                u = units[uid]
                score.total_workload_tokens += u.n_tokens
                if u.tier == Tier.DROP:
                    score.prefill_paid_s += pi_u * u.n_tokens
                    # Re-prefill brings the unit back to HBM.
                    u.tier = Tier.HBM
                    usage.used_bytes[Tier.HBM] = usage.used_bytes.get(Tier.HBM, 0) + u.n_bytes

        action = policy.decide(state)
        score.r2 += _apply_action(units, usage, action, costs)
        score.n_actions += len(action.assignments)
        prev_t = t

    score.trace_duration_s = events[-1][0] if events else 0.0
    return score


def main() -> None:
    cfg = WorldConfig()
    rng = random.Random(20260523)
    costs = default_costs()
    units = _build_units(cfg, rng)
    cap = {Tier.HBM: cfg.hbm_cap, Tier.DRAM: cfg.dram_cap, Tier.DISK: cfg.disk_cap}
    used = {Tier.HBM: 0, Tier.DRAM: 0, Tier.DISK: 0}
    for u in units.values():
        used[u.tier] = used.get(u.tier, 0) + u.n_bytes
    # bw_free is keyed by (src, dst); value is the *currently free* bandwidth
    # (bytes/sec). Initialize to bw_total per pair so the steady state lets
    # migrations happen at line rate; the cost function penalizes saturation.
    bw_free = {pair: bw for pair, bw in costs.bw.items()}
    usage = TierUsage(used_bytes=used, capacity_bytes=cap, bw_free=bw_free)
    events = _build_event_stream(cfg, units, rng)
    policies = [
        LRUPolicy(),
        ThunderAgentPolicy(),
        InferCeptPolicy(),
        ContinuumPolicy(ttl_seconds=90.0, pin_threshold=0.4),
        KVFlowPolicy(),
        OursGreedyPolicy(costs, prefill_cost_per_token=cfg.prefill_cost_per_token),
    ]

    print(f"# Workload: {cfg.n_sessions} sessions × {cfg.units_per_session} units, "
          f"{cfg.n_events} events, HBM cap {cfg.hbm_cap / 2**30:.1f} GB")
    print(f"# Reward = r1 (saved prefill, +) − r2 (migration, −) − r3 (holding, −)")
    print(f"# total_runtime_s = trace_duration + prefill_paid_s + r2 (excludes r3 holding)")
    print()

    header = (
        f"{'policy':<14} {'r1_saved':>11} {'r2_migr':>10} {'r3_hold':>10} "
        f"{'reward':>11} {'hit%':>6} {'prefill_paid':>13} {'total_runtime':>14} "
        f"{'throughput':>12}"
    )
    print(header)
    print("-" * len(header))
    scores: List[PolicyScore] = []
    for p in policies:
        s = _simulate_policy(p, units, usage, events, costs, cfg.prefill_cost_per_token)
        scores.append(s)
        total = s.n_hits + s.n_misses
        hit_pct = (100.0 * s.n_hits / total) if total else 0.0
        print(
            f"{s.name:<14} {s.r1:>11.3e} {s.r2:>10.3e} {s.r3:>10.3e} "
            f"{s.reward:>11.3e} {hit_pct:>6.1f} {s.prefill_paid_s:>13.3e} "
            f"{s.total_runtime_s:>14.3e} {s.throughput_tok_per_s:>12.2e}"
        )

    print()
    ours = next(s for s in scores if s.name == "ours_greedy")
    print("# Relative to ours_greedy:")
    print(f"  {'policy':<14} {'rel_reward':>10} {'rel_runtime':>11} {'rel_throughput':>14}")
    for s in scores:
        rel_r = s.reward / ours.reward if ours.reward else 0
        rel_run = s.total_runtime_s / ours.total_runtime_s if ours.total_runtime_s else 0
        rel_tp = s.throughput_tok_per_s / ours.throughput_tok_per_s if ours.throughput_tok_per_s else 0
        print(f"  {s.name:<14} {rel_r:>10.3f} {rel_run:>11.3f} {rel_tp:>14.3f}")


if __name__ == "__main__":
    main()
