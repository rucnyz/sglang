"""
Adapter that turns a ``baselines.Policy``-style value rule into the
``score_for_eviction(node, layer) -> float`` callable that sglang's
patched ``UnifiedRadixCache`` (env var ``SGLANG_KV_POLICY_MODULE``)
consumes.

`node` is an ``sglang.srt.mem_cache.unified_radix_cache.UnifiedTreeNode``;
`layer` is ``sglang.srt.mem_cache.unified_cache_components.EvictLayer``
(.DEVICE for HBM, .HOST for DRAM).  Lower score = evict first (matches
the default LRU semantics where ``last_access_time`` is the heap key).

We expose four callables so the same launch script can swap policies by
flipping one env var:

  - ``lru_score``                  (= stock sglang behaviour; sanity check)
  - ``lfu_score``                  (least-frequently-used first)
  - ``recency_freq_score``         (LFU normalised by age; Continuum-ish)
  - ``ours_greedy_score``          (paper §7 per-unit value, the load-bearing one)

For Run H we wire ``ours_greedy_score``.  The other three are there so a
real-serving ablation matrix is one env-var change away.

Post-T17 note: residence is now a SET per DESIGN §5.  The scorer is
called per-layer per-node, so we construct a single-tier-residence
ReuseUnit reflecting just the layer being scored — the value rule
asks 'what's V_u if this layer's bytes were the only copy?'.  The
holding-cost denominator (authoritative_tier) collapses to the layer
under inspection.  T29's policy-module plugin will pass the daemon's
hint-table state alongside so the scorer can use the full residence
view; for now the inline scorer is layer-local.
"""
from __future__ import annotations

import os
from typing import Any

from .base import ReuseUnit, Scope, Tier, UnitType
from .costs import default_costs
from .ours_greedy import (
    OursGreedyPolicy,
    holding_unit_cost,
    reload_cost,
)


# ---- tunable knobs (overridable via env so we don't need to recompile) ----

_BYTES_PER_TOKEN = int(os.environ.get("AGINFER_BYTES_PER_TOKEN", "2048"))
_PI_U = float(os.environ.get("AGINFER_PI_U", "5e-5"))  # prefill cost (s/tok)

_COSTS = default_costs()

# The FULL subpool name on a single-stack-attention cache matches
# sglang's `ComponentType.FULL` → "full" (see
# UnifiedRadixCache._aginfer_subpool_name).  SWA / Mamba hybrids will
# need this scorer extended; T45/T46 picks up.
_FULL_SUBPOOL = "full"


def _node_n_tokens(node: Any, layer_name: str) -> int:
    """Tokens currently held on the requested layer by the Full component."""
    # ComponentType.FULL is enum value 0 == BASE_COMPONENT_TYPE.
    cd = node.component_data[0]
    if layer_name == "DEVICE":
        return len(cd.value) if cd.value is not None else 0
    if layer_name == "HOST":
        return len(cd.host_value) if cd.host_value is not None else 0
    return 0


def _node_to_unit(node: Any, layer_name: str, current_counter: int) -> ReuseUnit:
    """Approximate a ``ReuseUnit`` from a ``UnifiedTreeNode`` for the
    layer being scored.

    Single-tier residence on purpose: the inline scorer is layer-
    local (asked 'what's the eviction cost of THIS layer's bytes').
    The full multi-tier residence view lives in the daemon's
    SchedulerState and the daemon's V_u is what's authoritative for
    migrate decisions; this scorer is just the cache's heap key.
    """
    n_tokens = _node_n_tokens(node, layer_name)
    age = max(1, current_counter - int(node.last_access_time))
    hits = int(node.hit_count)
    # Reuse-frequency proxy for p_hat: bounded into [0, 1].
    p_hat = min(1.0, hits / age) if age > 0 else 0.0
    # Poisson rate proxy: hits per access-tick (>=1e-3 floor).
    lam = max(1e-3, hits / age) if age > 0 else 1e-3
    tier = Tier.HBM if layer_name == "DEVICE" else Tier.DRAM
    n_bytes_total = n_tokens * _BYTES_PER_TOKEN
    return ReuseUnit(
        id=str(node.id),
        type=UnitType.SESSION,
        scope=Scope.SESSION,
        n_tokens=n_tokens,
        n_bytes_by_tier={tier: {_FULL_SUBPOOL: n_bytes_total}},
        residence=[tier],
        age_seconds=float(age),
        p_hat=p_hat,
        lambda_rate=lam,
        holders=[],
    )


# Lazy global so the per-call hot path is cheap.
_OURS_POLICY: OursGreedyPolicy | None = None


def _get_ours() -> OursGreedyPolicy:
    global _OURS_POLICY
    if _OURS_POLICY is None:
        _OURS_POLICY = OursGreedyPolicy(_COSTS, prefill_cost_per_token=_PI_U)
    return _OURS_POLICY


# Imported lazily inside callables to avoid a hard dep on sglang import order
# when this module is loaded as a side effect of `import baselines` in the
# offline simulator.
def _current_time_counter() -> int:
    from sglang.srt.mem_cache.unified_cache_components import (
        get_and_increase_time_counter,
    )
    return int(get_and_increase_time_counter())


# ----------------------------- public scorers -----------------------------


def lru_score(node: Any, layer: Any) -> float:
    """Stock sglang behaviour. Provided for sanity checks."""
    return float(node.last_access_time)


def lfu_score(node: Any, layer: Any) -> float:
    """Least-frequently-used first (lower hit_count -> evict first)."""
    return float(node.hit_count)


def recency_freq_score(node: Any, layer: Any) -> float:
    """LFU normalised by age. Approximates a TTL/Continuum-style score."""
    age = max(1, _current_time_counter() - int(node.last_access_time))
    hits = int(node.hit_count)
    return float(hits) / float(age)


# T38 (#169) + #176 audit-2: hit-count tie-break bonus.  Small
# enough that age always dominates — for any pair where
# last_access_time differs by even 1, the bonus can't flip
# ordering.
#
# `node.hit_count` in sglang is an UNBOUNDED Python int (see
# unified_radix_cache.py:1703 `node.hit_count += 1` with no cap).
# A long-running deployment can plausibly accumulate hit_counts of
# 2^32 or higher on a hot shared prefix.  Pick a bonus exponent
# that keeps the realistic max bonus << 1.0:
#
#   * 2^-50 gives max bonus 2^50 * 2^-50 = 1.0 (HARD CAP at
#     2^50 ≈ 10^15 hits — still 5e4× below float64's 2^53
#     precision limit on last_access_time addition).
#   * For practical hit_counts (< 2^40, ~ 10^12), max bonus is
#     2^40 * 2^-50 = 2^-10 ≈ 0.001 — six orders of magnitude
#     below the 1.0 minimum age gap.
#
# Round-1 used 2^-31 assuming a 32-bit cap that doesn't exist;
# B4 (round-2) is the regression pin.
_DEFAULT_HIT_COUNT_BONUS = 2.0 ** -50


def default_policy_score(node: Any, layer: Any) -> float:
    """T38 (#169) default-policy scorer — the policy module that
    runs when no daemon is attached.  DESIGN §3 superset framing:
    sglang's historical LRU + hit_count-write-through behavior is
    expressed as aginfer's default policy module.

    Semantics:
      * Base score = ``last_access_time`` (identical to ``lru_score``
        → matches stock sglang behavior in the limit).
      * Hit-count tie-break bonus: ``hit_count * 2^-31``.  Strictly
        smaller than 1.0 (the minimum gap between two distinct
        ``last_access_time`` integers), so two nodes with different
        ``last_access_time`` are ordered by age regardless of hits.
        Two nodes at the SAME ``last_access_time`` (a tie in the
        baseline scorer) are ordered with the higher-hit one kept
        longer.

    Lower score = evict first.  The bonus is conservative on purpose:
    we want LRU-equivalent behavior in 99% of cases AND a sensible
    tie-break for hot leaves whose access timestamp happens to
    coincide (e.g. shared prefix nodes refreshed in the same batch).
    """
    return (
        float(node.last_access_time)
        + float(node.hit_count) * _DEFAULT_HIT_COUNT_BONUS
    )


def ours_greedy_score(node: Any, layer: Any) -> float:
    """Paper §7 per-unit value rule, served as a sglang eviction heap key.

    Heap is a *min*-heap; the smallest key is popped (= evicted) first.
    Stock sglang uses ``last_access_time`` as the key, so the oldest leaf
    leaves first.  Our analogue: directly return ``V_u(authoritative_tier)``
    (saved-prefill term minus the holding tax) so the *least valuable*
    leaf -- the one whose contribution to future hit-rate is smallest --
    is popped first.  Higher V_u stays in the cache.
    """
    layer_name = layer.name.upper()
    now = _current_time_counter()
    u = _node_to_unit(node, layer_name, now)
    pi_u = _PI_U
    tier = u.authoritative_tier
    # Saved prefill at current tier (vs DROP) -- bigger = more valuable to keep.
    save_prefill = u.p_hat * (
        reload_cost(u, Tier.DROP, _COSTS, pi_u)
        - reload_cost(u, tier, _COSTS, pi_u)
    )
    # Holding tax (occupancy-weighted) -- we don't have live pool occupancy
    # here so use h_base * b_u * 1/lambda as the per-unit-time cost
    # amortised over an expected reuse interval (paper §7).
    h_base = _COSTS.h_base[tier]
    hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
    hold = h_base * u.n_bytes * hold_time
    value = save_prefill - hold
    return float(value)
