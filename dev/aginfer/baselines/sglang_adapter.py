"""
Adapter that turns a ``baselines.Policy``-style value rule into the
``score_for_eviction(node, layer) -> float`` callable that sglang's patched
``UnifiedRadixCache`` (env var ``SGLANG_KV_POLICY_MODULE``) consumes.

`node` is an ``sglang.srt.mem_cache.unified_radix_cache.UnifiedTreeNode``;
`layer` is ``sglang.srt.mem_cache.unified_cache_components.EvictLayer``
(.DEVICE for HBM, .HOST for DRAM).  Lower score = evict first (matches the
default LRU semantics where ``last_access_time`` is the heap key).

We expose four callables so the same launch script can swap policies by
flipping one env var:

  - ``lru_score``                  (= stock sglang behaviour; sanity check)
  - ``lfu_score``                  (least-frequently-used first)
  - ``recency_freq_score``         (LFU normalised by age; Continuum-ish)
  - ``ours_greedy_score``          (paper §7 per-unit value, the load-bearing one)

For Run H we wire ``ours_greedy_score``.  The other three are there so a
real-serving ablation matrix is one env-var change away.

Note on typed reuse units: the V1 wiring treats every tree node as
``UnitType.SESSION / Scope.SESSION`` -- sglang's existing prefix radix
doesn't carry the (platform / tool_def / subagent_ctx) tags from
paper §2.1 yet.  Adding that tagging is a separate piece of work that
lives at request-ingest time (see paper §2.4).  For now the policy
operates on token count + age + hit_count, which is already enough to
exercise the §7 value rule end-to-end.
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
    """Approximate a ``ReuseUnit`` from a ``UnifiedTreeNode``."""
    n_tokens = _node_n_tokens(node, layer_name)
    age = max(1, current_counter - int(getattr(node, "last_access_time", 0)))
    hits = int(getattr(node, "hit_count", 0))
    # Reuse-frequency proxy for p_hat: bounded into [0, 1].
    p_hat = min(1.0, hits / age) if age > 0 else 0.0
    # Poisson rate proxy: hits per access-tick (>=1e-3 floor).
    lam = max(1e-3, hits / age) if age > 0 else 1e-3
    tier = Tier.HBM if layer_name == "DEVICE" else Tier.DRAM
    return ReuseUnit(
        id=str(getattr(node, "id", "?")),
        type=UnitType.SESSION,
        scope=Scope.SESSION,
        n_tokens=n_tokens,
        n_bytes=n_tokens * _BYTES_PER_TOKEN,
        tier=tier,
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
    try:
        from sglang.srt.mem_cache.unified_cache_components import (
            get_and_increase_time_counter,
        )

        # peek without consuming: use the underlying counter if exposed,
        # otherwise read-and-increment (consuming is acceptable for our
        # estimation purposes).
        return int(get_and_increase_time_counter())
    except Exception:
        return 0


# ----------------------------- public scorers -----------------------------


def lru_score(node: Any, layer: Any) -> float:
    """Stock sglang behaviour. Provided for sanity checks."""
    return float(getattr(node, "last_access_time", 0))


def lfu_score(node: Any, layer: Any) -> float:
    """Least-frequently-used first (lower hit_count -> evict first)."""
    return float(getattr(node, "hit_count", 0))


def recency_freq_score(node: Any, layer: Any) -> float:
    """LFU normalised by age. Approximates a TTL/Continuum-style score."""
    age = max(1, _current_time_counter() - int(getattr(node, "last_access_time", 0)))
    hits = int(getattr(node, "hit_count", 0))
    return float(hits) / float(age)


def ours_greedy_score(node: Any, layer: Any) -> float:
    """Paper §7 per-unit value rule, served as a sglang eviction heap key.

    Heap is a *min*-heap; the smallest key is popped (= evicted) first.
    Stock sglang uses ``last_access_time`` as the key, so the oldest leaf
    leaves first.  Our analogue: directly return ``V_u(τ_current)``
    (saved-prefill term minus the holding tax) so the *least valuable*
    leaf -- the one whose contribution to future hit-rate is smallest --
    is popped first.  Higher V_u stays in the cache.
    """
    layer_name = getattr(layer, "name", str(layer)).upper()
    now = _current_time_counter()
    u = _node_to_unit(node, layer_name, now)
    pi_u = _PI_U
    # Saved prefill at current tier (vs DROP) -- bigger = more valuable to keep.
    save_prefill = u.p_hat * (
        reload_cost(u, Tier.DROP, _COSTS, pi_u) - reload_cost(u, u.tier, _COSTS, pi_u)
    )
    # Holding tax (occupancy-weighted) -- we don't have live tier usage here,
    # so use h_base * b_u * 1/lambda as the per-unit-time cost amortised over
    # an expected reuse interval (paper §7).  Bigger = more expensive to keep.
    h_base = _COSTS.h_base.get(u.tier, 0.0)
    hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
    hold = h_base * u.n_bytes * hold_time
    value = save_prefill - hold
    return float(value)
