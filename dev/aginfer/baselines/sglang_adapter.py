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

# #208 const-V_u isolation arm: when set, the inline scorer neutralises the
# reuse-prediction signal (p_hat / lambda → constant) so V_u depends only on
# size/tier.  The daemon's build_paper_state honours the same env var.
_CONST_VU = bool(os.environ.get("AGINFER_CONST_VU"))
if _CONST_VU:
    # Observable activation marker (#208) — so the const_vu arm's neutralised
    # ranking is unambiguous in the sglang log, no proc-env archaeology needed.
    import sys as _sys
    print("[aginfer] AGINFER_CONST_VU active — inline V_u reuse signal "
          "neutralised (p_hat=lambda=1.0) (#208)", file=_sys.stderr, flush=True)

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
    if _CONST_VU:
        # #208 const-V_u isolation arm: neutralise the reuse-prediction
        # signal (p_hat / lambda → constant) so V_u depends only on
        # size/tier.  Eviction ranking becomes reuse-blind while the
        # multi-tier machinery still runs — isolates "does the value
        # RANKING help" from "does the machinery help".
        p_hat, lam = 1.0, 1.0
    else:
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
    # #193: PEEK, not get_and_increase.  The eviction scorers are READS
    # ("what is the eviction cost of this node now?"); advancing the
    # global counter once per scored node would (a) make the V_u heap
    # key order-dependent (node i scored at now=base+i, so two
    # same-(last_access,hits) nodes score differently by iteration
    # position) and (b) pollute the access clock published as
    # `time_counter` in /aginfer/state.  peek_time_counter gives all
    # nodes in one heap build the SAME `now` → deterministic ordering,
    # no clock pollution.
    from sglang.srt.mem_cache.unified_cache_components import (
        peek_time_counter,
    )
    return int(peek_time_counter())


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


def default_policy_score(node: Any, layer: Any) -> float:
    """T38 (#169) default-policy scorer — the policy module that runs
    when no daemon is attached.  DESIGN §3 superset framing: sglang's
    historical eviction heuristic IS aginfer's default policy module.

    The default eviction value is **bare ``last_access_time``** — the
    LRU-equivalent V_u ("last_access as p_hat surrogate", DESIGN §3).
    Lower score = evict first, so the oldest leaf leaves first, exactly
    like stock sglang.  This MUST stay byte-identical to sglang's
    in-process ``_default_eviction_score`` so "aginfer disabled" and
    "aginfer default policy" are one code path (verify/t28 A3 is the
    cross-tree drift guard).

    hit_count is intentionally NOT part of the eviction score: DESIGN
    §3 uses hit_count only in the WRITE-THROUGH trigger
    (``should_write_through`` / #178).  (#177 removed an earlier
    ``+ hit_count·2^-50`` eviction tie-break — it was non-functional
    below the float64 ULP at realistic ``last_access_time`` AND near-
    pointless: the cache stamps ancestor nodes 1e-5 apart, so exact
    ties are effectively absent for realistic counters; only at extreme
    counter magnitudes (≳2^40, where the 1e-5 spacing ULP-collapses) do
    ties occur, and there stock LRU ties arbitrarily too.)
    """
    return float(node.last_access_time)


def default_policy_should_write_through(node: Any, threshold: int) -> bool:
    """T28/#178 default-policy WRITE-THROUGH trigger — the adapter-side
    mirror of sglang's in-process ``_default_should_write_through``.
    Completes the DESIGN §3 "default policy module" on the adapter side
    (eviction = ``default_policy_score``; write-through = this).

    Signature ``(node, threshold) -> bool`` matches the
    ``SGLANG_WRITE_THROUGH_MODULE`` plugin contract.  Default behaviour
    is the historical ``hit_count >= write_through_threshold``.  An
    aginfer V_u-aware version (``V_u(res ∪ {DRAM}) > V_u(res)``) plugs
    into the same hook once the hint-table consumer exists (T27 #188).
    """
    return int(node.hit_count) >= int(threshold)


def ours_greedy_score(node: Any, layer: Any) -> float:
    """Paper §7 per-unit value rule, served as a sglang eviction heap key.

    Heap is a *min*-heap; the smallest key is popped (= evicted) first.
    Stock sglang uses ``last_access_time`` as the key, so the oldest leaf
    leaves first.  Our analogue: directly return ``V_u(authoritative_tier)``
    (saved-prefill term minus the holding tax) so the *least valuable*
    leaf -- the one whose contribution to future hit-rate is smallest --
    is popped first.  Higher V_u stays in the cache.
    """
    u = _node_to_unit(node, layer.name.upper(), _current_time_counter())
    return _v_u_from_unit(u)


def _v_u_from_unit(u: ReuseUnit) -> float:
    """Paper §7 eviction value of a unit at its (single) residence tier.
    Heap key: lower = evict first; higher V_u stays in the cache.

    Shared by ``ours_greedy_score`` (p_hat/lambda derived from the
    node's hits/age) and ``hint_v_u`` (p_hat/lambda from the daemon's
    pushed hint) so the two CANNOT drift — there is exactly one V_u
    formula (verify/t27 A2 is the drift guard)."""
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
    return float(save_prefill - hold)


def hint_v_u(node: Any, layer: Any, hint: Any) -> float:
    """T27 (#188, DESIGN §3/§10) hint-aware eviction value.  Same paper-
    §7 V_u as ``ours_greedy_score``, but ``p_hat`` / ``lambda`` come
    from the daemon's pushed ``hint`` ({"p_hat", "lambda", "stamp"})
    instead of the local hits/age proxy — so daemon hints actually
    change eviction order.

    ``hint=None`` (no daemon entry for this unit yet) → graceful
    fallback to the local hits/age derivation (== ``ours_greedy_score``;
    NEVER bare LRU — DESIGN §3 "eviction never falls back to LRU on
    absent hints").  With unit-birth seeding (the cache seeds p_hat≈1
    on creation) absent hints are rare in practice."""
    u = _node_to_unit(node, layer.name.upper(), _current_time_counter())
    if hint is not None:
        u.p_hat = float(hint["p_hat"])
        u.lambda_rate = float(hint["lambda"])
    return _v_u_from_unit(u)
