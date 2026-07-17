"""#268 — predict_evict_cost_us(pool="mamba") must price an INTERNAL mamba
victim by the whole-prefix re-prefill TOTAL, not c_m alone (cross-fire
over-harvest). Distinct from #298, which is the parallel degenerate-curve gate
on the *migrate* actuator cost `c_migrate` (still open, not touched here).

`predict_evict_cost_us(pool="mamba")` prices an internal mamba victim (a node
whose mamba snapshot is dropped but whose KV value stays for descendants) by
`c_m(s_b)` alone, modelling the eviction as "loses only the snapshot." But per
the single-curve recovery model (paper §3: attention and recurrent layers
interleave, so recovering ANY evicted state needs the whole-prefix re-prefill),
recovering that snapshot costs the prefix TOTAL `c_kv(s_b) + c_m(s_b)` — the
same cost `eviction_priority()` already uses for the LPB sort key and the same
a leaf victim pays. The c_m-only term also violates
`predict_evict_cost_us`'s own contract that the priced set equals the evicted
set ranked by `eviction_priority`.

Under the real hybrid calibration `κ_M = 0` (#276) the gap is the whole cost:
`c_m ≡ 0`, so an internal victim is priced 0 and the Budgeter's m2k NB gate
reads draining it as free → over-harvest (#268).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mamba_evict_predictor import _add_node, _make_hybrid_cache  # noqa: E402

from sglang.srt.budgeter.cost_model import get_cost_curves  # noqa: E402


def test_internal_mamba_victim_priced_by_total():
    curves = get_cost_curves()
    # LRU mode so the oldest mamba node (mid) is picked first and, with its
    # child still cached, stays an INTERNAL victim (snapshot dropped, KV kept).
    c = _make_hybrid_cache("lru")
    mid = _add_node(c, c.root_node, [1, 2], t=1.0)   # oldest → internal victim
    _add_node(c, mid, [3, 4], t=2.0)                 # child leaf, newer

    leaf_v, internal_v, swept = c._plan_mamba_eviction(1)
    internal_keys = [tuple(n.key.token_ids) for n in internal_v]
    assert internal_keys == [(1, 2)], (
        f"setup: mid=[1,2] should be the sole internal victim, got "
        f"internal={internal_keys} leaf={[tuple(n.key.token_ids) for n in leaf_v]}"
    )

    predicted = c.predict_evict_cost_us(1, pool="mamba")
    s_b = 2
    total_us = (curves.c_kv_ms(s_b) + curves.c_m_ms(s_b)) * 1000.0   # n_b=1 (LRU)
    c_m_only_us = curves.c_m_ms(s_b) * 1000.0
    print(f"  predicted={predicted:.1f}us  total(c_kv+c_m)={total_us:.1f}us  "
          f"c_m-only={c_m_only_us:.1f}us")

    assert abs(predicted - total_us) < 1e-6, (
        f"internal mamba victim priced {predicted:.1f}us; must be the whole-"
        f"prefix total {total_us:.1f}us (recovering the snapshot needs a full "
        f"re-prefill), not c_m-only {c_m_only_us:.1f}us (#268). Under κ_M=0 the "
        f"missing c_kv IS the whole cost → drain reads as free → over-harvest."
    )
    print("  PASS  internal mamba victim priced by whole-prefix total")


if __name__ == "__main__":
    test_internal_mamba_victim_priced_by_total()
