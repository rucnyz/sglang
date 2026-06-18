"""own_evict_when_hot (§own_evict_when_hot) — Admitter prefers own-evict when src cache is hot.

design.md §own_evict_when_hot (negative): when `i_src` has hot cache (high-recompute,
expensive to evict) and `i_dst` has cold cache, own-evict on `i_dst`
must be cheaper than `cross-evict` (transfer wall + losing hot block
on src). The Admitter should pick own-evict — NOT blindly cross-evict
just because c^xfer is small.

This is the negative companion to cost_picks_xfree (positive): cost_picks_xfree asserts that when
i_src has FREE pages, Admitter does take cross-free; own_evict_when_hot asserts that
when i_src has no FREE pages but expensive-to-evict CACHE, the
Admitter does NOT take cross-evict.

Test style: calls the REAL `Admitter.decide(...)` with synthetic
realistic inputs. No scheduler, no live workload — exercise the
decision function with the same numeric protocol Phase 5's
`decide_for_req` uses.

Pass criteria (per §own_evict_when_hot):
  - Core: src hot + dst cold + src.free=0 → decision == 'own_evict'
  - Positive baseline: src has FREE pages → decision == 'cross_free'
  - Cold src + dst cold + src has FREE → decision == 'cross_free'
  - All expensive → decision == 'defer'
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.admitter import Admitter
from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model


def _warm_admitter(*, c_xfer_per_page_us: float = 100.0):
    """Build an Admitter with the EWMA pre-warmed so cross-* gating
    won't suppress decisions."""
    reset_cost_model()
    cm = get_cost_model()
    # 5 observations of (c_xfer_per_page_us × 12 chunks) each → warm.
    for _ in range(5):
        cm.update_xfer(total_us=c_xfer_per_page_us * 12, n_chunks=12)
    return Admitter(cost_model=cm)


def _decide(adm, *, x_tokens, dst_free, dst_evictable,
            src_free, src_evictable,
            c_evict_dst_us, c_evict_src_us,
            c_xfer_per_page_us=100.0, queue_len=0,
            tokens_per_page=1024):
    return adm.decide(
        x_tokens=x_tokens,
        dst_pool="kv", src_pool="mamba",
        dst_free=dst_free, dst_evictable=dst_evictable,
        src_free=src_free, src_evictable=src_evictable,
        queue_len=queue_len,
        c_evict_dst_us=c_evict_dst_us,
        c_evict_src_us=c_evict_src_us,
        c_xfer_per_page_us=c_xfer_per_page_us,
        tokens_per_page=tokens_per_page,
    )


# ---------------- Tests ----------------

def test_1_core_hot_src_cold_dst_picks_own_evict():
    """own_evict_when_hot core scenario: dst is full (own_free=inf), src has NO free
    pages but a lot of hot cache (cross_free=inf, c_evict_src is huge),
    dst has cold cache (c_evict_dst is small).

    Expected:
      own_evict cost      = 500 µs (cold dst)
      cross_evict cost    = 100 µs/pg × 12 pages + 8000 µs = 9200 µs
      → pick own_evict
    """
    adm = _warm_admitter()
    dec = _decide(
        adm,
        x_tokens=2048,            # demands 2 pages, rounds to 12 (lcm)
        dst_free=0,               # dst full
        dst_evictable=10000,      # plenty of cold dst cache
        src_free=0,               # src has no free pages
        src_evictable=10000,      # but plenty of hot cache
        c_evict_dst_us=500.0,     # cold → cheap to evict
        c_evict_src_us=8000.0,    # hot → expensive to evict
        c_xfer_per_page_us=100.0,
    )
    # Math check:
    # own_free        = inf       (dst_free=0)
    # own_evict       = 500       (c_evict_dst)
    # cross_free      = inf       (src_free=0)
    # cross_evict     = 1200+8000 = 9200
    # defer           = 0         (queue_len=0)
    # → defer wins! That's actually a problem with queue_len=0 (defer cost=0).
    # Need realistic queue_len so own_evict beats defer.
    # Repeat with a small queue:
    dec = _decide(
        adm,
        x_tokens=2048, dst_free=0, dst_evictable=10000,
        src_free=0, src_evictable=10000,
        c_evict_dst_us=500.0, c_evict_src_us=8000.0,
        c_xfer_per_page_us=100.0,
        queue_len=20,             # 20 × w_q (~50µs) = 1000 µs
    )
    assert dec.action == "own_evict", (
        f"Expected own_evict (500µs) to beat cross_evict (9200µs) and "
        f"defer (1000µs); got {dec.action}.\n  costs: {dec.candidate_costs_us}"
    )
    # And cross_evict was indeed considered (not inf):
    assert dec.candidate_costs_us["cross_evict"] < float("inf")
    assert (
        dec.candidate_costs_us["own_evict"]
        < dec.candidate_costs_us["cross_evict"]
    ), "own_evict must be strictly cheaper than cross_evict in this case"
    print("  PASS  1  core: hot src + cold dst → own_evict beats cross_evict")


def test_2_positive_baseline_src_has_free_picks_cross_free():
    """Positive companion to §cost_picks_xfree: src has plenty of FREE pages →
    cross_free is feasible AND cheap; Admitter must take it."""
    adm = _warm_admitter()
    dec = _decide(
        adm,
        x_tokens=2048, dst_free=0, dst_evictable=10000,
        src_free=10000,           # src has FREE pages
        src_evictable=0,
        c_evict_dst_us=500.0,
        c_evict_src_us=float("inf"),
        c_xfer_per_page_us=100.0,
        queue_len=10,
    )
    # cross_free = 100 × 12 = 1200 µs
    # own_evict  = 500 µs
    # Hmm — own_evict is cheaper. Need to make dst more expensive.
    dec = _decide(
        adm,
        x_tokens=2048, dst_free=0, dst_evictable=10000,
        src_free=10000,
        src_evictable=0,
        c_evict_dst_us=5000.0,    # hot dst now
        c_evict_src_us=float("inf"),
        c_xfer_per_page_us=100.0,
        queue_len=10,
    )
    assert dec.action == "cross_free", (
        f"Expected cross_free (1200µs) to beat own_evict (5000µs); "
        f"got {dec.action}.\n  costs: {dec.candidate_costs_us}"
    )
    print("  PASS  2  baseline: src has free + dst hot → cross_free wins")


def test_3_cold_src_cold_dst_with_src_free_picks_cross_free():
    """When src is cold AND has FREE pages, cross_free still wins as
    long as transfer cost stays competitive with own-evict."""
    adm = _warm_admitter()
    dec = _decide(
        adm,
        x_tokens=2048, dst_free=0, dst_evictable=10000,
        src_free=10000,
        src_evictable=10000,
        c_evict_dst_us=3000.0,
        c_evict_src_us=100.0,     # cold src — but doesn't matter for cross_free
        c_xfer_per_page_us=100.0,
        queue_len=5,
    )
    # cross_free  = 1200
    # own_evict   = 3000
    # cross_evict = 1200 + 100 = 1300
    # → cross_free wins (1200 < 1300)
    assert dec.action == "cross_free", (
        f"Expected cross_free=1200 to beat own_evict=3000 and "
        f"cross_evict=1300; got {dec.action}.\n  costs: {dec.candidate_costs_us}"
    )
    print("  PASS  3  cold src + cold dst + src has free → cross_free")


def test_4_everything_expensive_picks_defer():
    """When all fire-actions are expensive and queue is short, defer
    wins (queue waiting is cheaper than burning capacity)."""
    adm = _warm_admitter()
    dec = _decide(
        adm,
        x_tokens=2048, dst_free=0, dst_evictable=10000,
        src_free=10000, src_evictable=10000,
        c_evict_dst_us=50000.0,
        c_evict_src_us=50000.0,
        c_xfer_per_page_us=5000.0,  # 5000 × 12 = 60000 µs
        queue_len=10,                # 10 × w_q ~= 500 µs
    )
    assert dec.action == "defer", (
        f"Expected defer (500µs) to beat all fire actions (≥50000µs); "
        f"got {dec.action}.\n  costs: {dec.candidate_costs_us}"
    )
    print("  PASS  4  all expensive + short queue → defer")


def test_5_own_free_dominates_when_feasible():
    """The ALWAYS-WIN scenario: when dst has free capacity, own_free=0
    always wins (no need to fire or evict)."""
    adm = _warm_admitter()
    dec = _decide(
        adm,
        x_tokens=2048,
        dst_free=10000,           # plenty of own-free
        dst_evictable=10000,
        src_free=10000,
        src_evictable=10000,
        c_evict_dst_us=500.0,
        c_evict_src_us=500.0,
        c_xfer_per_page_us=100.0,
        queue_len=10,
    )
    assert dec.action == "own_free", (
        f"own_free must win when dst has capacity; got {dec.action}"
    )
    assert dec.candidate_costs_us["own_free"] == 0.0
    print("  PASS  5  own_free=0 dominates when dst has capacity")


def test_6_d6n_falsification_signal():
    """Borderline case + falsification: with a slightly-hot src that
    just edges out own_evict, the Admitter picks own_evict. If a bug
    undercounts c_evict_src to 0, the decision flips to cross_evict
    — this is what we'd see in production logs as the breakage signal.

    Math (note: decide() uses n_pages = ceil(X/tps) = 2, NOT lcm-rounded):
      cross_evict = c_xfer × 2 + c_evict_src
      own_evict   = c_evict_dst
    """
    adm = _warm_admitter()
    # Pick c_evict_dst=1500 and c_evict_src=1400 + c_xfer=100/pg:
    #   cross_evict = 200 + 1400 = 1600
    #   own_evict   = 1500
    # → own_evict wins by 100 µs margin.
    dec = _decide(
        adm,
        x_tokens=2048,
        dst_free=0, dst_evictable=10000,
        src_free=0, src_evictable=10000,
        c_evict_dst_us=1500.0,
        c_evict_src_us=1400.0,
        c_xfer_per_page_us=100.0,
        queue_len=200,            # 200 × w_q must beat own_evict for defer
    )
    assert dec.action == "own_evict", (
        f"Borderline: own_evict=1500 should beat cross_evict=1600; "
        f"got {dec.action}.\n  costs: {dec.candidate_costs_us}"
    )
    # Falsification trigger: c_evict_src undercounted to 0 → cross_evict
    # becomes 200, which is way under own_evict (1500). Decision flips.
    dec_bug = _decide(
        adm,
        x_tokens=2048,
        dst_free=0, dst_evictable=10000,
        src_free=0, src_evictable=10000,
        c_evict_dst_us=1500.0,
        c_evict_src_us=0.0,       # BUG: undercounted
        c_xfer_per_page_us=100.0,
        queue_len=200,
    )
    assert dec_bug.action == "cross_evict", (
        f"With c_evict_src=0 bug, cross_evict=200 must win; "
        f"got {dec_bug.action} — falsification signal mis-wired"
    )
    print("  PASS  6  falsification signal: c_evict_src=0 bug flips decision")


def main():
    tests = [
        test_1_core_hot_src_cold_dst_picks_own_evict,
        test_2_positive_baseline_src_has_free_picks_cross_free,
        test_3_cold_src_cold_dst_with_src_free_picks_cross_free,
        test_4_everything_expensive_picks_defer,
        test_5_own_free_dominates_when_feasible,
        test_6_d6n_falsification_signal,
    ]
    print(f"\nD6n — Admitter prefers own-evict when src cache hot "
          f"(n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nD6n: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
