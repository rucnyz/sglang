"""Reproducing tests for #312 — budgeter over-drains mamba → crash.

e2e symptom (cc_zero_downside inter_admitter, τ-refactor code, 2026-06-08):
  AssertionError: "Can not alloc mamba cache" at
  mamba_radix_cache.cache_unfinished_req — fork_from returned None and
  evict_mamba could free nothing (it skips locked / mamba_lock_ref>0 nodes).
  The budgeter fired mamba_to_kv on a near-full mamba pool; a later burst
  spiked active demand on the SHRUNKEN pool → no fork slot → SIGQUIT.

Mechanism (derived from sglang code, not e2e tuning):
  A fork fails iff `mamba free == 0` AND no UNLOCKED evictable cache. The m2k
  cross-fire reduces mamba_pool.live_size (allocatable capacity; size stays
  constant). A max_running burst RE-PREFILLS (drained cold cache is
  irrelevant), needing max_running FRESH active slots drawn from
  (live_size − protected); the protected (locked) slots stay reserved. So the
  exact floor is:
      live_size >= max_running + mamba_protected_size() + fork_headroom

`_mamba_drain_floor(live_size, floor_slots, slots_per_page, requested_pages)`
caps the m2k drain (in PAGES) so the post-drain live_size stays >= floor,
converting the slot headroom to pages via slots_per_page (the mamba arena
tokens_per_chunk — NOT assumed 1).

Test-first protocol: each test below bakes in the RED condition (the pre-fix
behaviour violates the invariant) next to the GREEN assertion (the fix holds
it), so the test demonstrably catches the bug it guards.
"""
import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
from sglang.srt.budgeter.agent import _mamba_drain_floor


def test_floor_caps_drain_to_keep_live_above_floor():
    """A fire wants 8 pages; live=6, floor=4, tps=1. A no-floor drain
    (min(8,6)=6) leaves live=0 < floor → next fork crashes (RED). The floor
    caps to (6-4)//1=2 → live stays 4 >= floor (GREEN)."""
    live, floor, tps, req = 6, 4, 1, 8

    no_floor = min(req, live)                      # pre-#312 behaviour
    assert live - no_floor < floor, (
        f"setup: no-floor drain should breach (left {live - no_floor} < {floor})")

    drained = _mamba_drain_floor(live, floor, tps, req)
    assert live - drained * tps >= floor, (
        f"FLOOR BREACH (#312): drained {drained}p×{tps} leaves "
        f"{live - drained*tps} < floor {floor}")
    assert drained == 2, f"expected 2, got {drained}"


def test_tps_gt1_no_overshoot():
    """F2 (the load-bearing unit fix): the cap must convert pages→slots via
    tokens_per_chunk. A PAGE-unit cap (the pre-F2 bug) drains pages×tps slots,
    overshooting the SLOT floor by tps× when tps>1 → re-opens #312. The
    tps-aware cap does not.

    RED: page-unit cap min(req, live-floor)=8 pages × tps=12 = 96 slots →
         live 256-96=160 < floor 200.
    GREEN: tps-aware cap (256-200)//12=4 pages × 12 = 48 → 256-48=208 >= 200.
    """
    live, floor, tps, req = 256, 200, 12, 8

    buggy_pages = min(req, max(0, live - floor))   # pre-F2 page-unit cap
    assert live - buggy_pages * tps < floor, (
        f"setup: page-unit cap should overshoot at tps={tps} "
        f"(left {live - buggy_pages*tps} >= floor {floor} would prove nothing)")

    pages = _mamba_drain_floor(live, floor, tps, req)
    assert live - pages * tps >= floor, (
        f"F2 BREACH: tps-aware cap drained {pages}p×{tps}={pages*tps} slots, "
        f"live {live} -> {live - pages*tps} < floor {floor}")
    assert pages == (live - floor) // tps, f"expected {(live-floor)//tps}, got {pages}"


def test_unknown_granularity_fails_closed():
    """slots_per_page <= 0 (unknown granularity) → refuse the fire (return 0),
    NOT drain (which would be unbounded). Fail-closed."""
    assert _mamba_drain_floor(256, 100, 0, 8) == 0
    assert _mamba_drain_floor(256, 100, -1, 8) == 0


def test_floor_reached_returns_zero():
    """live_size <= floor → no safe drain → 0 → caller must NOT fire m2k."""
    assert _mamba_drain_floor(100, 100, 1, 8) == 0
    assert _mamba_drain_floor(90, 100, 1, 8) == 0
    assert _mamba_drain_floor(100, 100, 12, 8) == 0


def test_invariant_over_grid():
    """Post-drain live_size >= floor for every (live, floor, tps, req) — the
    cap never lets pages×tps breach the floor (or it refused, drain=0)."""
    for live in range(0, 400, 11):
        for floor in range(0, 300, 17):
            for tps in (1, 4, 12):
                for req in (1, 4, 8, 100):
                    d = _mamba_drain_floor(live, floor, tps, req)
                    assert 0 <= d <= req
                    assert live - d * tps >= floor or d == 0
                    if live > floor:
                        assert live - d * tps >= floor


if __name__ == "__main__":
    import traceback
    f = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", n)
            except AssertionError:
                f += 1; print("FAIL", n); traceback.print_exc()
    sys.exit(1 if f else 0)
