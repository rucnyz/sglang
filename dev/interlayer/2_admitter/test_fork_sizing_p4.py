"""P4.2 — fork-lifecycle sizing for the symmetric Admitter (Phase 4).

P4.1 routes a mamba-scarce arrival to grow-mamba/defer, but its scarcity test
used `_mamba_arrival_need_slots = 1` — only the req's ACTIVE SSM slot. A hybrid
req's lifecycle also forks ONE cache slot at `cache_unfinished_req` (it copies
its prefix state into a new locked radix node while keeping its active slot —
net +1). So a req admitted when mamba has exactly 1 free slot gets its active
slot, then its mid-prefill fork finds the pool full → "Can not alloc mamba
cache" (#312). The arrival is scarce for its FULL lifecycle (active + fork),
not just the active slot.

Contract this test pins: mamba scarcity is measured against
`active(1) + fork(1)`, so a single free slot is still scarce (admitting would
crash the fork); two free slots cover the lifecycle and admit normally.

Test-first: with need=1 (P4.1) the one-free-slot arrival reports own_free
(RED — admits into a pool that crashes the fork). With fork-aware sizing it is
scarce → grow/defer (GREEN).
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/2_admitter")

from test_symmetric_admit_p4 import _decide  # noqa: E402  (reuse the harness)


def test_one_free_slot_is_still_scarce_for_the_fork():
    """mamba_free=1 covers the active slot but NOT the caching fork. Admitting
    (own_*) crashes the fork (#312). With an empty queue the safe, cost-optimal
    choice is defer; the hard requirement is: NOT own_*.

    RED (need=1): mamba_free 1 >= 1 -> not scarce -> own_free.
    GREEN (need=active+fork): 1 < 2 -> scarce -> defer (queue empty)."""
    dec = _decide(kv_free=200000, kv_evictable=0,
                  mamba_free=1, mamba_evictable=0, queue_len=0)
    assert dec is not None
    assert not dec.action.startswith("own_"), (
        f"a one-free-slot arrival must not be admitted (own_*) — its cache "
        f"fork would crash (#312); got {dec.action} "
        f"dst={getattr(dec,'dst_pool',None)}")


def test_one_free_slot_under_pressure_grows_mamba():
    """Same one-free-slot arrival but with a queue backlog (a burst): deferring
    is costly, so the Admitter grows mamba from KV to make room for the
    lifecycle (active + fork)."""
    dec = _decide(kv_free=200000, kv_evictable=0,
                  mamba_free=1, mamba_evictable=0, queue_len=200)
    assert dec is not None
    assert getattr(dec, "dst_pool", None) == "mamba", (
        f"one-free-slot burst must grow mamba (active+fork); got "
        f"dst={getattr(dec,'dst_pool','<missing>')} action={dec.action}")
    assert dec.action.startswith("cross_")


def test_two_free_slots_cover_the_lifecycle():
    """mamba_free=2 = active(1) + fork(1): the full lifecycle fits, so a normal
    own_free admission is correct (no needless grow)."""
    dec = _decide(kv_free=200000, kv_evictable=0,
                  mamba_free=2, mamba_evictable=0, queue_len=0)
    assert dec is not None
    assert dec.action == "own_free", (
        f"two free slots cover active+fork; expected own_free, got {dec.action}")


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failures else 0)
