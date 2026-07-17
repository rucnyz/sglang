"""P4-(b) — fork-failure self-heal: grow mamba from KV before asserting.

The #312 crash is `assert mamba_value_forked is not None, "Can not alloc mamba
cache"` in `MambaRadixCache.cache_unfinished_req`: the caching fork needs one
mamba slot, the pool is full, and `evict_mamba` finds no UNLOCKED cold cache to
reclaim. The Budgeter working-set floor keeps mamba above this line, but the
floor is a static reservation that can't be lowered (the fork happens
mid-prefill, where the Admitter's arrival-time grow has no hook). This wires a
grow hook AT the fork-failure point: when evict can't free a slot, fire a
synchronous k2m grow (KV→mamba) and retry, instead of asserting. That keeps the
cache (not a best-effort skip) AND lets the floor drop (a later step).

`_fork_mamba_with_recovery(mamba_value)` is the extracted recovery:
  fork → (None) evict cold + fork → (None) grow-from-KV + fork → (None).
The grow hook (`_mamba_grow_hook(n_slots) -> bool`) is injected by the Budgeter
when its cross-pool actuator chain is built. On final failure the recovery
returns None (#329 degrade): the caller (`cache_unfinished_req`) skips caching
this snapshot and the request continues on its live state, so an over-drained
pool back-pressures instead of crashing.

Test-first: the grow step lets the wired hook free a slot so the fork succeeds
(GREEN, vs the #312 assert). When no slot can be freed the fork returns None
(the #329 degrade, vs the old fail-loud assert).
"""
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache


class _StubMambaPool:
    """fork_from returns None while empty; `_free` slots are consumed per fork.
    The grow hook adds slots (simulating a k2m transfer landing)."""

    def __init__(self, free=0):
        self._free = free
        self.fork_calls = 0

    def fork_from(self, mamba_value):
        self.fork_calls += 1
        if self._free <= 0:
            return None
        self._free -= 1
        return ("slot", self.fork_calls)  # a non-None forked handle


def _cache(*, pool_free, evict_frees, grow_frees):
    """A MambaRadixCache with __init__ bypassed, wired only for the
    fork-recovery path. `evict_frees`/`grow_frees` = how many slots the stubbed
    evict / grow hook make available when called."""
    c = MambaRadixCache.__new__(MambaRadixCache)
    pool = _StubMambaPool(free=pool_free)
    c.req_to_token_pool = types.SimpleNamespace(mamba_pool=pool)

    def _evict(params):
        pool._free += evict_frees

    c.evict = _evict  # stub: frees `evict_frees` slots (0 = no cold cache)

    hook_calls = {"n": 0}
    if grow_frees is None:
        c._mamba_grow_hook = None       # stock: no grow hook wired
    else:
        def _hook(n_slots):
            hook_calls["n"] += 1
            pool._free += grow_frees
            return grow_frees > 0
        c._mamba_grow_hook = _hook
    return c, pool, hook_calls


def test_grow_from_kv_when_evict_finds_nothing():
    """#312 regime: pool empty, evict frees nothing (no unlocked cold cache),
    grow hook wired. Must fire the grow (which frees a slot) and the fork then
    succeeds — NOT assert.

    RED: without the grow step the second fork is None → assert "Can not alloc
    mamba cache". GREEN: hook called, fork succeeds."""
    c, pool, hook_calls = _cache(pool_free=0, evict_frees=0, grow_frees=1)
    forked = c._fork_mamba_with_recovery(mamba_value=("src", 0))
    assert forked is not None, "fork must succeed via the k2m grow hook (#312)"
    assert hook_calls["n"] == 1, (
        f"the grow hook must be fired exactly once when evict can't free a "
        f"slot; got {hook_calls['n']}")


def test_evict_alone_suffices_no_grow_fired():
    """When evicting cold cache frees a slot, the fork succeeds on the
    evict-retry and the grow hook is NOT fired (grow is the last resort)."""
    c, pool, hook_calls = _cache(pool_free=0, evict_frees=1, grow_frees=1)
    forked = c._fork_mamba_with_recovery(mamba_value=("src", 0))
    assert forked is not None
    assert hook_calls["n"] == 0, (
        f"grow must not fire when evict already freed a slot; got {hook_calls['n']}")


def test_free_slot_no_recovery_needed():
    """Pool has a free slot → first fork succeeds, no evict/grow."""
    c, pool, hook_calls = _cache(pool_free=1, evict_frees=0, grow_frees=0)
    forked = c._fork_mamba_with_recovery(mamba_value=("src", 0))
    assert forked is not None
    assert pool.fork_calls == 1 and hook_calls["n"] == 0


def test_no_hook_returns_none():
    """Stock sglang / Budgeter off: no grow hook. Pool empty + evict frees
    nothing → fork returns None (#329 degrade), the caller skips caching. The
    degrade is unconditional, not gated on a hook being wired. (With the
    Budgeter off the pool is never over-drained, so this branch is unreachable
    in practice; pinned for completeness.)"""
    c, pool, hook_calls = _cache(pool_free=0, evict_frees=0, grow_frees=None)
    assert c._fork_mamba_with_recovery(mamba_value=("src", 0)) is None


def test_grow_fires_but_still_no_slot_returns_none():
    """The grow hook runs but frees nothing (KV couldn't donate) → the fork
    still fails → returns None (#329 degrade: back-pressure, never assert)."""
    c, pool, hook_calls = _cache(pool_free=0, evict_frees=0, grow_frees=0)
    assert c._fork_mamba_with_recovery(mamba_value=("src", 0)) is None
    assert hook_calls["n"] == 1, "the grow hook should have been attempted once"


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
            except Exception:
                failures += 1
                print("ERROR", name)
                traceback.print_exc()
    sys.exit(1 if failures else 0)
