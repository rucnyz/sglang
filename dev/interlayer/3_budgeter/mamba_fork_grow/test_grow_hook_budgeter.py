"""P4-(b) — BudgetAgent._grow_mamba_from_kv: the synchronous k2m grow the
MambaRadixCache fork-failure hook calls.

When the caching fork can't get a mamba slot and evict finds no cold cache,
`MambaRadixCache._fork_mamba_with_recovery` calls `_mamba_grow_hook(n_slots)`.
The Budgeter wires that hook to `_grow_mamba_from_kv`, which builds a
`kv_to_mamba` FirePlan for `ceil(n_slots / mamba_tokens_per_chunk)` chunks
(rounded up to the actuator LCM) and executes it — growing mamba from KV. It
returns True iff the fire granted pages (so the caller knows to retry the
fork), False otherwise (no actuator chain / build refused / fire aborted /
zero granted), in which case the caller falls through to the original assert.

Test-first: the method does not exist yet (RED); after implementing it builds
the right kv_to_mamba plan and reports grant success.
"""
import math
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/2_admitter")

from sglang.srt.budgeter.agent import BudgetAgent  # noqa: E402
from test_sync_fire import FakePlanner, FakeActuator  # noqa: E402


def _agent(*, mamba_tps, lcm_kv=1, lcm_mamba=1, abort=False, kv_avail=1_000_000):
    import types
    a = BudgetAgent.__new__(BudgetAgent)
    a._fire_planner = FakePlanner()
    a._actuator = FakeActuator(n_kv_subpools=lcm_kv, n_mamba_subpools=lcm_mamba,
                               abort=abort)
    a._mamba_tokens_per_chunk = mamba_tps
    # The k2m grow bounds its request to KV's idle slack (#318), reading the KV
    # allocator's available_size; provide ample slack so the bound never caps
    # the grant under test. _kv_tokens_per_chunk / headroom complete the bound.
    a._kv_tokens_per_chunk = 1
    a._xfer_grow_headroom_slots = 32
    a.scheduler = types.SimpleNamespace(
        token_to_kv_pool_allocator=types.SimpleNamespace(
            available_size=lambda: kv_avail,
        ),
    )
    return a


def test_grow_builds_kv_to_mamba_of_ceil_need_over_tps():
    """need=2 slots, tps=1, lcm=1 → grow 2 chunks; tps=2 → 1 chunk. Direction
    kv_to_mamba. Returns True (FakeActuator grants)."""
    for tps, expect in ((1, 2), (2, 1)):
        a = _agent(mamba_tps=tps)
        ok = a._grow_mamba_from_kv(2)
        assert ok is True, f"grant>0 must return True (tps={tps})"
        calls = a._fire_planner.calls
        assert len(calls) == 1, f"one build expected; got {calls}"
        direction, n_pages = calls[0][0], calls[0][1]
        assert direction == "kv_to_mamba", f"must grow mamba; got {direction}"
        assert n_pages == math.ceil(2 / tps), (
            f"tps={tps}: expected {math.ceil(2/tps)} chunks, got {n_pages}")


def test_grow_rounds_up_to_actuator_lcm():
    """A 1-chunk need on an actuator whose LCM is 12 rounds up to 12 (atomic
    cross-pool transfer unit), not 1."""
    a = _agent(mamba_tps=1, lcm_kv=4, lcm_mamba=6)  # lcm(4,6)=12
    a._grow_mamba_from_kv(1)
    n_pages = a._fire_planner.calls[0][1]
    assert n_pages == 12, f"must round up to the actuator LCM 12; got {n_pages}"


def test_grow_returns_false_when_no_chain():
    """No actuator/planner wired → cannot grow → False (caller asserts)."""
    a = BudgetAgent.__new__(BudgetAgent)
    a._fire_planner = None
    a._actuator = None
    a._mamba_tokens_per_chunk = 1
    assert a._grow_mamba_from_kv(2) is False


def test_grow_returns_false_when_fire_aborts():
    """Fire aborts (e.g. KV couldn't donate) → False, so the fork falls
    through to the assert rather than retrying a fork that still has no slot."""
    a = _agent(mamba_tps=1, abort=True)
    assert a._grow_mamba_from_kv(2) is False


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError:
                failures += 1; print("FAIL", name); traceback.print_exc()
            except Exception:
                failures += 1; print("ERROR", name); traceback.print_exc()
    sys.exit(1 if failures else 0)
