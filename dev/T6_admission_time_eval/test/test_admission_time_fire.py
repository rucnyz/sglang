"""T6 unit test: BudgetAgent.try_admission_time_fire dispatches to the
cross-pool actuator and returns the right bool.

Constructs a BudgetAgent stand-in (with a fake _xpool_actuator) and
exercises the public method directly — no SGLang server.
"""

import os
import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


class _FakeXpoolActuator:
    """Records calls; returns canned stats."""
    def __init__(self):
        self.calls = []
        self.canned_stats = {"unmapped_total": 30, "granted_total": 30}

    def kv_to_mamba_chunks(self, n):
        self.calls.append(("kv_to_mamba", n))
        return self.canned_stats

    def mamba_to_kv_chunks(self, n):
        self.calls.append(("mamba_to_kv", n))
        return self.canned_stats


def make_agent_with_flag(admission_time_fire: str, xpool_planner: bool = True):
    """Construct a minimally-initialized BudgetAgent without the
    scheduler dependency."""
    os.environ["SGLANG_BUDGETER"] = "0"  # don't actually open log file
    os.environ["SGLANG_ADMISSION_TIME_FIRE"] = admission_time_fire
    from sglang.srt.budgeter.agent import BudgetAgent

    class _FakeScheduler:
        pass

    agent = BudgetAgent(_FakeScheduler())
    # Override fields the constructor would have set from a real scheduler.
    agent.xpool_planner_enabled = xpool_planner
    agent._xpool_actuator = _FakeXpoolActuator()
    return agent


def main():
    # Case A: env on + planner on + actuator wired → fire dispatches.
    a = make_agent_with_flag("1", xpool_planner=True)
    ok = a.try_admission_time_fire(direction="rec_to_kv", n_chunks=2)
    assert ok, "should return True when actuator commits"
    assert a._xpool_actuator.calls == [("mamba_to_kv", 2)], \
        f"unexpected calls: {a._xpool_actuator.calls}"
    print(f"[case A: env on, fire commits] OK, calls = {a._xpool_actuator.calls}")

    # Case B: env off → no-op, return False.
    b = make_agent_with_flag("0", xpool_planner=True)
    ok2 = b.try_admission_time_fire(direction="kv_to_rec", n_chunks=1)
    assert not ok2, "env off should return False"
    assert b._xpool_actuator.calls == [], "env off should not call actuator"
    print(f"[case B: env off] no-op, calls = {b._xpool_actuator.calls}")

    # Case C: planner disabled → no-op even with env on.
    c = make_agent_with_flag("1", xpool_planner=False)
    ok3 = c.try_admission_time_fire("rec_to_kv", 1)
    assert not ok3
    assert c._xpool_actuator.calls == []
    print(f"[case C: planner off] no-op, calls = {c._xpool_actuator.calls}")

    # Case D: actuator says "no commit" (unmapped=0, granted=0) → False.
    d = make_agent_with_flag("1", xpool_planner=True)
    d._xpool_actuator.canned_stats = {"unmapped_total": 0, "granted_total": 0}
    ok4 = d.try_admission_time_fire("rec_to_kv", 1)
    assert not ok4, "no-commit should return False"
    print(f"[case D: actuator no-commit] returned False as expected")

    # Case E: reentrancy guard — recursion into try_admission_time_fire
    # while another fire is in progress returns False.
    e = make_agent_with_flag("1", xpool_planner=True)
    e._emergency_fire_in_progress = True
    ok5 = e.try_admission_time_fire("rec_to_kv", 1)
    assert not ok5
    assert e._xpool_actuator.calls == []
    print(f"[case E: reentrancy guard] returned False, no actuator call")

    # Case F: bad direction → False with warning.
    f = make_agent_with_flag("1", xpool_planner=True)
    ok6 = f.try_admission_time_fire("nonsense", 1)
    assert not ok6
    print(f"[case F: bad direction] returned False")

    print("\nT6 try_admission_time_fire unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
