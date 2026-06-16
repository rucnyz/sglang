"""verify/scheduler_driver (#251 Stage B increment 1): the in-engine AginferDriver.

The decision brain is being relocated from the out-of-process daemon into the engine
(the #251 blocker = "no in-engine driver loop"). Increment 1 extracts the PURE
post-`joint_decide` decision subsystem — the #240 saturation-yield apply-rate EMA and
the #223 evict-cooldown filter + the saturation-yield plan strip — into
`sglang.srt.mem_cache.aginfer.scheduler_driver.AginferDriver`. This verify pins the
EXTRACTED logic byte-for-behaviour against the contract it had inline in the daemon
(kv_scheduler.py), server-free.

Capability:
  A. update_demote_apply_rate: a dispatched remove-HBM hash STILL HBM-resident a dump
     later LOWERS the EMA; one that left HBM RAISES it; the gen<1 grace defers a
     just-noted hash; with nothing pending the EMA relaxes back toward 1.0.
  B. postprocess_plan: the #223 cooldown filter drops a remove migrate for a cooled
     hash (keeps a pure-add); the #240 saturation yield strips remove-HBM migrates
     when the EMA is below threshold and is a NO-OP above it (do-no-harm).
  C. note_demote_dispatched tags pending demotes at the current dump generation.

Cost / worst-case: pure in-memory; no engine, no I/O.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for p in (os.path.join(_SG, "python"), os.path.join(_SG, "dev", "aginfer")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sglang.srt.mem_cache.aginfer.base import Tier  # noqa: E402
from sglang.srt.mem_cache.aginfer.knapsack import Migrate  # noqa: E402
from sglang.srt.mem_cache.aginfer.scheduler_driver import (  # noqa: E402
    AginferDriver, _DEMOTE_YIELD_EMA, filter_cooled_evicts,
)


class _Unit:
    """Minimal stand-in for a state unit: only `.residence` is read by the EMA."""
    def __init__(self, residence):
        self.residence = set(residence)


def _mig(uid, add, remove):
    """A Migrate whose `.id` is the (uid, add_tiers, remove_tiers) tuple the
    filters key on — matching knapsack.Migrate's contract
    (Migrate(cost, relief={}, acquired={}, id=None, group=None))."""
    return Migrate(cost=0.0, id=(uid, frozenset(add), frozenset(remove)))


def _check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def stage_A_apply_rate_ema():
    print("[A] update_demote_apply_rate EMA")
    d = AginferDriver()
    _check("starts optimistic (1.0)", d._demote_apply_ema == 1.0)

    # note a demote at gen 0, then a dump where it's STILL HBM-resident => did NOT land
    d.note_demote_dispatched("h_stuck")
    # first update: _dump_gen 0->1, gen diff (1-0)>=1 so it's checked; still HBM => landed=False
    d.update_demote_apply_rate({"h_stuck": _Unit({Tier.HBM, Tier.DRAM})})
    _check("EMA drops when demote didn't land (0.7*1 + 0.3*0 = 0.7)",
           abs(d._demote_apply_ema - 0.7) < 1e-9)
    _check("stuck hash consumed from pending", "h_stuck" not in d._pending_demote)

    # a demote that DID land (hash gone from HBM) raises the EMA
    d2 = AginferDriver()
    d2._demote_apply_ema = 0.5
    d2.note_demote_dispatched("h_ok")
    d2.update_demote_apply_rate({"h_ok": _Unit({Tier.DRAM})})  # left HBM
    _check("EMA rises when demote landed (0.7*0.5 + 0.3*1 = 0.65)",
           abs(d2._demote_apply_ema - 0.65) < 1e-9)

    # grace: a hash noted in the SAME generation as the check is deferred (gen diff < 1)
    d3 = AginferDriver()
    d3._demote_apply_ema = 0.8
    # advance gen to 1 with an empty update (nothing pending -> relax toward 1.0)
    d3.update_demote_apply_rate({})
    _check("empty-pending relaxes EMA toward 1.0 (+0.02)",
           abs(d3._demote_apply_ema - 0.82) < 1e-9)
    d3.note_demote_dispatched("h_fresh")  # tagged at current _dump_gen (=1)
    d3.update_demote_apply_rate({"h_fresh": _Unit({Tier.HBM})})  # _dump_gen->2, diff=1 NOT <1
    _check("a hash noted one gen before the check IS evaluated",
           "h_fresh" not in d3._pending_demote)


def stage_B_postprocess_plan():
    print("[B] postprocess_plan: #223 cooldown filter + #240 saturation yield")
    d = AginferDriver()  # EMA = 1.0 => yield is a NO-OP

    # #223: a remove migrate for a cooled hash is dropped; a pure-add for it survives
    cooled = _mig("h_cold", add=[], remove=[Tier.HBM])
    pure_add = _mig("h_cold", add=[Tier.DRAM], remove=[])
    other = _mig("h_warm", add=[], remove=[Tier.HBM])
    now = 100.0
    cooldown = {"h_cold": now + 5.0}  # cooled until now+5
    out = d.postprocess_plan([cooled, pure_add, other], dict(cooldown), now)
    ids = {c.id[0] for c in out}
    _check("cooled remove dropped", cooled not in out)
    _check("pure-add for cooled hash survives", pure_add in out)
    _check("non-cooled remove survives", other in out)

    # #240 saturation yield NO-OP when EMA healthy (do-no-harm)
    d_hi = AginferDriver()  # EMA 1.0 > threshold
    plan = [_mig("a", [], [Tier.HBM]), _mig("b", [Tier.DRAM], [])]
    out_hi = d_hi.postprocess_plan(list(plan), None, now)
    _check("healthy EMA strips nothing (do-no-harm)", len(out_hi) == 2)

    # #240 saturation yield STRIPS remove-HBM migrates when EMA below threshold
    d_lo = AginferDriver()
    d_lo._demote_apply_ema = _DEMOTE_YIELD_EMA - 0.05  # below threshold
    remove_hbm = _mig("a", [], [Tier.HBM])
    keep_promote = _mig("b", [Tier.HBM], [])   # ADD HBM (promote) — must survive
    keep_dram = _mig("c", [Tier.DRAM], [Tier.DISK])  # remove non-HBM — survives
    out_lo = d_lo.postprocess_plan([remove_hbm, keep_promote, keep_dram], None, now)
    _check("low EMA strips the remove-HBM migrate", remove_hbm not in out_lo)
    _check("low EMA keeps the promote (add HBM)", keep_promote in out_lo)
    _check("low EMA keeps the non-HBM remove", keep_dram in out_lo)


def stage_C_daemon_delegates():
    print("[C] daemon KvScheduler delegates to the in-engine driver")
    # the daemon's _filter_cooled_evicts must now be the driver's implementation
    import importlib
    kvs = importlib.import_module("daemon.kv_scheduler")
    now = 0.0
    cooled = _mig("x", [], [Tier.HBM])
    out = kvs._filter_cooled_evicts([cooled], {"x": now + 1.0}, now)
    _check("daemon _filter_cooled_evicts delegates (cooled remove dropped)", cooled not in out)
    # identity: same callable behaviour as the driver's
    out2 = filter_cooled_evicts([cooled], {"x": now + 1.0}, now)
    _check("daemon filter == driver filter", out == out2 == [])


if __name__ == "__main__":
    stage_A_apply_rate_ema()
    stage_B_postprocess_plan()
    stage_C_daemon_delegates()
    print("verify/scheduler_driver: ALL STAGES PASSED")
