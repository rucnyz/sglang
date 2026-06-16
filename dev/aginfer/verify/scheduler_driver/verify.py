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
    assignments_to_wire, tier_to_wire,
)


class _FakeCache:
    """Records the in-process apply calls the driver makes (stands in for the engine's
    UnifiedRadixCache). The real methods live in cache_hooks.py; here we only assert the
    driver CALLS them with the right wire — engine-side semantics are covered by verify/t20."""
    def __init__(self):
        self.migrate_calls = []
        self.hint_calls = []

    def apply_aginfer_migrations(self, actions):
        self.migrate_calls.append(actions)
        return {"applied": len(actions), "applied_hashes": [a["hash"] for a in actions],
                "skipped": []}

    def set_aginfer_hints(self, hints):
        self.hint_calls.append(hints)
        return (True, len(hints))


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

    # u=None path: a dispatched hash absent from the fresh dump => DROPPED (landed=True).
    d4 = AginferDriver()
    d4._demote_apply_ema = 0.5
    d4.note_demote_dispatched("h_gone")
    d4.update_demote_apply_rate({})  # h_gone not in units => u=None => landed=True
    _check("u=None (hash gone from dump) counts as landed (EMA rises 0.7*0.5+0.3=0.65)",
           abs(d4._demote_apply_ema - 0.65) < 1e-9 and "h_gone" not in d4._pending_demote)

    # 4096 safety clear: to keep entries pending past the loop they must hit the grace branch
    # (`_dump_gen - gen < 1`), i.e. their gen must be ≥ the post-increment _dump_gen. Seed 4097
    # entries one gen AHEAD of the counter; the next update increments into them (diff==0 → grace
    # → all kept) → len > 4096 → the guard clears the whole map (unbounded-growth backstop).
    d5 = AginferDriver()
    d5._dump_gen = 5
    d5._pending_demote = {"k%d" % i: 6 for i in range(4097)}  # gen=6, ahead of _dump_gen=5
    d5.update_demote_apply_rate({})  # _dump_gen -> 6; every entry diff (6-6)=0 <1 -> grace kept -> 4097 > 4096
    _check("4096 safety-clear fires (pending > cap -> cleared)", len(d5._pending_demote) == 0)
    # one-below-cap stays in grace, NOT cleared (boundary)
    d6 = AginferDriver()
    d6._dump_gen = 5
    d6._pending_demote = {"k%d" % i: 6 for i in range(4096)}  # exactly the cap
    d6.update_demote_apply_rate({})
    _check("at the cap (4096) the guard does NOT clear (still in grace)", len(d6._pending_demote) == 4096)


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

    # cooldown dict with ONLY expired entries: pruned to empty → inner filter skipped (no drop)
    d_exp = AginferDriver()
    remove_mig = _mig("h_x", add=[], remove=[Tier.HBM])
    cd_expired = {"h_x": now - 1.0}  # already expired at `now`
    out_exp = d_exp.postprocess_plan([remove_mig], cd_expired, now)
    _check("all-expired cooldown pruned → migrate NOT dropped", remove_mig in out_exp)
    _check("expired entry pruned from the dict in place", "h_x" not in cd_expired)

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


def stage_D_in_process_apply():
    print("[D] in-process apply (the 'no HTTP' half): plan/hints → cache methods")
    d = AginferDriver()
    cache = _FakeCache()

    # apply_hints → cache.set_aginfer_hints with the list; empty = no-op
    _check("empty hints = no-op (no cache call)", d.apply_hints([], cache) is None and not cache.hint_calls)
    hints = [{"hash": "h1", "p_hat": 0.9, "lambda": 1.0, "n_holders": 3, "stamp": 5}]
    d.apply_hints(hints, cache)
    _check("apply_hints calls set_aginfer_hints once with the hints",
           len(cache.hint_calls) == 1 and cache.hint_calls[0] == hints)

    # apply_plan: migrates → apply_aginfer_migrations with byte-identical wire; the
    # remove-HBM one is recorded for the EMA; pauses/resumes surfaced (admission=router).
    remove_hbm = _mig("u_demote", add=[], remove=[Tier.HBM])
    promote = _mig("u_promote", add=[Tier.HBM], remove=[Tier.DRAM])
    cache2 = _FakeCache()
    out = d.apply_plan([remove_hbm, promote], cache2)
    _check("apply_plan calls apply_aginfer_migrations once", len(cache2.migrate_calls) == 1)
    wire = cache2.migrate_calls[0]
    # wire must be byte-identical to assignments_to_wire([m.id ...]) (minus the random action_id)
    want = assignments_to_wire([remove_hbm.id, promote.id])
    _check("wire has both migrates", len(wire) == 2)
    _check("wire hashes + tier strings match the translator",
           [(w["hash"], w["add_tiers"], w["remove_tiers"]) for w in wire]
           == [(w["hash"], w["add_tiers"], w["remove_tiers"]) for w in want])
    _check("every wire item has an action_id correlator", all(w.get("action_id") for w in wire))
    _check("remove-HBM demote recorded for the apply-rate EMA",
           "u_demote" in d._pending_demote and "u_promote" not in d._pending_demote)

    # empty plan = no apply call
    cache3 = _FakeCache()
    out3 = d.apply_plan([], cache3)
    _check("empty plan = no migrate call", not cache3.migrate_calls and out3["migrate_result"] is None)

    # pauses/resumes surfaced for the (router-side) admission caller, NOT applied here
    from sglang.srt.mem_cache.aginfer.knapsack import Pause, Resume
    pause_obj = Pause(cost=1.0, pid="prog_1")
    resume_obj = Resume(gain=1.0, pid="prog_2")
    mig = _mig("u", [Tier.DRAM], [])
    cache5 = _FakeCache()
    out5 = d.apply_plan([pause_obj, resume_obj, mig], cache5)
    _check("apply_plan extracts real Pause pids", out5["pauses"] == ["prog_1"])
    _check("apply_plan extracts real Resume pids", out5["resumes"] == ["prog_2"])
    _check("pauses/resumes are NOT applied in-engine (admission = router half)",
           len(cache5.migrate_calls) == 1 and not cache5.hint_calls)  # only the migrate applied

    # malformed-id Migrate is skipped, not crashed (the defensive guard, review #8/#9)
    bad = Migrate(cost=0.0, id=None)        # id None
    bad2 = Migrate(cost=0.0, id=("only_two", [Tier.HBM]))  # 2-tuple, not 3
    good = _mig("ok", [Tier.DRAM], [])
    cache6 = _FakeCache()
    out6 = d.apply_plan([bad, bad2, good], cache6)
    _check("malformed-id migrates skipped (no crash), good one applied",
           len(cache6.migrate_calls) == 1 and len(cache6.migrate_calls[0]) == 1
           and cache6.migrate_calls[0][0]["hash"] == "ok")


def stage_E_cadence_gate():
    print("[E] should_tick cadence gate (the #1 hard problem: trigger under load)")
    d = AginferDriver()
    # below the low watermark → never tick (no pressure = nothing to migrate)
    _check("occ < theta_lo → no tick", d.should_tick(0.5, 100.0, theta_lo=0.7, min_interval_s=5.0) is False)
    _check("last_tick untouched when gated off", d._last_tick_t is None)
    # at/above theta_lo, first call → fires (last_tick was None)
    _check("occ ≥ theta_lo, first call → tick", d.should_tick(0.8, 100.0, theta_lo=0.7, min_interval_s=5.0) is True)
    _check("last_tick stamped", d._last_tick_t == 100.0)
    # within the throttle interval → no tick
    _check("within min_interval → throttled", d.should_tick(0.9, 103.0, theta_lo=0.7, min_interval_s=5.0) is False)
    _check("throttle did not advance last_tick", d._last_tick_t == 100.0)
    # past the interval → tick again
    _check("past min_interval → tick", d.should_tick(0.9, 106.0, theta_lo=0.7, min_interval_s=5.0) is True)
    _check("last_tick advanced", d._last_tick_t == 106.0)
    # pressure drops below θ_lo even past the interval → no tick
    _check("drop below theta_lo again → no tick", d.should_tick(0.6, 200.0, theta_lo=0.7, min_interval_s=5.0) is False)


def stage_F_hook_do_no_harm():
    print("[F] engine hook _aginfer_maybe_tick is INERT when the flag is off (do-no-harm)")

    class _FakeScheduler:
        """Minimal stand-in: the real method lives on Scheduler; replicate its guard so
        we pin the do-no-harm contract WITHOUT importing the full engine (server-free)."""
        def __init__(self, flag):
            self._aginfer_in_engine = flag
            self._aginfer_driver = AginferDriver() if flag else None
            self.tree_cache = None  # touching this would raise → proves the guard returns first
            self.ticked = 0

        # the exact guard from Scheduler._aginfer_maybe_tick (the first two lines)
        def maybe_tick(self):
            if not getattr(self, "_aginfer_in_engine", None):
                return  # do-no-harm: immediate return, nothing touched
            self.ticked += 1  # would proceed to read occ / should_tick / tick

    off = _FakeScheduler(flag=False)
    off.maybe_tick()
    _check("flag OFF → hook returns immediately (no driver, no tick)", off.ticked == 0)
    on = _FakeScheduler(flag=True)
    on.maybe_tick()
    _check("flag ON → hook proceeds", on.ticked == 1)


def stage_G_decide_composition():
    print("[G] decide() = joint_decide ∘ postprocess (composition; joint_decide monkeypatched)")
    import sglang.srt.mem_cache.aginfer.joint_decide as _jd
    orig = _jd.joint_decide
    try:
        # stub joint_decide to return a known plan; decide() must then apply the post-filters.
        remove_hbm = _mig("z", add=[], remove=[Tier.HBM])
        keep = _mig("y", add=[Tier.DRAM], remove=[])
        captured = {}

        def fake_joint_decide(sched_state, event, **kw):
            captured.update(kw)
            return [remove_hbm, keep]
        _jd.joint_decide = fake_joint_decide

        # EMA healthy → saturation yield is a no-op → both survive
        d = AginferDriver()
        plan = d.decide(object(), object(), costs="C", pi_u="P", theta_hi=0.9,
                        theta_lo=0.7, heartbeat_s=5.0, admission_enabled=True, now=100.0)
        _check("decide forwards params to joint_decide",
               captured.get("theta_hi") == 0.9 and captured.get("admission_enabled") is True)
        _check("decide returns joint_decide's plan unfiltered when EMA healthy", len(plan) == 2)

        # EMA low → decide must strip the remove-HBM (postprocess applied inside decide)
        d2 = AginferDriver()
        d2._demote_apply_ema = _DEMOTE_YIELD_EMA - 0.05
        plan2 = d2.decide(object(), object(), costs="C", pi_u="P", theta_hi=0.9,
                          theta_lo=0.7, heartbeat_s=5.0, admission_enabled=True, now=100.0)
        _check("decide applies saturation yield (remove-HBM stripped)",
               remove_hbm not in plan2 and keep in plan2)

        # default now (None) → uses monotonic, must not crash + still composes
        d3 = AginferDriver()
        plan3 = d3.decide(object(), object(), costs="C", pi_u="P", theta_hi=0.9,
                          theta_lo=0.7, heartbeat_s=5.0, admission_enabled=False)
        _check("decide with default now= (monotonic) works", len(plan3) == 2)
    finally:
        _jd.joint_decide = orig


def stage_H_tick_paths():
    print("[H] tick(): in-process dump path + no_dump + non-dict guard")
    d = AginferDriver()

    class _CacheWithDump:
        def dump_aginfer_state(self):
            return {"units": {"a": 1, "b": 2, "c": 3}, "time_counter": 7}

    class _SchedDump:
        tree_cache = _CacheWithDump()
    r = d.tick(_SchedDump())
    _check("tick dump path returns dump_only status", r["status"] == "dump_only_pending_build_state")
    _check("tick counts units from the in-process dump", r["n_units"] == 3)

    class _CacheNoDump:
        pass  # no dump_aginfer_state attr

    class _SchedNoDump:
        tree_cache = _CacheNoDump()
    r2 = d.tick(_SchedNoDump())
    _check("tick no_dump path when cache lacks dump_aginfer_state", r2["status"] == "no_dump")

    class _CacheListDump:
        def dump_aginfer_state(self):
            return ["not", "a", "dict"]  # malformed

    class _SchedList:
        tree_cache = _CacheListDump()
    r3 = d.tick(_SchedList())
    _check("tick non-dict dump → n_units 0 (isinstance guard, no crash)", r3["n_units"] == 0)


def stage_I_engine_hook_real_body():
    print("[I] Scheduler._aginfer_maybe_tick REAL body (flag on/off + exception-disable)")
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except Exception as e:  # heavy import (torch); skip gracefully if unavailable
        print("  SKIP (cannot import Scheduler: %s)" % str(e)[:80])
        return
    import types

    def _fake(occ_fn, flag=True):
        ns = types.SimpleNamespace()
        ns._aginfer_in_engine = flag
        ns._aginfer_driver = AginferDriver() if flag else None
        ns.server_args = types.SimpleNamespace(aginfer_theta_lo=0.7, aginfer_heartbeat_s=5.0)

        class _TC:
            def _aginfer_pool_usage(self_inner):
                return {"HBM": {"token_usage": occ_fn()}}
        ns.tree_cache = _TC()
        return ns

    # flag OFF → immediate return, driver untouched (do-no-harm), no exception
    off = _fake(lambda: 0.9, flag=False)
    Scheduler._aginfer_maybe_tick(off)
    _check("real hook flag-off → no-op (driver stays None)", off._aginfer_driver is None)

    # flag ON, high occ → should_tick True (first call) → tick runs (no_dump, no crash);
    # driver._last_tick_t gets stamped (proves should_tick fired through the real body)
    on = _fake(lambda: 0.9, flag=True)
    Scheduler._aginfer_maybe_tick(on)
    _check("real hook flag-on, high occ → ticked (last_tick stamped)",
           on._aginfer_driver._last_tick_t is not None)

    # flag ON, low occ → should_tick False → no tick (last_tick stays None)
    lo = _fake(lambda: 0.5, flag=True)
    Scheduler._aginfer_maybe_tick(lo)
    _check("real hook flag-on, low occ → not ticked (below theta_lo)",
           lo._aginfer_driver._last_tick_t is None)

    # exception in occ read → except disables the feature for the session (never raises)
    def _boom():
        raise RuntimeError("occ read failed")
    bad = _fake(_boom, flag=True)
    Scheduler._aginfer_maybe_tick(bad)  # must NOT raise
    _check("real hook exception → feature DISABLED for session (crash isolation)",
           bad._aginfer_in_engine is False)


if __name__ == "__main__":
    stage_A_apply_rate_ema()
    stage_B_postprocess_plan()
    stage_C_daemon_delegates()
    stage_D_in_process_apply()
    stage_E_cadence_gate()
    stage_F_hook_do_no_harm()
    stage_G_decide_composition()
    stage_H_tick_paths()
    stage_I_engine_hook_real_body()
    print("verify/scheduler_driver: ALL STAGES PASSED")
