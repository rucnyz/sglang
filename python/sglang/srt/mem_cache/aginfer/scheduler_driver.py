"""aginfer in-engine scheduler driver (refactor #251 Stage B, increment 1).

THE entry point the daemon's out-of-process orchestrator is being dissolved into.
Today the decision brain still runs in dev/aginfer/daemon/kv_scheduler.py over HTTP;
this module begins relocating that brain in-process so the engine can eventually
drive `joint_decide` itself with NO daemon (the #251 blocker = "no in-engine driver
loop"; this is its first load-bearing piece).

Increment 1 extracts the PURE post-`joint_decide` decision subsystem — the part that
turns a fresh state + event into a final plan — with NO transport (no fetch, no
outbound POST). It is byte-for-faithful with the inline logic it replaces in
`KvScheduler.handle()` (lines ~1431-1467 + `_update_demote_apply_rate`); the daemon
now delegates to it, so behaviour is unchanged and the existing verify suite is the
regression gate. Later increments fold in the belief plane + the cadence hook + the
in-process apply, then the daemon deletes.

Responsibilities owned here (the saturation-yield / cooldown decision subsystem):
  * `update_demote_apply_rate(units)` — the #240 apply-rate EMA (did our recent
    remove-HBM demotes actually land?), driven once per fresh dump.
  * `decide(sched_state, event, ...)` — run the in-engine `joint_decide`, then
    apply the #223 evict-cooldown filter and the #240 saturation yield, returning
    the final plan (may be empty = no-op, the do-no-harm floor).

State is reconstructed-from-scratch on restart (no persistence), exactly as the
daemon's was — see `_update_demote_apply_rate`'s original docstring.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sglang.srt.mem_cache.aginfer.base import Tier
from sglang.srt.mem_cache.aginfer.knapsack import Migrate, Pause, Resume


def tier_to_wire(tier: Tier) -> str:
    """Tier enum → the DESIGN §6 wire string. Single-sourced here (the daemon
    re-exports as `_tier_to_wire`)."""
    return {Tier.HBM: "HBM", Tier.DRAM: "DRAM", Tier.DISK: "DISK", Tier.DROP: "DROP"}[tier]


def assignments_to_wire(assignments) -> List[Dict[str, Any]]:
    """Translate ``[(unit_hash, add_tiers, remove_tiers), ...]`` → DESIGN §6
    ``apply_aginfer_migrations`` action dicts. Each carries add/remove tier lists
    plus an opaque ``action_id`` correlator (echoed back in APPLY_FAILED webhooks).
    Moved verbatim from daemon/kv_scheduler.py (single source; daemon re-exports)."""
    import uuid
    return [
        {
            "hash": uhash,
            "add_tiers": [tier_to_wire(t) for t in add],
            "remove_tiers": [tier_to_wire(t) for t in remove],
            "action_id": uuid.uuid4().hex,
        }
        for uhash, add, remove in assignments
    ]


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# #240 saturation yield: when the recent remove-HBM apply-rate EMA falls below this,
# the engine is racing sglang's own eviction lock on apply, so strip explicit demotes
# from the plan (sglang's V_u-guided reactive eviction frees the same idle tails). The
# constant mirrors the daemon's AGINFER_DEMOTE_YIELD_EMA (kv_scheduler.py).
_DEMOTE_YIELD_EMA = _env_float("AGINFER_DEMOTE_YIELD_EMA", "0.4")


def filter_cooled_evicts(plan: List[Any], cooldown: Dict[str, float],
                         now: float) -> List[Any]:
    """#223: drop any migrate that REMOVES a tier for a hash currently in the TOCTOU
    evict cooldown. Pure-add migrates (write-through) for a cooled hash still pass —
    only the failing remove is backed off. Expired entries are ignored (pruned by the
    caller). Moved verbatim from daemon/kv_scheduler.py:_filter_cooled_evicts."""
    out: List[Any] = []
    for c in plan:
        # a live Migrate's id is always (uid, add_tiers, remove_tiers) (ours_greedy.py:355);
        # guard the unpack so a malformed/None id can never crash the eviction path — such a
        # migrate just isn't a cooldown candidate and passes through (same as the live result).
        cid = getattr(c, "id", None)
        if isinstance(c, Migrate) and isinstance(cid, tuple) and len(cid) >= 3:
            uid, _add, remove = cid[0], cid[1], cid[2]
            if remove and cooldown.get(uid, 0.0) > now:
                continue
        out.append(c)
    return out


class AginferDriver:
    """In-engine decision driver. Owns the saturation-yield / cooldown subsystem and
    composes `joint_decide` into a final plan. Stateful (the apply-rate EMA), single
    instance per scheduler — NOT thread-safe (the engine scheduler loop is single
    threaded; the daemon consumer is too)."""

    def __init__(self) -> None:
        # #240 apply-rate EMA state (was KvScheduler._pending_demote / _demote_apply_ema
        # / _dump_gen). EMA starts optimistic (1.0 = "demotes land") so the first
        # pressure event is free to demote; it decays only on observed futile demotes.
        self._pending_demote: Dict[str, int] = {}
        self._demote_apply_ema: float = 1.0
        self._dump_gen: int = 0
        # #251 Stage B increment 3: the in-engine trigger throttle (last tick wall-time).
        self._last_tick_t: Optional[float] = None

    # -- cadence gate (the #1 hard problem: trigger under never-idle high load) ----
    # on_idle() only fires when the engine is FULLY idle — exactly the low-pressure
    # regime where aginfer has nothing to do; under flood (where the win lives) the loop
    # is never idle. So the driver rides the per-iteration hook (next to the webhook fire,
    # which runs busy-OR-idle), but is PRESSURE-GATED (only when occ ≥ theta_lo — no
    # pressure ⇒ nothing to migrate) and INTERVAL-THROTTLED (≥ min_interval since the last
    # tick) so it never runs every step (the dump is 5-50ms, #160) nor competes with the
    # hot path under headroom. Pure + deterministic ⇒ unit-tested.
    def should_tick(self, occ: float, now: float, *, theta_lo: float,
                    min_interval_s: float) -> bool:
        if occ < theta_lo:
            return False  # below the low watermark: no pressure, no work
        if self._last_tick_t is not None and (now - self._last_tick_t) < min_interval_s:
            return False  # throttle: too soon since the last tick
        self._last_tick_t = now
        return True

    # -- #240 saturation yield: measure whether dispatched demotes actually landed --
    def update_demote_apply_rate(self, units: Dict[str, Any]) -> None:
        """Did our recently-dispatched remove-HBM demotes actually land? A dispatched
        hash STILL HBM-resident a dump-generation later did not apply (lock-race vs
        sglang's own eviction). Update the apply-rate EMA the saturation yield reads.
        The EMA tolerates the dump's <1s eventual-consistency lag. Reconstructed from
        scratch on restart. Verbatim from daemon/kv_scheduler.py:_update_demote_apply_rate."""
        self._dump_gen += 1
        if not self._pending_demote:
            # Recovery drift: with nothing pending, slowly relax the EMA back toward
            # 1.0 so a sustained yield re-probes demotes once pressure eases — else the
            # yield (which suppresses dispatch, hence new measurements) locks in forever.
            if self._demote_apply_ema < 1.0:
                self._demote_apply_ema = min(1.0, self._demote_apply_ema + 0.02)
            return
        for h, gen in list(self._pending_demote.items()):
            if self._dump_gen - gen < 1:
                continue  # give the apply at least one fresh dump to show up
            u = units.get(h)
            landed = (u is None) or (Tier.HBM not in u.residence)
            self._demote_apply_ema = (
                0.7 * self._demote_apply_ema + 0.3 * (1.0 if landed else 0.0))
            del self._pending_demote[h]
        if len(self._pending_demote) > 4096:  # safety: never grow unbounded
            self._pending_demote.clear()

    def note_demote_dispatched(self, uid: str) -> None:
        """Record a remove-HBM migrate we just dispatched, tagged with the current dump
        generation, so a later `update_demote_apply_rate` can check if it landed. (The
        daemon does this in `_dispatch_plan`; exposed here so the apply-rate state lives
        with the EMA that consumes it.)"""
        self._pending_demote[uid] = self._dump_gen

    def postprocess_plan(self, plan: List[Any], evict_cooldown: Optional[Dict[str, float]],
                         now: float) -> List[Any]:
        """The pure plan post-processing the daemon ran inline after joint_decide:
        (1) #223 evict-cooldown filter, (2) #240 saturation yield. Returns the final
        plan (possibly empty). `evict_cooldown` may be None/empty (no cooldown)."""
        if evict_cooldown:
            # prune expired entries in place, then filter
            for _h in [h for h, exp in evict_cooldown.items() if exp <= now]:
                evict_cooldown.pop(_h, None)
            if evict_cooldown:
                plan = filter_cooled_evicts(plan, evict_cooldown, now)
        # #240 SATURATION YIELD: if recent explicit demotes aren't landing (EMA below
        # threshold → racing sglang's lock at apply), strip remove-HBM migrates. sglang's
        # V_u-guided reactive eviction frees the same idle tails; keep promotes/pauses/
        # resumes. Value-optimal do-no-harm (DESIGN §9 saturation yield).
        if plan and self._demote_apply_ema < _DEMOTE_YIELD_EMA:
            from sglang.srt.mem_cache.aginfer._metrics import m as _m
            _before = len(plan)
            plan = [c for c in plan if not (
                isinstance(c, Migrate)
                and isinstance(getattr(c, "id", None), tuple)
                and len(c.id) >= 3 and Tier.HBM in c.id[2])]
            if len(plan) < _before:
                _m("demote_saturation_yield",
                   ema=round(self._demote_apply_ema, 3),
                   stripped=_before - len(plan))
        return plan

    # -- in-process apply (the "no HTTP" half of the #251 blocker) ---------------
    # The daemon translates a plan into wire dicts and POSTs them over HTTP through the
    # bridge to the engine's apply_aginfer_migrations / set_aginfer_hints. In-engine
    # there is no process boundary: the driver calls those cache methods DIRECTLY. The
    # wire dicts are byte-identical to the HTTP path's, so the engine side is unchanged.
    def apply_hints(self, hints: List[Dict[str, Any]], cache) -> Any:
        """Push the V_u hints in-process (was: outbound HTTP PUT /aginfer/hints →
        bridge → set_aginfer_hints). Empty list = no-op. `cache` is the engine's
        UnifiedRadixCache (exposes set_aginfer_hints)."""
        if not hints:
            return None
        return cache.set_aginfer_hints(hints)

    def apply_plan(self, plan: List[Any], cache) -> Dict[str, Any]:
        """Apply a joint_decide mixed plan in-process. MIGRATIONS go straight to
        `cache.apply_aginfer_migrations` (was: outbound HTTP POST /aginfer/migrate).
        Remove-HBM migrates are recorded for the #240 apply-rate EMA. PAUSE/RESUME are
        the ADMISSION axis — NOT enforced in-engine (the ingress gate lives at the
        Dynamo router, #251 verified split); they are surfaced for the caller to route.
        Returns {"migrate_result", "pauses": [pid...], "resumes": [pid...]}."""
        # guard the id unpack: a live Migrate's id is always a 3-tuple (ours_greedy.py:355);
        # skip any malformed/None-id migrate rather than crash the apply path (it carries no
        # uid, so it could not be applied anyway). Same defensive contract as the eviction path.
        migrates = [c for c in plan if isinstance(c, Migrate)
                    and isinstance(getattr(c, "id", None), tuple) and len(c.id) >= 3]
        pauses = [getattr(c, "pid", None) for c in plan if isinstance(c, Pause)]
        resumes = [getattr(c, "pid", None) for c in plan if isinstance(c, Resume)]
        migrate_result = None
        if migrates:
            assignments = [m.id for m in migrates]
            wire = assignments_to_wire(assignments)
            for uid, _add, remove in assignments:
                if remove and Tier.HBM in remove:
                    self.note_demote_dispatched(uid)
            migrate_result = cache.apply_aginfer_migrations(wire)
        return {"migrate_result": migrate_result,
                "pauses": [p for p in pauses if p is not None],
                "resumes": [p for p in resumes if p is not None]}

    def decide(self, sched_state, event, *, costs, pi_u, theta_hi: float,
               theta_lo: float, heartbeat_s: float, admission_enabled: bool,
               evict_cooldown: Optional[Dict[str, float]] = None,
               now: Optional[float] = None) -> List[Any]:
        """Run the in-engine `joint_decide` over the union action space, then
        post-process (cooldown + saturation yield). Returns the final mixed plan.
        This is the in-process equivalent of the daemon's handle() decision body
        (kv_scheduler.py:1431-1467) — same call, same filters, no transport."""
        from sglang.srt.mem_cache.aginfer.joint_decide import joint_decide
        plan = joint_decide(
            sched_state, event,
            costs=costs,
            pi_u=pi_u,
            theta_hi=theta_hi,
            theta_lo=theta_lo,
            heartbeat_s=heartbeat_s,
            admission_enabled=admission_enabled,
        )
        if now is None:
            import time as _time
            now = _time.monotonic()
        return self.postprocess_plan(plan, evict_cooldown, now)

    # -- the in-process tick (composes dump → build_paper_state → decide → apply) --
    # Called from the engine scheduler loop's per-iteration hook when should_tick() is
    # true. This is the in-process replacement for the daemon's event_router → handle()
    # cycle. The DECIDE + APPLY pieces are already in-engine (decide / apply_plan /
    # apply_hints above); the missing piece is build_paper_state (still daemon-side,
    # kv_scheduler.py:624) — it relocates in the next increment, at which point the
    # commented body below activates. Until then the tick exercises the in-process
    # dump path (the engine producing its own s_t with no HTTP) and is a value no-op
    # — so turning the flag on is safe (it cannot mis-decide without the builder).
    def tick(self, scheduler) -> Dict[str, Any]:
        """One in-process scheduling tick. Returns a small status dict (for the hook's
        metric/log). MUST NOT raise into the scheduler loop — the caller wraps it, but
        keep it defensive too."""
        dump_fn = getattr(scheduler.tree_cache, "dump_aginfer_state", None)
        if dump_fn is None:
            return {"status": "no_dump"}  # non-Unified cache: in-engine aginfer N/A
        state_json = dump_fn()  # the engine's own s_t, in-process (no /aginfer/state HTTP)
        # build_paper_state is not yet in-engine (increment 4). Until then we do not
        # decide/apply — fire the trigger + the in-process dump only. When the builder
        # lands, this becomes: sched_state = build_paper_state(state_json, ...);
        # plan = self.decide(sched_state, event, ...); self.apply_plan(plan, cache);
        # self.apply_hints(hints_from_state(sched_state), cache).
        n_units = len(state_json.get("units", {})) if isinstance(state_json, dict) else 0
        return {"status": "dump_only_pending_build_state", "n_units": n_units}
