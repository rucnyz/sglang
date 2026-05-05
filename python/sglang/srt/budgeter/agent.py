"""BudgetAgent — in-process pool-pressure observer.

The scheduler instantiates one of these and calls `tick()` on every event-loop
iteration. The agent rate-limits internally: it only does real work every
`tick_interval_s` seconds. Each tick snapshots per-pool state to a JSONL and
optionally drives cross-pool VMM transfers via the planner / actuator path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


# Stats fields the budgeter requires from `scheduler.stats`. Validated
# once at health-check time; subsequent snapshots access them directly.
_REQUIRED_STATS_FIELDS = (
    "max_total_num_tokens",
    "kv_used_tokens",
    "kv_evictable_tokens",
    "kv_available_tokens",
    "token_usage",
    "full_token_usage",
    "swa_token_usage",
    "mamba_usage",
    "cache_hit_rate",
    "num_running_reqs",
    "num_queue_reqs",
    "num_paused_reqs",
    "num_retracted_reqs",
    "gen_throughput",
)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


def _json_default(obj: Any) -> Any:
    """Serializer fallback: unwrap dataclass-like SchedulerStats fields to scalars.

    `num_running_reqs`/`num_queue_reqs` are `QueueCount` instances exposing a
    `.total` int (and an optional priority breakdown). For the budgeter we only
    need the total. Other unknown objects fall back to their string repr.
    """
    total = getattr(obj, "total", None)
    if isinstance(total, (int, float)):
        return total
    return str(obj)


class BudgetAgent:
    """Observes pool pressure each scheduler tick.

    Cheap to construct; cheap to `tick()` (rate-limited internally). Holds a
    handle to the scheduler so it can drive cross-pool VMM transfers via
    `_maybe_xpool_planner` (paper §sec:design-l2). Snapshot-only when the
    cross-pool actuator can't be wired (e.g. non-hybrid model, or
    SGLANG_ARENA_SHARED=0).
    """

    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        # Required scheduler deps; populated on first tick by _do_health_check.
        # If any required dep is missing the agent hard-disables itself.
        self._tree_cache = None
        self._health_checked = False

        self.enabled = _env_flag("SGLANG_BUDGETER", False)
        self.tick_interval_s = _env_float("SGLANG_BUDGETER_TICK_S", 30.0)
        self.log_path = _env_str(
            "SGLANG_BUDGETER_LOG", "/tmp/sglang_budgeter.jsonl"
        )
        self._last_tick = 0.0
        self._tick_count = 0
        self._last_evicted_cumulative = 0
        # JSONL snapshot logging. log_enabled flips True only after a
        # successful open(); read paths gate on the flag, never on the
        # raw file handle.
        self.log_enabled = False
        self._log_fp = None

        # KV-side arena actuator (KVArenaActuator). Lazy-built on first
        # use; serves as the KV per-pool actuator for the cross-pool
        # fire path (propagates capacity changes back to the allocator).
        self._arena_actuator = None

        # Cross-pool transfer actuator. Requires SGLANG_ARENA_SHARED=1 at
        # engine boot; lazy-built on first tick.
        self._xpool_actuator = None

        # Cross-pool planner state (paper §sec:design-l2). Lazy-built on
        # first tick; runs every tick whenever the budgeter is enabled.
        self._xpool_planner = None

        # T6 (paper §3.2.4): admission-time fire enable.
        self.admission_time_fire_enabled = _env_flag(
            "SGLANG_ADMISSION_TIME_FIRE", False
        )
        # Single in-flight emergency fire at a time — avoids the same
        # admission event triggering N parallel actuator calls.
        self._emergency_fire_in_progress = False

        if self.enabled:
            try:
                self._log_fp = open(self.log_path, "a", buffering=1)
                self.log_enabled = True
            except OSError as e:
                logger.warning("BudgetAgent: failed to open %s: %s", self.log_path, e)
            logger.info(
                "BudgetAgent enabled (tick=%.1fs, log=%s, pid=%d)",
                self.tick_interval_s, self.log_path, os.getpid(),
            )
            # Register process-singleton so admission-time hooks can
            # find the agent without restructuring tree_cache.
            from sglang.srt.budgeter import _set_budget_agent_singleton
            _set_budget_agent_singleton(self)
            if self.admission_time_fire_enabled:
                logger.info(
                    "T6 admission-time fire enabled — admission gate may "
                    "trigger cross-pool transfers on demand"
                )

    # ---- Public API used from scheduler.event_loop_* ----

    def _do_health_check(self) -> bool:
        """One-shot schema check on first tick. SGLang init order
        guarantees scheduler.stats / .tree_cache / .token_to_kv_pool_allocator
        are non-None by the time the event loop runs (otherwise scheduler
        init would have crashed). The real failure mode this guards is
        upstream renaming a field on `SchedulerStats`: we check every
        field the snapshot path reads and hard-disable on drift so the
        error surfaces as one log line instead of per-tick AttributeError."""
        stats = self.scheduler.stats
        missing = [f"scheduler.stats.{f}" for f in _REQUIRED_STATS_FIELDS
                   if not hasattr(stats, f)]
        if missing:
            logger.error(
                "BudgetAgent health check failed — SchedulerStats schema drift: %s",
                ", ".join(missing),
            )
            return False
        tree_cache = self.scheduler.tree_cache
        # Pre-init counters on tree_cache. The eviction / retract sites
        # lazy-create these on the first event; pre-creating here lets
        # the snapshot path read attributes directly without getattr.
        if not hasattr(tree_cache, "_admission_cumulative_evicted_tokens"):
            tree_cache._admission_cumulative_evicted_tokens = 0
        if not hasattr(tree_cache, "_slow_recovery_len_kv_ewma"):
            tree_cache._slow_recovery_len_kv_ewma = 0.0
        if not hasattr(tree_cache, "_slow_recovery_len_rec_ewma"):
            tree_cache._slow_recovery_len_rec_ewma = 0.0
        if not hasattr(tree_cache, "_slow_recovery_len_retract_ewma"):
            tree_cache._slow_recovery_len_retract_ewma = 0.0
        self._tree_cache = tree_cache
        return True

    def tick(self) -> None:
        """Called every scheduler iteration. Internally rate-limited."""
        if not self.enabled:
            return
        if not self._health_checked:
            if not self._do_health_check():
                # Hard-disable: the scheduler is missing dependencies the
                # budgeter requires. Don't silently degrade — refuse to run
                # so an operator can see the error in the log.
                self.enabled = False
                logger.error(
                    "BudgetAgent: hard-disabled (health check failed). "
                    "Set the missing scheduler deps and restart."
                )
                return
            self._health_checked = True
        now = time.time()
        if now - self._last_tick < self.tick_interval_s:
            return
        self._last_tick = now
        self._tick_count += 1
        try:
            snapshot = self._snapshot(now)
        except Exception as e:  # never break the scheduler hot path
            logger.warning("BudgetAgent.tick snapshot failed: %s", e, exc_info=True)
            return

        # Planner-driven cross-pool transfers (paper §sec:design-l2).
        try:
            self._maybe_xpool_planner(snapshot)
        except Exception as e:
            logger.warning("BudgetAgent xpool planner failed: %s", e, exc_info=True)

        if self.log_enabled:
            try:
                self._log_fp.write(json.dumps(snapshot, default=_json_default) + "\n")
            except Exception as e:
                logger.warning("BudgetAgent.tick write failed: %r", e)

    def _ensure_arena_actuator(self) -> None:
        if self._arena_actuator is not None:
            return
        alloc = self.scheduler.token_to_kv_pool_allocator
        pool = alloc.get_kvcache()
        # Hybrid models route through HybridLinearKVPool (a thin wrapper):
        # the actual MHATokenToKVPool with `_kv_arena` is `pool.full_kv_pool`.
        # Single-pool (non-hybrid) models put the arena directly on `pool`.
        kv_pool = pool
        if not hasattr(kv_pool, "_kv_arena") and hasattr(pool, "full_kv_pool"):
            kv_pool = pool.full_kv_pool
        if not hasattr(kv_pool, "_kv_arena"):
            return
        from sglang.srt.arena.kv_actuator import KVArenaActuator
        self._arena_actuator = KVArenaActuator(kv_pool, alloc)
        logger.info("BudgetAgent: arena actuator attached (max=%d)",
                    self._arena_actuator.max_tokens)

    def _ensure_xpool_actuator(self) -> None:
        """Lazily attach the cross-pool transfer actuator (paper §sec:design-l2).

        Walks the scheduler to find the hybrid pool (full_kv_pool +
        mamba_pool), then wraps both arenas in a CrossPoolTransferActuator
        that writes through the shared SharedHandlePool.
        """
        if self._xpool_actuator is not None:
            return
        alloc = self.scheduler.token_to_kv_pool_allocator
        pool = alloc.get_kvcache()
        full_kv = getattr(pool, "full_kv_pool", None)
        mamba_pool = getattr(pool, "mamba_pool", None)
        if full_kv is None or mamba_pool is None:
            return
        kv_arena = getattr(full_kv, "_kv_arena", None)
        mamba_arena = getattr(mamba_pool, "_mamba_temporal_arena", None)
        if kv_arena is None or mamba_arena is None:
            return
        shared = kv_arena._arena._external_pool
        if shared is None:
            logger.warning(
                "BudgetAgent xpool: KV arena has no external SharedHandlePool; "
                "is SGLANG_ARENA_SHARED=1 set?"
            )
            return
        if mamba_arena._arena._external_pool is not shared:
            logger.warning(
                "BudgetAgent xpool: KV and mamba arenas use different "
                "SharedHandlePool instances — cross-pool transfer disabled."
            )
            return
        from sglang.srt.arena.cross_pool_actuator import (
            CrossPoolTransferActuator,
        )
        from sglang.srt.arena.mamba_actuator import MambaArenaActuator
        self._ensure_arena_actuator()
        kv_act = self._arena_actuator
        try:
            mamba_act = MambaArenaActuator(mamba_pool)
        except RuntimeError as e:
            logger.warning(
                "MambaArenaActuator build failed (%s); cross-pool transfer "
                "disabled — under live traffic without per-pool capacity "
                "propagation, allocator can hand out unmapped slots.", e,
            )
            return
        self._xpool_actuator = CrossPoolTransferActuator(
            kv_arena=kv_arena,
            mamba_arena=mamba_arena,
            shared_pool=shared,
            kv_actuator=kv_act,
            mamba_actuator=mamba_act,
        )
        logger.info(
            "BudgetAgent xpool: actuator attached (kv_act=%s, mamba_act=%s)",
            kv_act is not None, mamba_act is not None,
        )

    def _ensure_t8_state(self) -> bool:
        """Lazily build the T8 plan-based-fire state: FirePlanner,
        SchedulerOwnerProvider, KVPageMigrator, and a drain callback
        that delegates to the radix tree's per-page evict.

        Returns True iff state is fully wired and the T8 path can fire.
        Returns False (and logs once) if any prerequisite is missing.
        """
        if getattr(self, "_t8_state", None) is not None:
            return self._t8_state is not False
        # Use False as a sentinel for "tried, gave up" so we don't retry.
        self._t8_state = False
        if self._xpool_actuator is None:
            return False
        if self._xpool_actuator.kv_actuator is None:
            logger.info("T8: no kv_actuator wired — staying on legacy path")
            return False
        try:
            from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
            from sglang.srt.budgeter.scheduler_owner_provider import (
                SchedulerOwnerProvider,
            )
        except ImportError as e:
            logger.warning("T8: import failed: %r", e)
            return False
        provider = SchedulerOwnerProvider(
            scheduler=self.scheduler,
            kv_actuator=self._xpool_actuator.kv_actuator,
            mamba_actuator=self._xpool_actuator.mamba_actuator,
        )
        planner = XPoolFirePlanner(
            kv_actuator=self._xpool_actuator.kv_actuator,
            mamba_actuator=self._xpool_actuator.mamba_actuator,
            owner_provider=provider,
        )

        self._t8_state = {"planner": planner, "provider": provider}
        logger.info("T8: state wired — fires will route through execute(plan)")
        return True

    def _maybe_t8_fire(self, direction: str, unit: int, snapshot: dict):
        """Run a fire through the plan-based path. Returns a stats dict
        in the legacy shape so callers' snapshot logging works unchanged,
        or None if wiring isn't ready."""
        if not self._ensure_t8_state():
            return None
        st = self._t8_state
        actuator = self._xpool_actuator
        plan = st["planner"].build(
            direction=direction, n_pages_target=unit,
        )
        if plan is None:
            snapshot["xpool_t8_skipped"] = "plan_refused"
            return {
                "direction": direction,
                "unmapped_total": 0,
                "granted_total": 0,
                "kv_capacity_tokens": actuator.kv.current_capacity_tokens(),
                "mamba_capacity_tokens": actuator.mamba.current_capacity_tokens(),
                "free_after_grow": actuator.shared.free_count(),
                "skipped": "t8_plan_refused",
            }
        try:
            res = actuator.execute(plan)
        except RuntimeError as e:
            logger.error("T8 execute(plan) failed: %r", e)
            snapshot["xpool_t8_error"] = repr(e)
            return None
        snapshot["xpool_t8_plan_seq"] = res.plan_seq
        snapshot["xpool_t8_aborted"] = res.aborted
        if res.aborted:
            snapshot["xpool_t8_abort_reason"] = res.abort_reason
            return {
                "direction": direction,
                "unmapped_total": 0,
                "granted_total": 0,
                "kv_capacity_tokens": actuator.kv.current_capacity_tokens(),
                "mamba_capacity_tokens": actuator.mamba.current_capacity_tokens(),
                "free_after_grow": actuator.shared.free_count(),
                "skipped": f"t8_aborted: {res.abort_reason}",
            }
        return {
            "direction": direction,
            "unmapped_total": res.unmapped_pages,
            "granted_total": res.granted_pages,
            "kv_capacity_tokens": actuator.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": actuator.mamba.current_capacity_tokens(),
            "free_after_grow": actuator.shared.free_count(),
            "shrink_us": res.unmap_us,
            "grow_us": res.map_us,
            "fire_total_us": res.total_us,
        }

    def _estimate_post_fire_mamba_cap(self) -> int:
        """Predict the mamba-pool slot cap immediately after a
        successful mamba_to_kv fire of the configured unit size.
        Used by the pre-fire flush to compute the slot threshold for
        slot-id-targeted eviction. Returns 0 if the actuator has no
        mamba state (in which case targeted flush degrades to no-op).
        """
        try:
            actuator = self._xpool_actuator
            if actuator is None:
                return 0
            mamba_act = actuator.mamba_actuator
            if mamba_act is None or not hasattr(mamba_act, "pool"):
                return 0
            current_cap = mamba_act.live_capacity_tokens()
            unit = self._xpool_planner.config.dst_chunks_per_action
            n_mamba = actuator.n_mamba_subpools
            n_kv = actuator.n_kv_subpools
            # mamba_to_kv shrinks src (mamba) by the dst-anchored amount
            # divided by the src/dst sub-pool ratio. Per actuator code:
            # n_per_src_subpool = ceil(unit * n_kv / n_mamba)
            n_per_src = max(1, (unit * n_kv + n_mamba - 1) // n_mamba)
            mamba_pool = mamba_act.pool
            tokens_per_chunk = max(
                1, getattr(mamba_pool, "size", 384) // max(1,
                    getattr(actuator, "_init_chunks_per_pool", 3))
            )
            shrink_tokens = n_per_src * tokens_per_chunk
            return max(1, current_cap - shrink_tokens)
        except Exception:
            return 0

    def try_admission_time_fire(
        self,
        direction: str = "rec_to_kv",
        n_chunks: int = 1,
    ) -> bool:
        """T6 (paper §3.2.4): on-demand cross-pool fire from the admission
        gate. Skips the 30 s cooldown and hysteresis (the request-side
        wait penalty has already exceeded those by construction — paper
        Eq.~\\ref{eq:nb-direction-gate}).

        Returns True iff a fire was actually committed (bytes moved).
        Safe to call from the scheduler hot path; no-op when:
          - SGLANG_ADMISSION_TIME_FIRE=0 (default)
          - another emergency fire is already in flight on this thread
          - cross-pool slack is insufficient (planner declines)

        `direction` is "kv_to_rec" or "rec_to_kv". `n_chunks` is the
        size of the requested transfer in chunks.
        """
        if not self.admission_time_fire_enabled:
            return False
        if self._emergency_fire_in_progress:
            return False  # Reentrancy guard
        self._emergency_fire_in_progress = True
        try:
            self._ensure_xpool_actuator()
            if self._xpool_actuator is None:
                return False
            if direction == "kv_to_rec":
                t8_dir = "kv_to_mamba"
            elif direction == "rec_to_kv":
                t8_dir = "mamba_to_kv"
            else:
                logger.warning(
                    "try_admission_time_fire: unknown direction %r", direction
                )
                return False
            stats = self._maybe_t8_fire(t8_dir, n_chunks, snapshot={})
            if stats is None:
                return False
            unmapped = stats.get("unmapped_total", 0) if stats else 0
            granted = stats.get("granted_total", 0) if stats else 0
            committed = unmapped > 0 and granted > 0
            logger.info(
                "T6 admission-time fire: dir=%s n=%d committed=%s "
                "unmapped=%d granted=%d",
                direction, n_chunks, committed, unmapped, granted,
            )
            return committed
        except Exception as e:
            logger.warning(
                "try_admission_time_fire failed: %r", e, exc_info=True
            )
            return False
        finally:
            self._emergency_fire_in_progress = False

    def _maybe_xpool_planner(self, snapshot: dict) -> None:
        """Planner-driven cross-pool transfers (paper §sec:design-l2).

        Attaches the cross-pool actuator on first call, then a
        CrossPoolPlanner reads pressure signals from the snapshot and
        decides direction. Per-pool actuators always propagate capacity
        changes back to the allocators so transfers are safe under live
        traffic.
        """
        self._ensure_xpool_actuator()
        if self._xpool_actuator is None:
            return
        if self._xpool_planner is None:
            from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
            self._xpool_planner = CrossPoolPlanner()

        # Pressure-signal extraction. Snapshot's `token_usage` and
        # `mamba_usage` are sampled instantaneously; under low-frequency
        # ticking they often miss in-flight bursts. Read directly from
        # the pools/allocators to get the LIVE state at tick time, then
        # peak-track across consecutive ticks (peaks decay exponentially)
        # so the planner sees the recent maximum, not the just-now zero.
        alloc = self.scheduler.token_to_kv_pool_allocator
        kv_pool = alloc.get_kvcache()
        mamba_pool = getattr(kv_pool, "mamba_pool", None)

        usage_kv_inst = 0.0
        usage_mamba_inst = 0.0
        usage_mamba_active_inst = 0.0
        live = getattr(alloc, "live_size", alloc.size)
        avail = alloc.available_size()
        if live > 0:
            usage_kv_inst = max(0.0, min(1.0, (live - avail) / live))
        if mamba_pool is not None:
            ms_live = getattr(mamba_pool, "live_size", mamba_pool.size)
            ms_avail = mamba_pool.available_size()
            if ms_live > 0:
                usage_mamba_inst = max(0.0, min(1.0, (ms_live - ms_avail) / ms_live))
            # Active-slot count: 1 per running req via req.mamba_pool_idx.
            # The MambaPool's available_size() doesn't distinguish active
            # slots from snapshot slots (radix-tree-cached states); both
            # consume `free_slots`. For the saturation guard we want the
            # admission-gating signal (active slots), not the cache-fill
            # signal (snapshots), because shrinking the cache-snapshot
            # side is cheap (eats c_m × P_loss in NB) while shrinking
            # active slots forces queueing breakdown.
            try:
                active_count = 0
                for r in (getattr(self.scheduler.running_batch, "reqs", None) or []):
                    mi = getattr(r, "mamba_pool_idx", None)
                    if mi is None:
                        continue
                    active_count += 1
                if ms_live > 0:
                    usage_mamba_active_inst = max(
                        0.0, min(1.0, active_count / float(ms_live))
                    )
            except Exception:
                # Conservative fallback: assume same as total usage so the
                # guard's behavior is never worse than the pre-split path.
                usage_mamba_active_inst = usage_mamba_inst

        # Exponential decay for peak tracking. Per-tick decay; with
        # 0.5s tick and decay=0.6, half-life ~1 tick (0.5s) — fast enough
        # that an idle phase clears stale peaks but slow enough that
        # short bursts are still visible.
        if not hasattr(self, "_xpool_peak_kv"):
            self._xpool_peak_kv = 0.0
            self._xpool_peak_mamba = 0.0
        decay = float(os.environ.get("SGLANG_XPOOL_PEAK_DECAY", "0.6"))
        self._xpool_peak_kv = max(usage_kv_inst, self._xpool_peak_kv * decay)
        self._xpool_peak_mamba = max(usage_mamba_inst, self._xpool_peak_mamba * decay)
        usage_kv = self._xpool_peak_kv
        usage_mamba = self._xpool_peak_mamba
        snapshot["xpool_plan_usage_kv_inst"] = usage_kv_inst
        snapshot["xpool_plan_usage_mamba_inst"] = usage_mamba_inst
        # Mamba active-slot usage: admission-gate signal, distinct from
        # total mamba pool fill (which includes radix-tree snapshots).
        # Used by NB direction-aware gate's saturation guard so that
        # m2k fires aren't blocked by a snapshot-saturated mamba pool
        # whose active-slot count is well below high-water (snapshots
        # can be evicted cheaply by the radix tree's own LRU).
        snapshot["xpool_plan_usage_mamba_active_inst"] = usage_mamba_active_inst
        snapshot["usage_mamba_active"] = usage_mamba_active_inst

        # S_edge phase-transition signal (paper §sec:design-l2-actuator).
        # Maintain low-pass EMA of the two pools' usage and compute the
        # absolute change since last tick. When |Δu| > θ_edge on either
        # pool, mark this tick as edge-active so the adapter contributes
        # its bounded one-tick edge_us benefit. The signal is bounded
        # to one tick by construction (θ-trigger only fires when the
        # gradient crosses, not while it stays elevated).
        ema_alpha = float(os.environ.get("SGLANG_XPOOL_EMA_ALPHA", "0.4"))
        theta_edge = float(os.environ.get("SGLANG_XPOOL_THETA_EDGE", "0.10"))
        if not hasattr(self, "_xpool_ema_kv"):
            self._xpool_ema_kv = usage_kv_inst
            self._xpool_ema_mamba = usage_mamba_inst
            self._xpool_prev_ema_kv = usage_kv_inst
            self._xpool_prev_ema_mamba = usage_mamba_inst
        prev_ema_kv = self._xpool_ema_kv
        prev_ema_mamba = self._xpool_ema_mamba
        self._xpool_ema_kv = (
            ema_alpha * usage_kv_inst + (1.0 - ema_alpha) * prev_ema_kv
        )
        self._xpool_ema_mamba = (
            ema_alpha * usage_mamba_inst + (1.0 - ema_alpha) * prev_ema_mamba
        )
        du_kv = self._xpool_ema_kv - prev_ema_kv
        du_mamba = self._xpool_ema_mamba - prev_ema_mamba
        edge_active = (abs(du_kv) > theta_edge) or (abs(du_mamba) > theta_edge)
        snapshot["xpool_ema_kv"] = self._xpool_ema_kv
        snapshot["xpool_ema_mamba"] = self._xpool_ema_mamba
        snapshot["xpool_du_kv"] = du_kv
        snapshot["xpool_du_mamba"] = du_mamba
        snapshot["xpool_edge_active"] = edge_active

        # Engine-agnostic gate: snapshot is passed through to the
        # planner, which delegates pressure-signal extraction to its
        # `EnginePressureAdapter` (default `SGLangPressureAdapter`).
        # Adapter reads num_evicted_tokens_recent / num_retracted_reqs /
        # num_paused_reqs / num_queue_reqs from the snapshot and returns
        # PressureSignals; planner sums and compares to chunk_cost_us.
        def _scalar_or_total(v) -> int:
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return int(t)
            return int(v) if isinstance(v, (int, float)) else 0

        qdepth = _scalar_or_total(snapshot.get("num_queue_reqs", 0))

        decision = self._xpool_planner.decide(
            usage_kv, usage_mamba,
            queue_depth=qdepth,
            snapshot=snapshot,
            edge_active=edge_active,
        )
        snapshot["xpool_plan_direction"] = decision.direction or "none"
        snapshot["xpool_plan_reason"] = decision.reason
        snapshot["xpool_plan_usage_kv"] = decision.usage_kv
        snapshot["xpool_plan_usage_mamba"] = decision.usage_mamba
        snapshot["xpool_plan_queue_depth"] = decision.queue_depth

        if decision.direction is None:
            return

        # Paper §sec:design-l2 drain protocol: BEFORE the actuator's
        # `_drain_complete` check, force-flush tree-cache evictable
        # entries. SGLang's radix prefix cache holds slot ids on behalf
        # of completed-but-cached prefix paths; from the allocator's
        # perspective those slot ids are "in_use" (not in free/capped/
        # release/free_group), but they're not held by any live request
        # — they're held by the tree. If we don't flush them, an unmap
        # of chunks above new_cap will kill slots the tree still
        # references; a subsequent cache hit dereferences unmapped
        # memory → cudaErrorIllegalAddress.
        #
        # `evict_from_tree_cache(tree, BIG)` walks the radix tree and
        # frees every refcount-zero entry's KV via `allocator.free()`,
        # which routes through the cap-aware path → ids above new_cap
        # land in `_capped_pages`. After this, the only "in_use_above"
        # pages are slot ids genuinely held by active in-flight
        # requests; drain check correctly waits for those.
        # Pre-fire: flush tree_cache entries on the SHRINKING pool's
        # side. Without this, drain inspector sees in-flight reqs as
        # the only "above-cap" holders, but tree_cache snapshot nodes
        # quietly hold pool slots above new_cap and the actuator's
        # cuMemUnmap kills bytes the cache still references on the next
        # cache-hit. Symmetric handling for both directions: kv_to_mamba
        # evicts the FULL (paged-attention KV) side; mamba_to_kv evicts
        # the MAMBA (recurrent state snapshot) side.
        if decision.direction == "kv_to_mamba":
            try:
                # MambaRadixCache (the hybrid SGLang prefix cache) splits
                # `evictable_size` into separate full-pool and mamba-pool
                # accessors and raises NotImplementedError on the unified
                # one. Use the full-pool size (which is what KV-side
                # eviction acts on) and call evict_full directly.
                tc = self._tree_cache
                full_evictable = (
                    tc.full_evictable_size()
                    if hasattr(tc, "full_evictable_size")
                    else (tc.evictable_size() if hasattr(tc, "evictable_size") else 0)
                )
                if full_evictable > 0:
                    if hasattr(tc, "evict_full"):
                        tc.evict_full(full_evictable)
                    else:
                        from sglang.srt.mem_cache.common import (
                            evict_from_tree_cache,
                        )
                        evict_from_tree_cache(tc, full_evictable)
                    snapshot["xpool_pre_fire_evicted"] = full_evictable
            except Exception as e:  # noqa
                import traceback
                logger.warning(
                    "Pre-fire tree_cache flush failed: %r\n%s",
                    e, traceback.format_exc(),
                )
                snapshot["xpool_pre_fire_evicted_error"] = repr(e)
        elif decision.direction == "mamba_to_kv":
            try:
                tc = self._tree_cache
                mamba_evictable = (
                    tc.mamba_evictable_size()
                    if hasattr(tc, "mamba_evictable_size")
                    else 0
                )
                # Mamba-side pre-fire flush. Goal: clear refcount-0
                # snapshot nodes whose slot id falls in the over-cap
                # range so the actuator's drain inspector can commit
                # the cuMemUnmap. Two regressions to avoid:
                #   (1) Greedy generic eviction (mamba_evictable_size
                #       across the whole pool) wipes the multi-turn
                #       prefix cache and regresses steady-state TTFT
                #       (v5 measurement: -16% TPS).
                #   (2) Capped generic eviction (small N, LRU/HPB
                #       order) preserves cache but only stochastically
                #       hits the over-cap range; drain rarely passes
                #       (v6 measurement: 1 fire_w_mv per cell, neutral).
                # Right answer: targeted eviction — evict only nodes
                # whose mamba slot id exceeds the new cap. Falls back
                # to capped generic eviction on engines/tree-caches
                # that don't expose the targeted API.
                if mamba_evictable > 0:
                    # Compute new mamba cap (in slots) from current
                    # actuator state; fire will shrink to this.
                    new_cap_slots = self._estimate_post_fire_mamba_cap()
                    flush_cap = int(os.environ.get(
                        "SGLANG_XPOOL_MAMBA_FLUSH_CAP", "256"
                    ))
                    if hasattr(tc, "evict_mamba_above_slot"):
                        evicted = tc.evict_mamba_above_slot(
                            new_cap_slots, max_to_evict=flush_cap,
                        )
                        snapshot["xpool_pre_fire_evicted_mamba"] = evicted
                        snapshot["xpool_pre_fire_mamba_evictable_total"] = mamba_evictable
                        snapshot["xpool_pre_fire_mamba_threshold"] = new_cap_slots
                    elif hasattr(tc, "evict_mamba"):
                        to_evict = min(mamba_evictable, flush_cap)
                        tc.evict_mamba(to_evict)
                        snapshot["xpool_pre_fire_evicted_mamba"] = to_evict
                        snapshot["xpool_pre_fire_mamba_evictable_total"] = mamba_evictable
            except Exception as e:  # noqa
                import traceback
                logger.warning(
                    "Pre-fire mamba tree_cache flush failed: %r\n%s",
                    e, traceback.format_exc(),
                )
                snapshot["xpool_pre_fire_evicted_error"] = repr(e)

        unit = self._xpool_planner.config.dst_chunks_per_action
        # Paper §sec:design-l2: planner-driven fires use the direct single-
        # direction `*_chunks(unit)` path (grow dst by `unit` chunks per
        # dst-subpool). The balanced wrapper exists for round-trip
        # oscillator demos but isn't appropriate for planner-driven
        # firing — its lcm-balanced multiplier can demand more chunks
        # than available, causing src to shrink past static_min and bail.
        # T8: every fire goes through the plan-based execute(plan) path.
        # _maybe_t8_fire returns a stats dict (with skipped="..." when
        # the planner refuses), never None — caller has nothing to
        # fall back to since the legacy heuristic path is gone.
        stats = self._maybe_t8_fire(decision.direction, unit, snapshot)
        if stats is None:
            # T8 wiring failed (no scheduler / no actuator). Skip this
            # tick rather than crashing.
            snapshot["xpool_skipped"] = "t8_wiring_unavailable"
            return
        snapshot["xpool_plan_executed"] = True
        snapshot["xpool_direction"] = stats["direction"]
        snapshot["xpool_unmapped_total"] = stats["unmapped_total"]
        snapshot["xpool_granted_total"] = stats["granted_total"]
        snapshot["xpool_kv_capacity_tokens"] = stats["kv_capacity_tokens"]
        snapshot["xpool_mamba_capacity_tokens"] = stats["mamba_capacity_tokens"]
        snapshot["xpool_free_handles"] = stats["free_after_grow"]
        # Stage 1 calibration: real cuMemUnmap/cuMemMap wall time
        if "shrink_us" in stats:
            snapshot["xpool_shrink_us"] = stats["shrink_us"]
            snapshot["xpool_grow_us"] = stats["grow_us"]
            snapshot["xpool_fire_total_us"] = stats["fire_total_us"]
            # Runtime self-calibration (paper §sec:design-l2-firegate):
            # feed the observed wall-time into the process-wide EWMA so
            # the next gate evaluation sees the live-traffic actuator
            # cost instead of the conservative cold-start initial.
            n_chunks = max(1, int(stats.get("granted_total", 0)) or
                              int(stats.get("unmapped_total", 0)) or
                              int(unit))
            try:
                from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost
                get_runtime_actuator_cost().update(
                    total_us=float(stats["fire_total_us"]),
                    n_chunks=n_chunks,
                )
            except Exception:
                pass
        if "skipped" in stats:
            snapshot["xpool_skipped"] = stats["skipped"]

    # ---- Internal ----

    def _snapshot(self, now: float) -> dict:
        """Capture all signals the cross-pool planner / pressure adapter consume."""
        sched = self.scheduler
        stats = sched.stats

        snap: dict[str, Any] = {
            "ts": round(now, 3),
            "tick": self._tick_count,
            "max_total_num_tokens": stats.max_total_num_tokens,
            "kv_used_tokens": stats.kv_used_tokens,
            "kv_evictable_tokens": stats.kv_evictable_tokens,
            "kv_available_tokens": stats.kv_available_tokens,
            "token_usage": stats.token_usage,
            "full_token_usage": stats.full_token_usage,
            "swa_token_usage": stats.swa_token_usage,
            "mamba_usage": stats.mamba_usage,
            "cache_hit_rate": stats.cache_hit_rate,
            "num_running_reqs": stats.num_running_reqs,
            "num_queue_reqs": stats.num_queue_reqs,
            "num_paused_reqs": stats.num_paused_reqs,
            "num_retracted_reqs": stats.num_retracted_reqs,
            "gen_throughput": stats.gen_throughput,
            "unified_radix": (
                self._tree_cache.__class__.__name__ == "UnifiedRadixCache"
            ),
        }

        # Paper §sec:design-l2 SGLang adapter: tree-cache eviction is the
        # primary admission-pressure relief mechanism. The cumulative
        # counter is maintained by `check_decode_mem` in
        # schedule_batch.py; we emit the per-tick delta as
        # `num_evicted_tokens_recent` for the SGLangPressureAdapter to
        # convert into benefit-microseconds via prefill_save_us_per_token.
        cum_evict = self._tree_cache._admission_cumulative_evicted_tokens
        last = self._last_evicted_cumulative
        recent = max(0, cum_evict - last)
        self._last_evicted_cumulative = cum_evict
        snap["num_evicted_tokens_recent"] = recent
        snap["num_evicted_tokens_cumulative"] = cum_evict

        # Pool-occupancy metrics (paper §motivation, Figure
        # bubble_two_workloads): (pool.size - pool.available_size()) /
        # pool.size = used / total, INCLUDING radix-tree-cached prefix/
        # snapshots — unlike the scheduler's `full_token_usage` /
        # `mamba_usage` which subtract evictable size for the
        # admission-pressure framing. Occupancy is the right "real bubble"
        # measure for the motivation figure. Computed unconditionally —
        # does NOT depend on whether the cross-pool actuator is initialized.
        alloc = self.scheduler.token_to_kv_pool_allocator
        kv_total = alloc.size
        kv_avail = alloc.available_size()
        if kv_total > 0:
            snap["pool_occupancy_kv"] = max(
                0.0, min(1.0, (kv_total - kv_avail) / kv_total)
            )
        kv_pool = alloc.get_kvcache()
        mamba_pool = getattr(kv_pool, "mamba_pool", None)
        if mamba_pool is not None:
            m_total = mamba_pool.size
            m_avail = mamba_pool.available_size()
            if m_total > 0:
                snap["pool_occupancy_mamba"] = max(
                    0.0, min(1.0, (m_total - m_avail) / m_total)
                )

        # Paper §sec:design-formalism-offline: c_i(L) is evaluated at the
        # EWMA mean recovery length \\bar L_i per pool i ∈ {KV, rec}.
        # Each EWMA is fed by the actual L of pool-specific events:
        #   - kv:  prefix-tree leaf evict (re-prefill length)
        #   - rec: prefix-tree mamba snapshot evict (chunked-scan distance)
        # The retract EWMA is a separate retract-pressure signal (req kicked
        # out under KV pressure) for the SGLang adapter — NOT \\bar L_rec.
        # Cold-start fallback: SGLANG_XPOOL_DEFAULT_L so the gate has a
        # usable c_i evaluation point before the first event lands.
        tc = self._tree_cache
        default_L = float(os.environ.get("SGLANG_XPOOL_DEFAULT_L", "4096"))
        kv_L = tc._slow_recovery_len_kv_ewma if tc._slow_recovery_len_kv_ewma > 0 else default_L
        rec_L = tc._slow_recovery_len_rec_ewma if tc._slow_recovery_len_rec_ewma > 0 else kv_L
        retract_L = (
            tc._slow_recovery_len_retract_ewma if tc._slow_recovery_len_retract_ewma > 0 else kv_L
        )
        snap["slow_recovery_len_kv"] = kv_L
        snap["slow_recovery_len_rec"] = rec_L
        snap["slow_recovery_len_retract"] = retract_L

        # \hat v_{prefix}(m) reporter (paper §sec:design-l1, Eq:vprefix-est).
        # Diagnostic only — emitted to jsonl, not consumed by the gate.
        # Available on MambaRadixCache (and Hi*); other tree caches don't
        # expose this signal yet.
        if hasattr(self._tree_cache, "estimate_v_prefix_marginal"):
            try:
                snap["v_prefix_marginal"] = self._tree_cache.estimate_v_prefix_marginal()
            except Exception as e:  # never break the snapshot path
                snap["v_prefix_marginal_error"] = str(e)

        return snap

    def close(self) -> None:
        if self.log_enabled:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self.log_enabled = False
            self._log_fp = None
