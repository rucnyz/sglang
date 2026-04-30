"""BudgetAgent — in-process pool-pressure observer (Phase 2a, read-only).

The scheduler instantiates one of these and calls `tick()` on every event-loop
iteration. The agent rate-limits internally: it only does real work every
`tick_interval_s` seconds.

Phase 2a (this file): SNAPSHOT ONLY. We log per-pool state to a JSONL and verify
that the scheduler's hot path is not affected. No actuation.

Phase 2b/2c will add `LagrangePolicy.compute_evict_targets()` + actuators on top
of this scaffolding.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    handle to the scheduler so it can later actuate via `tree_cache.evict(...)`,
    `lora_manager.set_capacity(...)`, etc.

    For now (Phase 2a) this is read-only — it logs a JSONL line per
    `tick_interval_s` and never mutates state.
    """

    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        # Cached references; refreshed lazily because some are created late.
        self._tree_cache = None
        self._lora_manager = None

        self.enabled = _env_flag("SGLANG_BUDGETER", False)
        self.actuate_enabled = _env_flag("SGLANG_BUDGETER_ACTUATE", False)
        self.tick_interval_s = _env_float("SGLANG_BUDGETER_TICK_S", 1.0)
        self.log_path = _env_str(
            "SGLANG_BUDGETER_LOG", "/tmp/sglang_budgeter.jsonl"
        )
        self._last_tick = 0.0
        self._tick_count = 0
        self._log_fp = None

        # Phase 2b policy + actuation state. Lazily initialized so the import
        # cost is paid only once and only when the budgeter is enabled.
        self._policy = None
        self._evict_params_cls = None
        if self.enabled and self.actuate_enabled:
            from sglang.srt.budgeter.policy import PressurePolicy
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            self._policy = PressurePolicy()
            self._evict_params_cls = EvictParams

        # Phase 2e.4.d arena actuator. Lazy-built on first tick once
        # the scheduler has populated `token_to_kv_pool` and
        # `token_to_kv_pool_allocator`. SGLANG_BUDGETER_ARENA=1 turns
        # this on; SGLANG_BUDGETER_ARENA_DEMO=1 oscillates capacity for
        # a smoke demo.
        self.arena_enabled = _env_flag("SGLANG_BUDGETER_ARENA", False)
        self.arena_demo = _env_flag("SGLANG_BUDGETER_ARENA_DEMO", False)
        self._arena_actuator = None
        self._arena_phase = 0  # for the oscillator

        # Phase 2e.5.6.2 cross-pool transfer demo. Requires
        # SGLANG_ARENA_SHARED=1 at engine boot. Each budgeter tick alternates
        # one balanced kv→mamba and one balanced mamba→kv transfer; the
        # planner/policy logic for what to actually transfer is the next
        # milestone (real pressure signals via LagrangePlanner).
        self.xpool_demo = _env_flag("SGLANG_BUDGETER_XPOOL_DEMO", False)
        self._xpool_actuator = None
        self._xpool_phase = 0
        # Balanced-unit multiplier. Default 1 → smallest leftover-free
        # round-trip the actuator supports (uses lcm of sub-pool counts).
        self._xpool_unit = int(os.environ.get("SGLANG_BUDGETER_XPOOL_UNIT", "1"))
        # Phase 2e.5.6.3: when SGLANG_BUDGETER_XPOOL_COORDINATED=1, the
        # cross-pool actuator is constructed with per-pool actuators
        # (KVArenaActuator + MambaArenaActuator) so capacity changes
        # propagate to allocators. Otherwise (legacy / 2e.5.6.2 path),
        # only raw chunk movement happens, which is only safe in idle
        # windows.
        self.xpool_coordinated = _env_flag("SGLANG_BUDGETER_XPOOL_COORDINATED", False)

        # Phase 2e.5.6.3.c: planner-driven cross-pool transfers based on
        # real per-pool pressure signals (replaces the oscillator).
        # Requires xpool actuator + per-pool actuators to be useful.
        self.xpool_planner_enabled = _env_flag("SGLANG_BUDGETER_XPOOL_PLANNER", False)
        self._xpool_planner = None  # lazy-built on first tick

        if self.enabled:
            try:
                self._log_fp = open(self.log_path, "a", buffering=1)
            except OSError as e:
                logger.warning("BudgetAgent: failed to open %s: %s", self.log_path, e)
                self._log_fp = None
            logger.info(
                "BudgetAgent enabled (actuate=%s, tick=%.1fs, log=%s, pid=%d)",
                self.actuate_enabled, self.tick_interval_s, self.log_path, os.getpid(),
            )

    # ---- Public API used from scheduler.event_loop_* ----

    def tick(self) -> None:
        """Called every scheduler iteration. Internally rate-limited."""
        if not self.enabled:
            return
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
        # Phase 2b: actuate via UnifiedRadixCache.evict.
        if self.actuate_enabled and self._policy is not None:
            try:
                self._maybe_evict(snapshot)
            except Exception as e:
                logger.warning("BudgetAgent actuation failed: %s", e, exc_info=True)

        # Phase 2e.4.d: arena actuator (cross-pool VMM-aware resize).
        if self.arena_enabled:
            try:
                self._maybe_arena_actuate(snapshot)
            except Exception as e:
                logger.warning("BudgetAgent arena actuation failed: %s", e, exc_info=True)

        # Phase 2e.5.6.2: cross-pool KV ↔ mamba transfer demo.
        if self.xpool_demo:
            try:
                self._maybe_xpool_actuate(snapshot)
            except Exception as e:
                logger.warning("BudgetAgent xpool actuation failed: %s", e, exc_info=True)

        # Phase 2e.5.6.3.c: planner-driven cross-pool transfers.
        if self.xpool_planner_enabled:
            try:
                self._maybe_xpool_planner(snapshot)
            except Exception as e:
                logger.warning("BudgetAgent xpool planner failed: %s", e, exc_info=True)

        if self._log_fp is not None:
            try:
                self._log_fp.write(json.dumps(snapshot, default=_json_default) + "\n")
            except Exception as e:
                logger.warning("BudgetAgent.tick write failed: %r", e)

    def _ensure_arena_actuator(self) -> None:
        if self._arena_actuator is not None:
            return
        sched = self.scheduler
        alloc = getattr(sched, "token_to_kv_pool_allocator", None)
        if alloc is None:
            return
        pool = alloc.get_kvcache() if hasattr(alloc, "get_kvcache") else None
        if pool is None:
            return
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

    def _maybe_arena_actuate(self, snapshot: dict) -> None:
        self._ensure_arena_actuator()
        if self._arena_actuator is None:
            return
        if self.arena_demo:
            # Oscillator: full -> half -> full -> half on each ARENA tick.
            self._arena_phase = (self._arena_phase + 1) % 2
            target = (
                self._arena_actuator.max_tokens
                if self._arena_phase == 0
                else self._arena_actuator.max_tokens // 2
            )
            actual = self._arena_actuator.set_capacity_tokens(target)
            snapshot["budgeter_arena_target"] = target
            snapshot["budgeter_arena_actual"] = actual
            snapshot["budgeter_arena_phase"] = self._arena_phase

    def _ensure_xpool_actuator(self) -> None:
        """Phase 2e.5.6.2: lazily attach the cross-pool transfer actuator.

        Walks the scheduler to find the hybrid pool (full_kv_pool +
        mamba_pool), then wraps both arenas in a CrossPoolTransferActuator
        that writes through the shared SharedHandlePool.
        """
        if self._xpool_actuator is not None:
            return
        sched = self.scheduler
        alloc = getattr(sched, "token_to_kv_pool_allocator", None)
        if alloc is None:
            return
        pool = alloc.get_kvcache() if hasattr(alloc, "get_kvcache") else None
        if pool is None:
            return
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
        kv_act = None
        mamba_act = None
        if self.xpool_coordinated:
            self._ensure_arena_actuator()
            kv_act = self._arena_actuator
            from sglang.srt.arena.mamba_actuator import MambaArenaActuator
            try:
                mamba_act = MambaArenaActuator(mamba_pool)
            except RuntimeError as e:
                logger.warning(
                    "MambaArenaActuator build failed (%s); falling back to "
                    "raw chunk-move path (idle-window-only).", e,
                )
                mamba_act = None
        self._xpool_actuator = CrossPoolTransferActuator(
            kv_arena=kv_arena,
            mamba_arena=mamba_arena,
            shared_pool=shared,
            kv_actuator=kv_act,
            mamba_actuator=mamba_act,
        )
        logger.info(
            "BudgetAgent xpool: actuator attached, oscillator unit=%d, "
            "coordinated=%s (kv_act=%s, mamba_act=%s)",
            self._xpool_unit, self.xpool_coordinated,
            kv_act is not None, mamba_act is not None,
        )

    def _maybe_xpool_actuate(self, snapshot: dict) -> None:
        """Phase 2e.5.6.2 demo. Safe-by-design: only transfers when there are
        zero running/queued requests (i.e., during warmup or quiescent
        windows). Live-serving cross-pool resize requires the scheduler to
        know about the shrunken capacity (via KVArenaActuator and a
        not-yet-implemented MambaArenaActuator); that's the next milestone.
        Until then we restrict to safe windows so the demo doesn't unmap
        slots the engine is using.
        """
        self._ensure_xpool_actuator()
        if self._xpool_actuator is None:
            return

        # Safety gate: only act when nothing is in flight. snapshot fields
        # may be QueueCount objects with a `.total` attribute; unwrap to int.
        def _to_int(v):
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return int(t)
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0
        n_running = _to_int(snapshot.get("num_running_reqs", 0))
        n_queued = _to_int(snapshot.get("num_queue_reqs", 0))
        if n_running > 0 or n_queued > 0:
            snapshot["xpool_skipped"] = "engine_busy"
            snapshot["xpool_state"] = self._xpool_actuator.state()
            return

        # Oscillator: balanced kv→mamba, then balanced mamba→kv, repeat.
        # Balanced wrappers use lcm-aware sub-pool unit sizes so a
        # round-trip leaves both pools and the shared free list at their
        # starting state — no drift, no leftover handles accumulating.
        self._xpool_phase = (self._xpool_phase + 1) % 2
        if self._xpool_phase == 1:
            stats = self._xpool_actuator.balanced_kv_to_mamba(self._xpool_unit)
        else:
            stats = self._xpool_actuator.balanced_mamba_to_kv(self._xpool_unit)

        # Inline a few key fields into the snapshot for easy grep.
        snapshot["xpool_direction"] = stats["direction"]
        snapshot["xpool_unmapped_total"] = stats["unmapped_total"]
        snapshot["xpool_granted_total"] = stats["granted_total"]
        snapshot["xpool_kv_capacity_tokens"] = stats["kv_capacity_tokens"]
        snapshot["xpool_mamba_capacity_tokens"] = stats["mamba_capacity_tokens"]
        snapshot["xpool_free_handles"] = stats["free_after_grow"]

    def _maybe_xpool_planner(self, snapshot: dict) -> None:
        """Phase 2e.5.6.3.c: planner-driven cross-pool transfers.

        Reuses `_ensure_xpool_actuator` to attach the cross-pool actuator,
        then a CrossPoolPlanner reads pressure signals from the snapshot
        and decides direction. Compared to the oscillator
        (SGLANG_BUDGETER_XPOOL_DEMO=1), transfers are now sparse — only
        when a pool is actually stressed.

        Safety: same gate as the oscillator path — skip when engine is
        busy. Capacity-coordinated path (the actuator updates per-pool
        allocators) is implied by SGLANG_BUDGETER_XPOOL_COORDINATED=1.
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
        sched = self.scheduler
        alloc = getattr(sched, "token_to_kv_pool_allocator", None)
        kv_pool = alloc.get_kvcache() if alloc and hasattr(alloc, "get_kvcache") else None
        mamba_pool = None
        if kv_pool is not None:
            mamba_pool = getattr(kv_pool, "mamba_pool", None)

        usage_kv_inst = 0.0
        usage_mamba_inst = 0.0
        if alloc is not None:
            live = getattr(alloc, "live_size", alloc.size)
            avail = alloc.available_size()
            if live > 0:
                usage_kv_inst = max(0.0, min(1.0, (live - avail) / live))
        if mamba_pool is not None:
            ms_live = getattr(mamba_pool, "live_size", mamba_pool.size)
            ms_avail = mamba_pool.available_size()
            if ms_live > 0:
                usage_mamba_inst = max(0.0, min(1.0, (ms_live - ms_avail) / ms_live))

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

        # Setting 4 follow-up: pass queue_depth so the planner has an
        # admission-pressure signal at saturation (V≈usage proxy alone
        # is saturation-blind).
        _q = snapshot.get("num_queue_reqs", 0)
        _qt = getattr(_q, "total", None)
        qdepth = int(_qt) if isinstance(_qt, (int, float)) else (
            int(_q) if isinstance(_q, (int, float)) else 0
        )

        decision = self._xpool_planner.decide(usage_kv, usage_mamba, queue_depth=qdepth)
        snapshot["xpool_plan_direction"] = decision.direction or "none"
        snapshot["xpool_plan_reason"] = decision.reason
        snapshot["xpool_plan_usage_kv"] = decision.usage_kv
        snapshot["xpool_plan_usage_mamba"] = decision.usage_mamba
        snapshot["xpool_plan_queue_depth"] = decision.queue_depth

        if decision.direction is None:
            return

        # Engine-busy safety gate (same as oscillator path).
        def _to_int(v):
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return int(t)
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0
        n_running = _to_int(snapshot.get("num_running_reqs", 0))
        n_queued = _to_int(snapshot.get("num_queue_reqs", 0))
        if n_running > 0 or n_queued > 0:
            snapshot["xpool_plan_skipped"] = "engine_busy"
            return

        unit = self._xpool_planner.config.dst_chunks_per_action
        if decision.direction == "mamba_to_kv":
            stats = self._xpool_actuator.balanced_mamba_to_kv(unit)
        else:  # kv_to_mamba
            stats = self._xpool_actuator.balanced_kv_to_mamba(unit)
        snapshot["xpool_plan_executed"] = True
        snapshot["xpool_direction"] = stats["direction"]
        snapshot["xpool_unmapped_total"] = stats["unmapped_total"]
        snapshot["xpool_granted_total"] = stats["granted_total"]
        snapshot["xpool_kv_capacity_tokens"] = stats["kv_capacity_tokens"]
        snapshot["xpool_mamba_capacity_tokens"] = stats["mamba_capacity_tokens"]
        snapshot["xpool_free_handles"] = stats["free_after_grow"]

    def _maybe_evict(self, snapshot: dict) -> None:
        """Phase 2b actuation: ask the policy what to evict; call tree_cache.evict."""
        if self._tree_cache is None:
            return
        target = self._policy.decide(snapshot, self._tick_count)
        # Annotate the snapshot with the decision so the JSONL records what we did.
        snapshot["budgeter_decision"] = target.reason
        snapshot["budgeter_evict_kv"] = target.num_tokens
        snapshot["budgeter_evict_swa"] = target.swa_num_tokens
        snapshot["budgeter_evict_mamba"] = target.mamba_num
        if target.is_noop():
            return
        try:
            params = self._evict_params_cls(
                num_tokens=target.num_tokens,
                swa_num_tokens=target.swa_num_tokens,
                mamba_num=target.mamba_num,
            )
            result = self._tree_cache.evict(params)
            # Prefer the actual evicted counts back from the cache, when it returns them.
            actually_kv = getattr(result, "num_tokens_evicted", target.num_tokens)
            actually_mamba = getattr(result, "mamba_num_evicted", target.mamba_num)
            snapshot["budgeter_actually_evicted_kv"] = actually_kv
            snapshot["budgeter_actually_evicted_mamba"] = actually_mamba
            logger.debug(
                "Budgeter evicted kv=%s mamba=%s reason=%s",
                actually_kv, actually_mamba, target.reason,
            )
        except Exception as e:
            logger.warning("Budgeter evict() raised: %r", e, exc_info=True)
        # Phase 2a: no actuation.

    # ---- Internal ----

    def _snapshot(self, now: float) -> dict:
        """Capture all signals Phase 2's policy might consume."""
        sched = self.scheduler
        stats = getattr(sched, "stats", None)

        # Lazy: tree_cache and lora_manager get assigned partway through scheduler init
        if self._tree_cache is None:
            self._tree_cache = getattr(sched, "tree_cache", None)
        if self._lora_manager is None:
            tp_worker = getattr(sched, "tp_worker", None)
            if tp_worker is not None:
                model_runner = getattr(tp_worker, "_model_runner", None) or getattr(
                    tp_worker, "model_runner", None
                )
                if model_runner is not None:
                    self._lora_manager = getattr(model_runner, "lora_manager", None)

        snap: dict[str, Any] = {
            "ts": round(now, 3),
            "tick": self._tick_count,
        }

        if stats is not None:
            for k in (
                # KV
                "max_total_num_tokens",
                "kv_used_tokens",
                "kv_evictable_tokens",
                "kv_available_tokens",
                "token_usage",
                "full_token_usage",
                "swa_token_usage",
                # SSM / mamba
                "mamba_usage",
                # LoRA
                "lora_pool_slots_used",
                "lora_pool_slots_total",
                "lora_pool_utilization",
                # cache
                "cache_hit_rate",
                # queue / running
                "num_running_reqs",
                "num_queue_reqs",
                "num_paused_reqs",
                "num_retracted_reqs",
                # throughput
                "gen_throughput",
            ):
                if hasattr(stats, k):
                    snap[k] = getattr(stats, k)

        # Whether tree_cache supports unified eviction (will matter for 2b)
        snap["unified_radix"] = bool(
            self._tree_cache
            and self._tree_cache.__class__.__name__ == "UnifiedRadixCache"
        )
        snap["lora_present"] = self._lora_manager is not None

        # Phase 3.b (paper §4.2 Eq 4.4): V_prefix' marginal-value report.
        # Available on MambaRadixCache (and Hi*); other tree caches don't
        # expose this signal yet.
        if (
            self._tree_cache is not None
            and hasattr(self._tree_cache, "estimate_v_prefix_marginal")
        ):
            try:
                snap["v_prefix_marginal"] = self._tree_cache.estimate_v_prefix_marginal()
            except Exception as e:  # never break the snapshot path
                snap["v_prefix_marginal_error"] = str(e)

        return snap

    def close(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
