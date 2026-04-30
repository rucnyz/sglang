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
        if not hasattr(pool, "_kv_arena"):
            return
        from sglang.srt.arena.kv_actuator import KVArenaActuator
        self._arena_actuator = KVArenaActuator(pool, alloc)
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

        return snap

    def close(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
