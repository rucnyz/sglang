"""BudgetAgent — in-process pool-pressure observer + cross-pool actuator.

The scheduler instantiates one of these and calls `tick()` on every event-loop
iteration. The agent rate-limits internally: it only does real work every
`tick_interval_s` seconds. Each tick:
  1. snapshots per-pool state into the JSONL log,
  2. asks the PaybackPlanner (Budgeter) if a cross-pool transfer is warranted,
  3. converts the decision to a FirePlan via XPoolFirePlanner,
  4. executes the plan via XPoolActuator (cuMemUnmap / cuMemMap).

Enabled with SGLANG_HIMA=1 (which auto-promotes SGLANG_ARENA_SHARED=1
in memory_pool.py module-load — both pools must be arena-backed for the
actuator to work). Disabled by default → tick() is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Stats fields the budgeter requires from `scheduler.stats`. Validated
# once at health-check time; subsequent snapshots access them directly.
_REQUIRED_STATS_FIELDS = (
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


def _mamba_drain_floor(live_size_slots, floor_slots, slots_per_page, requested_pages):
    """Cap an m2k drain (in PAGES) so post-drain capacity stays >= floor.

    A mamba→kv fire unmaps `pages × slots_per_page` mamba SLOTS, shrinking the
    pool's allocatable capacity (live_size). `floor_slots` is the minimum
    capacity mamba must retain = the live working set `(m_used − evictable) +
    headroom` (see `_mamba_working_set_floor_slots`), so the active SSM states
    and a later cache_unfinished_req fork never fail ("Can not alloc mamba
    cache").

    Cap the PAGE count so the post-drain capacity stays >= floor, converting
    the slot headroom to pages via `slots_per_page` (mamba arena
    tokens_per_chunk — NOT assumed to be 1):
        max_drain_slots = max(0, live_size_slots - floor_slots)
        max_drain_pages = max_drain_slots // slots_per_page
        return min(requested_pages, max_drain_pages)
    Returns 0 → the caller must NOT fire m2k (floor reached, or unknown
    granularity → fail closed). Floors on live_size (the quantity the
    cross-fire actually shrinks; size is constant). design.md §"Allocator
    floor: working set only".
    """
    if int(slots_per_page) <= 0:
        return 0  # unknown slot granularity → fail closed (refuse m2k)
    max_drain_slots = max(0, int(live_size_slots) - int(floor_slots))
    return min(int(requested_pages), max_drain_slots // int(slots_per_page))


class BudgetAgent:
    """Per-tick pool-pressure observer + cross-pool actuator dispatcher.

    Cheap to construct; cheap to `tick()` (rate-limited internally). Holds a
    handle to the scheduler so the planner / actuator can read live pool
    state. Snapshot-only when SGLANG_HIMA is unset (default).
    """

    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        # Required scheduler deps; populated on first tick by _do_health_check.
        # If any required dep is missing the agent hard-disables itself.
        self._tree_cache = None
        self._health_checked = False

        self.enabled = True
        # SGLANG_HIMA_NO_BUDGETER=1 — the "w/o Budgeter" ablation cell:
        # suppress the tick-path PaybackPlanner (no background pool
        # resizing) while keeping everything the Admitter depends on —
        # the actuator chain build + Admitter wire-in, apply_pending_fires,
        # telemetry, and the fire worker. (Previously this env gated
        # BudgetAgent construction entirely in the scheduler, which left
        # admitter.actuator=None and silently degraded the Admitter to
        # observational mode, so the cell measured LPB-only.)
        self.planner_disabled = (
            os.environ.get("SGLANG_HIMA_NO_BUDGETER") == "1"
        )
        # Polling interval (seconds). This is a pure SAMPLING RATE, not a
        # behaviour knob: the planner prices signals as per-second rates and
        # gates the cooldown in wall-clock seconds, so decisions are
        # invariant to τ (see design.md §"Three separated concerns"). Default
        # 1 s gives good signal resolution within the amortization window at
        # negligible overhead.
        self.tick_interval_s = _env_float("SGLANG_HIMA_TICK_S", 1.0)
        self.log_path = _env_str(
            "SGLANG_HIMA_LOG", "/tmp/sglang_budgeter.jsonl"
        )
        self._last_tick = 0.0
        self._tick_count = 0
        self._last_evicted_cumulative = 0
        # Per-tick eviction-rate trackers for the grow-side signal:
        # last-seen cumulative cache evictions per pool.
        self._last_evicted_mamba_slots = 0
        self._last_evicted_kv_tokens = 0
        # Same, for the reuse-weighted LPB LOSS (us) — the accurate eviction-
        # cost signal the PaybackPlanner consumes (replaces the raw counts).
        self._last_evicted_kv_lpb_loss = 0.0
        self._last_evicted_mamba_lpb_loss = 0.0
        # JSONL snapshot logging. log_enabled flips True only after a
        # successful open(); read paths gate on the flag, never on the
        # raw file handle.
        self.log_enabled = False
        self._log_fp = None

        # PaybackPlanner (Budgeter) + XPoolFirePlanner + XPoolActuator. All
        # lazy-built on first tick where the agent observes pool state
        # (pool internals may not be fully wired at __init__).
        self._planner = None        # PaybackPlanner
        self._fire_planner = None   # XPoolFirePlanner (decision -> FirePlan)
        self._actuator = None       # XPoolActuator (FirePlan -> exec)
        self._kv_act = None         # KVArenaActuator (per-pool)
        self._mamba_act = None      # MambaArenaActuator (per-pool)
        self._owner_provider = None # SchedulerOwnerProvider
        self._stage0_handler = None # SchedulerStage0Handler (Drain/Migration)
        # Mamba slots per arena chunk (the k2m grow transfer unit). Set from
        # the arena when the actuator chain is built; 1 (atomic) until then.
        self._mamba_tokens_per_chunk = 1
        # KV tokens per arena chunk (the m2k/k2m transfer unit on the KV side).
        # Set from the KV arena at chain-attach; used by the KV working-set
        # floor that bounds a k2m KV drain. 1 (atomic) until then.
        self._kv_tokens_per_chunk = 1
        # On-demand grow headroom: an on-demand grow hook drains only the
        # DONOR's idle slack above (current working set + this many slots), so a
        # small burst between fires still fits without re-firing. Small because
        # the symmetric grow hook on each side is the real burst safety net.
        self._xfer_grow_headroom_slots = int(
            os.environ.get("SGLANG_XPOOL_GROW_HEADROOM_SLOTS", "32")
        )
        # KV working-set floor burst term: a k2m drain must always leave one
        # INDIVISIBLE prefill chunk of KV free+evictable, or alloc_token_slots
        # OOMs — a chunk cannot be back-filled on demand once the mamba donor
        # is saturated (the reactive _grow_kv_from_mamba hook returns False).
        # Unlike the mamba floor's small fork-headroom, the KV headroom IS the
        # chunk (chunked_prefill_size; max_prefill_tokens when chunking is off).
        _sa = getattr(self.scheduler, "server_args", None)
        _cps = getattr(_sa, "chunked_prefill_size", None)
        self._kv_prefill_chunk_floor = int(
            _cps if _cps and _cps > 0 else (getattr(_sa, "max_prefill_tokens", 0) or 8192)
        )
        # Wall-clock of the first planner tick — anchors the warmup
        # fire-suppression window (SGLANG_HIMA_FIRE_WARMUP_S).
        self._first_tick_monotonic: Optional[float] = None
        # Latched after the first time _ensure_actuator_chain fails so
        # the WARNING fires once, not on every tick.
        self._chain_unavailable_warned = False
        # Latched after the first time the c_m migrate probe fails.
        self._migrate_probe_warned = False
        # Boot-time c^xfer + κ_i probes run once after the actuator chain
        # builds, gated behind SGLANG_HIMA_BOOT_PROBE.
        self._boot_probe_done = False
        self._boot_probe_warned = False

        # Dynamic-admission-cap state. The Budgeter polls mamba_pool.size
        # each tick; on change, it resizes per-req arrays (ReqToTokenPool,
        # FutureMap, HybridReqToTokenPool) so admission can scale with the
        # actuator's pool growth. See dev/interlayer/dyn_admission_cap/.
        # `None` until first observation; then tracks last-seen mamba size.
        self._last_mamba_size: Optional[int] = None
        self._mamba_per_req_ratio: Optional[int] = None
        self._user_max_running: Optional[int] = None
        # fire. The planner's decision is direction-only; this scalar
        # sets the magnitude. Default 4 chunks ≈ 8 MiB per fire.
        # Mamba working-set floor burst-headroom. The floor is
        # `live_size >= (m_used − evictable) + this` = the live working set
        # (active SSM states + locked cache) plus a burst buffer. The buffer
        # absorbs a small admission burst's fresh active slots before the
        # `_mamba_active_grow_hook` (which recovers larger bursts from idle KV)
        # has to fire — so it is thrash avoidance, not the safety mechanism.
        # Default 32 slots; env-tunable.
        self._mamba_fork_headroom_slots = int(os.environ.get(
            "SGLANG_XPOOL_MAMBA_FLOOR_SLOTS", "32"))

        # Async fire worker — runs the expensive cuMemUnmap+cuMemMap+sync
        # phase off the scheduler thread so the scheduler thread only pays
        # the cap-barrier cost (~few hundred us) per fire. cap-barrier
        # alone is sufficient to make to-be-unmapped pages safe from
        # concurrent allocations, so the scheduler can resume admitting
        # requests while the worker does the multi-ms cuMem* work.
        #
        # Queue depth is small (default 4) because the planner's wall-clock
        # cooldown (cooldown_s, default 10 s) makes back-to-back fires
        # rare; if the queue does fill, _maybe_fire skips the new fire (logged).
        self._fire_queue_max = int(os.environ.get(
            "SGLANG_HIMA_FIRE_QUEUE_MAX", "4"))
        self._fire_async_enabled = _env_flag(
            "SGLANG_HIMA_FIRE_ASYNC", True
        )
        self._fire_queue: "Optional[queue.Queue]" = None
        self._fire_worker: "Optional[threading.Thread]" = None
        # True while the worker is inside execute_async (the physical
        # cuMemUnmap/Map + cap mutation). Single-writer (the worker thread),
        # read lock-free by has_inflight_fires() so the scheduler treats an
        # in-flight fire as non-idle and won't flush_cache mid-unmap.
        self._fire_executing = False

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
            from sglang.srt.budgeter import _set_budget_agent_singleton
            _set_budget_agent_singleton(self)

        self._assert_pools_arena_capable_or_die()

    def _assert_pools_arena_capable_or_die(self) -> None:
        """Boot-time loud-inert guard (dev/interlayer/5_mla_arena).

        Under SGLANG_HIMA=1 on a hybrid model, a KV or mamba pool that is
        not arena-backed makes the actuator chain PERMANENTLY unavailable:
        the server still boots, logs "HiMA enabled", runs LPB + telemetry,
        and silently never does any cross-pool work (audited failure mode
        on MLA-hybrid models before MLATokenToKVPool grew an arena branch).
        Fail at boot instead; SGLANG_HIMA_ALLOW_INERT=1 opts a deliberate
        partial-stack ablation back into the old behavior.
        """
        if os.environ.get("SGLANG_HIMA") != "1":
            return  # observation-mode agent on a stock server: no claim made
        req_pool = self.scheduler.req_to_token_pool
        mamba_pool = getattr(req_pool, "mamba_pool", None)
        if mamba_pool is None:
            # KV-only model: HiMA has no second pool; chain build will log
            # "no mamba_pool" per tick-path. Not an arena-capability bug.
            logger.warning(
                "SGLANG_HIMA=1 on a KV-only model (%s has no mamba_pool): "
                "no cross-pool control is possible.",
                type(req_pool).__name__,
            )
            return
        kv_pool = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
        inner_kv = getattr(kv_pool, "full_kv_pool", kv_pool)
        kv_arena = inner_kv._kv_arena
        mamba_arena = mamba_pool._mamba_temporal_arena
        if kv_arena is not None and mamba_arena is not None:
            return
        msg = (
            "SGLANG_HIMA=1 but the pools are not arena-backed "
            f"(kv pool {type(inner_kv).__name__}._kv_arena="
            f"{'ok' if kv_arena is not None else 'None'}, "
            f"mamba_pool._mamba_temporal_arena="
            f"{'ok' if mamba_arena is not None else 'None'}). "
            "The cross-pool actuator chain can NEVER build in this "
            "configuration — the run would silently degrade to "
            "LPB+telemetry only. Launch with SGLANG_ARENA_SHARED=1 pool "
            "construction (and for MLA models a chunk size that divides "
            "the per-token bytes, e.g. SGLANG_ARENA_CHUNK_BYTES=18874368 "
            "for Kimi-Linear), or set SGLANG_HIMA_ALLOW_INERT=1 to "
            "acknowledge an intentionally inert run."
        )
        if os.environ.get("SGLANG_HIMA_ALLOW_INERT") == "1":
            logger.error("BudgetAgent (ALLOW_INERT): %s", msg)
            return
        raise RuntimeError(f"BudgetAgent: {msg}")

    # ---- Public API used from scheduler.event_loop_* ----

    def _do_health_check(self) -> bool:
        """One-shot schema check on first tick. SGLang init order
        guarantees scheduler.stats / .tree_cache / .token_to_kv_pool_allocator
        are non-None by the time the event loop runs (otherwise scheduler
        init would have crashed). The real failure mode this guards is
        upstream renaming a field on `SchedulerStats`: we check every
        field the snapshot path reads and hard-disable on drift so the
        error surfaces as one log line instead of per-tick AttributeError.
        """
        stats = self.scheduler.metrics_reporter.stats
        missing = [f"metrics_reporter.stats.{f}" for f in _REQUIRED_STATS_FIELDS
                   if not hasattr(stats, f)]
        if missing:
            logger.error(
                "BudgetAgent health check failed — SchedulerStats schema drift: %s",
                ", ".join(missing),
            )
            return False
        tree_cache = self.scheduler.tree_cache
        # Pre-init counters on tree_cache. The eviction sites lazy-create
        # these on the first event; pre-creating here lets the snapshot
        # path read attributes directly.
        if not hasattr(tree_cache, "_admission_cumulative_evicted_tokens"):
            tree_cache._admission_cumulative_evicted_tokens = 0
        self._tree_cache = tree_cache
        # The m2k working-set floor reads tree_cache.mamba_evictable_size()
        # and mamba_pool.available_size / ._mamba_temporal_arena / .live_size
        # DIRECTLY each fire (no getattr fallback — fail loud). When a mamba
        # pool is present the agent CAN take that path, so require the API here
        # and hard-disable on absence with one log line — same contract as the
        # stats-schema check above, instead of a per-tick AttributeError that
        # tick() swallows.
        mamba_pool = getattr(
            self.scheduler.token_to_kv_pool_allocator.get_kvcache(),
            "mamba_pool", None,
        )
        mamba_allocator = getattr(
            self.scheduler.req_to_token_pool, "mamba_allocator", None
        )
        if mamba_pool is not None:
            missing_mamba = [
                name
                for name, present in (
                    ("tree_cache.mamba_evictable_size",
                     hasattr(tree_cache, "mamba_evictable_size")),
                    ("mamba_pool.available_size",
                     hasattr(mamba_pool, "available_size")),
                    ("mamba_pool._mamba_temporal_arena",
                     hasattr(mamba_pool, "_mamba_temporal_arena")),
                    ("mamba_allocator.live_size", hasattr(mamba_pool, "live_size")),
                )
                if not present
            ]
            if missing_mamba:
                logger.error(
                    "BudgetAgent health check failed — mamba cross-fire API "
                    "missing (the m2k working-set floor reads these): %s",
                    ", ".join(missing_mamba),
                )
                return False
        return True

    def tick(self) -> None:
        """Called every scheduler iteration. Internally rate-limited."""
        if not self.enabled:
            return
        if self._actuator is not None:
            self._actuator.apply_pending_fires()
        if not self._health_checked:
            if not self._do_health_check():
                self.enabled = False
                logger.error(
                    "BudgetAgent: hard-disabled (health check failed). "
                    "Fix the scheduler deps and restart."
                )
                return
            self._health_checked = True
        now = time.time()
        if now - self._last_tick < self.tick_interval_s:
            return
        # Wall seconds since the previous tick — the planner prices signals
        # as per-second rates (÷dt) and advances its cooldown clock by this.
        # On the first tick (`_last_tick == 0`) use the nominal interval.
        dt = (now - self._last_tick) if self._last_tick > 0.0 else self.tick_interval_s
        self._last_tick = now
        self._tick_count += 1
        try:
            snapshot = self._snapshot(now)
            snapshot["dt"] = dt
        except Exception as e:  # never break the scheduler hot path
            logger.warning("BudgetAgent.tick snapshot failed: %s", e, exc_info=True)
            return
        # Resize per-req arrays BEFORE deciding to fire — so the planner's
        # next decision (and the snapshot already-taken) sees the new cap.
        # Safe: this method polls mamba_pool.size and is idempotent; runs
        # on scheduler thread between batches.
        try:
            self._maybe_update_admission_cap()
        except Exception as e:
            logger.warning(
                "BudgetAgent admission-cap update failed: %s", e, exc_info=True
            )
        try:
            self._maybe_fire(snapshot)
        except Exception as e:
            logger.warning("BudgetAgent fire path failed: %s", e, exc_info=True)
        if self.log_enabled:
            try:
                self._log_fp.write(json.dumps(snapshot, default=_json_default) + "\n")
            except Exception as e:
                logger.warning("BudgetAgent.tick write failed: %r", e)

    # ---- Dynamic admission cap (couples mamba pool size with running cap) ----

    def _sync_admission_gate(self, new_cap: int) -> None:
        """Keep the live prefill-admission gate in lockstep with the cap.

        The scheduler admits a new prefill only while
        `running_bs < get_global_server_args().pp_max_micro_batch_size`
        (`Scheduler.get_num_allocatable_reqs`). That value is derived ONCE at
        boot (`max_running_requests // pp_size`) and, at tp1 / non-PP, is the
        actual concurrency ceiling. Bumping `max_running_requests` alone leaves
        the gate frozen, so a k2m grow would admit no extra requests (the
        concurrency win would be a no-op). Mirror the boot derivation so the
        gate tracks the cap both on grow and shrink.
        """
        from sglang.srt.server_args import get_global_server_args

        pp_size = max(1, int(self.scheduler.ps.pp_size))
        # v0.5.16: resolved server_args is read-only; override() is the
        # audited post-resolution mutation point (plain setattr raises).
        get_global_server_args().override(
            "hima-admission-cap",
            pp_max_micro_batch_size=max(new_cap // pp_size, 1),
        )

    def _maybe_update_admission_cap(self) -> None:
        """Resize per-req arrays so admission can follow actuator-driven
        mamba pool growth / shrinkage.

        Idempotent: compares current `mamba_pool.size` to the last
        observation; on change, calls `grow()` / `shrink()` on
        `ReqToTokenPool` and `FutureMap`. Runs on the scheduler thread
        between batches — safe with respect to attention backends'
        live tensor reads (per
        `dev/interlayer/dyn_admission_cap/audit_consumers.md`).
        """
        sched = self.scheduler
        kv_pool = sched.token_to_kv_pool_allocator.get_kvcache()
        # mamba_pool is the ONE genuinely-conditional attribute here:
        # HybridLinearKVPool has it; MHATokenToKVPool (KV-only models)
        # does not. Class difference, not API drift — explicit check.
        if not hasattr(kv_pool, "mamba_pool") or kv_pool.mamba_pool is None:
            return
        mamba_pool = kv_pool.mamba_pool
        mamba_allocator = self.scheduler.req_to_token_pool.mamba_allocator
        pool = sched.req_to_token_pool
        # Pool must be in dynamic-cap mode (constructed with max_size > size).
        # In back-compat mode _va_arena is explicitly None.
        if pool._va_arena is None:
            return

        # First tick: lazily snapshot init state.
        if self._last_mamba_size is None:
            # Read mamba's LIVE capacity, not its pre-allocated max:
            # `mamba_pool.size` is the pre-allocated upper bound, while
            # live_size is the current admission-relevant cap.
            self._last_mamba_size = int(mamba_allocator.live_size)
            # The user ceiling is --max-running-requests when set; when unset it
            # is None (the resolved boot cap lives on sched.max_running_requests,
            # not on server_args), so the growth ceiling is the pre-reserved
            # pool.max_size (the dynamic-cap headroom).
            user_cap = sched.server_args.max_running_requests
            self._user_max_running = (
                int(user_cap) if user_cap is not None else int(pool.max_size)
            )
            init_max_running = int(pool.size)
            self._boot_max_running = init_max_running
            self._mamba_per_req_ratio = max(
                1, int(self._last_mamba_size) // max(1, init_max_running)
            )
            logger.info(
                "[admission-cap] init: mamba_live=%d max_running=%d "
                "ratio=%d user_ceiling=%d pool_max=%d mamba_phys_max=%d",
                self._last_mamba_size, init_max_running,
                self._mamba_per_req_ratio, self._user_max_running,
                pool.max_size, int(mamba_pool.size),
            )
            return

        current_mamba_size = int(mamba_allocator.live_size)
        if current_mamba_size == self._last_mamba_size:
            return

        ratio = self._mamba_per_req_ratio
        ceiling = self._user_max_running
        # Compute new admission cap from CURRENT mamba size. Bounded by
        # both the user ceiling and the pool's pre-reserved max.
        # Floor: m2k fires shrink mamba but should not reduce max_running
        # below boot value UNLESS mamba is physically too small to support
        # it. Physical minimum = 2 slots/request (active + fork).
        # This prevents unnecessary max_running drops while protecting
        # against "Can not alloc mamba cache" crashes.
        boot_cap = self._boot_max_running
        physical_safe_cap = current_mamba_size // 2
        floor = min(boot_cap, physical_safe_cap)
        new_cap = min(ceiling, pool.max_size, current_mamba_size // ratio)
        new_cap = max(new_cap, floor)

        if new_cap > pool.size:
            old_size = pool.size
            pool.grow(new_cap)
            # FutureMap needs no grow: it is built from the ReqToTokenPool and
            # sizes its buffers off the full (max_size+1) req_to_token range,
            # so it already covers every req_pool_idx the grown cap hands out.
            sched.max_running_requests = new_cap
            self._sync_admission_gate(new_cap)
            logger.info(
                "[admission-cap] grew pool.size %d -> %d "
                "(mamba %d -> %d, ratio=%d)",
                old_size, new_cap, self._last_mamba_size,
                current_mamba_size, ratio,
            )
        elif new_cap < pool.size:
            try:
                old_size = pool.size
                pool.shrink(new_cap)
                # FutureMap doesn't shrink (no API + circular buffer
                # tail can stay unused harmlessly).
                sched.max_running_requests = new_cap
                self._sync_admission_gate(new_cap)
                logger.info(
                    "[admission-cap] shrunk pool.size %d -> %d "
                    "(mamba %d -> %d)",
                    old_size, new_cap, self._last_mamba_size,
                    current_mamba_size,
                )
            except RuntimeError as e:
                # Slot still held — defer shrink, retry next tick.
                logger.info(
                    "[admission-cap] shrink to %d blocked (slot held); "
                    "will retry next tick: %s", new_cap, e,
                )
                # Don't update _last_mamba_size, so next tick re-evaluates.
                return

        self._last_mamba_size = current_mamba_size

    # ---- PaybackPlanner + XPoolFirePlanner + XPoolActuator chain ----

    def _wire_admitter(self) -> None:
        """Plumb the actuator chain into the scheduler's Admitter — the
        moment lcm_pages first becomes knowable.

        The Admitter's `decide()` needs `lcm_pages` to price c_xfer at the
        rounded page count the actuator will actually fire; without this
        push, `self.lcm_pages` stays at the default 1 and under-prices
        cross-* by an LCM factor. Scheduler.init_running_status
        always sets `self.admitter` (to an instance or None) before the
        first `tick()`, so direct attribute access is the right shape.
        Idempotent: re-running after the first wire is a no-op.
        """
        adm = self.scheduler.admitter
        if adm is None or adm.actuator is not None:
            return
        adm.actuator = self._actuator
        adm.planner = self._fire_planner
        # Page-state ground truth so decide_for_req reads the SAME
        # fully-free / Migration-consolidatable page counts the planner
        # selects from (Admitter feasibility and planner page
        # selection agree by construction, no slot-vs-page mismatch).
        adm.owner_provider = self._owner_provider
        adm.lcm_pages = self._actuator.lcm_pages
        # Route Admitter cross-* fires through the SAME async cap_barrier +
        # shared-worker path the Budgeter tick uses (no 10-30ms sync stall
        # on the scheduler thread).
        adm._fire_submit = self._submit_admitter_fire
        logger.info(
            "BudgetAgent: Admitter wired (lcm_pages=%d, async_fire=%s)",
            adm.lcm_pages, self._fire_async_enabled,
        )

    def _submit_admitter_fire(self, plan):
        """Submit an Admitter-selected FirePlan through the same path as a
        Budgeter tick fire. Returns ``(aborted, sync_result)``:
          - async (default): cap_barrier inline, hand off to the shared fire
            worker; returns ``(False, None)`` on enqueue, ``(True, None)`` if
            cap_barrier aborts or the queue is full. The dst capacity is
            applied by ``apply_pending_fires`` at the next scheduler iteration
            (before the re-queued request is served); the worker warms the
            c^xfer EWMA on completion.
          - sync fallback (``SGLANG_HIMA_FIRE_ASYNC=0``): full execute inline;
            returns ``(result.aborted, result)`` so the Admitter can price the
            realized transfer.
        Stamps the PaybackPlanner cooldown clock on a successful async
        submit so the next Budgeter tick cannot immediately fire the reverse
        direction (k2m<->m2k oscillation)."""
        if not self._fire_async_enabled:
            result = self._actuator.execute(plan)
            return (result.aborted, result)

        self._ensure_fire_worker()
        token = self._actuator.cap_barrier(plan)
        if token.aborted:
            return (True, None)
        try:
            self._fire_queue.put_nowait(token)
        except queue.Full:
            # Roll back the cap-barrier so the reserved pages return to the
            # free list instead of leaking (mirror the Budgeter tick path).
            alloc = getattr(token.src_act, "allocator", None)
            if alloc is not None and hasattr(alloc, "unmark_pages_capped"):
                try:
                    alloc.unmark_pages_capped(token.cap_t)
                except Exception:
                    logger.exception(
                        "Admitter fire: cap-barrier rollback failed after "
                        "queue-full (seq=%d)", token.plan.plan_seq,
                    )
            logger.warning(
                "Admitter fire: queue full (depth=%d max=%d); deferring "
                "(seq=%d)", self._fire_queue.qsize(), self._fire_queue_max,
                token.plan.plan_seq,
            )
            return (True, None)
        # Cooldown interlock: the planner's decide() gates on
        # clock_s - _last_fire_clock < cooldown_s, where clock_s is the
        # snapshot's wall-clock "ts" (time.time(); see tick()/_snapshot).
        # Stamp the SAME clock so the next Budgeter tick is cooldown-gated
        # against this Admitter fire and cannot immediately fire the reverse
        # direction (k2m<->m2k oscillation).
        if self._planner is not None:
            self._planner._last_fire_clock = time.time()
        return (False, None)

    def _log_chain_unavailable(self, snapshot: dict, reason: str) -> None:
        """Stash the actuator-chain-unavailable reason in the snapshot
        and WARNING-log the first occurrence. Distinct from
        `fire_abort_reason`/`fire_aborted` which only mark a planner-
        wanted-but-failed fire (set in `_maybe_fire`)."""
        snapshot["chain_unavailable_reason"] = reason
        if not self._chain_unavailable_warned:
            logger.warning(
                "BudgetAgent: actuator chain unavailable — %s. "
                "Admitter cross-* will not fire until this is resolved.",
                reason,
            )
            self._chain_unavailable_warned = True

    def _ensure_actuator_chain(self, alloc, kv_pool, mamba_pool,
                               snapshot: dict) -> bool:
        """Lazy-build (KVArenaActuator, MambaArenaActuator, XPoolActuator,
        SchedulerOwnerProvider, XPoolFirePlanner) and push the actuator
        chain into the Admitter.

        Returns True iff the chain is ready (either was already built
        or built successfully here). False = the pool isn't arena-backed
        and no actuator can fire today. Stores abort reasons in
        `snapshot` so the JSONL log gives a debuggable record.

        Pulled out of `_maybe_fire` so the wire-in is decoupled from
        the planner's fire decision — admitter must
        receive `lcm_pages` on the first tick, not on the first fire.
        """
        if self._actuator is not None:
            return True
        from sglang.srt.arena.kv_actuator import KVArenaActuator
        from sglang.srt.arena.mamba_actuator import MambaArenaActuator
        from sglang.srt.arena.xpool_actuator import XPoolActuator
        from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
        from sglang.srt.budgeter.scheduler_owner_provider import (
            SchedulerOwnerProvider,
        )
        from sglang.srt.budgeter.scheduler_stage0_handler import (
            SchedulerStage0Handler,
        )

        # Chain-build failure does NOT set `fire_aborted` here — that
        # JSONL signal means "planner chose a direction but the fire
        # couldn't proceed", and the chain build now runs on EVERY tick
        # (before the planner decides). Stash the reason under a
        # distinct key; `_maybe_fire` promotes it to `fire_abort_reason`
        # iff the planner actually wanted to fire. First failure is logged
        # WARNING once via the gate below.
        if mamba_pool is None:
            self._log_chain_unavailable(snapshot,
                "no mamba_pool — model is KV-only")
            return False
        # mamba_pool != None ⟹ kv_pool is the hybrid wrapper (it owns
        # `mamba_pool`); the real backing MHATokenToKVPool with
        # `_kv_arena` is at `kv_pool.full_kv_pool`. Direct attribute
        # access: HybridLinearKVPool.__init__ assigns `full_kv_pool`
        # unconditionally; any KeyError here is a load-bearing
        # invariant break and should crash loudly.
        inner_kv = kv_pool.full_kv_pool
        # MHATokenToKVPool / MambaPool both unconditionally init these
        # attrs to None when not arena-backed (memory_pool.py
        # `_create_buffers` / `MambaPool.__init__`). Direct access.
        kv_arena = inner_kv._kv_arena
        mamba_arena = mamba_pool._mamba_temporal_arena
        if kv_arena is None or mamba_arena is None:
            self._log_chain_unavailable(snapshot, (
                f"pools not arena-backed (kv_arena={kv_arena!s:.20s} "
                f"mamba_arena={mamba_arena!s:.20s}; "
                f"SGLANG_ARENA_SHARED env not active at pool init?)"
            ))
            return False
        shared_pool = kv_arena._arena._external_pool
        if mamba_arena._arena._external_pool is not shared_pool:
            self._log_chain_unavailable(snapshot,
                "KV/mamba arenas on different shared pools")
            return False

        self._kv_act = KVArenaActuator(pool=inner_kv, allocator=alloc)
        self._mamba_act = MambaArenaActuator(pool=mamba_pool)
        # Stage-0 collaborator: drives sglang's radix-cache eviction
        # (Drain), allocates a free dst mamba slot, and rewrites the
        # in-flight req pointer after a byte-exact `migrate_slot`
        # (Migration). The actuator invokes it only when a plan carries
        # non-empty drains/migrations (cross_evict / cross_migrate); the
        # free-only path never touches it.
        self._stage0_handler = SchedulerStage0Handler(
            scheduler=self.scheduler,
            kv_actuator=self._kv_act,
            mamba_actuator=self._mamba_act,
        )
        # k2m serving floor: one full prefill chunk + a decode step for
        # every running request must always remain allocatable in the KV
        # pool, or alloc_token_slots OOMs with nothing evictable and the
        # scheduler dies (see XPoolActuator.kv_serving_floor_tokens).
        _sa = self.scheduler.server_args
        _kv_floor = 2 * int(_sa.chunked_prefill_size or 8192) + int(
            getattr(self.scheduler, "max_running_requests", 0) or 0
        )
        self._actuator = XPoolActuator(
            kv_arena=kv_arena, mamba_arena=mamba_arena,
            shared_pool=shared_pool,
            kv_actuator=self._kv_act, mamba_actuator=self._mamba_act,
            stage0_handler=self._stage0_handler,
            kv_serving_floor_tokens=_kv_floor,
        )
        logger.info(
            "XPoolActuator: kv_serving_floor_tokens=%d "
            "(2*chunked_prefill + max_running)", _kv_floor,
        )
        model_runner = getattr(
            getattr(self.scheduler, "model_worker", None), "model_runner", None
        )
        self._owner_provider = SchedulerOwnerProvider(
            scheduler=self.scheduler,
            kv_actuator=self._kv_act,
            mamba_actuator=self._mamba_act,
        )
        self._fire_planner = XPoolFirePlanner(
            kv_actuator=self._kv_act,
            mamba_actuator=self._mamba_act,
            owner_provider=self._owner_provider,
        )
        self._mamba_tokens_per_chunk = int(mamba_arena.tokens_per_chunk)
        self._kv_tokens_per_chunk = int(kv_arena.tokens_per_chunk)
        # Wire the fork-failure grow hook: when a caching fork can't get a
        # mamba slot and evict finds no cold cache, MambaRadixCache calls this
        # to synchronously grow mamba from KV and retry, instead of asserting
        # "Can not alloc mamba cache". Only wired once the actuator chain
        # exists; stays None on stock sglang / Budgeter off.
        # On-demand grow hooks gate (isolation knob). Default ON. When
        # SGLANG_XPOOL_ONDEMAND_GROW=0 the alloc-fail paths fall back to their
        # native asserts (no synchronous cross-pool grow), making the arena
        # behave like the static split for grows — used to A/B whether the
        # on-demand k2m/m2k grow path is what strands capped pages.
        _ondemand_grow = _env_flag("SGLANG_XPOOL_ONDEMAND_GROW", True)
        if _ondemand_grow:
            self._tree_cache._mamba_grow_hook = self._grow_mamba_from_kv
            # Symmetric KV grow hook: the alloc path calls it to grow the
            # arena's live KV from idle mamba (m2k) when the live cap is
            # exhausted, instead of crashing at the budgeter-fire-paced cap.
            self.scheduler.token_to_kv_pool_allocator._kv_grow_hook = (
                self._grow_kv_from_mamba
            )
            # Symmetric mamba-active grow hook: the active-slot mamba
            # alloc calls it to grow mamba from idle KV (k2m) when the live mamba
            # cap is exhausted, instead of asserting "Not enough space".
            self.scheduler.req_to_token_pool._mamba_active_grow_hook = (
                self._grow_mamba_from_kv
            )
        else:
            logger.info(
                "BudgetAgent: on-demand cross-pool grow hooks DISABLED "
                "(SGLANG_XPOOL_ONDEMAND_GROW=0)"
            )
        logger.info("BudgetAgent: XPoolActuator chain attached")
        self._wire_admitter()
        self._run_migrate_probe(mamba_pool)
        self._run_boot_probes()
        return True

    def _mamba_working_set_floor_slots(self, m_used: int, evictable: int) -> int:
        """The m2k mamba floor in SLOTS: the LIVE working set the drain must
        retain (design.md §"Allocator floor: working set only").

        The irreducible reservation is `m_used − evictable` = every used slot
        that is NOT a donatable, unlocked cached snapshot = active running SSM
        states + locked protected cache. Plus a fixed `_mamba_fork_headroom_slots`
        burst buffer. Donatable = live_size − floor = available + evictable −
        headroom: the free slack plus the unlocked evictable snapshots the
        plan's Drain stage frees before unmap (the cross-pool donor supply).

        The nominal `max_running_requests` is deliberately NOT reserved: it is
        a static, workload-independent concurrency cap, and in a KV-bound
        regime (KV binds at a few long requests) it far exceeds the actual
        active set, so reserving it over-reserves the whole pool and blocks m2k
        entirely. A burst beyond the headroom is recovered on demand by
        `_mamba_active_grow_hook` (grows mamba from idle KV) — the same safety
        net the on-demand m2k path relies on, so the headroom is thrash
        avoidance, not the safety mechanism, and stays fixed.
        """
        return max(0, int(m_used) - int(evictable)) + self._mamba_fork_headroom_slots

    def _kv_working_set_floor_slots(self, kv_used: int) -> int:
        """The k2m KV floor in TOKENS: reserve the running KV working set PLUS
        one indivisible prefill chunk of GENUINELY-FREE capacity.

        Unlike the mamba floor (`_mamba_working_set_floor_slots`, which credits
        evictable because its drain stage evicts+donates cache), do NOT credit
        evictable KV cache: a k2m fire donates FREE pages only (the cache stays
        as admission's own eviction buffer), and the swarm's shared-prefix
        cache is COW-locked by the very batch that needs the chunk, so
        crediting it lets the drain leave `free + evictable >= chunk` where the
        evictable then evaporates -> `free < chunk` -> alloc_token_slots OOM
        (the KV-side twin of the #339 COW-source over-count). `kv_used` already
        includes the cache, so `kv_used + chunk` leaves >= chunk of
        genuinely-free capacity; the cache is donated gradually as admission
        evicts it (low-value cache first, by LPB) and the freed pages become
        the next fire's donor supply."""
        return int(kv_used) + self._kv_prefill_chunk_floor

    def _grow_mamba_from_kv(self, n_slots: int) -> bool:
        """Synchronously grow mamba by ~`n_slots` slots, transferring chunks
        from KV (k2m). Called from `MambaRadixCache._fork_mamba_with_recovery`
        as the last resort before its "Can not alloc mamba cache" assert
        — it lets a caching fork survive a pool the Budgeter has drained
        to its working set. Runs on the scheduler thread; the actuator
        serializes against the Budgeter worker via its `_fire_inflight` lock
        (same path the Admitter's sync fire uses).

        Returns True iff the fire granted pages (caller retries the fork),
        False on no chain / planner refusal / abort / zero grant (caller falls
        through to the assert). Grows by `ceil(n_slots / tokens_per_chunk)`
        chunks rounded up to the actuator LCM (the atomic cross-pool unit).
        """
        if self._fire_planner is None or self._actuator is None:
            return False
        tps = max(1, int(self._mamba_tokens_per_chunk))
        n_chunks = max(1, (int(n_slots) + tps - 1) // tps)
        lcm = max(1, int(self._actuator.lcm_pages))
        n_pages = max(lcm, ((n_chunks + lcm - 1) // lcm) * lcm)
        # KV working-set floor, the on-demand counterpart of the tick k2m
        # bound in `_maybe_fire`. This k2m grow harvests KV's GENUINELY-FREE
        # pages to feed mamba; bound the drain by the same KV floor (running
        # working set + one indivisible prefill chunk) so it never shrinks KV
        # below a prefill chunk. Refuse when KV has no donatable slack (caller
        # falls through to its assert / fail).
        try:
            kv_alloc = self.scheduler.token_to_kv_pool_allocator
            kv_slots_per_page = max(1, int(self._kv_tokens_per_chunk))
            kv_live = int(kv_alloc.live_size)
            kv_avail = int(kv_alloc.available_size())
            kv_used = max(0, kv_live - kv_avail)
            floor_slots = self._kv_working_set_floor_slots(kv_used)
            n_pages = _mamba_drain_floor(
                kv_live, floor_slots, kv_slots_per_page, n_pages,
            )
            if n_pages <= 0:
                return False
        except Exception:
            logger.exception("BudgetAgent._grow_mamba_from_kv floor calc failed")
            return False
        try:
            plan = self._fire_planner.build(
                "kv_to_mamba", n_pages, allow_drain=True, allow_migrate=False,
            )
            if plan is None:
                return False
            result = self._actuator.execute(plan)
            if result.aborted:
                return False
            return int(result.granted_pages) > 0
        except Exception:
            logger.exception("BudgetAgent._grow_mamba_from_kv failed")
            return False

    def _grow_kv_from_mamba(self, n_tokens: int) -> bool:
        """Synchronously grow KV by ~`n_tokens` tokens, transferring chunks from
        mamba (m2k). The symmetric counterpart of `_grow_mamba_from_kv`,
        installed as the KV allocator's `_kv_grow_hook` and called from the
        alloc path when the arena's LIVE KV cap is exhausted but boot-deferred /
        mamba-idle capacity exists. Without it the arena's live KV grows
        only at the Budgeter's fire cadence; under a fast long-context fill the
        live cap exhausts and alloc crashes, where the full-size static pool
        does not. Runs on the scheduler thread; the actuator serializes against
        the Budgeter worker via its _fire_inflight lock.

        Returns True iff the fire granted pages (caller re-evicts + retries
        alloc), False on no chain / planner refusal / abort / zero grant.
        """
        if self._fire_planner is None or self._actuator is None:
            return False
        tps = max(1, int(self._kv_tokens_per_chunk))
        n_chunks = max(1, (int(n_tokens) + tps - 1) // tps)
        lcm = max(1, int(self._actuator.lcm_pages))
        n_pages = max(lcm, ((n_chunks + lcm - 1) // lcm) * lcm)
        # Drain mamba's donatable slack = free + unlocked-evictable cached
        # snapshots, keeping the LIVE working set (active + protected) plus a
        # burst headroom. Same working-set floor as the tick path. The
        # mamba active-slot alloc carries `_mamba_active_grow_hook` to recover a
        # burst from idle KV, so the nominal max_running is not reserved.
        # Refuse if mamba has no donatable slack (KV alloc then fails
        # gracefully).
        try:
            mamba_pool = self.scheduler.req_to_token_pool.mamba_pool
            mamba_allocator = self.scheduler.req_to_token_pool.mamba_allocator
            m_slots_per_page = max(1, int(self._mamba_tokens_per_chunk))
            m_live = int(mamba_allocator.live_size)
            m_avail = int(mamba_allocator.available_size())
            m_used = max(0, m_live - m_avail)
            evictable = int(self._tree_cache.mamba_evictable_size())
            floor_slots = self._mamba_working_set_floor_slots(m_used, evictable)
            n_pages_capped = _mamba_drain_floor(
                m_live, floor_slots, m_slots_per_page, n_pages
            )
            n_pages = n_pages_capped
            if n_pages <= 0:
                return False
        except Exception:
            logger.exception("BudgetAgent._grow_kv_from_mamba floor calc failed")
            return False
        try:
            plan = self._fire_planner.build(
                "mamba_to_kv", n_pages, allow_drain=True, allow_migrate=False,
            )
            if plan is None:
                return False
            result = self._actuator.execute(plan)
            if result.aborted:
                return False
            return int(result.granted_pages) > 0
        except Exception:
            logger.exception("BudgetAgent._grow_kv_from_mamba failed")
            return False

    def _run_boot_probes(self) -> None:
        """Boot-time `c^xfer` + `κ_i` calibration (design.md
        §"Boot-time probe"). Runs once, right after the actuator chain +
        c_m probe. Gated behind `SGLANG_HIMA_BOOT_PROBE=1` (default
        off) — the c^xfer probe fires the real actuator and the κ_i probe
        runs synthetic prefills, both of which an operator opts into after
        validating on their model/HW. Fail-closed: a probe exception
        leaves the conservative cold-start constant in place (c^xfer
        3000µs/page default; c_i builtin/env curves) and warns once.
        """
        if os.environ.get("SGLANG_HIMA_BOOT_PROBE") != "1":
            return
        if self._boot_probe_done:
            return
        self._boot_probe_done = True
        self._run_xfer_probe()
        self._run_recompute_probe()
        self._dump_probe_results()

    def _dump_probe_results(self) -> None:
        """Write the boot-probed c^xfer + c_m constants to the JSON path in
        `SGLANG_HIMA_PROBE_DUMP`, so the offline calibration driver
        (`dev/eval/cost_model/calibrate_profile.sh`) can fold them into a
        per-(model, GPU) cost profile. No-op when the env is unset (normal
        serving). c^xfer is the per-page seed the runtime EWMA drifts on
        top of; c_m is a fixed constant (`+inf` until probed → dumped as
        null so the profile omits an un-measured seed)."""
        path = os.environ.get("SGLANG_HIMA_PROBE_DUMP")
        if not path:
            return
        from sglang.srt.budgeter.cost_model import (
            get_migrate_cost,
            get_runtime_actuator_cost,
        )
        xfer = get_runtime_actuator_cost()
        migrate = get_migrate_cost()
        cm = migrate.mamba_per_slot_us
        # c^xfer: emit the measured wall only if the boot probe actually
        # seeded it (is_boot_seeded). is_calibrated gates on >=3 LIVE fires
        # and is always False at boot, so a FAILED probe leaves the
        # conservative default in current_us — None marks "not measured"
        # (mirrors the c_m +inf -> None convention so the profile omits an
        # un-measured seed).
        record = {
            "c_xfer_us_per_page": (
                xfer.current_us if xfer.is_boot_seeded else None
            ),
            "c_xfer_calibrated": xfer.is_calibrated,
            "c_m_us_per_slot": cm if cm != float("inf") else None,
            "c_m_calibrated": migrate.is_calibrated,
        }
        try:
            with open(path, "w") as f:
                json.dump(record, f, indent=2)
            logger.info(
                "BudgetAgent: boot-probe results dumped to %s: %s",
                path, record,
            )
        except Exception:
            logger.exception(
                "BudgetAgent: failed to write boot-probe dump to %s", path
            )

    def _run_xfer_probe(self, n_iters: int = 10, n_warmup: int = 2) -> None:
        """Probe `c^xfer` per page via self-reversing fires through the
        real `execute()` path. Each sample times a forward fire
        (kv→mamba of `lcm_pages`); a reverse fire (mamba→kv of exactly
        the moved count) restores the capacity split between samples.
        Seeds `RuntimeActuatorCost` with the median per-page wall.

        The full actuator wall (cap-barrier + verify + cuMemUnmap +
        cuMemMap) IS the design's `c^xfer`, and reusing `execute()` makes
        the probed quantity identical to what production fires pay. Unit:
        the runtime EWMA is per-PAGE (`update_xfer` is called with
        `n_chunks=granted_pages`), so we seed `wall_us / granted_pages`.
        """
        from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost

        actuator = self._actuator
        planner = self._fire_planner
        lcm = actuator.lcm_pages

        # The self-reversing probe relies on a working `mamba_to_kv` reverse
        # fire to restore the capacity split it perturbs; the actuator's verify
        # delegates to the mamba allocator's `count_reachable_capped`, so
        # mamba-as-source fires resolve without crashing.
        from sglang.srt.budgeter.boot_probe import balance_restore

        def _fire(direction: str, n_pages: int) -> int:
            plan = planner.build(direction=direction, n_pages_target=n_pages)
            if plan is None:
                return 0
            result = actuator.execute(plan)
            return int(result.granted_pages)

        def _kv_cap() -> int:
            return int(actuator.state()["kv_capacity_tokens"])

        def _restore(baseline: int) -> None:
            residual = balance_restore(
                baseline, _fire, _kv_cap, step_pages=lcm,
            )
            if residual != 0:
                logger.warning(
                    "c^xfer probe: kv capacity not fully restored "
                    "(residual %d tokens vs baseline %d); the budgeter's "
                    "steady-state rebalancing absorbs it",
                    residual, baseline,
                )

        baseline_kv = _kv_cap()
        try:
            for _ in range(n_warmup):
                _fire("kv_to_mamba", lcm)
                _restore(baseline_kv)

            per_page_samples = []
            for _ in range(n_iters):
                t0 = time.perf_counter_ns()
                moved = _fire("kv_to_mamba", lcm)
                wall_us = (time.perf_counter_ns() - t0) / 1000.0
                if moved > 0:
                    per_page_samples.append(wall_us / moved)
                _restore(baseline_kv)

            if not per_page_samples:
                raise RuntimeError(
                    "c^xfer probe: no fire moved any pages (planner built "
                    "no plan — pools may have no transferable free pages "
                    "at boot)"
                )
            per_page_samples.sort()
            median_us = per_page_samples[len(per_page_samples) // 2]
            get_runtime_actuator_cost().seed_from_boot_probe(median_us)
            logger.info(
                "c^xfer boot probe: %.1f µs/page (median over %d fires; "
                "min %.1f, max %.1f; kv capacity restored to %d)",
                median_us, len(per_page_samples),
                per_page_samples[0], per_page_samples[-1], _kv_cap(),
            )
        except Exception as e:
            if not self._boot_probe_warned:
                logger.warning(
                    "c^xfer boot probe failed (%s: %s); c^xfer stays at "
                    "the conservative cold-start default",
                    type(e).__name__, e,
                )
                logger.exception("c^xfer probe traceback")
                self._boot_probe_warned = True
        finally:
            # Always restore the capacity split, even if the measure loop
            # threw after a committed forward fire (a mid-loop
            # exception otherwise leaves a bounded one-way skew).
            try:
                _restore(baseline_kv)
            except Exception:
                logger.exception(
                    "c^xfer probe: final capacity restore failed; "
                    "budgeter steady-state rebalancing will absorb any skew"
                )

    def _run_recompute_probe(self) -> None:
        """No-op in production: `_make_prefill_timer` returns None (κ_i is
        calibrated OFFLINE via `dev/eval/cost_model/calibrate_profile.sh`,
        not by an in-engine probe — design.md §"Boot-time probe"), so this
        method early-returns before touching the cost curves.

        It remains as a unit-test seam: when a synthetic prefill timer is
        injected at `_make_prefill_timer`, it times re-prefill of L tokens
        through the KV / mamba stack, fits `c_KV`/`c_M` curves via
        `cost_model.fit_cost_curves`, and installs them (env-precedence
        respected by `set_cost_curves`). Fail-closed: a failure leaves the
        builtin/env curves in place + warns once.
        """
        from sglang.srt.budgeter.boot_probe import measure_recompute_curves
        from sglang.srt.budgeter.cost_model import (
            fit_cost_curves,
            set_cost_curves,
        )

        try:
            time_prefill_ms = self._make_prefill_timer()
            if time_prefill_ms is None:
                logger.info(
                    "κ_i boot probe: no synthetic-prefill timer available "
                    "for this worker; keeping builtin/env curves"
                )
                return
            kv_lengths = [256, 1024, 2048, 4096, 8192]
            m_lengths = [256, 2048, 8192]
            kv_samples, m_samples = measure_recompute_curves(
                time_prefill_ms, kv_lengths=kv_lengths, m_lengths=m_lengths,
            )
            curves = fit_cost_curves(kv_samples, m_samples, source="boot_probe")
            set_cost_curves(curves)
        except Exception as e:
            if not self._boot_probe_warned:
                logger.warning(
                    "κ_i boot probe failed (%s: %s); keeping builtin/env "
                    "cost curves",
                    type(e).__name__, e,
                )
                logger.exception("κ_i probe traceback")
                self._boot_probe_warned = True

    def _make_prefill_timer(self):
        """Return None: κ_i is calibrated OFFLINE, not by an in-engine
        probe (design.md "Boot-time probe"). A hybrid forward can't be
        split into the KV-L² and mamba-L stacks in-engine, so `κ_i` is
        fit by `dev/eval/cost_model/calibrate_profile.sh` from a
        pure-prefill `bench_one_batch` sweep and injected via
        `SGLANG_CSIGMA_*`. This hook stays as the documented seam:
        `_run_recompute_probe` early-returns on None (a clean no-op that
        preserves the env/builtin curves), and unit tests inject a
        synthetic timer here to exercise the fit path. Returning a real
        timer would re-enable an in-engine κ_i probe — deliberately not
        done."""
        return None

    def _run_migrate_probe(self, mamba_pool) -> None:
        """Run the boot-time `c_m` probe — mamba per-slot migration
        data-copy wall — and seed the cost model.

        Called once, immediately after the actuator chain builds (the
        same moment lcm_pages becomes knowable for the Admitter, see
        `_wire_admitter`). Failure does NOT crash the engine: the
        cold-start `c_migrate_us = +inf` already makes migrate
        candidates infeasible, and the cross-migrate path is the consumer.
        But the failure IS surfaced — a one-shot WARNING fires the
        first time so operators see it in the boot log, parallel to
        the `_chain_unavailable_warned` pattern. Subsequent calls
        (if the chain is rebuilt) won't re-warn.
        """
        from sglang.srt.budgeter.cost_model import get_migrate_cost
        from sglang.srt.budgeter.migrate_probe import measure_mamba_migrate
        migrate_cost = get_migrate_cost()
        if migrate_cost.is_env_pinned:
            logger.info(
                "BudgetAgent: c_m migrate probe skipped — c_m pinned via "
                "SGLANG_CM_MAMBA_PER_SLOT_US (env-precedence)"
            )
            return
        try:
            per_slot_us = measure_mamba_migrate(mamba_pool)
            migrate_cost.set_mamba(per_slot_us)
        except Exception as e:
            if not self._migrate_probe_warned:
                logger.warning(
                    "BudgetAgent: c_m migrate probe failed (%s: %s); "
                    "`c_migrate_us` stays +inf — Admitter migrate "
                    "candidates remain infeasible until probe succeeds",
                    type(e).__name__, e,
                )
                logger.exception("c_m probe traceback")
                self._migrate_probe_warned = True

    # Latches the one-time warning, per source pool, when the drain is
    # disabled by a degenerate cost curve — so operators see WHY cross-fire
    # went free-only without spamming every tick.
    _warned_degenerate_drain_mamba = False
    _warned_degenerate_drain_kv = False

    def _cross_drain_allowed(self, direction: str) -> bool:
        """Fail-closed gate for the cross-fire DRAIN (Stage-2 cold-cache
        eviction). Draining evicts cached entries from the SOURCE pool — m2k
        drains mamba, k2m drains KV — and is safe ONLY when the
        reuse-aware drain cost the NB subtracts can credibly price a HOT cache
        as expensive. Two independent degeneracies, each of which would
        re-open the starve regression, fall closed to free-only:

        1. Eviction policy. The drain cost weights victims by hit count
           `n_b`, which only LPB provides; under LRU `n_b ≡ 1` so it cannot
           tell hot from cold — the same reuse-awareness gate the grow
           benefit uses. Applies to BOTH directions.
        2. Cost curve. The source pool's recompute curve must be
           non-degenerate, else `c(L) ≈ 0` collapses the drain cost to ~0
           regardless of policy and defeats the gate. m2k checks the mamba
           curve (`m_alpha == m_beta == 0`, κ_M-zero); k2m checks the
           KV curve (`kv_alpha == kv_beta == kv_gamma == 0`).

        (Stage-3 LIVE-slot migration is a separate capability; this gate
        is only about the cold-cache DRAIN, which evicts cached — never
        live/graph-referenced — slots and is safe within the cross-fire
        model. Migration deliberately has NO analogous degenerate-curve gate:
        its cost `c_migrate` is a boot-probed scalar, not a recovery curve
        (cold-start `+inf`, fail-closed — see `BootProbedMigrateCost` and
        `test_migrate_probe`'s cold-start test), and it relocates LIVE state
        byte-exact with no eviction or recompute, so there is no hot-cache
        mispricing for such a gate to guard. The asymmetry with this drain gate
        is intentional, not a missing gate.)"""
        if direction not in ("mamba_to_kv", "kv_to_mamba"):
            return False
        tc = self._tree_cache
        if tc is None or not tc._should_use_lpb():
            return False
        from sglang.srt.budgeter.cost_model import get_cost_curves
        curves = get_cost_curves()
        # A cross-pool drain evicts cached leaves through the hybrid radix
        # cache's FULL eviction, which drops BOTH the leaf's KV tokens AND its
        # paired mamba snapshot. Re-prefilling that leaf recomputes the WHOLE
        # prefix, so its recompute cost is the TOTAL prefill wall — and the
        # offline κ calibration cannot split a hybrid forward into per-stack
        # costs, so by design it folds that total into the KV curve and sets
        # κ_M = 0 (`calibrate_kappa.py` "HYBRID CAVEAT"). The reuse-aware drain
        # cost is therefore `c_KV + c_M = total + 0`, and a NON-degenerate κ_KV
        # alone prices a hot cache as expensive — for BOTH directions, since
        # evict_full's per-leaf recompute is the same total either way.
        #
        # So the gate requires κ_KV non-degenerate; it does NOT require κ_M
        # non-degenerate — κ_M=0 is the EXPECTED post-calibration state (the
        # total is folded into κ_KV). κ_KV all-zero is the genuine degeneracy
        # (the folded total collapses ⇒ cannot gate hot cache) → fail closed.
        if (curves.kv_alpha == 0.0 and curves.kv_beta == 0.0
                and curves.kv_gamma == 0.0):
            if not BudgetAgent._warned_degenerate_drain_kv:
                BudgetAgent._warned_degenerate_drain_kv = True
                logger.warning(
                    "BudgetAgent: KV recompute cost curve is degenerate "
                    "(kv_alpha=kv_beta=kv_gamma=0) — the reuse-aware drain "
                    "cost collapses to ~0 and cannot gate hot-cache eviction "
                    "(the hybrid calibration folds the total prefix recompute "
                    "into κ_KV, so κ_KV=0 means NO cost signal). Cross-fire "
                    "drains stay FREE-ONLY. Re-calibrate c_KV (SGLANG_CSIGMA_KV_*)."
                )
            return False
        return True

    def _maybe_fire(self, snapshot: dict) -> None:
        """PaybackPlanner decides direction → XPoolFirePlanner builds a
        FirePlan from current ownership state → XPoolActuator executes
        it (cuMemUnmap source pages, cuMemMap them into dst pool).

        Lazy-built on first call. If the pool isn't arena-backed
        (SGLANG_ARENA_SHARED=1 was clobbered) the lazy build raises
        and the next tick's exception handler in tick() logs + skips."""
        alloc = self.scheduler.token_to_kv_pool_allocator
        kv_pool = alloc.get_kvcache()
        mamba_pool = getattr(kv_pool, "mamba_pool", None)
        mamba_allocator = getattr(self.scheduler.req_to_token_pool, "mamba_allocator", None)

        # Instantaneous usage per pool (observability; not consumed by
        # PaybackPlanner, but needed for snapshot JSONL and downstream
        # monitors).
        usage_kv_inst = 0.0
        usage_kv_active = 0.0
        usage_mamba_inst = 0.0
        usage_mamba_active = 0.0
        live = alloc.live_size
        avail = alloc.available_size()
        tc = self._tree_cache
        if live > 0:
            usage_kv_inst = max(0.0, min(1.0, (live - avail) / live))
            kv_cached = int(tc.full_evictable_size())
            usage_kv_active = max(
                0.0, min(1.0, (live - avail - kv_cached) / live)
            )
        if mamba_pool is not None:
            ms_live = mamba_allocator.live_size
            ms_avail = mamba_allocator.available_size()
            if ms_live > 0:
                usage_mamba_inst = max(0.0, min(1.0, (ms_live - ms_avail) / ms_live))
                m_cached = 0
                if hasattr(tc, "mamba_evictable_size"):
                    try:
                        m_cached = int(tc.mamba_evictable_size())
                    except Exception:
                        m_cached = 0
                usage_mamba_active = max(
                    0.0, min(1.0, (ms_live - ms_avail - m_cached) / ms_live)
                )
        snapshot["usage_kv_inst"] = usage_kv_inst
        snapshot["usage_kv_active"] = usage_kv_active
        snapshot["usage_mamba_inst"] = usage_mamba_inst
        snapshot["usage_mamba_active"] = usage_mamba_active

        # Lazy-build PaybackPlanner.
        if self._planner is None:
            from sglang.srt.budgeter.xpool_planner import PaybackPlanner
            from sglang.srt.budgeter.cost_model import (
                get_cost_curves,
                get_runtime_actuator_cost,
            )
            rac = get_runtime_actuator_cost()
            self._planner = PaybackPlanner(
                cost_curves=get_cost_curves(),
                fire_cost_us=rac.current_us if rac.is_boot_seeded else 5000.0,
            )
            logger.info("BudgetAgent: PaybackPlanner attached")

        # Build the actuator chain (+ Admitter wire-in) eagerly on the
        # first tick that gets here. Decoupling from the planner's
        # fire decision below is essential: the Admitter needs
        # `lcm_pages` from the first arrival onwards, not from the
        # first time the Budgeter decides to fire.
        # On failure (non-arena pool) the planner still runs for
        # observability; the fire path below short-circuits when
        # self._actuator is None.
        chain_ready = self._ensure_actuator_chain(
            alloc, kv_pool, mamba_pool, snapshot
        )

        # "w/o Budgeter" ablation: the chain above stays built (the
        # Admitter's per-arrival fires depend on it), but the tick-path
        # planner never decides/fires.
        if self.planner_disabled:
            snapshot["plan_direction"] = "none"
            snapshot["plan_reason"] = "budgeter disabled (SGLANG_HIMA_NO_BUDGETER)"
            return

        # Warmup fire-suppression window: the first seconds after boot are a
        # TRANSIENT (the initial admission wave briefly fills the mamba pool
        # and evicts a burst of deep snapshots), but a fire is IRREVERSIBLE
        # in practice — the free->free return path can never reclaim pages
        # from a snapshot-fragmented arena. On Kimi deep gates a single
        # warmup-priced k2m shrank KV 21.7% for the whole run (see
        # dev/interlayer/5_mla_arena/CALIBRATION.md). Observe (EWMAs still
        # update via decide() next tick), don't act. Window default 180 s
        # (>> 5 s EWMA tau, so the spike has fully decayed before acting).
        warmup_s = _env_float("SGLANG_HIMA_FIRE_WARMUP_S", 180.0)
        if warmup_s > 0:
            # Anchor on the first LOADED tick, not boot: the client may start
            # minutes after the server is ready, and the transient this window
            # exists to ride out is the initial ADMISSION WAVE (a boot-anchored
            # window expired 30 s before the wave's eviction spike on gate5).
            if self._first_tick_monotonic is None:
                if float(snapshot.get("num_running_reqs", 0) or 0) >= 8:
                    self._first_tick_monotonic = time.monotonic()
                else:
                    snapshot["plan_direction"] = "none"
                    snapshot["plan_reason"] = "warmup: awaiting first load"
                    return
            elapsed = time.monotonic() - self._first_tick_monotonic
            if elapsed < warmup_s:
                # Let the planner observe (EWMA warm-up) but veto any fire.
                _ = self._planner.decide(snapshot, float(snapshot.get("ts", 0.0)),
                                         float(snapshot.get("dt", 1.0)))
                snapshot["plan_direction"] = "none"
                snapshot["plan_reason"] = (
                    f"warmup fire-suppression ({elapsed:.0f}s < {warmup_s:.0f}s)"
                )
                return

        _p_t0 = time.perf_counter_ns()
        clock_s = float(snapshot.get("ts", 0.0))
        dt = float(snapshot.get("dt", 1.0))
        decision = self._planner.decide(snapshot, clock_s, dt)
        _p_t_decide = time.perf_counter_ns()
        snapshot["_probe_decide_us"] = (_p_t_decide - _p_t0) // 1000
        snapshot["plan_direction"] = decision.direction or "none"
        snapshot["plan_reason"] = decision.reason
        if decision.direction is None:
            return
        if not chain_ready:
            # Planner wanted to fire but actuator chain unavailable.
            # Promote `chain_unavailable_reason` (set by the eager
            # build attempt) to the canonical `fire_abort_reason`
            # only at this point, so JSONL consumers see fire_aborted
            # iff a fire was actually attempted.
            snapshot["fire_abort_reason"] = snapshot.get(
                "chain_unavailable_reason", "actuator chain unavailable"
            )
            snapshot["fire_aborted"] = True
            return

        # Ceiling-refuse: a grow fire is a no-op-at-best when the
        # DESTINATION pool is already at its page-id ceiling, and at worst
        # makes the arena cuMemMap chunks the allocator cannot represent
        # (token-slot id > max_size → unmark_pages_capped fail-fast /
        # orphaned handles). The two pools express "at ceiling" in opposite
        # conventions: the KV allocator freezes `size` at the ceiling and
        # tracks the live cap as `size - _capped_pages` (live_size), so
        # headroom exists iff `live_size < size`; MambaPool moves `size` as
        # the live cap up to `max_size`, so headroom exists iff
        # `size < max_size`.
        if decision.direction == "mamba_to_kv":
            if alloc.live_size >= alloc.size:
                snapshot["fire_direction"] = decision.direction
                snapshot["fire_aborted"] = True
                snapshot["fire_abort_reason"] = (
                    "kv at max_size ceiling — no grow headroom"
                )
                return
        elif decision.direction == "kv_to_mamba" and mamba_pool is not None:
            if int(mamba_pool.size) >= int(mamba_pool.max_size):
                snapshot["fire_direction"] = decision.direction
                snapshot["fire_aborted"] = True
                snapshot["fire_abort_reason"] = (
                    "mamba at max_size ceiling — no grow headroom"
                )
                return

        # Build FirePlan from current ownership state. The steady-state
        # rebalance must DRAIN cold cache, not only harvest genuinely-free
        # pages: at steady saturation the source pool's reclaimable slack is
        # its cold cached snapshots (full-but-quiescent), not free slots.
        # Budgeter fires are free→free ONLY: transfer free pages from the
        # source pool to the destination. No drain (no cache eviction), no
        # migrate (no live slot relocation). Non-destructive by construction.
        n_free = self._owner_provider.n_free_source_pages(
            decision.direction) if self._owner_provider else 0
        # m2k (grow KV from mamba): demand-driven (transfer all free).
        # k2m (grow mamba from KV): 1 LCM per fire for gradual convergence.
        if decision.direction == "mamba_to_kv":
            n_pages_target = n_free
        else:
            lcm = getattr(self._actuator, "lcm_pages", 48) if self._actuator else 48
            n_pages_target = min(n_free, lcm)
        if decision.direction == "mamba_to_kv" and mamba_pool is not None and n_pages_target > 0:
            live_size = int(mamba_allocator.live_size)
            m_used = max(0, live_size - int(mamba_allocator.available_size()))
            evictable = int(self._tree_cache.mamba_evictable_size())
            floor_slots = self._mamba_working_set_floor_slots(m_used, evictable)
            arena = mamba_pool._mamba_temporal_arena
            slots_per_page = int(arena.tokens_per_chunk) if arena is not None else 0
            n_pages_target = _mamba_drain_floor(
                live_size, floor_slots, slots_per_page, n_pages_target,
            )
        elif decision.direction == "kv_to_mamba" and n_pages_target > 0:
            # Symmetric KV working-set floor: the drain must leave the running
            # KV working set + one indivisible prefill chunk, or a later
            # alloc_token_slots OOMs (the swarm crash). Reuses the same
            # pool-agnostic _mamba_drain_floor clamp as the m2k branch.
            kv_live = int(alloc.live_size)
            kv_avail = int(alloc.available_size())
            kv_used = max(0, kv_live - kv_avail)
            floor_slots = self._kv_working_set_floor_slots(kv_used)
            n_pages_target = _mamba_drain_floor(
                kv_live, floor_slots, self._kv_tokens_per_chunk, n_pages_target,
            )
        if n_pages_target <= 0:
            snapshot["fire_direction"] = decision.direction
            snapshot["fire_aborted"] = True
            snapshot["fire_abort_reason"] = "no free source pages"
            return
        _p_t_build_start = time.perf_counter_ns()
        plan = self._fire_planner.build(
            direction=decision.direction,
            n_pages_target=n_pages_target,
        )
        _p_t_build_done = time.perf_counter_ns()
        snapshot["_probe_build_us"] = (_p_t_build_done - _p_t_build_start) // 1000
        if plan is None:
            snapshot["fire_direction"] = decision.direction
            snapshot["fire_aborted"] = True
            snapshot["fire_abort_reason"] = "fire_planner: no buildable plan this tick"
            return

        # Execute the fire. Two modes:
        #   - async (default): run cap-barrier sync on the scheduler thread,
        #     then hand the rest (cuMemUnmap+cuMemMap+sync+cap-bump) to a
        #     worker thread. The scheduler can resume admitting requests
        #     immediately after cap-barrier — those new requests will see
        #     the to-be-unmapped pages as gone (because mark_pages_capped
        #     already removed them from the allocator's free list).
        #   - sync (SGLANG_HIMA_FIRE_ASYNC=0): legacy path; the entire
        #     execute() blocks the scheduler thread (10-30ms typical).
        if self._fire_async_enabled:
            self._ensure_fire_worker()
            # 1. cap-barrier inline (synchronous on scheduler thread).
            #    cap_barrier returns a FireToken that the worker consumes.
            _p_t_cap_start = time.perf_counter_ns()
            token = self._actuator.cap_barrier(plan)
            _p_t_cap_done = time.perf_counter_ns()
            snapshot["_probe_cap_full_us"] = (_p_t_cap_done - _p_t_cap_start) // 1000
            if token.aborted:
                # verify failed; cap-barrier already rolled back. Surface
                # the abort in the snapshot just like the sync path.
                snapshot["fire_direction"] = token.plan.direction
                snapshot["fire_aborted"] = True
                snapshot["fire_abort_reason"] = token.abort_reason
                snapshot["fire_plan_seq"] = token.plan.plan_seq
                snapshot["fire_cap_barrier_us"] = token.cap_barrier_us
                return
            # 2. Hand off to worker; if the queue is full we have to
            #    roll back the cap-barrier (would otherwise hold pages
            #    out of free-list permanently).
            try:
                self._fire_queue.put_nowait(token)
                # Record the enqueue WITHOUT setting fire_direction —
                # D7's validator filters on (fire_direction != none and
                # not fire_aborted) and then checks per-fire unmap/grant.
                # The completion record (emitted by the worker) carries
                # those counts; we don't want the enqueue line to be
                # mistaken for a fire that "moved zero pages".
                snapshot["fire_enqueued"] = True
                snapshot["fire_enqueued_direction"] = token.plan.direction
                snapshot["fire_plan_seq"] = token.plan.plan_seq
                snapshot["fire_cap_barrier_us"] = token.cap_barrier_us
                snapshot["fire_queue_depth"] = self._fire_queue.qsize()
            except queue.Full:
                # Roll back cap-barrier: return capped pages to free.
                alloc = getattr(token.src_act, "allocator", None)
                if alloc is not None and hasattr(alloc, "unmark_pages_capped"):
                    try:
                        alloc.unmark_pages_capped(token.cap_t)
                    except Exception:
                        logger.exception(
                            "BudgetAgent: failed to roll back cap-barrier "
                            "after queue-full"
                        )
                snapshot["fire_direction"] = token.plan.direction
                snapshot["fire_aborted"] = True
                snapshot["fire_abort_reason"] = (
                    f"fire queue full (max={self._fire_queue_max}); "
                    f"skipped this tick"
                )
                snapshot["fire_plan_seq"] = token.plan.plan_seq
                snapshot["fire_cap_barrier_us"] = token.cap_barrier_us
                logger.warning(
                    "BudgetAgent: fire queue full (depth=%d max=%d); "
                    "skipping this tick's fire (seq=%d)",
                    self._fire_queue.qsize(), self._fire_queue_max,
                    token.plan.plan_seq,
                )
                return
        else:
            # Legacy synchronous path: full execute() on scheduler thread.
            shared = self._actuator.shared
            snapshot["fire_shared_free_before"] = shared.free_count()
            result = self._actuator.execute(plan)
            snapshot["fire_shared_free_after"] = shared.free_count()
            snapshot["fire_direction"] = result.direction
            snapshot["fire_aborted"] = result.aborted
            if result.aborted:
                snapshot["fire_abort_reason"] = result.abort_reason
            snapshot["fire_plan_seq"] = result.plan_seq
            snapshot["fire_unmapped_pages"] = result.unmapped_pages
            snapshot["fire_granted_pages"] = result.granted_pages
            snapshot["fire_cap_barrier_us"] = result.cap_barrier_us
            snapshot["fire_unmap_us"] = result.unmap_us
            snapshot["fire_map_us"] = result.map_us
            snapshot["fire_total_us"] = result.total_us

    # ---- Snapshot + close ----

    def _snapshot(self, now: float) -> dict:
        """Capture all signals the planner consumes."""
        sched = self.scheduler
        stats = sched.metrics_reporter.stats
        # stats.num_*_reqs are only refreshed in `_maybe_log_idle_metrics`
        # (every 30s, gated by --enable-metrics). Without that flag they
        # stay at 0 forever — which breaks the planner's queue/persist
        # signals. Read directly from the scheduler's live structures:
        #   running_batch.reqs : active req slots
        #   waiting_queue      : reqs admitted-but-not-yet-running
        def _count(v) -> int:
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return int(t)
            return int(v) if isinstance(v, (int, float)) else 0
        running_n = 0
        rb = getattr(sched, "running_batch", None)
        if rb is not None:
            running_n = len(getattr(rb, "reqs", []) or [])
        queue_n = len(getattr(sched, "waiting_queue", []) or [])
        snap: dict[str, Any] = {
            "ts": round(now, 3),
            "tick": self._tick_count,
            "max_total_num_tokens": getattr(sched, "max_total_num_tokens", 0),
            "kv_used_tokens": stats.kv_used_tokens,
            "kv_evictable_tokens": stats.kv_evictable_tokens,
            "kv_available_tokens": stats.kv_available_tokens,
            "token_usage": stats.token_usage,
            "full_token_usage": stats.full_token_usage,
            "swa_token_usage": stats.swa_token_usage,
            "mamba_usage": stats.mamba_usage,
            "cache_hit_rate": stats.cache_hit_rate,
            "num_running_reqs": running_n,
            "num_queue_reqs": queue_n,
            "max_running_mamba": int(self._last_mamba_size or 0) // max(1, self._mamba_per_req_ratio) if self._last_mamba_size else 0,
            "num_paused_reqs": _count(stats.num_paused_reqs),
            "num_retracted_reqs": _count(stats.num_retracted_reqs),
            "gen_throughput": stats.gen_throughput,
        }
        # Marginal fire deliverable (design.md §"NB is the net benefit of ONE
        # fire"). One fire moves all available source free pages. The planner
        # uses these counts to bound the admission-pressure benefit
        # (`pressure_to_σ`) to `min(1, fire_admit_σ / num_queue)`:
        # without the bound `pressure_to` prices the WHOLE queue, so a single
        # fire that relieves ~4 of 98 queued reqs is credited the full backlog
        # and the budgeter over-fires (m2k drains the recur cache for a
        # near-zero marginal KV gain). `fire_admit_kv` = grant_kv_tokens /
        # avg_kv_tokens_per_running_req; mamba is 1 slot per req so
        # `fire_admit_mamba` = the grant in slots directly. Missing field ⇒
        # planner caps at 1.0 (prices the whole queue).
        _running = max(1, running_n)
        _m_free = (self._owner_provider.n_free_source_pages("mamba_to_kv")
                   if self._owner_provider else 0)
        _k_free = (self._owner_provider.n_free_source_pages("kv_to_mamba")
                   if self._owner_provider else 0)
        _grant_kv_tok = _m_free * max(1, self._kv_tokens_per_chunk)
        _grant_m_slots = _k_free * max(1, self._mamba_tokens_per_chunk)
        _avg_kv_req = max(1.0, float(stats.kv_used_tokens) / _running)
        snap["fire_admit_kv"] = _grant_kv_tok / _avg_kv_req
        snap["fire_admit_mamba"] = float(_grant_m_slots)
        # Fire-planner refuse counter (design.md §"Page selection":
        # "increments the refuse-rate counter"). The observable signal for
        # "anywhere-free + Drain + Migration exhausted" — a sustained
        # non-zero rate flags pool-sizing / workload-composition drift.
        # None until the actuator chain (hence the planner) is built.
        if self._fire_planner is not None:
            snap["fire_refuse_count"] = int(self._fire_planner.refuse_count)

        # Eviction delta (used as a pressure proxy; cumulative counter is
        # maintained by `check_decode_mem` in schedule_batch.py).
        cum_evict = getattr(
            self._tree_cache, "_admission_cumulative_evicted_tokens", 0
        )
        last = self._last_evicted_cumulative
        snap["num_evicted_tokens_recent"] = max(0, cum_evict - last)
        snap["num_evicted_tokens_cumulative"] = cum_evict
        self._last_evicted_cumulative = cum_evict

        # Slow-recovery-length EWMAs — the per-eviction re-prefill /
        # rebuild length the planner's eviction-cost term c_σ(L) is
        # evaluated at (design.md §"Budgeter — steady-state pressure
        # rebalance"). Written onto tree_cache by record_recovery_len_kv
        # / _rec / _retract on every KV / mamba eviction or req retraction
        # (mem_cache/common.py), unconditionally init'd at cache __init__.
        # Without plumbing these the planner sees L=0 always → c_σ=0 → the
        # eviction-cost half of the NB is dead and only queue/persist
        # signals can fire.
        snap["slow_recovery_len_kv"] = float(
            self._tree_cache._slow_recovery_len_kv_ewma
        )
        snap["slow_recovery_len_rec"] = float(
            self._tree_cache._slow_recovery_len_rec_ewma
        )
        snap["slow_recovery_len_retract"] = float(
            self._tree_cache._slow_recovery_len_retract_ewma
        )

        # Per-tick cache-eviction RATE per pool — the grow-side signal
        # a pool actively shedding cache this tick is a candidate to
        # GROW (vs the recovery-length EWMAs above, which give the per-
        # eviction cost). Deltas of the cumulative counters written by
        # evict_mamba / evict_full.
        cum_m = self._tree_cache._cumulative_evicted_mamba_slots
        cum_k = self._tree_cache._cumulative_evicted_kv_tokens
        snap["mamba_evicted_slots_recent"] = max(0, cum_m - self._last_evicted_mamba_slots)
        snap["kv_evicted_tokens_recent"] = max(0, cum_k - self._last_evicted_kv_tokens)
        self._last_evicted_mamba_slots = cum_m
        self._last_evicted_kv_tokens = cum_k

        # Reuse-weighted LPB LOSS (us) shed per pool this tick — the ACCURATE
        # eviction-cost signal the PaybackPlanner uses for r_evict (a low-reuse
        # pool's churn carries ~0 loss, so it stops driving spurious fires).
        cum_kl = self._tree_cache._cumulative_evicted_kv_lpb_loss
        cum_ml = self._tree_cache._cumulative_evicted_mamba_lpb_loss
        snap["kv_evicted_lpb_loss_recent"] = max(0.0, cum_kl - self._last_evicted_kv_lpb_loss)
        snap["mamba_evicted_lpb_loss_recent"] = max(0.0, cum_ml - self._last_evicted_mamba_lpb_loss)
        self._last_evicted_kv_lpb_loss = cum_kl
        self._last_evicted_mamba_lpb_loss = cum_ml

        # Pool-occupancy metrics (paper §motivation, bubble figure):
        # (pool.size - pool.available_size()) / pool.size = used / total,
        # INCLUDING radix-tree-cached prefix/snapshots. Different from the
        # scheduler's `full_token_usage` / `mamba_usage` which subtract
        # evictable size for the admission-pressure framing.
        alloc = self.scheduler.token_to_kv_pool_allocator
        kv_total = alloc.size
        kv_avail = alloc.available_size()
        if kv_total > 0:
            snap["pool_occupancy_kv"] = max(
                0.0, min(1.0, (kv_total - kv_avail) / kv_total)
            )
        kv_pool = alloc.get_kvcache()
        mamba_pool = getattr(kv_pool, "mamba_pool", None)
        mamba_allocator = getattr(self.scheduler.req_to_token_pool, "mamba_allocator", None)
        if mamba_pool is not None:
            m_total = mamba_pool.size
            m_avail = mamba_allocator.available_size()
            if m_total > 0:
                snap["pool_occupancy_mamba"] = max(
                    0.0, min(1.0, (m_total - m_avail) / m_total)
                )
        return snap

    def close(self) -> None:
        # Tell the worker to exit (sentinel) and join briefly. The worker
        # is a daemon thread so this is best-effort; if it's mid-fire we
        # let it finish naturally on process teardown.
        if self._fire_queue is not None and self._fire_worker is not None:
            try:
                self._fire_queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass
            try:
                self._fire_worker.join(timeout=2.0)
            except Exception:
                pass
        if self.log_enabled:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self.log_enabled = False
            self._log_fp = None

    # ---- Async fire worker ----

    def _ensure_fire_worker(self) -> None:
        """Lazy-start the async fire worker thread. Called on first
        fire attempt so we don't spin up a worker for runs that never
        fire (e.g., budgeter enabled but workload doesn't saturate)."""
        if self._fire_worker is not None:
            return
        self._fire_queue = queue.Queue(maxsize=self._fire_queue_max)
        self._fire_worker = threading.Thread(
            target=self._fire_worker_loop,
            name="budgeter-fire-worker",
            daemon=True,
        )
        self._fire_worker.start()
        logger.info(
            "BudgetAgent: async fire worker started "
            "(queue_max=%d, enabled=%s)",
            self._fire_queue_max, self._fire_async_enabled,
        )

    def has_inflight_fires(self) -> bool:
        """True while a cross-pool fire is queued or executing on the worker —
        the KV/mamba pools are mid-mutation (cuMemUnmap/Map + cap state). The
        scheduler treats this as non-idle so a destructive flush_cache (which
        resets allocator capacity) can't run mid-unmap. False when the Budgeter
        is off (no fire queue)."""
        q = self._fire_queue
        return (q is not None and not q.empty()) or self._fire_executing

    def _fire_worker_loop(self) -> None:
        """Worker thread: pulls FireTokens off the queue and runs the
        cuMemUnmap+cuMemMap+cap-bump phase, writing each completed fire's
        record as its own JSONL line for the D7 byte-transfer validator.

        Drains one token at a time; the planner's cooldown keeps the
        queue mostly empty (worst-case 1-2 in-flight), so this single-
        threaded worker is the right structure: it serializes physical
        cuMem* ops, preventing two fires from competing for the same
        shared_pool free handles.
        """
        import torch
        torch.cuda.set_device(self.scheduler.ps.gpu_id)
        while True:
            token = self._fire_queue.get()
            if token is None:  # shutdown sentinel
                return
            try:
                shared_free_before = self._actuator.shared.free_count()
                self._fire_executing = True
                try:
                    result = self._actuator.execute_async(token)
                finally:
                    self._fire_executing = False
                shared_free_after = self._actuator.shared.free_count()
            except Exception:  # never let the worker die silently
                # Roll back cap_barrier: the src allocator's _capped_pages
                # still holds token.cap_t. Without unmark, these pages
                # leak permanently — alloc_lock/race.py test_2 reproduces this.
                try:
                    token.src_act.allocator.unmark_pages_capped(token.cap_t)
                except Exception:
                    logger.exception(
                        "BudgetAgent fire worker: rollback failed too — "
                        "%d pages may be permanently leaked from src pool",
                        int(token.cap_t.numel()),
                    )
                logger.exception(
                    "BudgetAgent fire worker: execute_async raised; "
                    "cap_barrier rolled back, discarding this fire"
                )
                continue
            completion_record = {
                "ts": round(time.time(), 3),
                "fire_completion": True,
                "fire_direction": result.direction,
                "fire_aborted": result.aborted,
                "fire_abort_reason": result.abort_reason,
                "fire_plan_seq": result.plan_seq,
                "fire_unmapped_pages": result.unmapped_pages,
                "fire_granted_pages": result.granted_pages,
                "fire_cap_barrier_us": result.cap_barrier_us,
                "fire_unmap_us": result.unmap_us,
                "fire_map_us": result.map_us,
                "fire_total_us": result.total_us,
                "fire_shared_free_before": shared_free_before,
                "fire_shared_free_after": shared_free_after,
                # Stage breakdown for win-attribution: how many of this fire's
                # freed pages came from cold-cache Drain vs LIVE-slot Migration
                # (vs plain anywhere-free). migrate_moves>0 ⟺ Stage-3 KV
                # migration actually fired.
                "fire_drain_pages": len(token.plan.drains),
                "fire_migrate_moves": len(token.plan.migrations),
            }
            # Producer-side wiring for the c^xfer EWMA. Skips aborted fires so
            # a rolled-back cap_barrier doesn't pollute the curve.
            if not result.aborted and result.granted_pages > 0:
                try:
                    from sglang.srt.budgeter.cost_model import get_cost_model
                    get_cost_model().update_xfer(
                        total_us=float(result.total_us),
                        n_chunks=int(result.granted_pages),
                    )
                except Exception:
                    logger.exception(
                        "cost_model.update_xfer failed (non-fatal); EWMA may "
                        "underestimate future fire costs."
                    )
            # Write the completion record as its own JSONL line so
            # validators (D7) that scan per-fire byte-transfer
            # invariants see the unmap/grant counts without having
            # to correlate with the enqueue-tick line.
            if self.log_enabled:
                try:
                    self._log_fp.write(
                        json.dumps(completion_record, default=_json_default) + "\n"
                    )
                except Exception as e:
                    logger.warning(
                        "BudgetAgent fire worker: log write failed: %r", e
                    )
