"""Upper-bound BudgetAgent.tick() per-call cost (no GPU, no server).

The idle_no_regression byte_transfer path runs tick() per scheduler event-loop iteration (with
internal rate-limit: full work only every tick_interval_s seconds).
This test measures both paths:

  fast path:  tick() called within tick_interval_s window → early return
  slow path:  tick() called past the window → full snapshot + planner.decide

If total tick cost over a 180-second run is bounded (say <100ms), the
+3% TTFT regression we see in idle_no_regression CANNOT be from budgeter — must be
arena tensor backing or noise.

Uses a stub scheduler that mimics the attributes BudgetAgent reads.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

# Enable budgeter so __init__ takes the enabled path
os.environ["SGLANG_HIMA"] = "1"
os.environ["SGLANG_HIMA_TICK_S"] = "1.0"
os.environ["SGLANG_BUDGETER_LOG"] = "/tmp/dasync_tick_cost.jsonl"

# Need to import sglang BEFORE creating the agent
from sglang.srt.budgeter.agent import BudgetAgent


class _QueueCount:
    """Mimic sglang's QueueCount with a .total attribute."""
    def __init__(self, n): self.total = n


class _Stats:
    """Stub for SchedulerStats; covers all fields BudgetAgent reads."""
    max_total_num_tokens = 100000
    kv_used_tokens = 50000
    kv_evictable_tokens = 10000
    kv_available_tokens = 40000
    token_usage = 0.5
    full_token_usage = 0.5
    swa_token_usage = 0.0
    mamba_usage = 0.4
    cache_hit_rate = 0.1
    num_running_reqs = _QueueCount(4)
    num_queue_reqs = _QueueCount(0)
    num_paused_reqs = 0
    num_retracted_reqs = 0
    gen_throughput = 100.0


class _TreeCache:
    _admission_cumulative_evicted_tokens = 0
    def full_evictable_size(self): return 5000
    def mamba_evictable_size(self): return 100


class _Alloc:
    """Stub for token_to_kv_pool_allocator."""
    size = 100000
    live_size = 100000
    def available_size(self): return 40000
    def get_kvcache(self): return _KVPool()


class _KVPool:
    """Stub MHATokenToKVPool — agent reads mamba_pool from this."""
    mamba_pool = None  # mamba-less stub; agent will skip mamba paths


class _Scheduler:
    stats = _Stats()
    tree_cache = _TreeCache()
    token_to_kv_pool_allocator = _Alloc()


def main():
    sched = _Scheduler()
    agent = BudgetAgent(sched)
    assert agent.enabled, "Agent should be enabled"

    # ---- Fast path: ticks within rate-limit window (~99% of real calls) ----
    # tick() should early-return after time.time() - last_tick < tick_interval_s.
    print("\n--- Fast path: rate-limited early return (tick called frequently) ---")
    # Prime: one real tick to set _last_tick
    agent.tick()

    N = 100000  # 100K calls — matches scheduler event-loop rate over ~100s
    start = time.perf_counter()
    for _ in range(N):
        agent.tick()  # all should hit "now - last_tick < 1.0" path
    fast_total = time.perf_counter() - start
    fast_per = fast_total / N * 1e9  # ns
    print(f"  N={N} calls, total={fast_total*1000:.2f}ms, per-call={fast_per:.0f}ns")
    print(f"  Over 180s scheduler run at 1000Hz event loop: ~180K calls × {fast_per:.0f}ns")
    print(f"    = {180_000 * fast_per / 1e6:.2f}ms total overhead from fast-path ticks")

    # ---- Slow path: full snapshot + planner.decide ----
    # Force a real tick by advancing _last_tick into the past.
    print("\n--- Slow path: full snapshot + planner.decide (1 per second) ---")
    slow_times = []
    for _ in range(20):
        agent._last_tick = 0  # force full tick path
        start = time.perf_counter()
        agent.tick()
        elapsed = time.perf_counter() - start
        slow_times.append(elapsed)
    mean = statistics.mean(slow_times)
    stdev = statistics.stdev(slow_times)
    median = statistics.median(slow_times)
    print(f"  N=20 full ticks, per-tick: mean={mean*1000:.2f}±{stdev*1000:.2f}ms  "
          f"median={median*1000:.2f}ms")
    full_180_total = mean * 180
    print(f"  Over 180s (180 full ticks @ 1Hz): {full_180_total*1000:.1f}ms total")

    # ---- Total + verdict ----
    total_overhead_ms = (180_000 * fast_per / 1e6) + (full_180_total * 1000)
    pct_of_180s = total_overhead_ms / 180_000 * 100
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    print(f"  Total budgeter overhead over 180s run: ~{total_overhead_ms:.1f}ms")
    print(f"  As fraction of wall time:              {pct_of_180s:.4f}%")
    print(f"")
    if pct_of_180s > 1.0:
        print(f"  → Budgeter overhead is non-trivial ({pct_of_180s:.2f}% of wall);")
        print(f"    plausibly contributes to idle_no_regression's +3% TTFT.")
    else:
        print(f"  → Budgeter overhead is bounded < 1% of wall — CANNOT be the")
        print(f"    source of idle_no_regression's +3% TTFT. The +3% must be from arena tensor")
        print(f"    backing or measurement noise.")
    agent.close()


if __name__ == "__main__":
    main()
