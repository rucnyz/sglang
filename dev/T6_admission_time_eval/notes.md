# T6 notes

## Three-piece delivery

| Part | File | Change |
|---|---|---|
| 1 | `python/sglang/srt/budgeter/__init__.py` | Process-singleton accessor `get_budget_agent()` so the alloc-time hook can find the agent without a tree_cache API change. |
| 2 | `python/sglang/srt/budgeter/agent.py` | `BudgetAgent.try_admission_time_fire(direction, n_chunks)` — synchronous on-demand fire. Skips cooldown / hysteresis (the request's wait penalty has already exceeded those guards by construction). Reentrancy-guarded. Logs commit/no-commit per call. |
| 3 | `python/sglang/srt/mem_cache/common.py` | One-line hook in `alloc_token_slots`: when allocator `available_size() < num_tokens`, call `try_admission_time_fire("rec_to_kv", 1)` BEFORE falling back to `evict_from_tree_cache`. |

## Verification

| Test | Path | Verifies | Result |
|---|---|---|---|
| `test_admission_time_fire.py` | direct method call | 6 cases: env-on commit, env-off no-op, planner-off no-op, actuator-no-commit, reentrancy guard, bad direction | PASS |
| `test_smoke.sh` | full engine boot | T1+T2+T3+T4+T5+T6 all 6 flags compose; "T6 admission-time fire enabled" log line confirms env path; 5 generates return cleanly | PASS (boot 110 s) |

## Honest scope

T6 ships the **on-demand fire mechanism** — admission detects shortfall,
calls `try_admission_time_fire`, the actuator runs immediately rather
than waiting for the 30 s control tick. It does NOT ship the full
per-request `min(c1, c2)` decision rule from paper §3.1.5; that
would require restructuring SGLang's admission policy in 5+ places.
The current implementation:

1. Hooks at `alloc_token_slots`'s `available_size() < num_tokens`
   guard, immediately before the existing `evict_from_tree_cache`
   call. If cross-pool fire commits, the existing eviction may not
   need to run as aggressively (more bytes available).
2. Uses a hard-coded direction ("rec_to_kv", n_chunks=1). The real
   c1/c2 comparison would pick direction adaptively per request.
   Adaptive direction is the natural T6.5 / T7 follow-on.

The mechanism + the wiring + the env path are what T6 commits to;
the adaptive policy is paper-modeling that runs against this surface.

## What this does NOT verify

The smoke does not exercise the hot path under admission saturation —
the `try_admission_time_fire` log line that's loaded by the hook only
fires when `available_size() < num_tokens`, which requires a workload
above the pool's free-block ceiling. T7 (M2 swarm conc=800) is that
workload; it'll grep server log for `T6 admission-time fire: dir=` and
count actual on-demand fires vs 30 s-tick fires.

The reentrancy guard wasn't stressed under multi-request concurrency
in a smoke (the per-request-thread admission path is single-threaded
in scheduler, so reentrancy is naturally bounded; the guard is
defense-in-depth).

## Status

T6 done. Stack now:

```
SGLANG_ARENA_SHARED=1                                # T1
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1                    # T2
SGLANG_SMART_OVERCAP=1                               # T3
SGLANG_ATOMIC_MIGRATION=1                            # T4
SGLANG_BUDGETER=1 + SGLANG_BUDGETER_XPOOL_PLANNER=1  # planner
SGLANG_ADMISSION_TIME_FIRE=1                         # T6
# T5 default 80 GiB VA headroom is implicit.
```

Real-workload validation: T7.
