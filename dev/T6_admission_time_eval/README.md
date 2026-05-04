# T6: Admission-time cost evaluation (move VMM ops off control loop)

Task #78. Paper §3.2.4: "`cuMemMap` and `cuMemUnmap` are issued only at
the admission gate, never inside compute kernels. The admission path
computes Eq.~\ref{eq:nb-direction}'s $\min(c_1, c_2)$ for the incoming
request, picks the cheapest source, and synchronously executes the
corresponding action."

## What's broken without T6

Today the cross-pool actuator only fires at the 30 s control tick.
That means:
- A request arriving at $t = 5\,\text{s}$ that needs more KV than the
  pool has currently mapped must **wait up to 25 s** for the next
  tick — admission queue grows in the meantime.
- The control tick isn't aware of the specific request's needs; it
  decides on aggregate trends. A burst of 10 KV-heavy requests in one
  scheduler step can't trigger an immediate fire.

## What T6 adds

`BudgetAgent.try_admission_time_fire(direction)` — synchronous "fire
NOW" entry point that the scheduler can call when an admission
attempt fails on a pool that has cross-pool slack:

```python
# Inside scheduler when admission can't satisfy a request
if budget_agent.try_admission_time_fire("rec_to_kv"):
    # actuator just moved bytes; retry alloc
    retry_alloc()
else:
    # no slack to take; queue the request
    queue.append(req)
```

Internally `try_admission_time_fire` runs the same `shrink_then_grow`
that the 30 s tick uses, with a "skip cooldown / hysteresis" override
since the request's wait penalty has already exceeded those guards by
construction.

## Implementation choice (narrow T6 scope)

Rather than rewiring the full per-request `min(c1, c2)` decision into
the admission hot path (which would require restructuring SGLang's
admission code in 5+ places), T6 ships a **lightweight hook**:

- `BudgetAgent.try_admission_time_fire(direction, n_chunks)` — public
  method that fires once if cross-pool conditions are favorable, else
  returns False. No-op when `SGLANG_ADMISSION_TIME_FIRE=0`.
- Add **one** call site in scheduler where admission detects
  insufficient capacity. The simplest safe insertion: in the
  scheduler's "low-water mark" eviction trigger, before falling
  back to `evict_from_tree_cache`, try the cross-pool fire first.

This keeps the on-demand-fire mechanism (T6's actual
contribution) without rewriting SGLang's admission policy. The
unified-cost decision rule from paper §3.1.5 remains the *modeling*
view; T6 is the *implementation* that lets the budgeter act on
admission events instead of only on the 30 s tick.

## Flag

`SGLANG_ADMISSION_TIME_FIRE=1` enables the on-demand fire path.
Default off — backward-compatible. Layered with T1–T5:

```
SGLANG_ARENA_SHARED=1
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1
SGLANG_SMART_OVERCAP=1
SGLANG_ATOMIC_MIGRATION=1
SGLANG_ADMISSION_TIME_FIRE=1
# (T5 default headroom 80 GiB stays implicit)
```

## How to verify

1. **Smoke** (`test/test_smoke.sh`): boot with all 6 flags on, serve a
   few prompts, confirm `T6 admission-time fire` log line is reachable
   (only fires when load triggers it; smoke alone won't, but the
   absence of crash on boot is the verification).
2. **Unit test** (`test/test_admission_time_fire.py`): direct-call
   `BudgetAgent.try_admission_time_fire` with a mock scheduler /
   actuator state, verify the call dispatches to the cross-pool
   actuator and returns the right bool.

End-to-end "admission-time fires actually shorten the queue" is T7.

## Status

- [x] Design note
- [x] `BudgetAgent.try_admission_time_fire(direction, n_chunks)`
  in `python/sglang/srt/budgeter/agent.py`; process-singleton
  registered in `BudgetAgent.__init__` via
  `python/sglang/srt/budgeter/__init__.py`
- [x] Single call site in `alloc_token_slots`
  (`python/sglang/srt/mem_cache/common.py`) hooked behind
  `SGLANG_ADMISSION_TIME_FIRE=1`
- [x] Smoke under T1–T6 flags: PASS, boot 110 s, "T6 admission-time
  fire enabled" log line confirmed
- [x] Direct unit test: 6 cases (env on/off, planner on/off,
  actuator commit/no-commit, reentrancy, bad direction) — PASS

## Followups

T7 (M2 swarm conc=800) is the workload that actually drives this
hook — bursts of sub-agent admissions overflow mamba pool, hook fires
cross-pool transfer immediately instead of waiting 30 s, queue drains
faster.
