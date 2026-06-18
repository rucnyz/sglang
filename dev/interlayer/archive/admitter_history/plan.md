# Admitter phased implementation plan

TDD: each phase lands tests BEFORE the implementation. Tests are
deep — they exercise the decision logic, not mocks.

## Phase 0 — Audits (DONE)

- [x] `audit_scheduler_hook.md` — hook point + demand X derivation
- [x] `audit_cost_model.md` — c^xfer wired, c^evict missing, etc.
- [x] `audit_radix_evict.md` — incremental sorted-index needed
- [x] `audit_sync_fire_path.md` — execute() reusable, but lock + reservation needed

## Phase 1 — Cost model facade + c^xfer producer wiring

### Goal

Add `CostModel` facade class with `c_xfer_us`, `c_recompute_us`,
`w_q_us`, `is_warmed_up`. Wire the c^xfer EWMA producer (calling
`update()` after each successful fire).

### Files

- `python/sglang/srt/budgeter/cost_model.py` — extend with `CostModel` facade
- `python/sglang/srt/budgeter/agent.py:~824` — call
  `cost_model.update_xfer(result.total_us, result.granted_pages)`
  after each non-aborted `execute_async`
- `dev/interlayer/admitter/test_cost_model_facade.py` — unit tests

### Tests (deep)

1. **EWMA convergence**: feed 10 samples of 1000 µs each → `c_xfer_us(1)`
   should be ≈ 1000 µs (α convergence)
2. **Warm-up gate**: returns False until 3 obs, True after. Exact boundary.
3. **`c_recompute_us` reads from CostCurves**: matches `c_kv_ms(L)` for
   KV pool, `c_m_ms(L)` for mamba pool
4. **`w_q_us` reads env**: SGLANG_XPOOL_QUEUE_WAIT_US override works
5. **Producer wiring**: simulate a fire (without actual cuMem ops),
   verify EWMA updated correctly

### Acceptance

- 5/5 tests pass
- No new D9b regression (c^xfer EWMA still self-suppresses spikes)

## Phase 2 — Skeleton Admitter (own-free / own-evict / defer)

### Goal

Land `admitter.py` with the no-cross-pool decision logic. Cross-*
candidates return cost = +inf for now. Lets us prove the framework
without the sync fire path complexity.

### Files

- `python/sglang/srt/budgeter/admitter.py` — `Admitter` class
- `dev/interlayer/admitter/test_admitter_no_cross.py` — unit tests

### Tests (deep)

1. **own-free wins when dst pool has free capacity**: stubbed
   allocator, X tokens demand < free → action='own_free'
2. **own-evict wins when own-free unavailable but evictable cache
   exists**: stubbed evict-cost < defer cost → action='own_evict'
3. **defer wins when nothing else works**: simulate full pool, no
   evictable → action='defer'
4. **Q · w_q computed correctly**: vary Q, verify defer cost scales
5. **No cross-* fires when not yet wired**: action never
   'cross_free' or 'cross_evict'
6. **Decision returned in <100 µs**: timing assertion on 1000 reps

### Acceptance

- 6/6 tests pass
- Performance: P99 decide() < 100 µs on the test bed

## Phase 3 — `c^evict_i(X)` snapshot + prefix-sum cache

### Goal

Implement the per-tick lazy snapshot of evictable leaves, sorted by
cost-per-token, with prefix-sum cache for O(log N) c_evict_us(X)
queries.

### Files

- `python/sglang/srt/budgeter/cost_model.py` — `EvictCostIndex` class
- `python/sglang/srt/budgeter/agent.py` — refresh index in `tick()`
- `dev/interlayer/admitter/test_evict_cost_index.py` — unit tests

### Tests (deep)

1. **Empty radix tree → c_evict = +inf**
2. **Single block → c_evict(X >= block size) = hit_prob × c_i(s)**
3. **Multi-block, sorted by cost → c_evict(X) walks cheapest first**
4. **Prefix sum correct after rebuild**: rebuild, verify against
   linear scan
5. **Lock-ref >0 excludes block**: block locked → not in index →
   c_evict ignores
6. **Refresh under 1 ms for 10⁴ blocks**: performance benchmark

### Acceptance

- 6/6 tests pass
- Refresh wall < 1 ms for 10⁴ blocks
- Query wall < 5 µs

## Phase 4 — Sync fire path

### Goal

Wire cross-free and cross-evict candidates. Add
`_fire_inflight` mutex on `XPoolActuator` to serialize against
`_fire_worker_loop`. Add reservation step.

### Files

- `python/sglang/srt/arena/xpool_actuator.py` — add `_fire_inflight` lock
- `python/sglang/srt/budgeter/agent.py:_fire_worker_loop` — acquire lock
- `python/sglang/srt/budgeter/admitter.py` — sync fire path
- `dev/interlayer/admitter/test_sync_fire.py` — unit tests

### Tests (deep)

1. **cross_free fires execute() and returns granted_pages**
2. **cross_evict fires src evict then execute()**
3. **Concurrent fire test**: thread A (Admitter) and thread B
   (Budgeter worker) both try to fire → serialized, no race
4. **Reservation works**: after sync fire, the triggering req sees
   the freshly-mapped pages reserved
5. **Min-LCM page rounding**: X < lcm → round up to lcm
6. **Sync fire wall ≤ 5 ms** for a 2-page transfer

### Acceptance

- 6/6 tests pass
- D9b (c^xfer EWMA spike test) still passes (no degradation)

## Phase 5 — Scheduler hook + logging

### Goal

Hook `Admitter.decide(req)` into
`scheduler.py:_add_request_to_queue:2212`. Add
`SGLANG_ADMITTER_LOG=path` JSONL output.

### Files

- `python/sglang/srt/managers/scheduler.py:_add_request_to_queue`
- `python/sglang/srt/budgeter/admitter.py` — JSONL log writer
- `dev/interlayer/admitter/test_scheduler_hook.py` — integration test

### Tests

1. **Hook is gated by SGLANG_BUDGETER=1**: when off, decide() never called
2. **Per-arrival latency**: P99 decide() < 100 µs under N=10⁴ reqs
3. **JSONL log records all decisions**

### Acceptance

- 3/3 tests pass
- D8 saturated still PASS (no scheduler-hot-path regression)

## Phase 6 — D6 / D6n / D3 validation

### Goal

The actual live tests of design.md §616/§803/§883. Each defines a
synthetic workload where the EXPECTED decision is provable
analytically. We assert the Admitter picks that decision.

### Tests

- `dev/interlayer/verify/D6/` — Admitter picks cross-free when cheap
- `dev/interlayer/verify/D6n/` — Admitter prefers own-evict when src
  cache hot
- `dev/interlayer/verify/D3/` — cross-free dominant winner under
  Lemma A1 conditions

### Acceptance

- All three D6 family tests pass per their PASS criteria

## Phase 7 — D10 re-run with Admitter

### Goal

Now that the Admitter provides synchronous fire trigger, the cost
model warms up. D10 should show the headline win:

> At least one of {mean TTFT -3%, p99 TTFT -3%, output_tps +3%,
> cache_hit +1pp} AND fire_count > 5

### Acceptance

- D10 PASS per §1113 spec
- `dev/interlayer/verify/D10/README.md` updated

## Out of scope (future work)

- Stage-0 calibration script (`dev/eval/cost_model/calibrate.py`)
  for `c_i(s)` curves — currently using BUILTIN_DEFAULT
- D11 burst-recovery — Admitter handles synchronously, but the
  burst-test infra needs separate work
- Disagg-mode arrival path — Admitter is NULL-disagg only for now
