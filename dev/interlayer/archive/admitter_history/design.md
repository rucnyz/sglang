# Admitter design (synthesized from audits)

## Spec recap (design.md §358)

On each arrival with demand X bytes for destination pool `i_dst`,
evaluate five candidates and pick the cheapest:

```
own-free:    cost = 0
own-evict:   cost = c^evict_dst(X)
cross-free:  cost = c^xfer(X)
cross-evict: cost = c^xfer(X) + c^evict_src(X)
defer:       cost = Q · w_q
```

Tie-break preference: own-free > cross-free > own-evict =
cross-evict (lowest cost wins).

## What exists & what's missing (audit synthesis)

| Component | State | Need |
|---|---|---|
| Hook into scheduler arrival | none | hook at `_add_request_to_queue:2212` |
| Demand X derivation | implicit (used by `PrefillAdder`) | reuse `extend_input_len` (post-cache) or `len(origin_input_ids)` (pre-cache) |
| `c^xfer` EWMA | scaffold exists in `cost_model.py:218` | wire producer from `_fire_worker_loop` |
| `c^evict_i(X)` | none | new incremental sorted-index in radix cache OR per-tick refresh |
| `c_i(s)` | curves struct exists; Stage-0 calibration script missing | use `BUILTIN_DEFAULT` for now; add calibration later |
| `w_q` | hardcoded 100 µs/req | reuse env knob `SGLANG_XPOOL_QUEUE_WAIT_US` |
| Defer | implicit (already in waiting_queue) | no change |
| Sync fire path | `XPoolActuator.execute(plan)` exists | + serialize against Budgeter worker + reservation step |

## Architecture

### Module layout

```
python/sglang/srt/budgeter/
├── admitter.py              (NEW) per-arrival decision + sync trigger
├── cost_model.py            (EXISTING) extend with c_evict facade + EWMA wiring
├── agent.py                 (PATCH) wire cost_model.update() after execute_async
├── xpool_actuator.py        (NO CHANGE) reuse execute()
└── fire_planner.py          (NO CHANGE) reuse build()
```

### Admitter API

```python
class Admitter:
    """Per-arrival cost-driven admission decision."""

    def __init__(self, scheduler, actuator, cost_model, owner_provider,
                 fire_planner):
        ...

    def decide(self, req: Req) -> AdmitterDecision:
        """Called by scheduler at _add_request_to_queue.
        Returns one of: own_free, own_evict, cross_free, cross_evict, defer.
        For cross_*, synchronously fires the actuator BEFORE returning.
        """
```

```python
@dataclass
class AdmitterDecision:
    action: str  # 'own_free' | 'own_evict' | 'cross_free' | 'cross_evict' | 'defer'
    reason: str
    candidate_costs_us: dict  # for logging / D6n test gate
    fire_result: Optional[FirePlanResult]  # if synchronous fire was triggered
```

### Hot-path budget

The Admitter runs on the **scheduler thread**, on every arrival.
Target ≤ **100 µs** P99. Sub-targets:

- demand X computation: <5 µs (1 attribute lookup)
- `c_xfer(X)`: <5 µs (already EWMA read + multiply)
- `c_evict_dst(X)`: <50 µs (sorted-index proposal; or 1-tick stale)
- `c_evict_src(X)`: <50 µs (same)
- defer cost: <5 µs (just queue-len lookup × w_q)
- arg-min + tie-break: trivial

Sync fire path (when triggered) is OUTSIDE the 100 µs budget:
budgeted at ≤ 1 ms (per design.md §382, ~10 µs/page).

### Cost model facade

Add to `cost_model.py`:

```python
class CostModel:
    def c_xfer_us(self, n_pages: int) -> float:
        """EWMA cost of moving n_pages via the actuator."""

    def c_evict_us(self, pool: Literal['kv', 'mamba'], x_tokens: int) -> float:
        """Expected recompute cost if evicting x_tokens cheapest blocks
        from pool. Returns +inf if pool has insufficient evictable
        capacity."""

    def c_recompute_us(self, pool: str, s_tokens: int) -> float:
        """Re-prefill wall for a sequence of length s. From CostCurves."""

    def w_q_us(self) -> float:
        """SLO penalty per req of queue wait."""

    def is_warmed_up(self) -> bool:
        """Returns False until c_xfer EWMA has >=3 observations.
        Admitter must NOT fire cross-* until True; degrades to
        own-free / own-evict / defer."""
```

### Two-pass eviction-cost approximation

For the hot path, the audits suggest:

**Pass 1 (per tick, lazy):** A background snapshot of evictable
leaves sorted by `eviction_priority`, with prefix-sum cache.
Refreshed on Budgeter tick (1 Hz). Per-snapshot cost: O(N log N)
once per second.

**Pass 2 (per arrival, hot):** Binary search the snapshot for X
tokens, return prefix sum. Sub-µs.

Staleness is bounded by 1 s. Worst-case error: a cache block was
just evicted and the snapshot says it's still there → admitter
overestimates evictable capacity → may trigger an extra fire.
Bounded by 1-tick worth of churn. Acceptable.

### Sync fire path safety

Per `audit_sync_fire_path.md`:

1. **Serialize against Budgeter worker**: the Admitter's fire MUST
   not race with `_fire_worker_loop`'s in-flight fire on
   `SharedHandlePool._free_handles`. Approach:

   - Add a `_fire_inflight: threading.Lock` on `XPoolActuator`
     (NEW). Both Admitter and `_fire_worker_loop` acquire it before
     `execute_async`. Lock hold time ≤ 1 ms; not a hot-path concern.

2. **Reserve granted slots**: after `execute()`, the new dst capacity
   is visible to other waiting arrivals. The triggering arrival
   needs the slots reserved BEFORE its admission resumes.

   - The Admitter, holding the dst allocator's `_alloc_lock`,
     reads `free_pages[:X]` immediately after `execute()` returns
     and IMMEDIATELY allocates them to the triggering req.

3. **Minimum plan size**: `≥ lcm(n_src_subpools, n_dst_subpools)`
   pages, OR round X up to that.

### Cost-curve bootstrap (chicken-and-egg)

The cost model needs ≥3 fires to warm up `c^xfer` EWMA. But the
Admitter is the SOURCE of those fires.

**Bootstrap protocol:**
- On boot, `is_warmed_up() = False`
- Until warmed up, `c^xfer = SGLANG_XPOOL_NB_CHUNK_COST_INIT_US`
  (default 3000 µs/page, conservative)
- This is HIGH → cross-free will lose to own-evict on most arrivals
- BUT: when `own-evict = +∞` (no evictable cache) AND no other
  option remains, the Admitter PROBES with cross-free even
  unwarmed. After 3 such probes, `is_warmed_up() = True` and
  steady-state behavior takes over.

This is the cold-start mechanism mentioned in design.md §356.

## Edge cases

| Case | Admitter behavior |
|---|---|
| Both pools full, no evictable | defer (queue cost ≪ +∞) |
| `is_warmed_up() = False` AND own-evict possible | own-evict (conservative) |
| `is_warmed_up() = False` AND own-evict impossible | cross-free PROBE (warm up) |
| KV-only model (no mamba) | only own-free / own-evict / defer; no cross-* |
| Disagg mode | TBD — `_add_request_to_queue` is NULL-disagg only per audit; disagg path doesn't hit Admitter for now |
| Budgeter fire in flight | Admitter blocks on `_fire_inflight` lock |
| Admitter fire in flight, Budgeter wants to fire | Budgeter sees lock held, defers to next tick |

## Logging

`SGLANG_ADMITTER_LOG=path` → JSONL with one record per `decide` call:

```json
{
  "ts": 1234567890.123,
  "req_id": "abc",
  "x_tokens": 256,
  "action": "cross_free",
  "candidate_costs_us": {
    "own_free": null, "own_evict": 12000, "cross_free": 3000,
    "cross_evict": 15000, "defer": 200
  },
  "fire_result": {"granted_pages": 2, "wall_us": 800},
  "warmed_up": true
}
```

D6 / D6n / D11 validators parse this to assert the chosen action.

## Risks

- **Cost-model garbage in → admission garbage out.** Stage-0
  calibration script is missing; we ship with `BUILTIN_DEFAULT`
  curves tuned for one HW SKU. Phase 5+ should add calibration.
- **`c^evict_i(X)` snapshot staleness.** 1-Hz refresh may diverge
  during bursts. Mitigation: probe-and-adjust EWMA on observed vs
  predicted post-eviction L.
- **Lock contention.** `_alloc_lock` + new `_fire_inflight` create
  3 lock acquisitions per cross-* fire. Bench under burst load.
- **Single-thread bottleneck.** Admitter is on scheduler thread.
  Per-arrival 100 µs × 1k arrivals/s = 100 ms/s = 10% scheduler
  budget. Acceptable but borderline. Profile in D6.

## Phasing (see plan.md)

1. Cost-model facade + c^xfer producer wiring + warm-up gate
2. Skeleton Admitter with own-free / own-evict / defer only (no
   cross-* fires) + unit tests for decision math
3. `c^evict_i(X)` snapshot + prefix-sum cache + unit tests
4. Sync fire path (cross-free / cross-evict) + serialization mutex
5. Scheduler hook at `_add_request_to_queue` + per-arrival logging
6. D6 (live), D6n (unit + live), D3 (live) validation
7. D10 re-run with Admitter; expect cost-model warm-up + headline win

Each phase has its own tests landed BEFORE implementation (TDD).
