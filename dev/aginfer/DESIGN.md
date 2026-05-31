# aginfer — design

External KV-cache scheduler for sglang serving multi-turn agentic
workloads.  This document specifies *what the system should be*, not
the path that got us here.  Where the implementation is a simplification
of what's described, the simplification is called out explicitly.

## 1. What it is

A daemon-side scheduler that decides, on every workload-relevant
event:

* which cache **units** belong in which **tier** (HBM / DRAM / Disk / Drop)
* which **programs** should be paused or resumed under memory pressure

The mechanism is **event-driven**, **per-unit value-based**, and
**externalised from sglang** so the inference engine stays general
while scheduling policy is free to iterate.

## 2. Workload model (the hard physical facts)

aginfer assumes the workload is multi-turn agentic inference:

* programs run tool-bound reasoning loops (typically tens of turns)
* each program accumulates KV state across turns; turn-tail value
  decays but isn't zero
* programs concurrently share large prefix material (system prompt,
  tool documentation, sub-agent context)
* tool execution puts a program off-GPU for 50 ms-30 s — its KV
  occupies HBM but is *idle* during that window
* program runtimes are heterogeneous (5x-100x dynamic range)

Three facts follow that drive every design choice:

1. **Per-unit KV value is highly non-uniform.**  Shared prefix used
   by 32 concurrent programs ≫ a 30-turn-old intermediate scratch
   from one program.  LRU treats them the same and gets it backwards.

2. **Cache value lives on a continuum, not binary keep/evict.**
   "1% chance of future reuse" doesn't deserve HBM but isn't worth
   dropping either.  Hardware exposes a tier hierarchy (HBM ≫ DRAM
   ≫ Disk in $/bandwidth); the decision space must match.

3. **KV is owned by programs.**  Per-unit micro-management can't
   express "pause this whole program because its tools are slow and
   its current value is below another program's".  A scheduler that
   can only evict bytes can't free coherent program slots.

## 3. Architecture

Three layers, three cadences (none of them periodic).

```
              harbor (agent trial containers)
                 │  OpenAI HTTP, extra_body.program_id
                 ▼
       ┌──────────────────────────────────────────────────┐
       │  aginfer-daemon                                   │
       │                                                   │
       │  proxy ── emits per-program lifecycle events ──┐  │
       │                                                ▼  │
       │  event_router (single-worker asyncio queue,       │
       │    serialised, idempotent)                        │
       │    │                                              │
       │    ▼                                              │
       │  on_event(e):                                     │
       │    state ← fetch /aginfer/state                   │
       │    D_t   ← decision_set(e.kind)                   │
       │    a_t   ← kv_scheduler.decide(state, D_t)        │
       │    POST /aginfer/migrate(a_t)                     │
       │    if e.kind ∈ {MEMORY_PRESSURE, PRESSURE_RESOLVED}: │
       │        admission.act(state)                       │
       │                                                   │
       │  program_tracker — REASONING / ACTING / PAUSED   │
       │    state machine, driven from the same events     │
       └─────────┬──────────────────────────┬──────────────┘
                 │ proxied requests         │ /aginfer/* admin HTTP
                 ▼                          ▼
       ┌──────────────────────────────────────────────────┐
       │  sglang  (V4-Flash, TP=2, HiCache + Mooncake)     │
       │                                                   │
       │  inline scorer (safety net)                       │
       │    — runs inside `drive_eviction`                 │
       │    — uses the same V_u rule with daemon-fed       │
       │      lambda/p_hat hints if available, otherwise   │
       │      cached LRU age                               │
       │                                                   │
       │  admin endpoints:                                 │
       │    GET  /aginfer/state                            │
      │    POST /aginfer/migrate                          │
       │                                                   │
       │  outbound webhook (mandatory):                    │
       │    POST <daemon>/aginfer/event                    │
       │    on every scheduler step, fire when the         │
       │    allocator-truth HBM occupancy crosses          │
       │    theta_hi up or back down                       │
       └──────────────────────────────────────────────────┘
```

| Layer | Where | Cadence | Decision granularity |
|---|---|---|---|
| inline scorer | inside sglang's `drive_eviction` | every alloc-failure | eviction heap key for this one evict |
| kv_scheduler | daemon | one per workload event | batch of `(hash, target_tier)` over D_t |
| admission_controller | daemon | one per pressure event | per-program pause / resume |

All three use the **same V_u rule**.  They differ in (a) what they
can act on, (b) what visibility they have, (c) what triggers them.

## 4. Events

Eight event kinds.  The daemon is **strictly reactive** to events;
the system has no internal timer.

| kind | emitted by | semantic |
|---|---|---|
| `SESSION_ARRIVAL` | proxy | first request of a new program |
| `LLM_PREFILL` | proxy | every chat completion call (state observation) |
| `TOOL_CALL_START` | proxy | program goes off-GPU to wait on a tool |
| `TOOL_CALL_END` | proxy | tool returned, program resuming |
| `SUB_DISPATCH_BLOCKING` | proxy | program dispatches a sync sub-agent |
| `SUB_DISPATCH_ASYNC` | proxy | program dispatches an async sub-agent |
| `MEMORY_PRESSURE` | sglang webhook | allocator-truth HBM occupancy crossed `theta_hi` upward |
| `PRESSURE_RESOLVED` | sglang webhook | crossed `theta_hi` downward |

Each event carries (`kind`, `session_id` if applicable, `payload`).
The payload may include event-specific context the decision rule can
exploit (see §7).

## 5. State surface — `/aginfer/state`

```json
{
  "page_size": int,
  "bytes_per_token": int,
  "time_counter": int,                   // monotonic access tick

  "tier_usage": {                        // RADIX-TREE view
    "HBM":  {"used_bytes": int, "cap_bytes": int},
    "DRAM": {"used_bytes": int, "cap_bytes": int},
    "DISK": {"used_bytes": int, "cap_bytes": int}
  },

  "pool_usage": {                        // ALLOCATOR-TRUTH view
    "HBM": {
      "used_bytes":      int,
      "cap_bytes":       int,
      "available_bytes": int,
      "evictable_bytes": int,
      "token_usage":     float           // = (cap - avail - evictable) / cap
                                          // matches sglang full_token_usage,
                                          // i.e. real pressure after eviction
    }
  },

  "units": [
    {"hash": str, "tier": "HBM"|"DRAM",
     "n_tokens": int, "n_bytes": int,
     "last_access_time": int, "hit_count": int,
     "session_ids": list[str]}
  ]
}
```

### Why two occupancy views

The radix tree contains **only committed prefix-shareable units**.
In-flight decode KV is allocator-owned but **not** in the tree.

* `tier_usage` (radix view) is the right input for **V_u migration
  value scoring** — V_u can only act on tree nodes, so the relevant
  cost is "how full is the tree's slice of HBM".
* `pool_usage` (allocator view) is the right input for **admission
  gating** — admission throttles whole programs, so the relevant
  pressure is "is the allocator running out of pages for new
  requests", which includes in-flight decode.

Mixing them either makes V_u over-eager (thinks HBM is empty when
it's actually decode-full) or makes admission asleep at the wheel
(thinks HBM is empty when it's actually pressured).  Two views,
two consumers, no overlap.

## 6. Action surface — `/aginfer/migrate`

Request:
```json
{"actions": [{"hash": str, "target_tier": "HBM"|"DRAM"|"DISK"|"DROP"}, ...]}
```

Response:
```json
{"applied": int, "applied_hashes": [str, ...],
 "skipped": [{"hash": str, "reason": str}, ...]}
```

### Skip-reason classes

* **`race:*`** — time-window race between daemon's state fetch and
  apply; tree mutated by concurrent evict / request.  Retryable.
  Daemon re-issues on the next event.
* **`promote_load_back_declined:<category>`** — `load_back()`
  declined cleanly, with `<category>` distinguishing the sub-cause
  (full-allocator alloc fail, SWA sub-pool evict short, etc.).
  Usually transient; surface for diagnosis.
* **`promote_raised:<exc>:<loc>:<msg>`** — load_back threw.
  Indicates an invariant break; investigate.
* **`unknown_target_tier:...`** / **`unsupported_tree_cache:...`** —
  contract violations; daemon misbehavior.  Halt loudly.

The daemon's retry / debug loop dispatches on the **class prefix**,
not the literal string.

## 7. Decision rule

Per event, the daemon constructs a **decision set** `D_t ⊆ units`
and runs `kv_scheduler.decide(state, D_t)`:

```python
def decide(state, D_t):
    plan = []
    for u in D_t:
        best_tier, best_score = u.tier, _net_value(u, u.tier, state)
        for τ in (HBM, DRAM, DISK, DROP):
            if τ == u.tier: continue
            if capacity_left[τ] < u.n_bytes: continue
            score = _net_value(u, τ, state)
            if score > best_score:
                best_score, best_tier = score, τ
        if best_tier != u.tier:
            plan.append((u.id, best_tier))
    return plan

_net_value(u, τ, state) = _value(u, τ, state) - migration_cost(u, u.tier, τ)
_value(u, τ, state)     = p_hat × (reload_from_DROP - reload_from_τ)
                          - h_τ(occupancy_of_τ) × bytes × hold_time
```

### D_t per event kind

| event | D_t |
|---|---|
| `SESSION_ARRIVAL` | shared prefix units (preload before first prefill) |
| `LLM_PREFILL` | ∅ (observation only; refresh program_tracker, no migrate) |
| `TOOL_CALL_START` | session tail units of the caller (demote candidate while idle) |
| `TOOL_CALL_END` | session tail units of the caller (promote candidate, about to reuse) |
| `SUB_DISPATCH_BLOCKING` | parent tail + shared prefix |
| `SUB_DISPATCH_ASYNC` | shared prefix only |
| `MEMORY_PRESSURE` | top-k by regret across all units |
| `PRESSURE_RESOLVED` | top-k by regret across all units |

### Inputs to `_net_value`

`_net_value` is parameterised by four quantities.  The ideal estimator
for each is:

| input | ideal estimator | current implementation |
|---|---|---|
| `p_hat(u)` — future reuse probability of unit u | online Bayesian update from observed accesses; bursty workloads use Hawkes (self-exciting) | `min(1.0, hits / age)` proxy — coarse |
| `λ(u)` — reuse rate (1 / expected reuse interval) | empirical Poisson / Hawkes fit per unit class | calibrated floor per program state (ACTING / REASONING) |
| `h_τ(occupancy)` — per-byte holding cost at tier τ as a function of how loaded τ is | live measurement: cost ∝ marginal value of a free byte at τ, observable from the allocator's recent pressure trajectory | static `cost × (1 + occupancy²)` |
| `bw_free(σ, τ)` — free bandwidth on σ↔τ link | live measurement: bytes/sec idle on the actual transfer path | static config default |

Inputs are crude proxies today; replacing them with the ideal
estimators is the highest-value scheduler improvement and **must
not require changing the decide() algorithm or the event/D_t
contract** — the system is parameterised on these inputs.

### Event-specific context

Events may carry decision-relevant context the rule can use beyond
the unit's intrinsic state.  Concrete case: `TOOL_CALL_START`
ideally carries the **tool's expected duration** (looked up from
the tool registry or historical mean) so `hold_time` for THIS
demote decision is the actual ETA — not the unit's global lambda.

* short ETA → demote not worth (transfer round-trip > savings)
* long ETA → demote profitable

This is the right model because hold_time is a per-decision physical
quantity, not a per-unit invariant.  When the event carries the
information, the scheduler must use it.

## 8. Admission

Triggered by `MEMORY_PRESSURE` (after kv_scheduler's per-unit
migrate has had a chance to relieve pressure).

```python
def admission_on_pressure(state):
    occ = state.pool_pressure[HBM]              # allocator truth
    if occ < theta_hi: return                   # kv_scheduler handled it
    while occ > theta_hi:
        victim = argmin_program(V_u_program(p) for p in active_programs)
        pause(victim)
        occ = recompute(state, paused={victim})
```

```python
def admission_on_resolved(state):
    while paused and state.pool_pressure[HBM] < theta_lo:
        candidate = oldest_paused_program
        if room_for(candidate, state):
            resume(candidate)
```

`V_u_program(p) = Σ_{u ∈ p.units} (V_u(u) / |u.session_ids|)`.

The `1 / |session_ids|` weight prevents shared prefix from being
double-counted across programs that reference it.  PAUSED programs
remain in the denominator because pausing one active program doesn't
free a unit still held by paused ones.

`theta_hi` and `theta_lo` are sglang/daemon-shared constants
(currently 0.85 / 0.70).  The sglang webhook uses `theta_hi` as
its OK↔HIGH threshold so the two systems agree on when admission
should be triggered.

## 9. Why two channels (unit migrate + program pause)

Two distinct levers cover the pressure spectrum:

* **unit migrate** *reorganises* existing KV across tiers.  Cost:
  one H↔D transfer per migrated unit.  Effective when there's
  *capacity to reorganise into*.
* **program pause** *throttles inflow*.  Cost: a program waits.
  Effective when there's no reorganisation that helps because
  inflow is the source of pressure.

Light pressure → unit migrate is enough.  Heavy pressure → migrate
exhausts options because every tier is full; pause must reduce
inflow.  The two are non-substitutable; both are needed.

Formally, the action space is the *union*
`A = {unit-level assignments} ∪ {program pause/resume}`; the MDP is
one decision problem, not two.  Implementation splits the modules
purely for code clarity.

## 10. Invariants

| invariant | enforced by |
|---|---|
| **Single-worker event loop**: handlers serialised; no concurrent migrate races | asyncio queue + single consumer |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent migrate**: re-applying the same action returns 200 with `applied=0` and a `race:*` skip | sglang `apply_aginfer_migrations` |
| **Threshold parity**: sglang's `--aginfer-theta-hi` and daemon's admission `theta_hi` are launched from the same value (currently 0.85) | launch scripts pass aligned flags |
| **Webhook mandatory**: sglang's launch scripts always pass `--aginfer-notify-url`; admission's trigger path is not optional | `launch_sglang_v4flash*.sh` |
| **Pool-truth admission**: admission reads `pool_usage.HBM.token_usage`, not `tier_usage` | daemon `admission_controller._hbm_occ` |
| **Tree-view V_u**: V_u migration scoring reads `tier_usage`, never `pool_usage` | daemon `OursGreedyPolicy._value` |
| **Inline-scorer safety net**: if the daemon is down or unreachable, sglang's eviction still uses the V_u rule via its inline scorer | `SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score` |

## 11. Failure modes

The system is designed so any single failure degrades to a
well-defined floor:

| failure | system degrades to |
|---|---|
| daemon process down | sglang's inline scorer alone (V_u rule, no program-level admission) |
| daemon up but `/aginfer/state` slow | event handler skips this round (no migrate); next event retries |
| daemon up but `/aginfer/migrate` 5xx | log + continue; the same hash is re-considered on the next event |
| webhook POST fails (network) | sglang retries 3× with exponential backoff; eventually drops |
| sglang scheduler crashes (subprocess exit) | watchdog restarts; daemon detects via /aginfer/state probe and re-syncs |
| inline scorer module fails to load | sglang halts at startup with a structured `kv_policy_loaded` error line — launch scripts grep and fail loudly |
| daemon thresholds drift from sglang's | webhook fires at sglang's `theta_hi` but daemon ignores below daemon's `theta_hi`; the launch contract prevents this by sourcing both from the same value |

Every layer is **independently disable-able** without breaking the
others — admission off + kv_scheduler on, HiCache off + admission
on, etc.  The composition is multiplicative in win, not in
correctness; turning a layer off only forfeits its incremental
contribution.

## 12. Where this design departs from prior work

Beyond the obvious (separate scheduler from engine), three choices
are non-obvious and intentional:

1. **Two-view occupancy.**  Standard cache schedulers conflate
   "what's in the cache structure" with "what the allocator says
   is full".  Agentic workloads diverge sharply between the two
   because in-flight decode KV inflates allocator pressure without
   appearing in the prefix tree.  Aginfer keeps both views
   addressable and routes each consumer to the correct one.

2. **Event-driven, not timer-driven.**  Polling at 5 s (TA-style)
   adds a configurable knob that always has wrong answers somewhere.
   Per-event reaction has no knob and lower latency at every load
   level.  The only event kinds not derivable from proxy state
   come from sglang via a mandatory webhook.

3. **Per-event decision set.**  The event already names a program
   and the program's recent state, so D_t doesn't need to scan
   all units — the event carries the right scope for free.  Plus
   event-specific context (ETA, sub-agent dispatch type) can
   inform the decision rule's per-decision parameters, not just
   which units to score.
