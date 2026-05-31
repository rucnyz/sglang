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
       │    a_t   ← kv_scheduler.decide(state, D_t, e)     │
       │    POST /aginfer/migrate(a_t)                     │
       │    program_tracker.advance(e)                     │
       │    admission.evaluate(state, e)                   │
       │      # runs every event (no internal timer)       │
       │                                                   │
       │  program_tracker — REASONING / ACTING / PAUSED   │
       │    state machine, driven from the same events     │
       └─────────┬──────────────────────────┬──────────────┘
                 │ proxied requests         │ /aginfer/* admin HTTP
                 ▼                          ▼
       ┌──────────────────────────────────────────────────┐
       │  sglang  (V4-Flash, TP=2, HiCache + Mooncake)     │
       │                                                   │
       │  inline scorer — V_u handler for the                │
       │    sglang-internal alloc-failure event            │
       │    (drive_eviction).  Same V_u rule as the daemon,│
       │    uses daemon-fed lambda/p_hat hints when fresh. │
       │                                                   │
       │  admin endpoints:                                 │
       │    GET  /aginfer/state                            │
       │    POST /aginfer/migrate                          │
       │                                                   │
       │  outbound webhook (mandatory):                    │
       │    POST <daemon>/aginfer/event                    │
       │    on every scheduler step, fire on               │
       │    OK↔HIGH (theta_hi) and HIGH↔OK (theta_lo)      │
       │    transitions of allocator-truth HBM occupancy.  │
       └──────────────────────────────────────────────────┘
```

| Layer | Where | Trigger event | Decision granularity |
|---|---|---|---|
| inline scorer | inside sglang's `drive_eviction` | sglang-internal alloc-failure | eviction heap key for this one evict |
| kv_scheduler | daemon | workload events (8 kinds, §4) | batch of `(hash, target_tier)` over D_t |
| admission_controller | daemon | every workload event | per-program pause / resume |

All three use the **same V_u rule**.  They live where they live
because of **event ownership**, not as fallbacks for each other:
the alloc-failure event is internal to sglang and not exposed over
HTTP, so its V_u handler must live in-process; the 8 workload events
are visible to the daemon's proxy / sglang webhook, so their V_u
handlers live in the daemon.

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
| `PRESSURE_RESOLVED` | sglang webhook | allocator-truth HBM occupancy crossed `theta_lo` downward (hysteresis) |

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
| `LLM_PREFILL` | ∅ (no migrate; the event still advances `program_tracker` to REASONING and admission re-evaluates) |
| `TOOL_CALL_START` | session tail units of the caller (demote candidate while idle) |
| `TOOL_CALL_END` | session tail units of the caller (promote candidate, about to reuse) |
| `SUB_DISPATCH_BLOCKING` | parent tail + shared prefix |
| `SUB_DISPATCH_ASYNC` | shared prefix only |
| `MEMORY_PRESSURE` | top-k by regret across all units |
| `PRESSURE_RESOLVED` | top-k by regret across all units |

### Inputs to `_net_value`

`_net_value` takes four quantities defined below.  The system is
parameterised on these inputs; any replacement estimator must fit
this interface without changing `decide()` or the event / D_t
contract.

#### `p_hat(u, Δt)` — probability u is accessed within Δt

A unit's reuse probability is the **disjunction over its holders'
access probabilities, each conditional on that holder's observable
program state**.  Per holder s ∈ `u.session_ids`:

```
p_access(u, s, Δt)  depends on s.program_state:
  REASONING  →  high if u is in s's recent prefix tail;
                decays with turn-distance from current tail
  ACTING     →  ≈ P(s's tool returns within Δt)
                ETA pulled from event payload or tool registry
  PAUSED     →  0   (no access until admission resume)
  ENDED      →  0
```

Aggregated across independent holders:

```
p_hat(u, Δt) = 1 - ∏_{s ∈ u.session_ids} (1 - p_access(u, s, Δt))
```

Three consequences fall out of this definition:

* PAUSED holders contribute exactly zero — admission's decisions
  feed straight into the next V_u, no warm-up.
* Shared prefix held by N concurrent programs aggregates correctly
  via the product — no ad-hoc `1/N` weighting needed.
* ACTING holders' contribution depends on the **specific tool's
  ETA**, not a per-unit static rate.

`Δt` is **the candidate decision's `hold_time`** — the same number
fed in as the cost-side `hold_time` (below), not a separate input.

> *Note:* Common stochastic-process formulations (Poisson / Hawkes
> rate fit) collapse all of the holder-state information into a
> single λ; this formulation does not because the holder state is
> directly observable.

#### `hold_time` — duration the candidate decision is being scored over

For a transfer decision the scheduler is right now considering,
`hold_time` is the **physical time window the unit would stay in
the candidate tier** before its next likely access.  When an event
carries a per-decision ETA (e.g. `TOOL_CALL_START` payload includes
the tool's expected duration), the scheduler **must** use it as
`hold_time` — the daemon's per-unit average is not a substitute
when a sharper per-event signal exists.

* short ETA → transfer round-trip > savings → don't demote
* long ETA → demote profits

#### `h_τ(occupancy)` — per-byte holding cost at tier τ

Cost ∝ marginal value of a free byte at τ, observable from the
allocator's recent pressure trajectory.  Same allocator-truth view
the admission controller reads (§5 `pool_usage`).

#### `bw_free(σ, τ)` — free bandwidth on σ↔τ link

Live bytes-per-second idle on the physical transfer path.

### Event-specific context

The event payload may carry decision-relevant context the rule
must exploit beyond a unit's intrinsic state.  Concrete cases:

* **`TOOL_CALL_START`** carries the tool's expected duration (tool
  registry lookup or historical mean per tool name).  The scheduler
  plugs this as `hold_time` for every demote decision over the
  caller's session tail.
* **`SUB_DISPATCH_*`** carries whether the sub-agent inherits the
  parent's prefix — drives which units in D_t need to stay HBM
  vs are safe to demote.

`hold_time` is a **per-decision physical quantity**, not a per-unit
invariant.  Per-unit averages are an acceptable input only when no
per-event signal exists.

## 8. Admission

Admission is a decision function: "given current `(programs,
HBM pressure, paused queue)`, which active programs should be
paused and which paused programs resumed?".  It must run on **every
event whose information could change that answer**, not just on a
single "pressure crossed" trigger.

### Triggers

| event | why admission's decision can change |
|---|---|
| `SESSION_ARRIVAL` | new program enters → expected HBM demand rises; if free capacity < forecast demand, **pre-pause** a low-V_u program before contention starts (proactive) |
| `LLM_PREFILL` | program-state advances to REASONING; the inputs to `V_u_program` for that program have changed |
| `TOOL_CALL_START` | program voluntarily idles → **pause cost ≈ 0** at this moment (no reasoning state is being interrupted); the cheapest possible time to free this program's HBM if its V_u is low |
| `TOOL_CALL_END` | program returns and must reason → reconsider whether any currently-paused program now outranks one of the actives that just came back |
| `MEMORY_PRESSURE` | sglang reports allocator at `theta_hi` → reactive pause |
| `PRESSURE_RESOLVED` | allocator dropped below `theta_lo` → consider resuming paused programs in V_u order |

### Decision rule (one definition; reused per trigger)

```python
def admission_evaluate(state):
    # state.pool_pressure[HBM] is the allocator truth (§5).
    occ = state.pool_pressure[HBM]
    forecast = occ + forecast_inflight_demand(state)

    while forecast > theta_hi and active_programs(state):
        victim = argmin(active_programs, key=V_u_program)
        if marginal_pause_cost(victim) > marginal_relief_value(victim, occ):
            break
        pause(victim); forecast -= victim.hbm_footprint
    while paused and forecast < theta_lo:
        candidate = argmax(paused, key=V_u_program)
        if not capacity_fits(candidate, state): break
        resume(candidate); forecast += candidate.hbm_footprint
```

* `forecast_inflight_demand` adds expected HBM growth from
  REASONING programs that haven't peaked yet (and at
  `SESSION_ARRIVAL` time, the incoming program's estimated need).
* `marginal_pause_cost(p)` is near zero when p is ACTING (idle
  anyway), positive when REASONING (interrupting work).  This is
  what makes TOOL_CALL_START such a cheap pause opportunity.
* `V_u_program(p)` is the program-level aggregate of unit V_u.
  Note that under the conditional p_hat formulation (§7), PAUSED
  programs' units already contribute 0 to V_u, so PAUSED programs
  naturally sort to the bottom of resume priority.

### Program aggregation

`V_u_program(p) = Σ_{u ∈ p.units} V_u(u)`.  No `1/|session_ids|`
weight is needed: the conditional p_hat from §7 is already a
holder-product, so shared-prefix attribution is built in.

### Threshold parity

`theta_hi` (up-crossing) and `theta_lo` (down-crossing) form the
admission hysteresis band.  sglang's webhook and the daemon's
admission controller must read the SAME values; this is launched
from a single source per the §10 invariant.

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
| **Single-worker event loop**: handlers serialised; no concurrent migrate races; no internal timer for any layer (kv_scheduler, admission, forecast refresh) — every recomputation is on event arrival | asyncio queue + single consumer |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent migrate**: re-applying the same action returns 200 with `applied=0` and a `race:*` skip | sglang `apply_aginfer_migrations` |
| **Threshold parity**: `theta_hi` and `theta_lo` are sourced from a single config; sglang's webhook and daemon's admission controller never read divergent values | launch scripts pass aligned `--aginfer-theta-*` flags |
| **Webhook mandatory**: sglang's launch contract always passes `--aginfer-notify-url`; the admission trigger path is not optional | `launch_sglang_v4flash*.sh` |
| **Pool-truth admission**: admission reads `pool_usage.HBM.token_usage`, not `tier_usage` | daemon `admission_controller._hbm_occ` |
| **Tree-view V_u**: V_u migration scoring reads `tier_usage`, never `pool_usage` | daemon `OursGreedyPolicy._value` |
| **Layer-disableable**: kv_scheduler, admission, and HiCache each have an independent enable flag; turning one off forfeits only its contribution, never breaks the others | daemon CLI + sglang flags |
