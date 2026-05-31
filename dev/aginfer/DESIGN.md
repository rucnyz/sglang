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
       │    program_tracker.advance(e)                     │
       │    plan  ← joint_decide(state, e)   # §9          │
       │      # one knapsack-greedy pass over the union    │
       │      # action space {unit migrate} ∪ {program     │
       │      # pause/resume}; selection key V_u/byte      │
       │    apply(plan):                                   │
       │      POST /aginfer/migrate for unit actions       │
       │      proxy.gate for pause/resume                  │
       │                                                   │
       │  program_tracker — REASONING / ACTING / PAUSED   │
       │    state machine, driven from the same events     │
       └─────────┬──────────────────────────┬──────────────┘
                 │ proxied requests         │ /aginfer/* admin HTTP
                 ▼                          ▼
       ┌──────────────────────────────────────────────────┐
       │  sglang (inference engine; HBM↔DRAM via HiCache,  │
       │          DRAM↔Disk via Mooncake)                   │
       │                                                   │
       │  inline scorer — V_u handler for the              │
       │    eviction-decision callsite (synchronous, in-   │
       │    process; the daemon's HTTP round-trip is too   │
       │    slow for this event).  Reads V_u inputs from   │
       │    a hint table pushed by the daemon (§6).        │
       │                                                   │
       │  admin endpoints:                                 │
       │    GET  /aginfer/state                            │
       │    POST /aginfer/migrate                          │
       │    PUT  /aginfer/hints                            │
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
| inline scorer | inside sglang's eviction-decision callsite | sglang-internal: free-pages-now | which unit(s) to drop to satisfy this allocation |
| kv_scheduler | daemon | workload events (10 kinds, §4) | batch of `(hash, target_tier)` over D_t |
| admission_controller | daemon | every workload event | per-program pause / resume |

All three use the **same V_u rule**.  They live where they live
because of **event ownership**, not as fallbacks for each other:
the eviction-decision callsite is sglang-internal, fires
synchronously on the scheduler step, and cannot wait for an HTTP
round-trip to the daemon, so its V_u handler must be in-process.
The 10 workload events are surfaced over HTTP (proxy + sglang
webhook), so their V_u handlers live in the daemon.

## 4. Events

Ten event kinds.  The daemon is **strictly reactive** to events; the
system has no internal timer.

| kind | emitted by | semantic |
|---|---|---|
| `SESSION_ARRIVAL` | proxy | first request of a new program |
| `LLM_PREFILL` | proxy | every chat completion call (state observation) |
| `TOOL_CALL_START` | proxy | program goes off-GPU to wait on a tool |
| `TOOL_CALL_END` | proxy | tool returned, program resuming |
| `SUB_DISPATCH_BLOCKING` | proxy | program dispatches a sync sub-agent |
| `SUB_DISPATCH_ASYNC` | proxy | program dispatches an async sub-agent |
| `SUB_RETURN` | proxy | sub-agent terminated (parent tail becomes a promote candidate; child output may become a new shared `subagent_ctx` unit) |
| `SESSION_END` | proxy | program terminated (no more arrivals); its session-scoped units become demote/drop candidates and contribute 0 to future `p_hat` |
| `MEMORY_PRESSURE` | sglang webhook | allocator-truth HBM occupancy crossed `theta_hi` upward |
| `PRESSURE_RESOLVED` | sglang webhook | allocator-truth HBM occupancy crossed `theta_lo` downward (hysteresis) |

Each event carries (`kind`, `session_id` if applicable, `payload`).
The payload may include event-specific context the decision rule can
exploit (see §7) — e.g. `TOOL_CALL_START` carries the tool's
expected duration, `SUB_DISPATCH_*` carries the inherit-prefix flag.

`SESSION_END` is non-trivial to detect from the OpenAI proxy: the
last request looks like any other.  The proxy commits a program as
ENDED when the harbor / agent client closes its session explicitly
(out-of-band signal), or — as a fallback — when no arrival has been
seen for `T_idle` (this is the **only** place the system uses a
time-based decision; `T_idle` is a deployment constant, not a tuning
knob).

## 5. State surface — `/aginfer/state`

```json
{
  "page_size": int,
  "bytes_per_token": int,              // n_bytes = n_tokens × bytes_per_token
                                        // is a derived quantity, not authoritative
  "time_counter": int,                  // monotonic access tick

  "tier_usage": {                       // RADIX-TREE view
    "HBM":  {"used_bytes": int, "cap_bytes": int},
    "DRAM": {"used_bytes": int, "cap_bytes": int},
    "DISK": {"used_bytes": int, "cap_bytes": int}    // zero-stub until L3 wired
  },

  "pool_usage": {                       // ALLOCATOR-TRUTH view
    "HBM": {
      "used_bytes":      int,
      "cap_bytes":       int,
      "available_bytes": int,
      "evictable_bytes": int,
      "token_usage":     float,         // = (cap - avail - evictable) / cap
                                         // matches sglang full_token_usage
      // SWA hybrid attention exposes the underlying sub-pools.
      // For non-SWA models these fields are absent; the daemon
      // gates `token_usage` regardless and ignores the sub-pool detail.
      "full_token_usage":      float?,
      "swa_token_usage":       float?,
      "swa_used_bytes":        int?,
      "swa_cap_bytes":         int?,
      "swa_available_bytes":   int?,
      "swa_evictable_bytes":   int?
    }
  },

  "per_program_usage": {                // PER-PROGRAM-OWNED view
    "<program_id>": {
      "hbm": {
        "committed_bytes": int,         // share of radix nodes attributed to
                                         // this program, with 1/holders weight
                                         // for nodes shared across programs
        "inflight_bytes":  int          // bytes of in-flight decode KV
                                         // (request-owned, not yet committed
                                         //  to the radix tree)
      },
      "dram": {
        "committed_bytes": int          // host pool share, same attribution
      },
      "state": "REASONING"|"ACTING"|"PAUSED"|"ENDED"
    }
  },

  "units": [
    {"hash": str, "tier": "HBM"|"DRAM",
     "n_tokens": int, "n_bytes": int,
     "last_access_time": int, "hit_count": int,
     "session_ids": list[str]}
  ],

  // Diagnostic: emitted by sglang only when the loaded tree cache
  // class lacks dump_aginfer_state.  Daemon halts loudly on this —
  // running aginfer against an incompatible cache is a deployment
  // bug, not a graceful-degradation case.
  "unsupported_tree_cache": str?
}
```

### Why three views

The radix tree contains **only committed prefix-shareable units**;
in-flight decode KV is allocator-owned but **not** in the tree.
Three distinct consumers in §7 / §8 need three distinct slices:

* `tier_usage` (radix view) — input for **V_u migration value
  scoring**.  V_u acts on tree nodes; the relevant cost is "how
  full is the tree's slice of HBM".
* `pool_usage` (allocator view) — input for **admission's
  pressure trigger**.  Admission decides whether to act based on
  total HBM pressure (including in-flight decode the radix view
  misses).
* `per_program_usage` (program view) — input for **admission's
  victim selection**.  Pausing a program frees its committed
  share + prevents its future in-flight growth; admission needs
  per-program footprint to know **which** pause yields the most
  HBM bytes per unit V_u_program lost.

A two-view design (only `tier_usage` + `pool_usage`) is
under-determined: admission knows the pool is pressured but has
no principled way to compare candidate victims by HBM relief.
Walking `units` + `session_ids` and summing only counts the
committed share, so a runaway-decode program with 80 K in-flight
bytes but 6 K committed prefix looks *smaller* than a quiet
program with 8 K cold prefix — exactly the wrong victim.

Mixing any two also breaks: V_u over-eager (decode-full HBM
looks empty in tree view), admission asleep (allocator-pressured
HBM looks empty in tree view), or admission picks the wrong
victim (sees committed share, not real footprint).  **Three
views, three consumers, no overlap.**

### Why pull (per-event re-fetch), not push-mirror

Every event handler re-fetches `/aginfer/state` at entry rather
than maintain a local mirror updated by delta-push from sglang.
The pull model is correct under three conditions, all of which
hold here:

* Event rate is high enough (10²/s in active workloads) that
  state drift between fetches is bounded by ~10 ms — small enough
  that a decision on slightly-stale state strictly dominates "no
  decision".
* The pull is also the snapshot-resync mechanism for daemon
  restart; eliminating it from the steady-state path would force
  a separate snapshot endpoint anyway.
* sglang's `dump_aginfer_state` is allocation-light (bytes path,
  ~ms-scale on 10⁴-node trees); the cost is acceptable per event.

A delta-push channel would shave the per-event latency but at the
cost of two state-tracking implementations (daemon mirror +
sglang authoritative), and the events-mirror skew is exactly the
class of bug the always-fresh invariant exists to rule out.

## 6. Action surfaces

Two daemon → sglang endpoints, both write-only from the daemon's
perspective.

### `POST /aginfer/migrate` — apply tier transitions

Request:
```json
{"actions": [{"hash": str, "target_tier": "HBM"|"DRAM"|"DISK"|"DROP"}, ...]}
```

Response:
```json
{"applied": int, "applied_hashes": [str, ...],
 "skipped": [{"hash": str, "reason": str}, ...]}
```

#### Skip-reason classes (canonical list in code)

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

### `PUT /aginfer/hints` — push V_u inputs to the inline scorer

Request: `{"hints": [{"hash": str, "p_hat": float, "lambda": float, "stamp": int}, ...]}`

Response: `{"applied": int}`.

The inline scorer fires on sglang's allocation-decision callsite,
which cannot wait for an HTTP round-trip to fetch fresh V_u inputs
from the daemon.  Hints solve that with a daemon→sglang push: the
daemon pushes the latest `p_hat` / `λ` for any unit whose value
has changed beyond a threshold; sglang's inline scorer reads the
hint table when computing eviction order.

Lifecycle:

* **Unit birth** (sglang creates a new tree node): sglang seeds the
  hint table locally with `p_hat = 1.0, λ = λ_NEW_UNIT_DEFAULT`
  — newborn units are at-least-once reusable by construction
  (the request that created them is still in-flight).  Initialised
  in-process, no daemon round-trip.
* **Daemon refresh**: on every event the daemon scores units in
  D_t (§7) and pushes the resulting `p_hat` / `λ` via `PUT
  /aginfer/hints` for any unit whose value changed beyond a
  threshold.  sglang's hint table is overwrite-by-stamp.
* **Unit death** (eviction / drop): sglang clears the hint entry.

There is no "missing hint" case: every live unit has either the
sglang-side birth default or a daemon-refined value.  This means
the inline scorer always has the same V_u inputs the daemon
would have used at the time of the last event — it is not a
defensive LRU fallback.

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

_net_value(u, τ, state) = _value(u, τ, state)
                          - migration_cost(u, u.tier, τ)
                          - unavailability_cost(u, u.tier, τ)

_value(u, τ, state)     = p_hat × (reload_from_DROP - reload_from_τ)
                          - h_τ(occupancy_of_τ) × bytes × hold_time

unavailability_cost(u, σ, τ) =
    p_hat(u, transfer_time(σ, τ))           # access lands during transfer
  × P(serve-from-σ fails | σ-write-policy)  # 0 under write-through
  × (reload_from_σ - 0)                     # penalty if it does fail
```

**Under our write-through HiCache semantic the unavailability
cost evaluates to 0** — see Transfer-window semantics below.  The
term is kept in the formula so the math stays correct when the
underlying write policy changes (zero-copy moves, non-write-through
caches, etc.) without rewriting `_net_value`.

### Transfer-window semantics

When unit u is mid-migrate σ → τ and an access to u arrives, the
lookup MUST serve from σ; the in-flight σ → τ transfer continues
uninterrupted and the access pays `σ_latency` (not `τ_latency`).
The next access — after transfer completes — pays `τ_latency`.

This is option **(C) race-serve-from-source**, and it strictly
dominates the alternatives:

| | this access | next access | bytes wasted |
|---|---|---|---|
| (A) wait for transfer | σ_latency + transfer_time | τ_latency | 0 |
| (B) cancel transfer | σ_latency | σ_latency | partial transfer |
| **(C) race** | σ_latency | τ_latency | 0 |

(C) requires σ to remain valid throughout the transfer window —
guaranteed by the write-through write policy.  Under (C):
`P(serve-from-σ fails) = 0`, so `unavailability_cost = 0` in
every term of `_net_value`.

> **Planned (sglang code lag).**  Today sglang's `match_prefix`
> path takes the host-only segment via `init_load_back`, which
> synchronously queues a new `load_back` and lets the forward
> pass wait on the CUDA stream `producer_event.finish_event` —
> functionally option (A) "wait for transfer", not (C).  This
> adds `transfer_time` to TTFT instead of the formula's
> `unavailability_cost = 0`.  Closing the gap requires sglang
> to expose a "DRAM-serve while load_back in-flight" branch in
> lookup; the design's `unavailability_cost` form already covers
> both (A) and (C) by parameterising on `P(serve-from-σ fails)`.

If the write policy ever changes such that σ is invalidated
during transfer (e.g. zero-copy move), `P(serve-from-σ fails) =
1` and the unavailability term kicks in fully — without any
change to the decision rule.

### D_t per event kind

| event | D_t |
|---|---|
| `SESSION_ARRIVAL` | shared prefix units (preload before first prefill) |
| `LLM_PREFILL` | ∅ (no migrate; the event still advances `program_tracker` to REASONING and admission re-evaluates) |
| `TOOL_CALL_START` | session tail units of the caller (demote candidate while idle); promote-ahead is scheduled here too, timed by the tool ETA so the unit lands before the next prefill |
| `TOOL_CALL_END` | session tail units of the caller (promote-now if ahead-of-time promote didn't catch up; otherwise no-op) |
| `SUB_DISPATCH_BLOCKING` | parent tail + shared prefix |
| `SUB_DISPATCH_ASYNC` | shared prefix only |
| `SUB_RETURN` | parent tail (promote candidate) + the child's output that just materialised as a new `subagent_ctx` unit (decide initial tier) |
| `SESSION_END` | session-scoped units of the ending program (demote or drop, depending on remaining holders) |
| `MEMORY_PRESSURE` | top-k by regret across all units |
| `PRESSURE_RESOLVED` | top-k by regret across all units |

`TOOL_CALL_START` carries the tool's expected duration; the
scheduler schedules the promote-back action for `T_start +
tool_ETA - load_back_latency` so the unit is ready exactly when
the next prefill arrives.  Waiting until `TOOL_CALL_END` to start
the promote means the prefill races the in-flight transfer.

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

Aggregated under an **independence-across-holders** assumption:

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

**Assumption: holders' access events are conditionally independent
given each holder's program-state.**  This is exact for
unrelated programs and approximate for sub-agent fan-out (where a
parent and its children's accesses are correlated through the
join).  When the assumption fails, the product underestimates
`p_hat` (treats correlated events as independent → product is
smaller than true OR-probability).  Conservative direction: the
scheduler will demote slightly too eagerly, never promote too
eagerly.  See §7 "Event-specific context" for how
`SUB_DISPATCH_*` payload can correct this when known a priori.

Common stochastic-process formulations (Poisson / Hawkes rate
fit) collapse holder-state information into a single λ; this
formulation does not, because holder state is directly observable.

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

## 8. Admission — program-level candidate generator

Admission is the **program-level candidate generator** that feeds
`joint_decide` (§9).  It does not run its own decision loop; the
loop lives in §9 and consumes its candidates alongside the
unit-level candidates from §7.

For each event, the admission generator emits:

* one `Pause(p, cost, relief)` candidate for every ACTIVE program p
* one `Resume(p, gain, re_use)` candidate for every PAUSED program p

where:

```
cost(pause p, event)   = V_u_program(p) + marginal_pause_cost(p, event)
relief(pause p, state) = marginal_relief_value(p, state)

gain(resume p, state)   = V_u_program(p)                  # V_u recovered
re_use(resume p, state) = marginal_relief_value(p, state) # HBM re-occupied
```

§9 then merges these with §7's unit candidates and sorts the
union by `cost / relief` ascending (knapsack-greedy).

### Component definitions

* `V_u_program(p) = Σ_{u ∈ p.units} V_u(u)`.  The conditional p_hat
  from §7 is already a holder-product, so shared-prefix attribution
  is built in; no `1/|session_ids|` weight is needed.  PAUSED
  programs' units already contribute 0 to V_u, so PAUSED programs
  naturally sort to the bottom of resume priority.
* `marginal_pause_cost(p, event)` — work-loss penalty of pausing p
  **at this event's moment**.  Near zero when the event is
  `TOOL_CALL_START` for p (p was going off-GPU anyway); positive
  when p is mid-REASONING (interrupting in-flight decode).
* `marginal_relief_value(p, state)` — HBM bytes pausing p would
  free or prevent.  Read from `state.per_program_usage[p].hbm`:
  * **inflight_bytes** — released when p's current request
    completes / is preempted; the pause prevents the program's
    NEXT request from re-allocating these.
  * **committed_bytes** — the program's exclusive share of radix
    nodes; if pausing p drops `len(session_ids)` of a node to 0,
    the node becomes evictable.
* `capacity_fits(p, state)` — true iff resuming p would leave
  `forecast(state)` still ≤ `theta_hi` after re-incorporating p's
  HBM footprint.  Resume candidates failing this gate are omitted
  by the generator (not in the candidate set).
* `forecast(state)` — `state.pool_pressure[HBM].token_usage +
  forecast_inflight_demand(state)`.  The second term is the
  expected HBM growth before the next event — for REASONING
  programs, `Σ max_remaining_tokens(p) × bytes_per_token` bounded
  by `max_completion_tokens`; for SESSION_ARRIVAL, the incoming
  program's seed estimate.

### Triggers

The candidate set changes on **every event** because at least one
of (active set, paused set, V_u_program ranking,
marginal_pause_cost, forecast) shifts.  The §9 loop runs every
event regardless of kind; the kind only affects what
candidates appear (e.g. TOOL_CALL_START makes p's pause cheap by
zeroing marginal_pause_cost; SESSION_END removes p's candidates
entirely).

### Threshold parity

`theta_hi` (up-crossing) and `theta_lo` (down-crossing) form the
admission hysteresis band.  sglang's webhook and the daemon's
admission controller must read the SAME values; this is launched
from a single source per the §10 invariant.

## 9. Joint decision over a union action space

Each event triggers ONE decision function over the **union action
space** `A = {unit-tier migrations} ∪ {program pause/resume}`.
The two lever types attack different parts of HBM (radix vs
in-flight) but contribute to the same scalar resource budget, so
choosing between them must be **joint**, not sequential.

### Why not sequential migrate-then-pause

Sequential decomposition (run kv_scheduler first, then run
admission on the post-migrate state — what an earlier version of
this design did) is Gauss-Seidel coordinate descent: optimise
unit-actions holding pause-set fixed, then optimise pauses
holding unit-actions fixed.  Two failure modes:

* **Redundant pay**: migrate a unit out of HBM (cost = V_u of that
  unit), then realise pausing the unit's program would have freed
  it anyway.  Both lever costs charged, only one needed.
* **Wrong lever order**: a runaway-decode program with tiny radix
  footprint but huge in-flight bytes is invisible to unit-migrate
  (the in-flight bytes aren't in the tree).  Sequential burns
  migration on small wins, then pauses the big lever later.
  Joint would pause first and skip the migrations.

These aren't pathological corner cases — they happen whenever a
single program contributes both migrate candidates and pause
relief, which is the common case in agent workloads.

### Joint decide

```python
def joint_decide(state, event):
    candidates: list[Candidate] = []

    # Unit-level: for each unit in D_t(event), each candidate tier.
    for u in decision_set(event, state.units):
        for τ in (HBM, DRAM, DISK, DROP):
            if τ == u.tier: continue
            cost   = _value(u, u.tier, state) - _value(u, τ, state)
            cost  += migration_cost(u, u.tier, τ)
            cost  += unavailability_cost(u, u.tier, τ)
            relief = bytes_freed_by_migrate(u, u.tier, τ)
            if relief > 0:
                candidates.append(Migrate(u.id, τ, cost, relief))

    # Program-level: each ACTIVE program is a pause candidate;
    # each PAUSED program is a resume candidate.
    for p in active_programs(state):
        cost   = V_u_program(p) + marginal_pause_cost(p, event)
        relief = marginal_relief_value(p, state)
        if relief > 0:
            candidates.append(Pause(p.id, cost, relief))
    for p in paused_programs(state):
        if not capacity_fits(p, state): continue
        gain   = V_u_program(p)                 # value recovered
        re_use = marginal_relief_value(p, state) # HBM re-occupied
        candidates.append(Resume(p.id, -gain, -re_use))   # negative-cost item

    # Knapsack-greedy on V_u-per-byte (same key as §7 / §8 A4).
    candidates.sort(key=lambda c: c.cost / c.relief if c.relief > 0 else math.inf)

    bytes_needed = max(0, forecast(state) - theta_hi * pool_cap(state))
    plan, freed = [], 0
    for c in candidates:
        if c.relief > 0 and freed >= bytes_needed: break
        plan.append(c); freed += c.relief
        if c.cost < 0: continue                  # Resume always welcome while capacity_fits

    return plan
```

The selection criterion is **V_u-per-byte ascending**, identical
across migration and pause candidates.  Resumes appear as
negative-cost items (they restore V_u rather than cost it), so
they sort to the front whenever capacity admits them.

### What collapses out

* **Trade gate**: in the sequential version §8 needed an
  explicit `if marginal_pause_cost > marginal_relief_value: break`
  guard to refuse a bad pause.  In the joint version, the
  sort-by-V_u/byte selection naturally orders bad trades after
  good ones; the loop's `freed >= bytes_needed` early-out is the
  only stopping criterion.
* **Composition order**: there's no "run kv_scheduler first
  then admission" — the two kinds of action are interleaved by
  cost/byte ranking.  An admission pause can come BEFORE a
  migrate in the same plan if it's cheaper per byte.
* **Per-trigger D_t for admission**: admission considers every
  active / paused program every event; D_t only constrains the
  unit-migrate candidate generator.

### Modules vs decisions

`kv_scheduler` and `admission_controller` remain as **candidate
generators** for the two action classes — the code split is
honest about responsibility (unit-level scoring vs program-level
scoring) and lets each be unit-tested.  But the decision happens
once, by `joint_decide`, not twice.  The action-application step
still uses the two existing transport surfaces (`POST /aginfer/
migrate` for unit actions; proxy gate for program pause/resume).

## 10. Invariants

| invariant | enforced by |
|---|---|
| **Single-worker event loop**: handlers serialised; no concurrent migrate races; no internal timer for any layer (kv_scheduler, admission, forecast refresh) — every recomputation is on event arrival | asyncio queue + single consumer |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent migrate**: re-applying the same action returns 200 with `applied=0` and a `race:*` skip | sglang `apply_aginfer_migrations` |
| **Threshold parity**: `theta_hi` and `theta_lo` are sourced from a single config; sglang's webhook and daemon's admission controller never read divergent values | launch scripts pass aligned `--aginfer-theta-*` flags |
| **Webhook mandatory**: sglang's launch contract always passes `--aginfer-notify-url`; the admission trigger path is not optional | `launch_sglang_v4flash*.sh` |
| **Pool-truth admission**: admission reads `pool_usage.HBM.token_usage`, not `tier_usage`; if the snapshot lacks `pool_usage` the daemon halts loudly (sglang too old / misconfigured) | daemon `admission_controller._hbm_occ` |
| **Tree-view V_u**: V_u migration scoring reads `tier_usage`, never `pool_usage` | daemon `OursGreedyPolicy._value` |
| **All traffic through daemon proxy**: every chat-completion to sglang arrives via the daemon's `/v1/chat/completions` proxy; direct-to-sglang clients are out of scope and would render admission's program-pause unenforceable | deployment topology |
| **Hint table covers every live unit**: sglang seeds `(p_hat=1, λ=λ_NEW)` on unit birth; the daemon refines via `PUT /aginfer/hints`; eviction never falls back to LRU on absent hints | sglang allocator hook + daemon hint pusher |
| **Layer enable flags**: HiCache, kv_scheduler, and admission each have an independent enable flag.  Admission can only fire when kv_scheduler is also on (admission's pre-pause migrate path requires the daemon's V_u machinery).  HiCache is independent of both | daemon CLI + sglang flags |
