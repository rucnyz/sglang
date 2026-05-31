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

**Out of scope**: multi-tenant fairness / per-program priority /
per-tenant SLOs.  The design optimises throughput on the shared
HBM resource and does not enforce a starvation guard for low-V_u
programs.  Fairness is the responsibility of an upstream
admission gating layer above harbor (route limiter, priority
queue, etc.) that pre-shapes the workload the daemon sees.

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
       │      # exact 0/1 knapsack (DP) over the union     │
       │      # action space {unit migrate} ∪ {program     │
       │      # pause/resume}; selection key V_u/byte      │
       │    apply(plan):                                   │
       │      POST /aginfer/migrate for unit actions       │
       │      proxy.gate for pause/resume                  │
       │                                                   │
       │  program_tracker — REASONING / ACTING /           │
       │    PAUSED / ENDED state machine, driven from the  │
       │    same events.  Stores `pre_pause_state` on the  │
       │    REASONING/ACTING → PAUSED transition (consumed │
       │    by §8 Resume gain counterfactual).             │
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

| Layer | Where | Trigger event | Role |
|---|---|---|---|
| inline scorer | inside sglang's eviction-decision callsite | sglang-internal: free-pages-now | scoring function for the allocator's eviction heap (operates on the hint table) |
| kv_scheduler | daemon | workload events (10 kinds, §4) | **unit-migrate candidate generator** consumed by §9 `joint_decide` |
| admission_controller | daemon | every workload event | **program pause / resume candidate generator** consumed by §9 `joint_decide` |

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
| `MEMORY_PRESSURE` | sglang webhook | allocator-truth HBM occupancy crossed `theta_hi` upward (HIGH band); while in HIGH or CRITICAL, sglang re-fires every `heartbeat_s` (default 5 s) so the daemon stays caught up if it missed earlier events |
| `PRESSURE_CRITICAL` | sglang webhook | allocator-truth HBM occupancy crossed `theta_crit` upward (above HIGH); same payload shape as `MEMORY_PRESSURE`, signals the daemon to escalate (skip headroom-only optimisations, prefer Pause over migrate even at higher V_u cost) |
| `PRESSURE_RESOLVED` | sglang webhook | allocator-truth HBM occupancy crossed `theta_lo` downward (back to OK band, hysteresis) |

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
    {"hash": str, "tier": "HBM"|"DRAM"|"DISK",
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

* **Unit birth** (sglang creates a new tree node): sglang seeds
  the hint table locally with `p_hat = 1.0` and a near-term
  expected-use signal — newborn units are at-least-once reusable
  by construction (the request that created them is still
  in-flight).  Initialised in-process, no daemon round-trip.
  Initial tier is HBM by allocator semantics (see §7 "Initial
  tier for newly created units").
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

The hint table is read by the inline scorer concurrently with the
daemon's `PUT`s.  See the §10 "Hint atomicity" invariant for the
per-key seqlock / CAS requirement.

### Multi-rank protocol (TP / EP / DP)

The daemon talks to a single sglang HTTP endpoint regardless of
how the inference engine is parallelised.  Per-rank coordination
happens inside sglang.

| parallelism | per-rank HBM holds | daemon's view | actions |
|---|---|---|---|
| TP > 1 | same logical unit's **different head-dim slice** on each rank | sglang's tokenizer-server fans `/aginfer/state` / `migrate` / `hints` out to all rank schedulers; the snapshot returned to the daemon is aggregated across ranks (`tier_usage`, `pool_usage`, `per_program_usage` summed) | every action is **all-rank atomic by semantic requirement** — see below |
| EP > 1 | prefix KV mirrored across ranks (same as TP); only the MoE expert weights / activations differ per rank | same as TP from the daemon's perspective — expert weights aren't in the daemon's scheduling scope | same as TP |
| DP > 1 | each DP replica has its own independent KV pool serving its own program subset | each DP replica is a **separate sglang endpoint** with its own daemon-sglang pairing; no cross-replica daemon coordination | independent per replica |

### Why TP forces all-rank-atomic actions

TP attention computes `softmax(QK^T)V` with each rank handling
`1/N` of the heads.  Every rank must have the unit's KV slice in
HBM at compute time, otherwise the rank without it stalls
loading from DRAM while the others wait for the all-reduce.

→ The state "unit U is in HBM" is **binary across the whole
N-rank ensemble** — either all ranks have U in HBM or none do.
A migrate or pause must apply atomically across all ranks.

This is **not** a per-rank-actions-deferred limitation; it's the
TP semantic.  A future "rank-targeted action" would mean
"different ranks hold different units" — under TP attention that
breaks correctness, not just performance.

A meaningful "cross-rank HBM tier" doesn't exist either: each
rank's HBM stores its own head-dim slice, slices aren't
substitutable across ranks, NVLink/NCCL is used for compute
all-reduce not for storage tiering.  There is no useful "borrow
rank 1's HBM space for rank 0's units" operation.

### Hint consistency across ranks

When `PUT /aginfer/hints` fans out, ranks may apply with slight
skew (one rank in CUDA graph capture, etc.).  This is
**eventual-consistent by design**: an inline scorer using a
stale hint may make a slightly sub-optimal eviction; the worst
case is one missed unit recoverable via re-prefill.  No strong
consistency / global lock — at hint update rate (≤ event rate ≈
10²/s), the daemon's next push will reach the lagging rank
within milliseconds.

## 7. Decision rule

### Symbols and units

All cost-side quantities are in **seconds** (paper §3 Reward
units: time saved/paid against wall-clock).  All relief / re_use
quantities are in **bytes**.  The joint knapsack (§9) mixes
seconds-cost with bytes-budget; the comparison is well-typed
because the constraint is a one-sided byte threshold.

| symbol | unit | meaning |
|---|---|---|
| `u.n_bytes`, `u.n_tokens` | bytes, tokens | unit size; one-of derivable as `n_bytes = n_tokens × bytes_per_token` |
| `p_hat(u, Δt)` | unitless ∈ [0,1] | conditional reuse probability over horizon Δt |
| `Δt` | seconds | decision look-ahead horizon (§7 inputs) |
| `hold_time` | seconds | expected residence in candidate tier (§7 inputs) |
| `reload_from(u, τ)`, `reload_from_DROP(u)` | seconds | per-paper-§3 reload costs `ρ_τ × n_tokens`, `π_u × n_tokens` |
| `h_τ(occ)` | $/byte/sec | per-byte holding cost at tier τ at occupancy `occ`; multiplied by bytes×seconds to get seconds-cost |
| `bw_free(σ, τ)` | bytes/sec | live free bandwidth on σ↔τ link |
| `transfer_bytes(u, σ, τ)` | bytes | `u.n_bytes` if `τ ≠ DROP` else 0 |
| `transfer_time(σ, τ)` | seconds | `transfer_bytes / bw_free` |
| `page_bytes` | bytes | DP quantisation granularity = `state.page_size × state.bytes_per_token` |
| `cost`, `gain`, `V_u`, `V_u_program` | seconds | net value at the same time-axis |
| `relief`, `re_use`, `bytes_moved`, `bytes_needed` | bytes | HBM-resource axis |
| `forecast(state)` | unitless ∈ [0,1] | occupancy fraction (matches `pool_usage.HBM.token_usage`) |

### Candidate types

```python
@dataclass(frozen=True)
class Migrate:
    hash: str                 # u.hash (sglang canonical key)
    target_tier: Tier
    cost: float               # seconds
    relief: int               # HBM bytes freed
    bytes_moved: int          # bytes the σ→τ transfer moves

@dataclass(frozen=True)
class Pause:
    program_id: str
    cost: float               # seconds (V_u_program + marginal_pause_cost)
    relief: int               # HBM bytes freed

@dataclass(frozen=True)
class Resume:
    program_id: str
    gain: float               # seconds (V_u_program_if_active)
    re_use: int               # HBM bytes re-occupied
```

### kv_scheduler is a candidate generator

Per event, the daemon constructs a **decision set** `D_t ⊆ units`
and emits **unit-level migrate candidates** for the joint decider
in §9.  `kv_scheduler` is a **candidate generator only** — it does
not select a plan or apply actions; that is §9's job.

```python
def migrate_candidates(state, D_t) -> list[Migrate]:
    out = []
    for u in D_t:
        for τ in (HBM, DRAM, DISK, DROP):
            if τ == u.tier: continue
            cost   = _value(u, u.tier, state) - _value(u, τ, state)
            cost  += migration_cost(u, u.tier, τ)
            cost  += unavailability_cost(u, u.tier, τ)
            relief = bytes_freed_by_migrate(u, u.tier, τ)
            if relief > 0:                     # only HBM-relieving migrates
                out.append(Migrate(
                    hash=u.hash,
                    target_tier=τ,
                    cost=cost,
                    relief=relief,
                    bytes_moved=transfer_bytes(u, u.tier, τ),
                ))
    return out

# V_u value at a tier (paper §7, conditional-p_hat form §7 above).
# Δt and hold_time are DISTINCT inputs (see §7 below) — Δt is the
# decision look-ahead window for p_hat, hold_time is the expected
# physical residence in τ.  Under Poisson they collapse to 1/λ;
# under our conditional formulation they do not.
_value(u, τ, state) =
    p_hat(u, Δt) × (reload_from_DROP(u) - reload_from(u, τ))
  - h_τ(occupancy_of_τ(state)) × u.n_bytes × hold_time

# Reload costs (paper §3 Tier Parameters, renamed for legibility):
reload_from(u, τ)    = ρ_τ × u.n_tokens     # per-token reload at tier τ
reload_from_DROP(u)  = π_u × u.n_tokens     # full re-prefill cost

# Tier occupancy ratio (derived from §5 tier_usage):
occupancy_of_τ(state) = state.tier_usage[τ].used_bytes
                      / state.tier_usage[τ].cap_bytes

# Bytes the σ→τ transfer moves over the link:
transfer_bytes(u, σ, τ):
    if τ == DROP:  return 0   # drop is metadata-only
    return u.n_bytes

# σ→τ transfer time at current link utilisation:
transfer_time(σ, τ) = transfer_bytes(u, σ, τ) / bw_free(σ, τ)

# Cost of the σ→τ transfer itself (BW link contention):
migration_cost(u, σ, τ) = transfer_bytes(u, σ, τ) / bw_free(σ, τ)
                          # seconds the link is held; paid against
                          # the same time-axis as prefill savings

unavailability_cost(u, σ, τ) =
    p_hat(u, transfer_time(σ, τ))           # access lands during transfer
  × P(serve-from-σ fails | σ-write-policy)  # 0 under write-through
  × reload_from(u, σ)                       # penalty if it does fail

# HBM-relief axis for the joint knapsack: a candidate frees HBM
# bytes iff its source tier is HBM.  Migrates whose source is
# DRAM / DISK return 0 here and are filtered out by the §9 loop.
bytes_freed_by_migrate(u, σ, τ):
    return u.n_bytes if σ == HBM else 0
```

The `if relief > 0` filter is load-bearing: it keeps the §9
knapsack focused on the actual bottleneck resource (HBM bytes).
A future redesign that admits DRAM-pressure or DISK-pressure
events would extend `joint_decide` with parallel knapsacks per
resource axis; the generator here would gate on each axis.

**Under our write-through HiCache semantic the unavailability
cost evaluates to 0** — see Transfer-window semantics below.  The
term is kept in the formula so the math stays correct when the
underlying write policy changes (zero-copy moves, non-write-through
caches, etc.) without rewriting `_value`.

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
every term of `_value`.

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

### `decision_set(event, state)` — D_t per event kind

```python
def decision_set(event, state):
    """The unit subset for which §9 will solve a knapsack on this event.
    Matches paper §4 table.  Takes `state` (not just `units`) because
    the pressure-event branch needs regret + top-k sizing, both of
    which read from `state`."""
    e, sid, units = event.kind, event.session_id, state.units
    match e:
        case SESSION_ARRIVAL:        return shared_prefix_units(units)
        case LLM_PREFILL:            return frozenset()
        case TOOL_CALL_START:        return session_tail(units, sid)
        case TOOL_CALL_END:          return session_tail(units, sid)
        case SUB_DISPATCH_BLOCKING:  return session_tail(units, sid) | shared_prefix_units(units)
        case SUB_DISPATCH_ASYNC:     return shared_prefix_units(units)
        case SUB_RETURN:             return session_tail(units, sid) | {child_output_unit(event)}
        case SESSION_END:            return session_scoped_units(units, sid)
        case MEMORY_PRESSURE | PRESSURE_RESOLVED:
            return top_k_by_regret(units, state, k=top_k_pressure(state))
```

| event | rationale |
|---|---|
| `SESSION_ARRIVAL` | shared prefix candidates (preload before first prefill) |
| `LLM_PREFILL` | ∅ (no migrate; the event still advances `program_tracker` to REASONING and admission re-evaluates) |
| `TOOL_CALL_START` | session tail = demote candidate while idle; promote-ahead is scheduled here too, timed by the tool ETA so the unit lands before the next prefill |
| `TOOL_CALL_END` | session tail = promote-now if ahead-of-time promote didn't catch up; otherwise no-op |
| `SUB_DISPATCH_BLOCKING` | parent tail + shared prefix |
| `SUB_DISPATCH_ASYNC` | shared prefix only |
| `SUB_RETURN` | parent tail (promote candidate) + the child's output that just materialised as a new `subagent_ctx` unit (decide initial tier) |
| `SESSION_END` | session-scoped units of the ending program (demote or drop, depending on remaining holders) |
| `MEMORY_PRESSURE`, `PRESSURE_RESOLVED` | top-k by regret across all units |

#### `regret(u, state)` and `top_k_pressure(state)`

For the two pressure events, the candidate set is restricted by
**regret** — the V_u headroom we'd recover by moving u to its best
alternative tier:

```
regret(u, state) = _value(u, u.tier, state)
                 - max_{τ ≠ u.tier} _value(u, τ, state)
```

`top_k_by_regret(units, k)` returns the k units of highest regret.
`k` is sized so the joint knapsack stays microsecond-scale:

```
bytes_needed(state)   = max(0, forecast(state) - theta_hi × cap_total)
                        # = same scalar §9 computes; reused here
mean_unit_bytes(state) = (Σ_{u ∈ state.units} u.n_bytes) / |state.units|
                        # arithmetic mean across all units in HBM+DRAM+DISK

top_k_pressure(state) =
    min(K_MAX,
        max(K_MIN, bytes_needed(state) / mean_unit_bytes(state) × K_SAFETY))
```

Defaults: `K_MAX = 256`, `K_MIN = 16`, `K_SAFETY = 4`.  These are
deployment constants, not workload tuning knobs — they bound the
DP table size (§9), not the optimality of the chosen subset.
`V_u(u)` is the shorthand for `_value(u, u.tier, state)` —
the unit's value at its current tier under the live state.

`TOOL_CALL_START` carries the tool's expected duration; the
scheduler schedules the promote-back action for `T_start +
tool_ETA - load_back_latency` so the unit is ready exactly when
the next prefill arrives.  Waiting until `TOOL_CALL_END` to start
the promote means the prefill races the in-flight transfer.

### Inputs to `_value`

`_value` takes five quantities defined below.  The system is
parameterised on these inputs; any replacement estimator must fit
this interface without changing the candidate-generator contract.

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

#### `Δt` — decision look-ahead horizon for `p_hat`

`Δt` is "how far into the future a reuse counts as relevant for
this decision".  It is **distinct from `hold_time` below**: a
short Δt favours promotion (anything in this window is value-add)
while a long Δt dilutes p_hat (more time for the access to *not*
happen).

Estimator priority (highest signal first):

1. **Event payload ETA**: `TOOL_CALL_START` carries
   `tool_eta` → Δt = `tool_eta` (the access we care about is the
   one that fires when the tool returns).
2. **program-state expected event distance**: from the holder's
   `program_tracker` history, the expected time to its next LLM
   event.  For a REASONING program mid-decode this is one decode
   step; for an ACTING program waiting on a tool whose ETA isn't
   in payload, use the tool's historical mean.
3. **Global default**: average inter-event spacing (10 ms-1 s
   range on agent workloads).  Only used when nothing better is
   known.

#### `hold_time` — expected duration in the candidate tier

`hold_time` is "how long the unit will actually stay in τ before
the next migrate/evict/drop reconsiders it".  This is the time
the `h_τ(occupancy) × bytes × hold_time` holding tax integrates
over — a **physical residence time**, not a prediction window.

Under Poisson-rate models the two quantities collapse to `1/λ`
because every access is an evaluation opportunity, and λ governs
both reuse frequency and reconsideration frequency.  The
conditional `p_hat` formulation doesn't share that property:
units are re-evaluated by the daemon's event loop (every event
where the unit is in D_t), which is **independent of the unit's
own access pattern**.  hold_time tracks the *evaluation* cadence;
Δt tracks the *access* horizon.

Estimator priority:

1. **Time until next event with u ∈ D_t**: estimable from
   D_t selection rules (§7) + the per-event-kind expected
   inter-event distance for the relevant holders.
2. **Coarse default**: average inter-event spacing across all
   events (same fallback as Δt's level 3).  In steady state on
   busy agent workloads, hold_time ≈ Δt; the two diverge sharply
   when the unit is held by a quiet program (long stretches with
   no D_t-relevant event).

The two are typically close on a busy workload but **must be
estimated independently**; conflating them is the Poisson
simplification the rest of §7 has already dropped.

#### Per-event override

When the event payload carries a sharper signal than either
estimator produces, the per-event signal wins.  Concrete:
`TOOL_CALL_START`'s tool ETA sets both Δt and hold_time for
demote decisions over the caller's session tail (the unit is
about to be idle for ETA and the relevant reuse window is also
ETA).  This is the common case where Δt ≈ hold_time arises
naturally — not from forcing them equal in the formula.

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

### Initial tier for newly created units

A new unit is born whenever a request commits fresh KV to the
radix tree — concretely on `LLM_PREFILL` commit (prefill output),
on decode-step commit, and on `SUB_RETURN` when the child's
output materialises as a `subagent_ctx` unit.

**Policy: new units are born in HBM** by sglang's allocator
semantics, with no per-unit scheduler decision.  The hint table
seed (§6) marks the unit as freshly accessed (`p_hat ≈ 1` for the
next event horizon) so the inline scorer's eviction heap doesn't
evict it before its first joint_decide evaluation.

The first joint_decide event that includes the unit in D_t may
immediately demote it (e.g. `SUB_DISPATCH_ASYNC` payload + parent
ACTING with a long tool ETA → low `p_hat` for the new
`subagent_ctx` → demote to DRAM).  Worst-case cost of this
"born-HBM-then-demote" path is one H→D transfer per unit that
would have been better-off born elsewhere — accepted as
overhead until measurement shows it matters.

> **Planned (future opportunity).**  The ideal would have the
> scheduler pick initial tier via the same `_value` argmin
> on the candidate's predicted p_hat at birth time — same
> formula as a migration decision, just applied before
> allocation.  Realising this requires extending sglang's
> allocator API with `alloc_at_tier(σ)`.  Not pursued until a
> workload demonstrates the born-HBM-then-demote overhead is
> material; on busy agent workloads where almost every newborn
> is about to be consumed by an in-flight request, HBM-default
> is correct anyway.

## 8. Admission — program-level candidate generator

Admission is the **program-level candidate generator** that feeds
`joint_decide` (§9).  It does not run its own decision loop; the
loop lives in §9 and consumes its candidates alongside the
unit-level candidates from §7.

For each event, the admission generator emits:

* one `Pause(p, cost, relief)` candidate for every ACTIVE program p
* one `Resume(p, gain, re_use)` candidate for every PAUSED program p

where:

```python
def pause_candidates(state, event):
    return [
        Pause(
            program_id = p.id,
            cost       = V_u_program(p)                       # work-loss (seconds)
                       + marginal_pause_cost(p, event),
            relief     = marginal_relief_value(p, state),     # HBM bytes (≥ 0)
        )
        for p in state.programs if p.state != PAUSED
    ]

def resume_candidates(state):
    return [
        Resume(
            program_id = p.id,
            gain       = V_u_program_if_active(                # counterfactual seconds
                             p, state, p.pre_pause_state),
            re_use     = marginal_relief_value(p, state),     # HBM bytes p will reclaim
        )
        for p in state.programs
        if p.state == PAUSED
        and capacity_fits(p, state)                            # see §8 Component defs
    ]
```

The Resume `gain` is a **counterfactual**: "if we resume p, how
much V_u does p produce?".  Computing `V_u_program(p)` on the
current state would return 0 because PAUSED holders' contribution
to every unit's `p_hat` is 0 by §7's conditional formulation —
all paused programs would tie at 0 and the headroom phase's
`gain / re_use` ordering would be undefined.

§9 consumes these candidates: pressure phase uses Pauses + §7's
Migrates; headroom phase uses Resumes.

### Component definitions

* `V_u_program(p) = Σ_{u ∈ p.units} V_u(u)` — computed against the
  current state, used as Pause's cost (the V_u we'd lose if p
  stops here).  The conditional p_hat from §7 is already a
  holder-product, so shared-prefix attribution is built in; no
  `1/|session_ids|` weight is needed.
* `V_u_program_if_active(p, state, hypothetical_state)` —
  counterfactual: compute V_u as if p's state field were
  overridden to `hypothetical_state` (and all other holders'
  states held fixed).  Used as Resume's gain.  The
  hypothetical is `p.pre_pause_state` — the state p was in when
  admission paused it — because resuming p restores exactly that
  pre-pause activity (a REASONING program that admission paused
  mid-decode goes back to REASONING; an ACTING program paused
  while awaiting a tool goes back to ACTING).  If `pre_pause_state`
  is missing (e.g. daemon restart lost it), the fallback is
  `REASONING` — the worst case for over-estimating gain, biased
  toward resuming sooner.
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
* `forecast(state)` — `state.pool_usage["HBM"].token_usage +
  forecast_inflight_demand(state)`.

  The second term is the **expected** HBM growth before the next
  event, summed over programs that will still be allocating:

  ```
  forecast_inflight_demand(state) =
    Σ_{p ∈ REASONING} E[remaining_tokens(p)] × bytes_per_token
  ```

  `E[remaining_tokens(p)]` is the expected residual decode length
  conditional on p's observable state.  Estimator priority:

  1. Event payload hint, if any (e.g. an in-band signal from the
     agent that the current decode will be short).
  2. p's own per-turn decode-length history.
  3. Workload-prior fit over recent same-class programs.
  4. **Bootstrap**: `max_completion_tokens` — only at cold-start
     before history accumulates.  A one-shot warning is logged so
     the bootstrap window is visible.

  Using **`max`** instead of `E[·]` (the bootstrap fallback as the
  steady-state rule) is wrong: it inflates the forecast by the
  ratio `max / mean`, which on agent workloads is ~5×.  Admission
  then over-pauses by ~5×, HBM is left under-utilized, and the
  daemon's whole throughput advantage is eaten by pessimism.  The
  error direction matters: under-estimating `E[remaining]` is
  recoverable (sglang's reactive `MEMORY_PRESSURE` webhook fires
  and admission catches up), over-estimating is a continuous
  throughput loss.  Same principle as §7's conditional `p_hat`:
  use observed conditional distributions, not constant upper
  bounds.

  At `SESSION_ARRIVAL` the incoming program has no history; the
  workload prior provides the seed `E[remaining_tokens]` for
  programs of the same class.

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
admission on the post-migrate state) is a Gauss-Seidel
coordinate descent: optimise unit-actions holding pause-set
fixed, then optimise pauses holding unit-actions fixed.  Two
failure modes:

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

Two phases, each is a **0/1 knapsack solved by exact DP** on the
relevant resource:

* **Pressure phase** runs when HBM pool is above `theta_hi`.
  Items: Pause programs + Migrate-HBM-out units.  Resource: HBM
  bytes.  Goal: free at least `bytes_needed` HBM, **minimising
  total V_u cost**.
* **Headroom phase** runs when HBM pool has dropped below
  `theta_lo`.  Items: Resume paused programs.  Resource: HBM
  bytes available before crossing `theta_hi` again.  Goal:
  **maximise total V_u gain** subject to bytes-re-occupied ≤
  free_room.

The two phases are mutually exclusive per event (forecast is
either too high or too low; in between is the hysteresis band
where neither phase has anything to do).

```python
def capacity_left_bytes(state, τ):
    """Bytes free in destination tier τ.  Pool view, not radix view —
    we want to know whether the τ allocator can actually accept
    `bytes_moved` more bytes, not whether the radix tree's slice of τ
    has room."""
    if τ in state.pool_usage:
        return (state.pool_usage[τ]["cap_bytes"]
              - state.pool_usage[τ]["used_bytes"])
    # DRAM / DISK tiers may only expose tier_usage:
    return (state.tier_usage[τ]["cap_bytes"]
          - state.tier_usage[τ]["used_bytes"])

def joint_decide(state, event):
    cap_total = state.pool_usage["HBM"]["cap_bytes"]
    cap_left  = {τ: capacity_left_bytes(state, τ)
                 for τ in (HBM, DRAM, DISK)}

    # ----- pressure phase: min-cost knapsack subject to relief >= bytes_needed -----
    bytes_needed = max(0, forecast(state) - theta_hi * cap_total)
    if bytes_needed > 0:
        cands  = kv_scheduler.migrate_candidates(state, decision_set(event, state))
        cands += admission.pause_candidates(state, event)
        cands  = [c for c in cands if c.relief > 0
                  and (not isinstance(c, Migrate) or cap_left[c.target_tier] >= c.bytes_moved)]
        return knapsack_min_cost(cands, bytes_needed, bucket_size=page_bytes)

    # ----- headroom phase: max-value knapsack subject to re_use <= free_room -----
    free_room = max(0, theta_lo * cap_total - forecast(state))
    if free_room > 0:
        cands = admission.resume_candidates(state)
        cands = [c for c in cands if c.re_use > 0]
        return knapsack_max_value(cands, free_room, bucket_size=page_bytes)

    return []                                            # hysteresis dead-zone
```

Both phases are **exact 0/1 knapsack** at K ≈ tens of items.
Concrete sizing: pressure-phase `bytes_needed` is at most
`(theta_hi - theta_lo) × cap_bytes` (one hysteresis band's worth);
quantised at `page_bytes` granularity that's ≈ 100 buckets;
candidates are tens.  DP table 30 × 100 = 3 000 cells, microseconds.

```python
def knapsack_min_cost(items, bytes_needed, bucket_size):
    """0/1 knapsack: subset S minimising Σ cost(s∈S)
    subject to Σ relief(s∈S) >= bytes_needed.
    Quantises relief at bucket_size so the DP table fits."""
    W = (bytes_needed + bucket_size - 1) // bucket_size
    K = len(items)
    INF = float("inf")
    dp = [[INF] * (W + 1) for _ in range(K + 1)]
    take = [[False] * (W + 1) for _ in range(K + 1)]
    for k in range(K + 1):
        dp[k][0] = 0
    for k in range(1, K + 1):
        r_bk = min(W, items[k - 1].relief // bucket_size)
        for w in range(W + 1):
            dp[k][w] = dp[k - 1][w]                            # skip
            w_prev   = max(0, w - r_bk)
            picked   = dp[k - 1][w_prev] + items[k - 1].cost
            if picked < dp[k][w]:
                dp[k][w] = picked
                take[k][w] = True
    # Infeasibility: even taking everything doesn't reach bytes_needed.
    # Return the all-take subset (best HBM we can free this event); the
    # next sglang webhook re-fire will surface the residual pressure.
    if dp[K][W] == INF:
        return list(items)
    # Reconstruct.
    chosen, w = [], W
    for k in range(K, 0, -1):
        if take[k][w]:
            chosen.append(items[k - 1])
            w = max(0, w - min(W, items[k - 1].relief // bucket_size))
    return chosen

def knapsack_max_value(items, budget_bytes, bucket_size):
    """Symmetric: subset maximising Σ gain(s) subject to Σ re_use(s) <= budget."""
    ...  # dual recurrence
```

#### Why exact DP, not greedy

LP-relaxation greedy (sort by `cost/relief`, take cheapest-per-byte
until budget met) is the standard 0/1 knapsack approximation.
Worst-case it's 2× off — e.g. budget 100, items A=(cost 10, relief
99) and B=(cost 11, relief 100): greedy picks A then B for total
cost 21, optimal is B alone at 11.

That worst case happens whenever a single sufficient item is
slightly less efficient per byte than a sequence of insufficient
items that together waste cost.  In admission this is the
"one-pause-could-have-solved-it but we did three migrates first"
scenario.  Not adversarially constructed — common when one runaway
program could cover the budget alone.

At K ≈ 30 and bucketised W ≈ 100, exact DP is microseconds — the
same order as greedy.  There is **no efficiency reason** to use
the approximation.  The exact form is the design.

#### Properties this satisfies

1. **Optimal cost / gain** under the knapsack formulation.  No
   1/2-approximation gap.
2. **Per-tier sub-budgets.**  `cap_left[τ]` tracks how many bytes
   remain in each destination tier; migrate candidates are
   filtered up-front if they'd overflow τ.  Without this guard
   the plan could schedule 50 HBM→DRAM moves into a DRAM
   that's already 95 % full.
3. **Pause/Migrate vs Resume disjoint.**  Pressure phase only
   handles `freeing` actions (Pause, HBM-out Migrate).  Headroom
   phase only handles `claiming` actions (Resume).  They never
   compete in the same knapsack.

#### Always-fresh state at the inter-event boundary

Exact DP solves one knapsack against a single snapshot of state.
Within that DP, candidates' `cost` / `relief` are evaluated **on
the snapshot fetched at event entry** — the formulation is
exact w.r.t. that snapshot, so no per-pick re-ranking is needed
(unlike a greedy loop which mutates state mid-pass).  The
always-fresh invariant (§10) is satisfied at the event boundary:
the next event's joint_decide will refetch state and re-solve
its knapsack from scratch.

> **Planned (code lag).**  The daemon's current `joint_decide`
> implementation uses the greedy `cost/relief` ordering instead
> of exact DP.  This is a pure code lag — exact DP at K ≈ 30,
> W ≈ 100 buckets runs in microseconds, the same order as
> greedy, so there is no efficiency reason to keep the
> approximation.  Replace with `knapsack_min_cost` /
> `knapsack_max_value` per the pseudo-code above.

### What collapses out

* **Trade gate**: the min-cost DP only includes an item in the
  optimal subset if it reduces total cost relative to the next-
  best subset achieving the same relief.  Any subset dominated
  by a cheaper one is rejected by construction — there's no
  need for the explicit `if pause.cost > pause.relief × (best alt
  cost/byte): break` guard a sequential greedy would require
  (which itself only type-checks once both sides are expressed in
  seconds/byte; cost is seconds, relief is bytes, so the gate
  collapses out at the DP's level rather than as a scalar
  comparison).
* **Composition order**: the DP enumerates subsets implicitly
  across the union; there's no "run kv_scheduler first then
  admission".  A Pause and a Migrate can both appear in the
  optimal subset, or only one, or neither — whichever combination
  minimises total cost.
* **Per-trigger D_t for admission**: admission considers every
  active / paused program every event; D_t only constrains the
  unit-migrate candidate generator.

### Modules vs decisions

`kv_scheduler` and `admission_controller` remain as **candidate
generators** — they expose `migrate_candidates(...)`,
`pause_candidates(...)`, `resume_candidates(...)`.  They don't
emit plans, they don't apply actions.  `joint_decide` (this
section) is the single decision function; sglang is the single
action sink (`POST /aginfer/migrate` for unit actions; proxy gate
for pause/resume).

## 10. Invariants

| invariant | enforced by |
|---|---|
| **Single-worker event loop**: handlers serialised; no concurrent migrate races; no internal timer in kv_scheduler, admission, forecast refresh, or program_tracker — every recomputation is on event arrival.  Sole exception: the proxy's `T_idle` SESSION_END detection fallback (§4) | asyncio queue + single consumer |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent migrate**: re-applying the same action returns 200 with `applied=0` and a `race:*` skip | sglang `apply_aginfer_migrations` |
| **Webhook mandatory**: every sglang launch passes `--aginfer-notify-url`; the admission trigger path is not optional | launch script per-deployment |
| **Pool-truth admission**: admission reads `state.pool_usage.HBM.token_usage`, not `tier_usage`; if the snapshot lacks `pool_usage` the daemon halts loudly (sglang too old / misconfigured) | daemon `admission_controller._hbm_occ` |
| **Tree-view V_u**: V_u migration scoring reads `tier_usage`, never `pool_usage` | daemon `OursGreedyPolicy._value` |
| **All traffic through daemon proxy**: every chat-completion to sglang arrives via the daemon's `/v1/chat/completions` proxy; direct-to-sglang clients are out of scope and would render admission's program-pause unenforceable | deployment topology |
| **Hint table covers every live unit**: sglang seeds a "fresh access just happened" entry on unit birth (`p_hat ≈ 1` for the next event horizon); the daemon refines via `PUT /aginfer/hints`; eviction never falls back to LRU on absent hints | sglang allocator hook + daemon hint pusher |
| **Hint atomicity**: the inline scorer's read of a hint entry and the daemon's `PUT` of a new entry are atomic per-key (read-modify-write would race a daemon update against an in-flight eviction).  Per-key seqlock or compare-and-swap suffices; full RW lock is overkill at 10²/s | sglang hint-table data structure |
| **Layer enable flags**: HiCache, kv_scheduler, and admission each have an independent enable flag.  Admission can only fire when kv_scheduler is also on (admission's pre-pause migrate path requires the daemon's V_u machinery).  HiCache is independent of both | daemon CLI + sglang flags |
| **Threshold convention**: launch operators must pass the SAME `theta_hi` / `theta_lo` values to sglang and the daemon.  Mismatch is a known footgun (sglang fires `MEMORY_PRESSURE` at its threshold but daemon refuses to act because its threshold is higher, leaving admission silently inert).  Not currently enforced by a single config source; weakening this from an invariant to a convention surfaces the gap honestly | launch scripts pass aligned `--aginfer-theta-*` flags + daemon CLI defaults match the sglang defaults |

## 11. Recovery (daemon restart / sglang restart)

The daemon's in-process state — `program_tracker` (REASONING /
ACTING / PAUSED / ENDED per program), the proxy's pause-gate
suspended request set, the `migrate` retry queue — is volatile.
Either side crashing leaves the other holding a stale view.

### Daemon restart

On crash + restart the daemon's recovery sequence is:

1. `GET /aginfer/state` — pulls the authoritative snapshot of
   `tier_usage`, `pool_usage`, `per_program_usage`, `units`, and
   `program_tracker`-equivalent state encoded in
   `per_program_usage[p].state`.  The state schema is designed so
   the daemon can rebuild its in-memory view from one fetch.
2. **Released pauses on restart**: the proxy gate is empty after
   restart (it's a Python set in the proxy process).  Programs
   that were PAUSED at crash time were holding their next request
   in the gate; that request is now lost.  Two choices, the
   second is required:
   * **wrong**: silently release all paused requests; harbor's
     next retry succeeds → user-visible inconsistency vs the
     daemon's intended pause.
   * **right**: any program with `per_program_usage[p].state ==
     PAUSED` at restart is re-registered as PAUSED in the
     daemon's tracker, and the proxy will gate that program's
     next arrival.  In-flight requests at crash time are
     considered failed (proxy didn't acknowledge); the client
     hits the gate when it retries.  This relies on the client
     using retry-on-timeout semantics — pure fire-and-forget
     clients lose the dropped requests.  The "All traffic
     through daemon proxy" deployment invariant (§10) implicitly
     mandates a retrying client; clients that don't retry are
     out of scope.

   `pre_pause_state` is **lost** on restart (it was in-process
   only).  The Resume gain counterfactual (§8) falls back to
   `REASONING` for any program re-registered as PAUSED without a
   known `pre_pause_state`.  REASONING is the safe fallback
   because it carries the highest expected `V_u_program_if_active`
   across the workload — a REASONING-on-resume program is by
   construction about to issue a decode step (highest p_hat for
   its tail).  Resume's argmax-gain prefers high counterfactual
   value, so the fallback biases toward **resuming sooner**
   rather than later — preferred to silently holding a program.
3. **In-flight migrate retries**: any migrate POSTs that were
   in-flight at crash are not retried — they are idempotent
   (§10), so re-issuing on the next event is harmless and
   correct.

### sglang restart

The daemon detects sglang restart via `/aginfer/state` failure
modes (connection refused, then 200 with reset counters).  On
the first 200 after a failure:

1. Daemon clears its hint-side tracking (every unit it had
   pushed a hint for is now gone; sglang's hint table is empty
   at startup).
2. Daemon re-pushes hints lazily, as units appear in subsequent
   `/aginfer/state` snapshots.
3. `program_tracker` state is preserved daemon-side; proxy gate
   re-applies to incoming requests.

### sglang's webhook fire-and-forget contract

If sglang fires a webhook while the daemon is down, the event is
**lost**.  This is acceptable because:

* admission's decision is event-driven by **any** of 10 event
  kinds; the next proxy event (any chat completion) re-evaluates.
* sglang continues firing on subsequent transitions (and
  heartbeats while in HIGH / CRITICAL), so the daemon picks up
  the next webhook within `heartbeat_s` ≈ 5 s.

No webhook persistence / replay is needed.  This is design, not
trade-off: a queue or WAL between sglang and daemon would add
mid-stream consistency to honour a single dropped event, at the
cost of a recoverable storage subsystem that complicates every
deployment.
