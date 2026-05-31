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
       │    same events.  On the REASONING/ACTING → PAUSED │
       │    transition it notifies sglang via              │
       │    PUT /aginfer/program_paused so `pre_pause_state`│
       │    is persisted authoritatively in                │
       │    per_program_usage (consumed by §8 Resume gain  │
       │    counterfactual; survives daemon restart).      │
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
| kv_scheduler | daemon | workload events (12 kinds, §4) | **unit-migrate candidate generator** consumed by §9 `joint_decide` |
| admission_controller | daemon | every workload event | **program pause / resume candidate generator** consumed by §9 `joint_decide` |

All three use the **same V_u rule**.  They live where they live
because of **event ownership**, not as fallbacks for each other:
the eviction-decision callsite is sglang-internal, fires
synchronously on the scheduler step, and cannot wait for an HTTP
round-trip to the daemon, so its V_u handler must be in-process.
The 12 workload events are surfaced over HTTP (proxy + sglang
webhook), so their V_u handlers live in the daemon.

### Aginfer is sglang's decision pipeline (superset framing)

aginfer is **not an optional add-on** layered on top of sglang's
native heuristics — it is the single decision pipeline that
sglang invokes for every cache-management choice.  sglang's
historical heuristics (LRU eviction, `hit_count >=
write_through_threshold` write-through trigger, automatic
on-access load_back) are expressed as aginfer's **default
policy module**: the policy module that runs when no daemon is
attached.

The implication is symmetric across three modes:

| mode | daemon | scorer / write-through policy | behaviour |
|---|---|---|---|
| **aginfer disabled** (baseline) | not running | aginfer default policy = LRU-equivalent V_u + hit_count write-through | identical to historical sglang vanilla |
| **aginfer in-process only** | not running | aginfer default policy | same as baseline; daemon-side benefits absent |
| **aginfer full** | running | daemon-pushed V_u hints + admission active | full pipeline |

There is **one code path** through the cache manager regardless
of mode; modes differ only in which V_u inputs the in-process
scorer reads from the hint table.  Experimental ablations
(baseline vs ours) flip a policy parameter, not a code path —
eliminating the "is the diff because of the policy or because
of daemon-RTT overhead" confound that would otherwise muddle
every comparison.

Two physical plugin points carry this:

* **Eviction scorer**: sglang's `SGLANG_KV_POLICY_MODULE`
  registers a scoring function called from the eviction-decision
  callsite.  Default module re-implements LRU as a V_u proxy
  (last_access as p_hat surrogate).  Aginfer registers its
  hint-table-aware V_u.
* **Write-through trigger**: sglang's HiCache invokes a
  `should_write_through(node)` function (new plugin point) when
  considering write_through_selective.  Default implementation
  is the historical `hit_count >= write_through_threshold`.
  Aginfer registers a V_u-aware version that triggers when
  `V_u(residence ∪ {DRAM}) > V_u(residence)`.

The same plugin pattern can be extended to future decision points
(predictive load_back, mooncake archive trigger) without
restructuring the framework.

## 4. Events

Twelve event kinds.  The daemon is **strictly reactive** to events;
the system has no internal timer.

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
| `PRESSURE_CRITICAL` | sglang webhook | allocator-truth HBM occupancy crossed `theta_crit` upward (above HIGH); same payload shape as `MEMORY_PRESSURE`.  Routing-priority signal: the event router preempts any pending lower-priority event and runs `joint_decide` immediately.  See §9 for why no decision-rule branch is needed (forecast + widened `Pause.relief` carry the urgency) |
| `PRESSURE_RESOLVED` | sglang webhook | allocator-truth HBM occupancy crossed `theta_lo` downward (back to OK band, hysteresis) |
| `APPLY_FAILED` | sglang webhook | a daemon-issued action could not be applied; payload carries `{endpoint, action_id, reason}` where `endpoint ∈ {migrate, program_paused, hints, thresholds}` and `reason` is the skip-class (§6 `POST /aginfer/migrate` reasons + the analogue for other endpoints).  The daemon's `joint_decide` re-evaluates on the next event handler entry; idempotent invariants (§10) let it safely re-issue any superseding action without explicit retry bookkeeping |

Each event carries (`kind`, `session_id` if applicable, `payload`).
The payload may include event-specific context the decision rule can
exploit (see §7) — e.g. `TOOL_CALL_START` carries the tool's
expected duration, `SUB_DISPATCH_*` carries the inherit-prefix flag.

`SESSION_END` is non-trivial to detect from the OpenAI proxy: the
last request looks like any other.  The proxy commits a program as
ENDED **only** when the harbor / agent client closes its session
explicitly via an out-of-band signal (harbor's `/aginfer/session_end`
endpoint).  An agent client that never signals leaves its program
in the tracker forever — that is a deployment bug to be fixed at
the client, not a graceful-degradation case for the scheduler.

No timer-based fallback exists: any time-based "looks idle" rule
would re-introduce the internal-timer pattern §10's invariant
exists to rule out, and would race against legitimate long-running
ACTING programs whose tools take minutes.

### Event router priority

The event_router is a two-priority asyncio queue (HIGH, NORMAL),
single consumer.  `PRESSURE_CRITICAL` is the only HIGH-priority
event; everything else is NORMAL.

* HIGH events **preempt the queue**: on arrival they jump ahead of
  any pending NORMAL events.  They never preempt an in-flight
  handler (handlers are serialised), but the next handler run will
  be the CRITICAL one regardless of arrival order.
* NORMAL events are FIFO.
* The router is the only place priority appears; `joint_decide`
  itself is priority-agnostic (it reads `state` and produces a
  plan; the urgency is encoded as a larger `bytes_needed` via
  `forecast`).

This keeps the decision rule pure: priority is a routing concern,
not a decision-rule concern.

## 5. State surface — `/aginfer/state`

```json
{
  "page_size": int,
  "time_counter": int,                  // monotonic access tick

  "tier_usage": {                       // RADIX-TREE view; aggregate per tier
    "HBM":  {"used_bytes": int, "cap_bytes": int},
    "DRAM": {"used_bytes": int, "cap_bytes": int},
    "DISK": {"used_bytes": int, "cap_bytes": int}    // zero-stub until L3 wired
  },

  "pool_usage": {                       // ALLOCATOR-TRUTH view; per-subpool
                                         // breakdown.  Each tier carries a
                                         // `subpools` dict keyed by
                                         // architecture-determined component
                                         // names; concrete subpool keys per
                                         // model class are listed in §12
                                         // Scenarios.  sglang's UnifiedRadixCache
                                         // exposes these via its component
                                         // registry (one component per
                                         // attention type per architecture;
                                         // see sglang unified_cache_components/).
    "HBM":  {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int}}},
    "DRAM": {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int}}},
    "DISK": {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int}}}
  },

  "per_program_usage": {                // PER-PROGRAM-OWNED view
    "<program_id>": {
      "hbm": {
        "committed": {                  // share of radix nodes attributed to
          "<subpool>": int              // this program per HBM subpool, with
        },                               // 1/holders weight for nodes shared
                                         // across programs
        "inflight":  {                  // request-owned bytes per HBM subpool
          "<subpool>": int              // (not yet committed to the radix
        }                                // tree).  Per subpool because Mamba
                                         //   inflight allocation differs from
                                         //   attention KV inflight growth.
      },
      "dram": {
        "committed": {"main": int}      // host pool share, same attribution
      },
      "state": "REASONING"|"ACTING"|"PAUSED"|"ENDED",
                                         // sglang derives REASONING/ACTING
                                         // locally from request liveness
                                         // (it sees every chat-completion
                                         //  via the daemon proxy).  PAUSED
                                         //  and ENDED are written by the
                                         //  daemon via §6
                                         //  `PUT /aginfer/program_paused`.
      "pre_pause_state":
            "REASONING"|"ACTING"|null,  // populated by daemon→sglang
                                         // PauseNotification on the
                                         // REASONING|ACTING → PAUSED
                                         // transition; null for non-PAUSED
                                         // programs.  Persists across daemon
                                         // restart because sglang is the
                                         // authoritative store.
      "unit_hashes": list[str]          // hashes of units owned by this program
                                         // (= {u.hash for u in units
                                         //         if program_id in u.session_ids}),
                                         // sglang materialises this list once
                                         // per state-dump so admission doesn't
                                         // walk `units` per candidate
    }
  },

  "units": [
    {"hash": str,
     "residence": ["HBM"|"DRAM"|"DISK", ...],  // SET of tiers the unit
                                                // currently occupies bytes in.
                                                // A unit can be in HBM+DRAM
                                                // simultaneously (post write-
                                                //   through, before HBM
                                                //   eviction), DRAM+DISK,
                                                //   etc.  Empty residence
                                                //   means the unit has been
                                                //   fully dropped from the
                                                //   radix tree (it should
                                                //   not appear in `units`
                                                //   at all in that case).
     "n_tokens": int,
     "n_bytes": {                             // per-(tier, subpool) bytes;
                                                // one outer entry per tier
                                                // in residence.  Subpool
                                                // keys match
                                                // pool_usage[tier].subpools.
       "<tier>": {"<subpool>": int}
     },
                                                // Per-architecture concrete
                                                //   shapes are in §12
                                                //   Scenarios.
     "last_access_time": int, "hit_count": int,
     "session_ids": list[str]}
  ],

  "link_stats": {                       // PHYSICAL-LINK view, input to V_u
    "<σ>-><τ>": {                       // one entry per directional link
                                         // (HBM->DRAM, DRAM->HBM, DRAM->DISK,
                                         //  DISK->DRAM); DROP has no link
      "peak_bw_bps":             int,   // theoretical link peak (PCIe gen5
                                         // x16, NVMe nominal); static per
                                         // deployment, set from operator
                                         // config or device probe
      "recent_throughput_bps":   int,   // EMA over the last ~100 ms window
                                         // of (bytes / elapsed) observed at
                                         // sglang's IO callsites (HiCache
                                         // dispatcher for HBM↔DRAM, Mooncake
                                         // put/get for DRAM↔DISK)
      "samples_in_window":       int    // 0 when link is idle; daemon uses
                                         // this to decide between
                                         // `recent_throughput_bps` (link
                                         // active) and `peak_bw_bps`
                                         // (link idle → assume free)
    }
  },

  "tier_holding_cost": {                // PER-(TIER, SUBPOOL) MARGINAL
                                         //   DISPLACEMENT, input to V_u's
                                         //   h_(τ, sp)(occ) term.  Per-subpool
                                         //   because Mamba snapshots and
                                         //   attention KV in the same tier
                                         //   compete for different physical
                                         //   bytes and saturate at different
                                         //   occupancies.
    "HBM": {
      "<subpool>": {"h_max_per_byte_sec": float}    // one entry per subpool
                                                     //   declared in
                                                     //   pool_usage.HBM
    },
    "DRAM": {"main": {"h_max_per_byte_sec": float}},
    "DISK": {"main": {"h_max_per_byte_sec": float}}
  },

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
* `pool_usage` (allocator view, **per-subpool**) — input for
  **admission's pressure trigger**.  Admission acts when **any**
  HBM subpool crosses its `theta_hi` threshold, not when the
  aggregate does — a Mamba snapshot pool at 95% with attention
  at 60% is the failure mode an aggregate view hides.
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

Daemon → sglang endpoints, write-only from the daemon's
perspective.

### Fire-and-forget delivery (all action endpoints)

Every daemon-issued action endpoint (`POST /aginfer/migrate`,
`PUT /aginfer/program_paused`, `PUT /aginfer/thresholds`,
`PUT /aginfer/hints`) is **fire-and-forget** from the event
handler's perspective:

1. Handler computes the plan and **enqueues each action** onto the
   daemon's outbound `asyncio.Queue`; the handler returns immediately.
2. A dedicated outbound worker task consumes the queue, issues
   the HTTP `POST` / `PUT`, and on 200 simply drops the response.
3. If sglang returns a non-2xx or the apply path raises **inside
   sglang**, sglang fires an `APPLY_FAILED` webhook back to the
   daemon (§4) carrying `{endpoint, action_id, reason}`.  The
   daemon treats the webhook like any other workload event — the
   next `joint_decide` re-evaluates the state and may re-issue a
   superseding action.  Idempotency (§10) makes re-issue safe.

This decouples handler latency from sglang processing latency.
Under a 200 ms sglang stall (CUDA-graph capture, prefill batch
serialization), a synchronous awaited POST would freeze the entire
event router; fire-and-forget keeps the inbound queue draining and
defers the action to the outbound worker.

Each action carries an `action_id` (UUID generated at the daemon)
so `APPLY_FAILED` can be correlated to a specific in-flight
attempt; the id is logged on both sides.

### `POST /aginfer/migrate` — apply residence-set transitions

Request:
```json
{"actions": [
  {"hash": str,
   "add_tiers":    ["HBM"|"DRAM"|"DISK", ...],   // tiers to add to residence
   "remove_tiers": ["HBM"|"DRAM"|"DISK", ...]}   // tiers to remove from residence
]}
```

The unit's residence set is updated atomically per action:
`new_residence = (old_residence ∪ add_tiers) \ remove_tiers`.
A few physical interpretations:

| `add_tiers` | `remove_tiers` | physical operation |
|---|---|---|
| `["DRAM"]` | `[]` | write-through to DRAM (create host backup, keep HBM live) |
| `[]` | `["HBM"]` | evict from HBM (DRAM backup retained if present) |
| `["DRAM"]` | `["HBM"]` | write-through then evict — the legacy "HBM→DRAM migrate" |
| `["HBM"]` | `[]` | load_back / predictive promote (HBM populated, DRAM kept) |
| `[]` | `["HBM","DRAM","DISK"]` | DROP from radix tree entirely |
| `["DISK"]` | `[]` | archive to disk via Mooncake; DRAM kept until DRAM pressure |

Resulting empty residence (`= []`) implies the unit's radix-tree
node is removed; sglang deletes the hash on the next state-dump
boundary.

Adding a tier that already exists in residence is a no-op; same for
removing one that isn't present.  All operations are idempotent
per the §10 invariant.

Response:
```json
{"applied": int, "applied_hashes": [str, ...],
 "skipped": [{"hash": str, "reason": str}, ...]}
```

#### Skip-reason classes (canonical list in code)

* **`race:*`** — time-window race between daemon's state fetch and
  apply; tree mutated by concurrent evict / request.  Retryable.
  Daemon re-issues on the next event.
* **`promote_load_back_declined:<category>`** — `load_back()` for
  an `add_tiers: ["HBM"]` action declined cleanly, with
  `<category>` distinguishing the sub-cause (full-allocator alloc
  fail, SWA sub-pool evict short, etc.).  Usually transient.
* **`promote_raised:<exc>:<loc>:<msg>`** — load_back threw.
  Indicates an invariant break; investigate.
* **`write_through_declined:<category>`** — `add_tiers: ["DRAM"]`
  (write-through) declined cleanly (host pool full, mooncake
  unreachable, etc.).  Usually transient.
* **`unknown_tier:...`** / **`unsupported_tree_cache:...`** —
  contract violations; daemon misbehavior.  Halt loudly.

The daemon's retry / debug loop dispatches on the **class prefix**,
not the literal string.

### `GET /aginfer/thresholds` — canonical hysteresis thresholds

The daemon is the source of truth for `theta_hi` / `theta_lo` /
`theta_crit` / `heartbeat_s`.  Three-stage lifecycle, no
startup coupling:

1. **Bootstrap fetch (sglang side).**  At sglang launch, sglang
   tries `GET /aginfer/thresholds`.  If the daemon is up, sglang
   writes the response to a local cache file
   (`<sglang-data>/aginfer_thresholds.json`).  If the daemon is
   down, sglang loads the existing cache and proceeds — sglang
   can start without the daemon, using the last-known thresholds.
   First-ever launch with no cache and no daemon is a deployment
   bug; sglang refuses with an explicit error.
2. **Daemon-side update broadcast.**  When the daemon's threshold
   config changes at runtime (operator restart with new defaults,
   reload via signal), the daemon `POST`s the new values to
   sglang's `PUT /aginfer/thresholds` endpoint and waits for ack
   before considering the change applied.  Sglang updates its
   cache file atomically.
3. **Mismatch is loud.**  If an operator passes `--aginfer-theta-*`
   to sglang AND the value disagrees with the daemon's view at the
   time of bootstrap fetch, sglang halts.  The daemon's view wins;
   the CLI flag is a deployment-intent assertion that must match.

Request: empty `GET`.

Response:
```json
{"theta_hi": float, "theta_lo": float,
 "theta_crit": float, "heartbeat_s": float}
```

`PUT /aginfer/thresholds` (daemon → sglang) has the same body.

### `PUT /aginfer/program_paused` — daemon→sglang pause notification

Daemon notifies sglang on every admission Pause / Resume transition
so sglang can persist `per_program_usage[p].state` and
`pre_pause_state`.  This makes `/aginfer/state` the single
authoritative store for both fields; daemon restart loses nothing.

Request:
```json
{"program_id": str,
 "transition": "PAUSE"|"RESUME"|"END",
 "pre_pause_state": "REASONING"|"ACTING"|null}
```

* `PAUSE`: payload carries `pre_pause_state` (the state the
  program was in immediately before admission paused it).  sglang
  sets `state = PAUSED`, `pre_pause_state = payload.pre_pause_state`.
* `RESUME`: daemon sends `pre_pause_state = null` (sentinel, not
  the restored state).  sglang reads its **own** stored
  `pre_pause_state` to determine the restored state, then sets
  `state = (the stored pre_pause_state)` and `pre_pause_state =
  null`.  The payload field is intentionally null-only on RESUME
  to make it impossible for the daemon to disagree with sglang's
  authoritative view.
* `END`: sglang sets `state = ENDED`; `pre_pause_state` is
  cleared.  Payload `pre_pause_state` is ignored.

Response: `{"applied": true}`.  Idempotent: re-applying the same
transition on a program already in the target state is a no-op
200.

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
skew (one rank in CUDA graph capture, etc.).  Hint propagation
is **per-rank atomic** (the §10 hint-atomicity invariant) but
**not cross-rank atomic**: rank 0 may see hint version `k` while
rank 1 still sees version `k-1` during the propagation window.

The semantic argument that this is the right model — not a
performance compromise:

* The inline scorer's input is a **value estimate**, not a
  ground-truth observation.  The "true" `p_hat` for a unit is
  unobservable (it depends on agent decisions outside aginfer's
  control).  Every hint is already an estimate of an unobservable;
  cross-rank temporal alignment of estimates doesn't add
  information.
* The same logical unit's KV bytes are mirrored across TP ranks
  (§6 multi-rank table).  An eviction on rank 0 invalidates the
  unit globally — the eviction itself is the cross-rank
  synchronisation point.  A per-rank scorer disagreement on
  *which* unit to evict next is resolved by whichever rank's
  eviction commits first; the others observe the eviction and
  rescore.
* A global synchronous PUT-barrier (wait for every rank to ack
  before considering the hint applied) would synchronise hint
  *propagation* but not the scorer's evaluation timing — ranks
  still evaluate the heap at slightly different wall-clock
  moments.  The barrier moves the inconsistency window without
  closing it.

**Risk this argument doesn't cover**: if two ranks score the same
heap at near-simultaneous wall-clock and pick **different** units
to evict because they're reading different hint versions, the
allocator may free unit A on rank 0 and unit B on rank 1, leaving
the logical KV in an inconsistent split.  Whether this can happen
depends on sglang's per-rank eviction commit protocol — the
all-rank atomicity is asserted by §6's "every action is all-rank
atomic by semantic requirement", but it's the cross-rank
coordination layer (tokenizer-server fanout) that enforces it,
not the hint table.

**Empirical question (T11+ workload).**  Verify the risk is
actually zero in T11 tests: run a workload with high hint churn
under TP > 1 and compare the set of evicted hashes seen by each
rank.  If a divergence is ever observed (rank 0 evicts hash X
that rank 1 didn't pick), the all-rank atomicity invariant is
broken and the design needs a stronger primitive.

## 7. Decision rule

### Symbols and units

All cost-side quantities are in **seconds** (paper §3 Reward
units: time saved/paid against wall-clock).  All relief / re_use
quantities are in **bytes**.  The joint knapsack (§9) mixes
seconds-cost with bytes-budget; the comparison is well-typed
because the constraint is a one-sided byte threshold.

| symbol | unit | meaning |
|---|---|---|
| `u.n_bytes`, `u.n_tokens` | nested-dict / tokens | `u.n_bytes` is a nested dict `{tier: {subpool: bytes}}` (per §5).  `bytes_at(u, τ)` = bytes u occupies at tier τ; `total_bytes(u, τ)` = sum across τ's subpools |
| `p_hat(u, Δt)` | unitless ∈ [0,1] | conditional reuse probability over horizon Δt |
| `Δt` | seconds | decision look-ahead horizon (§7 inputs) |
| `hold_time` | seconds | expected residence in candidate tier (§7 inputs) |
| `reload_from(u, τ)`, `reload_from_DROP(u)` | seconds | per-paper-§3 reload costs `ρ_τ × n_tokens`, `π_u × n_tokens` |
| `h_(τ, sp)(occ)` | sec / (byte × sec of holding) | per-(tier, subpool) marginal displacement cost at occupancy `occ`; one entry per subpool listed in `pool_usage[τ].subpools`.  Linear placeholder `h_(τ, sp)_max × occ` from `state.tier_holding_cost[τ][sp]`; final shape pending T12 calibration |
| `occupancy_of(τ, sp, state)` | unitless ∈ [0,1] | `pool_usage[τ].subpools[sp].used_bytes / .cap_bytes` |
| `bw_free(σ, τ)` | bytes/sec | live free bandwidth on σ↔τ link; reads `state.link_stats[σ→τ].recent_throughput_bps` when active, falls through to `peak_bw_bps` when link idle |
| `transfer_bytes(u, σ, τ)` | bytes | `bytes_at(u, σ)` — what the link physically moves when adding τ to residence (source is `authoritative_tier(residence)`) |
| `transfer_time(σ, τ)` | seconds | `transfer_bytes / bw_free` |
| `page_bytes` | bytes | DP quantisation granularity = sglang's allocator page size in bytes (model-architecture dependent; for paged-KV with non-uniform per-layer pools, the smallest page across all (tier, subpool) pairs governs) |
| `cost`, `gain`, `V_u`, `V_u_program` | seconds | net value at the same time-axis |
| `relief`, `re_use`, `bytes_moved`, `bytes_needed` | bytes | HBM-resource axis |
| `forecast(state)` | dict[subpool, bytes] | per-HBM-subpool predicted bytes at the next event arrival if no scheduling action is taken: `pool_usage.HBM.subpools[sp].used_bytes + forecast_inflight_demand(state)[sp]`.  Compare each entry against `theta_hi × subpools[sp].cap_bytes`.  Horizon = `forecast_horizon(state)`, see §8 |
| `forecast_horizon(state)` | seconds | expected time to next event: `min(heartbeat_s, 1 / recent_event_rate)`.  Bounded above by webhook heartbeat under HIGH/CRITICAL pressure (≈ 5 s) and below by typical event interval under normal load (≈ 10 ms) |
| `decode_throughput(p)` | tokens/sec | sglang-observed decode rate for p, used to cap `E[remaining_tokens]` by `horizon × throughput` |
| `prefill_throughput(state)` | bytes/sec | sglang-observed prefill rate, used to convert per-program in-flight bytes (summed across HBM subpools) into a re-prefill seconds-cost in `marginal_pause_cost` |

### Candidate types

```python
@dataclass(frozen=True)
class Migrate:
    hash: str                 # u.hash (sglang canonical key)
    add_tiers: tuple[Tier, ...]      # tiers to ADD to residence
    remove_tiers: tuple[Tier, ...]   # tiers to REMOVE from residence
    cost: float                       # seconds
    relief: dict[Tier, dict[str, int]]   # per-(tier, subpool) bytes freed at
                                          # the source side of the transition.
                                          # Keys: tier ∈ remove_tiers ∩ residence;
                                          # inner keys: subpools in that tier.
    acquired: dict[Tier, dict[str, int]] # per-(tier, subpool) bytes newly
                                          # occupied at destination tiers in
                                          # add_tiers (consumed against
                                          # destination cap in §9).

@dataclass(frozen=True)
class Pause:
    program_id: str
    cost: float               # seconds (V_u_program + marginal_pause_cost)
    relief: dict[str, int]    # per-HBM-subpool bytes
                              # = snapshot_relief + future_inflight_savings
                              # (§8 — each term is itself a per-subpool dict)

@dataclass(frozen=True)
class Resume:
    program_id: str
    gain: float               # seconds (V_u_program_if_active)
    re_use: dict[str, int]    # per-HBM-subpool bytes
                              # = expected_peak_hbm_after_resume (§8)
```

### kv_scheduler is a candidate generator

Per event, the daemon constructs a **decision set** `D_t ⊆ units`
and emits **unit-level migrate candidates** for the joint decider
in §9.  `kv_scheduler` is a **candidate generator only** — it does
not select a plan or apply actions; that is §9's job.

A `Migrate` candidate is **a residence-set transition** —
`(add_tiers, remove_tiers)` applied to `u.residence`.  The generator
enumerates the meaningful transitions and scores each against the
current state.

```python
# Authoritative tier: the highest-compute-readiness tier in u's
# residence.  HBM if present (compute-ready), else DRAM (must
# load_back to use), else DISK.  V_u's holding cost is paid only
# here — bytes in lower tiers are either backups (sunk cost, sglang
# keeps them precisely so future evict is cheap) or the active
# residence.
def authoritative_tier(residence):
    return HBM  if HBM  in residence else \
           DRAM if DRAM in residence else \
           DISK

# V_u under a hypothetical residence set, summed across the
# authoritative tier's subpools.  Reload cost is from the
# authoritative tier (= the tier serving the next access).
def _value(u, residence, state):
    a = authoritative_tier(residence)
    return  p_hat(u, Δt) * (reload_from_DROP(u) - reload_from(u, a)) \
          - sum(h_(a, sp)(occupancy_of(a, sp, state)) *
                u.n_bytes[a][sp] * hold_time
                for sp in u.n_bytes[a])

# Reload costs (paper §3 Tier Parameters, renamed for legibility):
reload_from(u, τ)    = ρ_τ × u.n_tokens     # per-token reload at tier τ
reload_from_DROP(u)  = π_u × u.n_tokens     # full re-prefill cost

# Per-subpool occupancy ratio (read from §5 pool_usage):
occupancy_of(τ, sp, state) =
      state.pool_usage[τ].subpools[sp].used_bytes
    / state.pool_usage[τ].subpools[sp].cap_bytes

# Bytes a unit occupies at tier τ (sum across τ's subpools);
# 0 if τ not in residence.
def bytes_at(u, τ):
    return sum(u.n_bytes.get(τ, {}).values())

# Link bandwidth cost of a transition.  Each tier added that wasn't
# already in residence costs a transfer over the relevant link;
# tiers removed from residence are metadata-only (no link traffic).
def migration_cost(u, add_tiers, remove_tiers, state):
    cost = 0.0
    for τ in add_tiers:
        if τ in u.residence:  continue          # already present, no-op
        source = authoritative_tier(u.residence)
        cost  += transfer_bytes(u, source, τ) / bw_free(source, τ, state)
    return cost                                  # removes are free (no link)

# Bytes the link physically moves (sum of u's per-subpool bytes
# at the source tier).
transfer_bytes(u, σ, τ) = bytes_at(u, σ)

# Per-(tier, subpool) bytes the residence-set transition frees.
# Returns a nested dict {tier: {subpool: bytes_freed}} so the §9
# DP can post each axis independently.
def bytes_freed_by_migrate(u, add_tiers, remove_tiers):
    freed = {}
    for τ in remove_tiers:
        if τ not in u.residence:  continue       # already absent, no-op
        freed[τ] = dict(u.n_bytes[τ])            # the τ-side bytes leave
    return freed

# Per-(tier, subpool) bytes the residence-set transition newly occupies
# at each destination tier (consumed against `cap_left[τ][sp]` in §9).
def bytes_acquired_by_migrate(u, add_tiers):
    acquired = {}
    for τ in add_tiers:
        if τ in u.residence:  continue           # already present, no-op
        source = authoritative_tier(u.residence)
        # Same physical bytes land on τ (write-through copies the unit
        # bit-for-bit; subpool layout is the same per architecture).
        acquired[τ] = dict(u.n_bytes[source])
    return acquired

unavailability_cost(u, add_tiers, remove_tiers) =
    p_hat(u, transfer_time(add_tiers, u, state))     # access during transfer
  × P(serve-from-source fails | write-through-semantics)  # 0 under write-through
  × reload_from(u, authoritative_tier(u.residence))

def migrate_candidates(state, D_t) -> list[Migrate]:
    """Enumerate the meaningful residence-set transitions for each u ∈ D_t.

    The candidate space is bounded by the small set of physically
    distinct transitions, not by the 2^|tiers| theoretical product:

      * `add  {DRAM}`               write-through (only if no DRAM yet)
      * `add  {DISK}`               archive (only if no DISK yet)
      * `remove {HBM}`              evict from HBM (only if DRAM ∪ DISK ≠ ∅,
                                       else this is a DROP candidate)
      * `remove {DRAM}`             evict from DRAM (only if DISK ≠ ∅ or
                                       HBM ≠ ∅; else DROP)
      * `remove {HBM, DRAM, DISK}`  DROP (whole-unit eviction)
      * `add  {HBM}`                load_back / predictive promote
                                       (only if HBM ∉ residence already)

    A meaningful pressure-relieving transition has nonempty
    bytes_freed in at least one (tier, subpool).
    """
    out = []
    for u in D_t:
        for (add, remove) in _meaningful_transitions(u):
            new_residence = (set(u.residence) | set(add)) - set(remove)
            if new_residence == set(u.residence):  continue   # no-op

            cost  = _value(u, set(u.residence), state) \
                  - _value(u, new_residence, state)
            cost += migration_cost(u, add, remove, state)
            cost += unavailability_cost(u, add, remove)

            relief   = bytes_freed_by_migrate(u, add, remove)
            acquired = bytes_acquired_by_migrate(u, add)
            if not any(b > 0 for subpools in relief.values()
                              for b in subpools.values()):
                continue                          # no pressure relieved
            out.append(Migrate(
                hash=u.hash,
                add_tiers=add,
                remove_tiers=remove,
                cost=cost,
                relief=relief,
                acquired=acquired,
            ))
    return out
```

**Action set semantics summary.**  Three physically distinct
"costs" decompose cleanly under the residence-set framing:

* **Adding** a tier that wasn't in residence is a write/copy — pays
  link bandwidth `bytes / bw_free(source, target)` and consumes
  destination subpool capacity.
* **Removing** a tier from residence is metadata-only — costs zero
  link bandwidth, frees the source's subpool capacity, and (if it
  was the authoritative tier) shifts the authoritative-tier role
  to the next-best surviving tier.
* **DROP** = `remove_tiers = residence`; the unit's hash leaves the
  radix tree.  Reload-from-DROP cost (`π_u × n_tokens`) is paid
  only on a future access that would have hit u.

The §9 DP's per-(tier, subpool) relief axes track `bytes_freed`
across all `Migrate` candidates simultaneously: a workload with
**one HBM subpool pressured and another spacious** (Mamba pool
full at 95% while attention `full` is 60%) sees the DP free
specifically Mamba bytes via Migrates whose `remove_tiers` reduce
Mamba residence.

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
        case MEMORY_PRESSURE | PRESSURE_CRITICAL | PRESSURE_RESOLVED:
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
| `MEMORY_PRESSURE`, `PRESSURE_CRITICAL`, `PRESSURE_RESOLVED` | top-k by regret across all units |

#### Helper definitions

```python
def shared_prefix_units(units):
    """Units held by ≥ 2 programs.  These are the cross-program
    prefix nodes whose V_u benefits from high p_hat aggregated
    over multiple holders (§7 conditional p_hat formulation)."""
    return frozenset(u for u in units if len(u.session_ids) >= 2)

def session_tail(units, sid):
    """The session's currently-held units, ordered by last access
    descending.  These are the demote candidates while p is
    ACTING (tool-bound) and the promote candidates ahead of p's
    next prefill.  No fixed-K cap — the joint knapsack (§9) will
    pick a subset under its byte budget."""
    return frozenset(u for u in units if sid in u.session_ids)

def session_scoped_units(units, sid):
    """Units held ONLY by this session — `session_ids == {sid}`.
    Used at SESSION_END: these units have no other holder, so
    after END they're either evicted (if all V_u<0) or demoted to
    the cheapest tier that still offers nonzero `p_hat` from the
    workload-prior (next program with this prefix shape will
    benefit from a warm DRAM/Disk copy)."""
    return frozenset(u for u in units if u.session_ids == {sid})

def child_output_unit(event):
    """The unit materialised by SUB_RETURN — the child sub-agent's
    final output, freshly committed to the radix tree as a new
    `subagent_ctx` unit.  Event payload carries its hash."""
    return event.payload["child_output_hash"]
```

#### `regret(u, state)` and `top_k_pressure(state)`

For the two pressure events, the candidate set is restricted by
**regret** — the V_u headroom we'd recover by moving u to its best
alternative tier:

```
regret(u, state) = _value(u, set(u.residence), state)
                 - max_{R' ∈ meaningful_neighbours(u.residence)}
                       _value(u, R', state)
```

`top_k_by_regret(units, k)` returns the k units of highest regret.
`k` is sized so the joint knapsack stays microsecond-scale:

```
bytes_needed(state)   = max(0, forecast(state) - theta_hi × cap_total)
                        # = same scalar §9 computes; reused here
mean_unit_bytes(state) = (Σ_{u ∈ state.units}
                            sum(bytes_at(u, τ) for τ in u.residence))
                       / |state.units|
                        # arithmetic mean across all units in HBM+DRAM+DISK

top_k_pressure(state) =
    min(K_MAX,
        max(K_MIN, bytes_needed(state) / mean_unit_bytes(state) × K_SAFETY))
```

Defaults: `K_MAX = 256`, `K_MIN = 16`, `K_SAFETY = 4`.  These are
deployment constants, not workload tuning knobs — they bound the
DP table size (§9), not the optimality of the chosen subset.
`V_u(u)` is the shorthand for `_value(u, set(u.residence), state)`
— the unit's value at its **current residence set** under the
live state.  Holding cost is paid only at the
`authoritative_tier(residence)`; lower-tier copies (e.g. DRAM
backup of an HBM-resident unit) are sunk-cost free per the §7
holding-cost rule.

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

#### `h_(τ, sp)(occupancy)` — per-byte holding cost at (tier, subpool)

Physical meaning: the seconds of opportunity cost per byte per
second of holding at subpool `sp` of tier `τ`, when that subpool
is at occupancy `occ`.  This is the marginal V_u of the byte
that would be displaced by adding one more byte to that subpool.
At low occupancy the displaced byte has near-zero value, at high
occupancy it has the V_u of the next-best resident in the same
subpool.

```
h_(τ, sp)(occ) = h_(τ, sp)_max × occ      # linear placeholder; see below
h_(τ, sp)_max  = state.tier_holding_cost[τ][sp].h_max_per_byte_sec
occ            = occupancy_of(τ, sp, state)
               = pool_usage[τ].subpools[sp].used_bytes
               / pool_usage[τ].subpools[sp].cap_bytes
```

Why per-subpool, not per-tier: in a hybrid Mamba+attention model,
the Mamba snapshot subpool and the attention `full` subpool can
sit at very different occupancies and have very different
displacement-cost curves (a Mamba snapshot is a much larger byte
unit and rarer-to-reuse).  Sharing one `h_τ` across both subpools
would mean Mamba pressure shows up as attention demotion or vice
versa.

The shape is **linear in occupancy as a placeholder** until T12
calibration determines the empirical curve.  Linear is the
simplest non-trivial monotone function — 0 cost when subpool is
empty (can't displace anything), `h_(τ, sp)_max` cost when full.

> **Planned (T12 calibration, per subpool).**  The true shape of
> `h_(τ, sp)(occ)` is the relationship between current subpool
> occupancy and the V_u of the marginal (lowest-V_u) resident
> **within that subpool** at that occupancy.  T12 measures this
> empirically per subpool: at every event during T9 / T11 runs,
> log `(subpool, occ, marginal_V_u)` triples across the workload
> mix.  Fit candidate shapes — linear `α × occ`, power
> `α × occ^γ`, hyperbolic `α / (1 - occ)` — and pick the one
> that minimises residual *per subpool*.  Hyperbolic is the most
> physically motivated candidate (diverges as occ → 1, matching
> the §9 admission cap), but the data picks the shape, not the
> prior.  Until T12 lands, linear with operator-calibrated
> per-subpool `h_max` is the spec.

#### `bw_free(σ, τ)` — free bandwidth on σ↔τ link

Live bytes-per-second available on the σ↔τ link, read directly
from `state.link_stats`:

```
bw_free(σ, τ) =
    state.link_stats["σ->τ"].recent_throughput_bps
        if state.link_stats["σ->τ"].samples_in_window > 0
    else
        state.link_stats["σ->τ"].peak_bw_bps   # link idle → assume free
```

Sourcing:

* **`peak_bw_bps`** is a deployment constant (PCIe gen5 ×16 for
  HBM↔DRAM is ~64 GB/s; NVMe drive nominal for DRAM↔DISK).  Set
  at sglang launch from operator config (`--aginfer-peak-bw-*`)
  or device probe.
* **`recent_throughput_bps`** is sglang's EMA over a ~100 ms
  window of `(bytes_moved / elapsed_time)` measured at the actual
  IO callsites: HiCache `write_backup` / `load_back` for
  HBM↔DRAM (CUDA-event bracketed), Mooncake `put` / `get` for
  DRAM↔DISK (wall-clock bracketed at the Python adapter
  boundary).  Each direction maintained separately.

The fallback to `peak_bw_bps` when no recent samples exist is not
a defensive fallback — an idle link genuinely has its full peak
available.  The samples are the dynamic correction, the peak is
the physical truth.

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

**Newborns are HBM-only by physical necessity.**  The forward
pass writes KV into HBM-backed tensors directly (GPU compute
writes go to device memory); there is no `alloc_at_tier(σ)` API
because there cannot be one — DRAM/Disk are pure storage
classes, only reachable through a load-back round-trip and
unable to host compute.  The first physical residence of every
unit is HBM-only by construction.

The first scheduling decision for a newborn is therefore **not
"where to allocate"** (the allocator has no choice) but
**"when to write-through to DRAM"** and **"when to evict from
HBM"** — two separate transitions on the unit's residence set.
Both are governed by the standard §7 / §8 / §9 machinery applied
to the unit's evolving residence: a unit's `_value` at HBM
becomes negative when its `p_hat` falls, and §9 schedules the
write-through (if not already backed) and HBM eviction.

## 8. Admission — program-level candidate generator

Admission is the **program-level candidate generator** that feeds
`joint_decide` (§9).  It does not run its own decision loop; the
loop lives in §9 and consumes its candidates alongside the
unit-level candidates from §7.

### Working view: iterate `state.per_program_usage`

All admission inputs live in sglang's `/aginfer/state`.
`pre_pause_state` is authored by the daemon (§6
`PUT /aginfer/program_paused`) but stored by sglang in
`per_program_usage[pid].pre_pause_state`, so admission reads it
the same way it reads every other field.  No daemon-tracker join,
no restart fallback.

```python
def _programs(state):
    """Iterate (pid, ProgramView) over state.per_program_usage."""
    for pid, pu in state.per_program_usage.items():
        yield ProgramView(
            id              = pid,
            state           = pu["state"],
            hbm             = pu["hbm"],
            dram            = pu["dram"],
            unit_hashes     = pu["unit_hashes"],
            pre_pause_state = pu["pre_pause_state"],
        )
```

For each event, the admission generator emits:

* one `Pause(p, cost, relief)` for every REASONING or ACTING program p
* one `Resume(p, gain, re_use)` for every PAUSED program p that
  fits `capacity_fits` (ENDED programs emit neither)

```python
def pause_candidates(state, event):
    return [
        Pause(
            program_id = p.id,
            cost       = V_u_program(p, state)                # work-loss (seconds)
                       + marginal_pause_cost(p, event),
            relief     = pause_relief(p, state),              # HBM bytes (≥ 0)
        )
        for p in _programs(state)
        if p.state in (REASONING, ACTING)                     # ENDED + PAUSED skipped
    ]

def resume_candidates(state):
    return [
        Resume(
            program_id = p.id,
            gain       = V_u_program_if_active(                # counterfactual seconds
                             p, state, p.pre_pause_state),
            re_use     = expected_peak_hbm_after_resume(p, state),
        )
        for p in _programs(state)
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

* `V_u_program(p, state) = Σ_{h ∈ p.unit_hashes} V_u(state.units[h])`
  — computed against the current state, used as Pause's cost (the
  V_u we'd lose if p stops here).  The conditional p_hat from §7
  is already a holder-product, so shared-prefix attribution is
  built in; no `1/|session_ids|` weight is needed.
* `V_u_program_if_active(p, state, hypothetical_state)` —
  counterfactual: compute V_u as if p's state field were
  overridden to `hypothetical_state` (and all other holders'
  states held fixed).  Used as Resume's gain.  The
  hypothetical is `p.pre_pause_state` — the state p was in when
  admission paused it — because resuming p restores exactly that
  pre-pause activity (a REASONING program that admission paused
  mid-decode goes back to REASONING; an ACTING program paused
  while awaiting a tool goes back to ACTING).  `pre_pause_state`
  is authoritatively stored in sglang's `per_program_usage`
  (written via `PUT /aginfer/program_paused`, §6) so daemon
  restarts never lose it.
* `marginal_pause_cost(p, state)` — work lost if we pause p now
  vs at the next natural off-GPU boundary.  The lost work is
  exactly p's current in-flight decode: the tokens decoded so
  far on the current request will be discarded and re-prefilled
  on resume.

  ```
  marginal_pause_cost(p, state) =
      sum(state.per_program_usage[p].hbm.inflight.values())
    / prefill_throughput(state)            # bytes/sec on the
                                            # prefill path
  ```

  Sum across HBM subpools because the in-flight decode allocates
  bytes to every subpool the architecture uses (attention `full`
  always, plus `swa` or `mamba` if hybrid).  All of those bytes
  represent decoded-so-far tokens that re-prefill on resume.

  This formulation is **event-independent** — `inflight` already
  encodes whether p is mid-decode.  Implications fall out naturally:

  * `TOOL_CALL_START`: at the instant of the event, p's request
    is completing; on the immediately-following event, all
    `inflight[sp] ≈ 0` for p → pause cost ≈ 0.  Matches the
    physical intuition: p was going off-GPU anyway.
  * `LLM_PREFILL` for p mid-decode: at least one `inflight[sp] > 0`
    → pause cost > 0 proportional to total-bytes-decoded-so-far.
  * `MEMORY_PRESSURE` / `PRESSURE_CRITICAL`: cost is whatever p's
    current inflight bytes happen to be; nothing special about
    the pressure event.
  * Resumed-but-no-request (F4): all `inflight[sp] == 0` → pause
    cost == 0 (pausing an idle program loses nothing).
All `forecast` / `relief` / `re_use` quantities are **per-HBM-subpool
dicts**.  A pause / migrate frees bytes from each HBM subpool it
contributes to, and admission gates per-subpool, so the bookkeeping
follows the subpool keys exposed by `state.pool_usage.HBM.subpools`.

* `pause_relief(p, state)` — per-HBM-subpool bytes pausing p saves
  the scheduler from holding, **including future-inflight savings
  the pause prevents**:

  ```
  pause_relief(p, state)[sp] =                    # per HBM subpool sp
      snapshot_relief(p, state)[sp]
    + future_inflight_savings(p, state)[sp]
  ```

  The `future_inflight_savings` term is what makes Pauses
  trajectory-strong (Migrates only deliver snapshot relief).
  This is *why* PRESSURE_CRITICAL doesn't need a special cost
  twist in §9: under high `forecast`, `future_inflight_savings`
  is large for actively-decoding programs, so Pauses win on
  relief/cost naturally.

  * `snapshot_relief(p, state)[sp]` — read from
    `state.per_program_usage[p].hbm`:
    * **inflight[sp]** — bytes p is currently using in subpool sp
      for in-flight decode; released when p's current request
      completes / is preempted.  Pause prevents the program's NEXT
      request from re-allocating these.
    * **committed[sp]** — p's exclusive share of subpool-sp radix
      bytes; if pausing p drops `len(session_ids)` of a node to 0
      on that subpool, the node becomes evictable on that subpool.
  * `future_inflight_savings(p, state)[sp]` — per-subpool projected
    growth pausing p averts.  Computed symmetrically to
    `forecast_inflight_demand` (below): cap `E[remaining_tokens(p)]`
    by `forecast_horizon × decode_throughput(p)` and multiply by
    p's per-token bytes contribution to subpool sp.  0 if p has
    `inflight[sp] == 0` (p isn't currently allocating in that
    subpool — Mamba state for a sequence is allocated once and
    fixed; attention `full` grows monotonically with decode).

* `expected_peak_hbm_after_resume(p, state)` — **incremental**
  per-HBM-subpool bytes p will allocate when resumed.  Only count
  bytes **not already in HBM**: a paused program whose units are
  still resident (because another holder kept them alive)
  consumes zero new HBM on resume on that subpool.

  ```
  expected_peak_hbm_after_resume(p, state)[sp] =
      Σ_{h ∈ p.unit_hashes}
          (bytes_at_subpool(state.units[h], "HBM", sp)
              if "HBM" not in state.units[h].residence
              else 0)
                                                  # only count units not
                                                  # currently HBM-resident
                                                  # (demoted, dropped, or
                                                  # absent).  For HBM+DRAM
                                                  # coexisting units, the
                                                  # HBM copy still satisfies
                                                  # resume needs — no new
                                                  # HBM bytes consumed.
    + future_inflight_savings(p, state)[sp]       # post-resume decode
                                                  # growth in subpool sp

  bytes_at_subpool(u, τ, sp) = u.n_bytes.get(τ, {}).get(sp, 0)
  ```

  Why "not already in HBM" is the correct filter: HBM bytes for a
  unit already resident are accounted in `pool_usage[HBM].subpools[sp].used_bytes`
  (the first term of `forecast`).  `capacity_fits` checks each
  subpool independently — re-adding bytes via `re_use[sp]` would
  double-count them and over-reject Resume candidates whose units
  are kept alive by other holders.

* `capacity_fits(p, state)` — true iff for **every** HBM subpool sp:
  `forecast(state)[sp] + expected_peak_hbm_after_resume(p, state)[sp]
   ≤ theta_hi × pool_usage[HBM].subpools[sp].cap_bytes`.
  Resume candidates failing this gate on any subpool are omitted
  (not in the candidate set).
* `forecast(state)` — per-HBM-subpool dict.  For each subpool sp:
  ```
  forecast(state)[sp] =
      state.pool_usage["HBM"].subpools[sp].used_bytes
    + forecast_inflight_demand(state)[sp]
  ```

  The second term is the **expected** HBM growth before the next
  event in subpool sp, summed over programs that **are actively
  decoding right now in that subpool**:

  ```
  forecast_inflight_demand(state)[sp] =
    Σ_p min(E[remaining_tokens(p)],
            forecast_horizon(state) × decode_throughput(p))
        × bytes_per_token_in_subpool(p, sp)
    for p in state.per_program_usage
    if p.hbm.inflight[sp] > 0
  ```
  where `bytes_per_token_in_subpool(p, sp)` is the model-architecture
  constant for how many bytes per decoded token p adds to subpool sp
  (attention `full` ≈ `2 × num_full_layers × hidden_dim × dtype_size`;
  SWA same but only retained within the window; Mamba state grows
  only at snapshot boundaries, **not** per-token, so for the `mamba`
  subpool this is 0 for in-flight forecast and snapshot allocations
  are handled separately at radix-tree commit time).

  Two refinements over the naive `Σ_{p ∈ REASONING} E[remaining]`:

  * **`inflight[sp] > 0` filter, not `state == REASONING`.**  A
    program can be REASONING but have no in-flight request
    (e.g. just-resumed but client hasn't retried yet) — those
    don't contribute to near-term HBM growth.  REASONING is a
    semantic label about what the program is trying to do; the
    physical question "is subpool sp growing for p RIGHT NOW" is
    answered by `inflight[sp]`.
  * **Capped by `forecast_horizon`.**  We forecast only as far as
    the next decision opportunity; beyond that, a new
    `joint_decide` will re-evaluate.  `E[remaining_tokens(p)]` is
    the full residual decode (can be thousands of tokens, many
    seconds), but the horizon is set by the next event arrival,
    so we cap at `horizon × decode_throughput(p)` tokens.

  ```
  forecast_horizon(state) =
      min(heartbeat_s,                     # bounded by webhook re-fire
          1.0 / recent_event_rate(state))  # typical when active
  ```

  Under HIGH/CRITICAL pressure, sglang re-fires every
  `heartbeat_s` (≈ 5 s default), so the horizon is at most that.
  Under normal load (10²/s event rate), the horizon shrinks to
  ~10 ms — well-aligned to the cadence of intra-decode growth.

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
space** `A = {unit migrate} ∪ {program pause/resume}` (per §3).
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

Two phases, each is a **0/1 knapsack solved by exact DP**.  The
pressure phase is genuinely multi-resource because items consume
bytes from *different* tier budgets (HBM-relief, DRAM-room, DISK-room),
so the DP enumerates over a 4-dimensional state.  The headroom
phase is single-resource (HBM-room only).

The decision rule is **event-priority-agnostic**: `PRESSURE_CRITICAL`
takes the same code path as `MEMORY_PRESSURE`.  Urgency enters
through `forecast(state)` (larger at higher occupancy), which
makes `bytes_needed` bigger, which makes Pauses dominate Migrates
in the DP via `pause_relief`'s `future_inflight_savings` term (§8).
The CRITICAL-specific behaviour lives in the event router (§4) —
CRITICAL events preempt the queue, they don't change the decision
function.

* **Pressure phase** runs when `forecast > theta_hi × cap_bytes`.
  Items: Pause programs + Migrate-HBM-out units.  Resources:
  HBM-relief budget + per-destination tier-room budgets (DRAM,
  DISK).  Goal: free at least `bytes_needed` HBM **while not
  overflowing destination tiers**, minimising total V_u cost.
* **Headroom phase** runs when `forecast < theta_lo × cap_bytes`.
  Items: Resume paused programs.  Resource: HBM-room available
  before crossing `theta_hi` again.  Goal: maximise total V_u
  gain subject to bytes-reclaimed ≤ free_room.

The two phases are mutually exclusive per event (forecast is
either too high or too low; in between is the hysteresis band
where neither phase has anything to do).

```python
def capacity_left_bytes(state, τ, sp):
    """Bytes free in subpool sp of destination tier τ."""
    s = state.pool_usage[τ]["subpools"][sp]
    return s["cap_bytes"] - s["used_bytes"]

def joint_decide(state, event):
    hbm_subpools  = state.pool_usage["HBM"]["subpools"]
    dram_subpools = state.pool_usage["DRAM"]["subpools"]
    disk_subpools = state.pool_usage["DISK"]["subpools"]

    # Pressure: per (HBM, subpool) target = max(0, forecast - theta_hi × cap).
    bytes_needed = {
        ("HBM", sp): max(0, forecast(state)[sp]
                            - theta_hi * hbm_subpools[sp]["cap_bytes"])
        for sp in hbm_subpools
    }
    if any(b > 0 for b in bytes_needed.values()):
        cap_left = {("DRAM", sp): capacity_left_bytes(state, "DRAM", sp)
                    for sp in dram_subpools} | \
                   {("DISK", sp): capacity_left_bytes(state, "DISK", sp)
                    for sp in disk_subpools}
        cands  = kv_scheduler.migrate_candidates(state, decision_set(event, state))
        cands += admission.pause_candidates(state, event)
        cands  = [c for c in cands
                  if any(b > 0
                         for sub in (c.relief if isinstance(c, Migrate)
                                     else {"HBM": c.relief}).values()
                         for b in sub.values())]
        return knapsack_min_cost_multi(
            items        = cands,
            bytes_needed = bytes_needed,             # keyed by (HBM, sp)
            cap_left     = cap_left,                 # keyed by (DRAM|DISK, sp)
            bucket_size  = page_bytes,
        )

    # Headroom: per HBM-subpool slack budget.  Resumes consume HBM
    # bytes in each subpool the resumed program's units live in.
    free_room = {
        ("HBM", sp): max(0, theta_lo * hbm_subpools[sp]["cap_bytes"]
                            - forecast(state)[sp])
        for sp in hbm_subpools
    }
    if all(r > 0 for r in free_room.values()):
        cands = admission.resume_candidates(state)
        cands = [c for c in cands if any(b > 0 for b in c.re_use.values())]
        return knapsack_max_value_multi(
            items       = cands,
            budget      = free_room,                 # keyed by (HBM, sp)
            bucket_size = page_bytes,
        )

    return []                                        # hysteresis dead-zone
```

A workload with a Mamba snapshot subpool at 95 % occupancy and an
attention `full` subpool at 60 % will see `bytes_needed[mamba] > 0
and bytes_needed[full] == 0` — the DP picks candidates that relieve
specifically `mamba` (Pauses of programs holding large mamba state,
Migrates of mamba-leaf units to DRAM/Disk), without disturbing
attention-resident units that are cheap and healthy.

Both phases are **exact 0/1 knapsack** at K ≈ tens of items
(K_MAX = 256 per §7 top-k cap).  Resource-axis count is
`|HBM subpools| + |DRAM subpools| + |DISK subpools|`, varying
per architecture (see §12 Scenarios for concrete counts).

Dense table size grows exponentially in axis count; the DP runs
**sparse** over a `dp` dict (cells reachable from any subset-take).
On representative candidate sets fewer than ~10⁵ tuples are
reachable, ≪ dense bound, at any practical axis count.

```python
def _bk(n, bucket_size):                            # round-down quantisation
    return n // bucket_size

def _bk_up(n, bucket_size):                         # round-up quantisation
    return (n + bucket_size - 1) // bucket_size

# Axis enumeration helpers — pressure phase has one >= axis per HBM
# subpool and one <= axis per (DRAM|DISK, subpool); headroom phase
# has one <= axis per HBM subpool only.
def _hbm_relief_axes(bytes_needed):                 # >= axes (frees HBM)
    return [("HBM", sp) for sp in bytes_needed]
def _dest_cap_axes(cap_left):                       # <= axes (destinations)
    return list(cap_left.keys())                    # [(DRAM, sp), (DISK, sp), ...]

def _relief_at(c, tier, sp):
    """Bytes the candidate frees on (tier, sp)."""
    if isinstance(c, Pause):
        return c.relief.get(tier, {}).get(sp, 0) if isinstance(c.relief.get(tier), dict) \
               else (c.relief.get(sp, 0) if tier == "HBM" else 0)
    return c.relief.get(tier, {}).get(sp, 0)

def _acquire_at(c, tier, sp):
    """Bytes the candidate adds to destination (tier, sp).  Pauses
    and pure-remove Migrates contribute 0."""
    if isinstance(c, Migrate):
        return c.acquired.get(tier, {}).get(sp, 0)
    return 0

def knapsack_min_cost_multi(items, bytes_needed, cap_left, bucket_size):
    """0/1 knapsack: subset S minimising Σ cost(s∈S) subject to
    (a) for each (HBM, sp) axis:
        Σ_{s∈S} relief[HBM][sp](s)   >= bytes_needed[(HBM, sp)]
    (b) for each (DRAM|DISK, sp) axis a:
        Σ_{s∈S} acquired[a](s)       <= cap_left[a]

    Pure-remove Migrates contribute 0 to (b) by construction.
    Pauses contribute relief on the HBM tier only.  Quantises
    relief round-DOWN (safe for >=); quantises destination
    consumption round-UP (safe for <=)."""
    relief_axes = list(bytes_needed.keys())          # list of (HBM, sp)
    cap_axes    = list(cap_left.keys())              # list of (DRAM|DISK, sp)
    W    = {a: _bk_up(bytes_needed[a], bucket_size) for a in relief_axes}
    Wcap = {a: _bk(cap_left[a], bucket_size)        for a in cap_axes}
    K    = len(items)
    INF  = float("inf")

    # dp state encodes (relief_buckets, cap_buckets) as a flat tuple.
    # Sparse — only reachable cells materialise.
    def zero_state():
        return (*(0 for _ in relief_axes), *(0 for _ in cap_axes))

    dp   = {zero_state(): 0.0}
    take = {}
    for k, c in enumerate(items, start=1):
        d_relief = tuple(_bk(_relief_at(c, tier, sp), bucket_size)
                         for (tier, sp) in relief_axes)
        d_cap    = tuple(_bk_up(_acquire_at(c, tier, sp), bucket_size)
                         for (tier, sp) in cap_axes)
        new_dp = dict(dp)
        for s, cost in dp.items():
            n = len(relief_axes)
            r_buckets, cap_buckets = s[:n], s[n:]
            r_new = tuple(min(W[relief_axes[i]], r_buckets[i] + d_relief[i])
                          for i in range(n))
            cap_new = tuple(cap_buckets[i] + d_cap[i]
                            for i in range(len(cap_axes)))
            if any(cap_new[i] > Wcap[cap_axes[i]] for i in range(len(cap_axes))):
                continue
            s_new = r_new + cap_new
            new_cost = cost + c.cost
            if new_cost < new_dp.get(s_new, INF):
                new_dp[s_new] = new_cost
                take[(k, s_new)] = True
        dp = new_dp

    # Feasible cells: every (HBM, sp) relief axis hit its bucket bound.
    full_r = tuple(W[a] for a in relief_axes)
    feasible = [(c, s) for s, c in dp.items() if s[:len(relief_axes)] == full_r]
    if feasible:
        _, s_pick = min(feasible)                    # min over cost
    else:
        # No subset satisfies every (HBM, sp) target under (b) caps.
        # Pick the cell maximising Σ relief buckets (closest to
        # satisfying), tie-break on min cost.  Every cell already
        # respects (b) by construction.  sglang's next webhook
        # re-fire surfaces residual pressure.
        s_pick = max(dp,
                     key=lambda s: (sum(s[:len(relief_axes)]),
                                    -dp[s]))
    chosen, k = [], K
    while k > 0:
        if take.get((k, s_pick)):
            c = items[k - 1]
            chosen.append(c)
            n = len(relief_axes)
            r_buckets   = list(s_pick[:n])
            cap_buckets = list(s_pick[n:])
            for i, (tier, sp) in enumerate(relief_axes):
                r_buckets[i] = max(0, r_buckets[i]
                                  - _bk(_relief_at(c, tier, sp), bucket_size))
            for i, (tier, sp) in enumerate(cap_axes):
                cap_buckets[i] -= _bk_up(_acquire_at(c, tier, sp), bucket_size)
            s_pick = tuple(r_buckets) + tuple(cap_buckets)
        k -= 1
    return chosen


def knapsack_max_value_multi(items, budget, bucket_size):
    """0/1 knapsack: subset S maximising Σ gain(s∈S) subject to
    for each (HBM, sp) axis a:
        Σ_{s∈S} re_use[sp](s)   <= budget[a]
    Quantises re_use round-UP (safe for <=).  Same sparse multi-axis
    DP shape as knapsack_min_cost_multi."""
    axes = list(budget.keys())                       # list of (HBM, sp)
    W    = {a: _bk(budget[a], bucket_size) for a in axes}
    K    = len(items)
    NEG  = float("-inf")

    dp   = {tuple(0 for _ in axes): 0.0}
    take = {}
    for k, c in enumerate(items, start=1):
        d = tuple(_bk_up(c.re_use.get(sp, 0), bucket_size)
                  for (tier, sp) in axes)
        new_dp = dict(dp)
        for s, gain in dp.items():
            s_new = tuple(s[i] + d[i] for i in range(len(axes)))
            if any(s_new[i] > W[axes[i]] for i in range(len(axes))):
                continue
            new_gain = gain + c.gain
            if new_gain > new_dp.get(s_new, NEG):
                new_dp[s_new] = new_gain
                take[(k, s_new)] = True
        dp = new_dp

    s_pick = max(dp, key=dp.get)
    chosen, k = [], K
    while k > 0:
        if take.get((k, s_pick)):
            c = items[k - 1]
            chosen.append(c)
            s_pick = tuple(s_pick[i] - _bk_up(c.re_use.get(axes[i][1], 0),
                                              bucket_size)
                           for i in range(len(axes)))
        k -= 1
    return chosen
```

`Migrate` carries a `target_subpool: str` field — when sglang's
unified cache moves a unit's bytes from one tier to another, the
destination tier's subpool layout determines where each component
of the unit lands.

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
> of exact DP, and treats the destination-tier sub-budgets as a
> per-item filter rather than as a multi-dimensional knapsack
> resource.  This is a pure code lag — both replacements run in
> the same microsecond order on K ≈ 30 candidates.  Replace with
> `knapsack_min_cost_multi` / `knapsack_max_value` per the
> pseudo-code above.

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
| **Single-worker event loop**: handlers serialised; no concurrent migrate races; no internal timer anywhere — kv_scheduler, admission, forecast refresh, program_tracker, and the proxy all recompute only on event arrival.  SESSION_END is signalled by the client explicitly (§4); there is no time-based fallback | asyncio queue + single consumer |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent daemon→sglang actions**: every endpoint accepts re-application — migrate (a no-op `applied=0` with a `race:*` skip when the target tier already matches), pause/resume (200 with `applied:false` when the program is already in the requested state), hint PUT (overwrite-by-stamp).  Same reasoning across all of them: the daemon may emit the same action twice across consecutive event handlers because state-dump propagation lags the daemon's just-emitted action; the sglang side must absorb this without error | sglang endpoints |
| **Outbound queue is volatile**: the daemon's outbound action queue lives in memory only.  On daemon crash, pending actions are lost — and that's correct.  After restart, the daemon's first `GET /aginfer/state` reads the live state; if the lost action was needed, `joint_decide` re-issues it.  No disk WAL exists for outbound actions, by the same first-principles argument that rules out a webhook persistence WAL (§11): every authoritative quantity already lives in sglang's state, the daemon is just a decision function over it | daemon outbound worker |
| **State-dump internal consistency**: a single `/aginfer/state` response is a snapshot taken under a single read-side lock on sglang's tree cache + allocator + per-program tables.  `units[*]`, `per_program_usage[*].unit_hashes`, and `pool_usage` always refer to the same logical timestamp; `state.units[h]` is safe to dereference for every `h ∈ per_program_usage[p].unit_hashes` | sglang `dump_aginfer_state` |
| **Hint clear ordering**: when sglang evicts unit u, the order is (1) inline scorer finishes its current heap-iteration read (using the hint that was live when u was picked); (2) eviction commits, allocator reclaims u's bytes; (3) hint entry for u is cleared.  This rules out a "scorer reads cleared hint" race; the scorer never sees a missing entry for a unit that's still in the heap | sglang inline scorer + allocator commit path |
| **Webhook mandatory**: every sglang launch passes `--aginfer-notify-url`; the admission trigger path is not optional | launch script per-deployment |
| **Pool-truth admission, per subpool**: admission's `forecast(state)` returns a dict over HBM subpools, each reading `state.pool_usage.HBM.subpools[sp].used_bytes` and `cap_bytes` (byte-denominated, since `forecast_inflight_demand` is byte-scaled and §9 budgets are bytes), never `tier_usage` (radix view).  Pressure fires when **any** subpool's forecast crosses `theta_hi`; this prevents the failure mode where a Mamba snapshot subpool at 95 % is hidden by an attention subpool at 60 %.  If the snapshot lacks `pool_usage.HBM.subpools` the daemon halts loudly | daemon `admission_controller.forecast` |
| **Physical inputs sourced from sglang**: `bw_free(σ, τ)` reads `state.link_stats[σ→τ]`, never an in-daemon estimate; `h_τ(occ)` reads `state.tier_holding_cost[τ].h_max_per_byte_sec`, never a constant baked into the daemon.  Sglang owns the measurements (HiCache + Mooncake instrumentation hooks expose link throughput; operator config sets per-tier `h_max`); the daemon consumes them.  If `link_stats` or `tier_holding_cost` are missing the daemon halts loudly | sglang instrumentation + operator config |
| **Tree-view V_u**: V_u migration scoring reads `tier_usage`, never `pool_usage` | daemon `OursGreedyPolicy._value` |
| **All traffic through daemon proxy**: every chat-completion to sglang arrives via the daemon's `/v1/chat/completions` proxy; direct-to-sglang clients are out of scope and would render admission's program-pause unenforceable | deployment topology |
| **Hint table covers every live unit**: sglang seeds a "fresh access just happened" entry on unit birth (`p_hat ≈ 1` for the next event horizon); the daemon refines via `PUT /aginfer/hints`; eviction never falls back to LRU on absent hints | sglang allocator hook + daemon hint pusher |
| **Hint atomicity**: the inline scorer's read of a hint entry and the daemon's `PUT` of a new entry are atomic per-key (read-modify-write would race a daemon update against an in-flight eviction).  Per-key seqlock or compare-and-swap suffices; full RW lock is overkill at 10²/s | sglang hint-table data structure |
| **Layer enable flags**: HiCache, kv_scheduler, and admission each have an independent enable flag.  Admission can only fire when kv_scheduler is also on (admission's pre-pause migrate path requires the daemon's V_u machinery).  HiCache is independent of both | daemon CLI + sglang flags |
| **Threshold parity**: sglang and the daemon use the SAME `theta_hi` / `theta_lo` / `theta_crit` / `heartbeat_s` values.  The daemon is the canonical source; sglang reads on bootstrap (or from a local cache file if the daemon is unreachable), and the daemon broadcasts runtime changes via `PUT /aginfer/thresholds` (§6).  sglang and daemon are not co-launched: each can restart independently while preserving the invariant | daemon `GET /aginfer/thresholds` + sglang local cache + daemon→sglang update broadcast |

## 11. Recovery (daemon restart / sglang restart)

`per_program_usage` (in sglang's `/aginfer/state`) is the
authoritative store for every program-level state field that
survives across crashes — `state`, `pre_pause_state`,
`unit_hashes`, `hbm`/`dram` footprint.  The daemon's in-process
`program_tracker` is a **derived cache**, rebuilt from
`/aginfer/state` on every restart; it never holds information
not also persisted in sglang.

The daemon's *truly* volatile state is narrower:
the proxy's pause-gate suspended request set (a Python set in the
proxy process) and the `migrate` retry queue.

### Daemon restart

On crash + restart the daemon's recovery sequence is:

1. `GET /aginfer/state` — pulls the authoritative snapshot of
   `tier_usage`, `pool_usage`, `per_program_usage` (including
   `state` and `pre_pause_state`), and `units`.  The daemon
   rebuilds its `program_tracker` cache by walking
   `per_program_usage`: each entry's `state` becomes the
   tracker's REASONING / ACTING / PAUSED / ENDED label.  No
   field is lost.
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

   `pre_pause_state` survives daemon restart because it is
   authoritatively stored in sglang's `per_program_usage` (§5,
   written via `PUT /aginfer/program_paused` on every pause /
   resume transition; §6).  The Resume gain counterfactual (§8)
   reads it directly from `/aginfer/state`; no in-process daemon
   memory is required.
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
**lost** — and that's by construction the correct behaviour, not
a trade-off.

The first-principles argument: an event in aginfer is **the
derivative of state**, and state is authoritative in sglang
(§5).  Every daemon handler re-fetches `/aginfer/state` at entry
(the always-fresh invariant, §10), so the handler's input is the
live state regardless of whether any specific event payload was
delivered.  A "lost" webhook just means the daemon never woke up
for that specific transition; the next event of any kind will
wake it, it re-fetches state, and the state already reflects
the missed transition.  In particular:

* sglang continues firing on subsequent transitions and on
  heartbeats while in HIGH / CRITICAL, so within `heartbeat_s` ≈
  5 s the daemon receives another webhook and re-fetches the
  state that already accounts for the missed event.
* Any chat-completion through the proxy also produces an event;
  on event rates of 10²/s the next wake-up is sub-second.

A persistent queue or WAL between sglang and daemon would be a
*duplicate authoritative store* for the same state sglang
already holds.  At-least-once delivery doesn't add information
to a system where the recipient re-reads the source on every
handler — the recipient already has at-least-once knowledge of
state via the pull.

## 12. Scenarios

The framework above is architecture-agnostic: it operates on
named-subpool dicts (`pool_usage.<tier>.subpools`, `u.n_bytes`,
`tier_holding_cost`).  Concrete sglang deployments instantiate
the dicts according to the model's attention / state-space
component mix.  Each scenario below lists exactly which keys
appear and how unit bytes split across them; the §7 / §8 / §9
machinery is unchanged.

### S1 — Single-stack attention (e.g. DeepSeek-V4-Flash / MLA, Llama, Qwen)

One attention component, no SWA, no Mamba.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"attn": {...}}` |
| `pool_usage.DRAM.subpools` | `{"attn": {...}}` |
| `pool_usage.DISK.subpools` | `{"attn": {...}}` |
| `units[i].n_bytes` | `{"attn": int}` for every unit |
| `tier_holding_cost.HBM` | `{"attn": h_max}` |
| §9 DP axis count | 3 (HBM-attn relief + DRAM-attn cap + DISK-attn cap) |

forecast / bytes_needed / capacity_fits each have a single
`attn` key.  The multi-axis sparse DP reduces to the 3-axis case
matching the round-6 multi-resource baseline.

### S2 — SWA-hybrid attention (e.g. Mistral, Gemma)

Two attention components: full-attention layers and sliding-window
layers, with separate sub-pools because SWA layers retain only the
last N tokens per sequence.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"full": {...}, "swa": {...}}` |
| `pool_usage.DRAM.subpools` | `{"full": {...}, "swa": {...}}` |
| `pool_usage.DISK.subpools` | `{"full": {...}, "swa": {...}}` |
| `units[i].n_bytes` | `{"full": F, "swa": S}` where `S > 0` only when the unit's tokens are still inside the SWA window; `S == 0` for aged-out units |
| §9 DP axis count | 5 (full + swa across HBM relief, DRAM cap, DISK cap; plus the same across DRAM and DISK destinations) |

Architecture-specific behaviour: when the SWA window slides past
a unit's tokens, sglang transitions the unit's `n_bytes.swa`
from `S` to `0` at the next state-dump.  The unit remains in
the radix tree (its `full` bytes are still valuable for prefix
reuse), but its SWA contribution is gone.  No special daemon
handler is needed; the next `joint_decide` event sees the new
`n_bytes` shape and re-scores naturally.

### S3 — Mamba+attention hybrid (e.g. Jamba, Zamba)

Mixed transformer layers with attention KV per token, and Mamba
SSM layers with state vectors snapshotted at radix-tree leaf
nodes (sglang's `MambaPool` / `hi_mamba_radix_cache.py`).

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"full": {...}, "mamba": {...}}` |
| `pool_usage.DRAM.subpools` | `{"full": {...}, "mamba": {...}}` |
| `pool_usage.DISK.subpools` | `{"full": {...}, "mamba": {...}}` |
| `units[i].n_bytes` for an intermediate node | `{"full": F, "mamba": 0}` (no snapshot) |
| `units[i].n_bytes` for a **leaf** node | `{"full": F, "mamba": M}` (snapshot present, size determined by `mamba_cache_chunk_size`) |
| §9 DP axis count | 5 (same shape as S2; subpool keys differ) |

Architecture-specific behaviour: only leaf nodes carry mamba
bytes (sglang stores exactly one mamba snapshot per leaf).  When
the radix tree extends a leaf (new tokens commit underneath), the
ex-leaf becomes intermediate and its mamba bytes either move to
the new leaf or are freed — sglang manages this transparently;
the daemon just observes the updated `n_bytes` shape on the next
state-dump.  In-flight Mamba state grows only at snapshot
boundaries, so `forecast_inflight_demand[mamba]` is 0 between
snapshots (§8).

### S4 — Mamba+SWA+full (future hybrid)

Three subpool keys in HBM; not yet observed in production models
but the framework supports it without modification.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"full": {...}, "swa": {...}, "mamba": {...}}` |
| `units[i].n_bytes` | `{"full": F, "swa": S, "mamba": M}` with `S = 0` when aged out of the SWA window and `M = 0` when the unit isn't a Mamba leaf |
| §9 DP axis count | 7 (3 HBM subpools × {HBM-relief, DRAM-cap, DISK-cap} minus the relief-axis overlap; see §9 axis counting) |

Sparse DP cell count remains the binding factor — at 7 axes the
dense table is intractable but the reachable-cell count under
agent workloads (where most candidates touch only 1–2 axes
nontrivially) stays in the 10⁵ envelope.

### Residence evolution

A unit's `residence` set evolves through the §6
`POST /aginfer/migrate` actions (and through sglang's own
default-policy plugin when no daemon is attached).  Canonical
lifecycle for an attention KV unit:

```
{HBM}                  fresh, just allocated by forward pass
  → {HBM, DRAM}        after `should_write_through(u)` triggers
                          (default: hit_count ≥ threshold;
                           aginfer: V_u(R∪DRAM) > V_u(R))
  → {DRAM}             after HBM pressure forces remove HBM
  → {DRAM, HBM}        re-accessed; load_back populates HBM,
                          DRAM kept as backup
  → ...                cycles between {DRAM} and {DRAM, HBM}
  → {DRAM, DISK}       on DRAM pressure: archive via Mooncake,
                          DRAM kept until DRAM pressure
  → {DISK}             on DRAM eviction
  → {}                 final DROP (radix-tree node removed)
```

Mamba snapshot units follow the same lifecycle but the bytes
involved live in different subpools (see S3).

### Scenario-independent assertions

Across S1 – S4 the daemon code path is identical: read
`pool_usage.HBM.subpools.keys()` at state-dump entry, iterate
over those keys for every per-subpool quantity (`forecast`,
`bytes_needed`, `cap_left`, `pause_relief`, `re_use`,
`tier_holding_cost`).  No scenario-specific branch lives in the
daemon — the keys come from the state-dump, the rest follows.
