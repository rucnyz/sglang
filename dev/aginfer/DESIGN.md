# aginfer — design

External KV-cache scheduler for sglang serving multi-turn agentic
workloads.  This document specifies *what the system should be*, not
the path that got us here.  Where the implementation is a simplification
of what's described, the simplification is called out explicitly.

## 1. What it is

A daemon-side scheduler that decides, on every workload-relevant
event:

* which cache **units** belong in which **tier** (HBM / DRAM / DISK / DROP)
* which **programs** should be paused or resumed under memory pressure

The mechanism is **event-driven**, **per-unit value-based**,
**anticipatory** — it schedules actions ahead of need (promote KV
before the reuse arrives, free HBM before a spike lands) via the
three-plane model (§3) — and **externalised from sglang** so the
inference engine stays general while scheduling policy is free to
iterate.

**Out of scope**: multi-tenant fairness / per-program priority /
per-tenant SLOs.  The design optimises throughput on the shared
HBM resource and does not enforce a starvation guard for low-V_u
programs.  Fairness is the responsibility of an upstream
admission gating layer above harbor (route limiter, priority
queue, etc.) that pre-shapes the workload the daemon sees.

## 2. Workload model — why value-aware, tiered, program-level

aginfer assumes **multi-turn agentic inference**: programs run tool-bound
reasoning loops over tens of turns, accumulating shared and per-program KV, each
off-GPU during its tool calls, with heterogeneous runtimes (5×–100×). Three facts
about this workload drive every design choice:

1. **Per-unit KV value is highly non-uniform.** A prefix shared by 32 concurrent
   programs ≫ a 30-turn-old scratch held by one. LRU treats them the same and
   gets it backwards → the decision needs a per-unit value `V_u`, not recency.

2. **Cache value is a continuum, not binary keep/evict.** "1% future-reuse"
   doesn't deserve HBM but isn't worth dropping either. Hardware exposes a tier
   hierarchy (HBM ≫ DRAM ≫ DISK in $/bandwidth) → the decision space is the
   4-tier residence set `{HBM, DRAM, DISK, DROP}`, not one bit.

3. **KV is owned by programs.** "Pause this whole program because its tools are
   slow and its value is below another's" cannot be expressed by evicting bytes →
   the design needs program-level pause/resume, not only unit migration.

The concrete agentic *signals* this exploits — typed tool gaps, loop determinism,
the fleet-shared prefix, context compaction, session / sub-agent lifecycle,
fan-out spikes — and the specific scenarios where each yields a *measurable* win
over a cached baseline (with workloads, metrics, and falsification tests) are
catalogued in **`wherewewin/`**. This section keeps only the three load-bearing
facts that justify the value model; the scenario detail lives there, not here.

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
       │          DRAM↔DISK via Mooncake)                   │
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
| kv_scheduler | daemon | workload events (13 kinds, §4) | **unit-migrate candidate generator** consumed by §9 `joint_decide` |
| admission_controller | daemon | every workload event | **program pause / resume candidate generator** consumed by §9 `joint_decide` |

All three use the **same V_u rule**.  They live where they live
because of **event ownership**, not as fallbacks for each other:
the eviction-decision callsite is sglang-internal, fires
synchronously on the scheduler step, and cannot wait for an HTTP
round-trip to the daemon, so its V_u handler must be in-process.
The 13 workload events are surfaced over HTTP (proxy + sglang
webhook), so their V_u handlers live in the daemon.

### Three planes: belief, decision, action-timeline

aginfer is structured as three decoupled planes. The split is what lets it be
**anticipatory** — promote before the request arrives, free HBM before a spike
lands, drop a corpse before it blocks live work — which is its whole value over
reactive HiCache, while still ruling out the failure mode the earlier "no
internal timer" rule guarded against.

1. **Belief plane — event-sourced, never time-inferred.** The daemon's model of
   the world (which units exist, their residence sets, holder counts, per-program
   lifecycle REASONING / ACTING / PAUSED / ENDED, per-unit value `V_u`) is updated
   **only** by events (§4) and the always-fresh `/aginfer/state` re-fetch. State is
   **never inferred from elapsed wall-clock** — there is no "this program LOOKS
   idle because N seconds passed" rule, because that races a legitimately-slow
   tool.

2. **Decision plane — event-triggered, value-optimal.** On each event,
   `joint_decide` (§9) chooses an optimal subset over the union action space
   {unit migrate, program pause/resume} given the current belief **and any
   prediction the event carries** (a `TOOL_CALL_START`'s tool ETA, a
   `SUB_DISPATCH_ASYNC`'s imminent-demand estimate). A chosen action may be tagged
   with a **due-time** (a "schedule a future action") — this is not a new DP axis,
   just an ordinary migrate/pause carrying when it should fire, handed to the
   action plane. A prediction is an *input to a decision*, never a substitute for
   observed state.

3. **Action-timeline plane — temporal execution, belief-validated.** A decided
   action may be **scheduled for a future moment** when that is when it should
   land: a predictive promote at `T_start + tool_ETA − load_back_latency` (§7), a
   pre-emptive HBM free just before a forecast spike (§8). Scheduled actions live
   in a small due-action min-heap and are dispatched **when the event stream next
   advances past their due time** — the serialized single-consumer event stream is
   the clock. Under the pressure where anticipation matters the event rate is high
   (so the dispatch jitter is small), and the §4 `heartbeat_s` re-fire under HIGH /
   CRITICAL gives a floor cadence. At dispatch every scheduled action is
   **re-validated against current belief** and dropped or adjusted if stale
   (idempotent, §10) — so time paces *execution* but never drives *state*.

The principle that governs all three (it **replaces** the older "the system has
no internal timer"):

> **Belief is event-sourced; actions may be time-scheduled but are
> belief-validated at execution.**

This admits anticipation (predictive promote, proactive room-making) while
forbidding exactly what the old rule forbade — inferring program *state* from
elapsed time. A scheduled action that has gone stale by fire time is a no-op on
re-validation, so the timeline can never desynchronise belief. The inline
scorer + kv_scheduler + admission below are the **decision** plane's candidate
generators; `/aginfer/state` + the 13 events are the **belief** plane;
`/aginfer/migrate` + pause/resume + the due-action heap are the **action** plane.

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
  `should_write_through(node, threshold)` function (new plugin
  point, env `SGLANG_WRITE_THROUGH_MODULE`) when considering
  write_through_selective.  Default implementation is the
  historical `hit_count >= write_through_threshold` (the
  `threshold` arg carries the cache's configured value).  Aginfer
  registers a V_u-aware version that triggers when
  `V_u(residence ∪ {DRAM}) > V_u(residence)` (and may ignore
  `threshold`).

Both plugin points live ONLY on `UnifiedRadixCache` — the cache
aginfer requires.  The entire aginfer surface (`/aginfer/state`
schema, hint table, migrate/program_paused/hints endpoints) is
UnifiedRadixCache-only; on any other tree cache every aginfer
endpoint returns `unsupported_tree_cache` unless
`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`.  The sibling stock caches
(`HiRadixCache`, `HiMambaRadixCache`) keep their hardcoded
`hit_count >= write_through_threshold` and stock LRU — they carry no
aginfer state and aginfer never instantiates them, so plugging them
would bolt isolated aginfer machinery onto code aginfer can't run on.
The plugin DEFAULTS are byte-identical
to the siblings' hardcoded behaviour, so the no-daemon baseline is the
same regardless of cache.

The same plugin pattern can be extended to future decision points
(predictive load_back, mooncake archive trigger) without
restructuring the framework.

## 4. Events

Thirteen event kinds.  **Belief is event-sourced**: the daemon updates its
world-model only on these events (plus the always-fresh state re-fetch), never
from elapsed wall-clock — see §3. The action-timeline plane may *execute* a
previously-decided action at a predicted time, but that paces execution only and
never changes how state is observed.

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
| `HASH_COLLISION` | sglang webhook | while building the `{hash → node}` map at the start of `apply_aginfer_migrations`, sglang observed two distinct radix-tree nodes mapping to the same `node.hash_value` — two distinct token prefixes produced the same SHA-256 chain.  Payload `{hash, node_a_summary, node_b_summary}`.  The daemon `fatal('hash_collision', ...)` with a forensic dump on receipt — this is a deployment-bug-class fault (§10), not absorbed.  Probability is < 10⁻²² at any tree size aginfer encounters; the event exists so the design fails loud if the assumption is ever wrong, not because it's expected |

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

No time-based **state inference** exists: any "looks idle / looks ended" rule
that guessed a program's lifecycle from elapsed time would race a legitimately
long-running ACTING program whose tool takes minutes — so lifecycle is
event-signalled (above), never timed. This constrains the **belief** plane only;
it does not forbid the **action-timeline** plane (§3) from *executing* an
already-decided action at a predicted time (e.g. a promote-ahead), which is
re-validated against belief at fire and is a no-op if stale.

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
  plan; higher occupancy simply makes more units net-positive to
  migrate out via `forecast`).

This keeps the decision rule pure: priority is a routing concern,
not a decision-rule concern.

## 5. State surface — `/aginfer/state`

```json
{
  "time_counter": int,                  // monotonic access tick

  "throughput_ema": {                   // sglang-observed throughput EMAs,
                                         // updated on every prefill / decode
                                         // step over a ~1 s window.  Input to
                                         // §7 marginal_pause_cost (prefill)
                                         // and §8 forecast_inflight_demand
                                         // (decode).
    "prefill_bps": float,               // global prefill bytes / sec
    "decode_per_program": {
      "<program_id>": float             // p's decode bytes / sec (tokens × bytes_per_token)
    }
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
                                         //
                                         // `page_bytes` is per-(tier, subpool)
                                         // because paged-KV allocators may use
                                         // different page sizes per subpool
                                         // (MLA's latent layers, GQA groups,
                                         // Mamba snapshot chunks); §9's DP
                                         // quantises each axis at its own
                                         // `page_bytes`.
    "HBM":  {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int,
                                     "page_bytes": int}}},
    "DRAM": {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int,
                                     "page_bytes": int}}},
    "DISK": {"subpools": {"<name>": {"used_bytes": int, "cap_bytes": int,
                                     "available_bytes": int,
                                     "evictable_bytes": int,
                                     "page_bytes": int}}}
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
        "committed": {"<subpool>": int} // host pool share, same attribution;
                                         //   subpool keys mirror
                                         //   pool_usage.DRAM.subpools per §12
      },
      "state": "REASONING"|"ACTING"|"PAUSED"|"ENDED",
                                         // exhaustive — these are the only
                                         //   states ever surfaced to the
                                         //   daemon.  Any sglang-internal
                                         //   intermediate (e.g. mid-preempt
                                         //   queueing) is collapsed into
                                         //   the nearest of these four at
                                         //   dump time.
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
      "time_since_last_sample_s": float // seconds since the most recent
                                         // bytes-moved sample on this link.
                                         // Cold-start (link never used since
                                         // boot): set to +Inf, which trips
                                         // the > LINK_IDLE_SECONDS branch in
                                         // §7 bw_free and returns peak_bw_bps
                                         // (correct — an unused link is idle
                                         // by definition).  Daemon
                                         // distinguishes "measured idle"
                                         // (gap > LINK_IDLE_SECONDS = 1.0)
                                         // from "freshly measured" by this
                                         // number — see §7 bw_free.
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
    "DRAM": {"<subpool>": {"h_max_per_byte_sec": float}},
    "DISK": {"<subpool>": {"h_max_per_byte_sec": float}}
  },

  // Diagnostic: emitted by sglang only when the loaded tree cache
  // class lacks dump_aginfer_state.  Daemon halts loudly on this —
  // running aginfer against an incompatible cache is a deployment
  // bug, not a graceful-degradation case.
  "unsupported_tree_cache": str?
}
```

### Why two views

The radix tree contains **only committed prefix-shareable units**;
in-flight decode KV is allocator-owned but **not** in the tree.
Two distinct consumers in §7 / §8 need two distinct slices:

* `pool_usage` (allocator view, **per-subpool**) — input for both
  **V_u migration value scoring** (the `h_(τ, sp)(occ)` term in §7
  reads `pool_usage[τ].subpools[sp]` directly so holding cost
  reflects actual allocator pressure, not just the radix tree's
  slice) and **admission's pressure trigger**.  Admission acts when
  **any** HBM subpool crosses its `theta_hi` threshold, not when
  the aggregate does — a Mamba snapshot pool at 95% with attention
  at 60% is the failure mode an aggregate view hides.
* `per_program_usage` (program view) — input for **admission's
  victim selection**.  Pausing a program frees its committed
  share + prevents its future in-flight growth; admission needs
  per-program footprint to know **which** pause yields the most
  HBM bytes per unit V_u_program lost.

A one-view design (only `pool_usage`) is under-determined:
admission knows the pool is pressured but has no principled way
to compare candidate victims by HBM relief.  Walking `units` +
`session_ids` and summing only counts the committed share, so a
runaway-decode program with 80 K in-flight bytes but 6 K committed
prefix looks *smaller* than a quiet program with 8 K cold prefix —
exactly the wrong victim.

Mixing the two also breaks: admission picks the wrong victim
(sees committed share, not real footprint).  **Two views, two
consumers, no overlap.**

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

1. Handler computes the plan (`joint_decide` output: a
   `list[Migrate | Pause | Resume]`), groups the items **by
   endpoint** (all `Migrate`s → one `POST /aginfer/migrate`
   batch; each `Pause`/`Resume` → one `PUT /aginfer/program_paused`
   call), and enqueues each batch onto the outbound
   `asyncio.Queue`; the handler returns immediately.
2. A dedicated outbound worker task consumes the queue, issues
   the HTTP `POST` / `PUT`, and on 200 simply drops the response.
3. If sglang returns a non-2xx or any individual item in a batch
   fails, sglang fires an `APPLY_FAILED` webhook back to the
   daemon (§4) carrying the batch envelope and a list of failed
   items with structured `reason` strings.  The daemon treats
   the webhook like any other workload event — the next
   `joint_decide` re-evaluates the state and may re-issue a
   superseding action.  Idempotency (§10) makes re-issue safe.

This decouples handler latency from sglang processing latency.
Under a 200 ms sglang stall (CUDA-graph capture, prefill batch
serialization), a synchronous awaited POST would freeze the entire
event router; fire-and-forget keeps the inbound queue draining and
defers the action to the outbound worker.

**Endpoint-aware coalescing + freshness-bounded actuation.**  The
outbound channel is single-flight at sglang (one control communicator;
apply is serialised against the scheduler loop, ≈ one iteration per
POST).  Because the daemon re-pushes a `hints` PUT *every event* (§10:
no daemon-side hint cache), hints dominate outbound traffic by ~100×,
and a naïve FIFO queue makes every time-sensitive `migrate` wait behind
that idempotent flood — ageing it until the radix tree diverges and the
apply-time leaf check (§7) rejects it.  The worker therefore drains the
queued burst each wake and emits **at most one dispatch per endpoint**,
by each endpoint's temporal semantics:

* `hints` — idempotent overwrite-by-stamp ⇒ merge the burst into ONE
  PUT, latest value per hash.  This collapses the flood so the channel
  un-clogs; a migrate no longer waits behind it.
* `migrate` — time-sensitive; merged into ONE POST per wake, latest
  decision per unit hash, after the freshness drop.  Coalescing makes a
  burst one round-trip, so the single-flight ceiling bounds *how often* a
  batch is sent, not how many actions it carries.  A migrate decided on a
  state snapshot is a prediction with a **validity horizon**: the worker
  drops an action whose queue-age exceeds a *generous* bound (a
  pathological-spike floor — sglang catastrophically stalled — not a tight
  knob masking ordinary latency).  This is the proactive complement to
  §10's reactive `APPLY_FAILED` re-issue: never actuate a decision already
  stale against the world it was decided on.  Dispatch order is liveness
  (`program_paused`) → eviction (`migrate`) → idempotent (`hints`), so
  time-sensitive intents never queue behind the flood.  The
  sustained-unreachable escalation (§10) counts failed POSTs **accumulated
  across wakes** (reset only on a 2xx); coalescing reduces the per-wake
  increment (≤ one per endpoint), not the signal — the streak still reaches
  the threshold under a sustained stall, just over more wakes.
* `program_paused` — liveness-critical; coalesced by pid (latest state
  wins), never dropped.

**Identifier model.**  Each outbound HTTP request carries a
`batch_id` (UUID generated at the daemon at enqueue time, written
into the request body envelope).  Each individual action inside
the batch also carries its own `action_id` (UUID, in the per-item
object).  This lets `APPLY_FAILED` correlate at either granularity:
the webhook payload names the `batch_id` once and lists per-item
`action_id` + `reason` for every item that failed.  Successful
items are not echoed back (silence == applied).

**Per-item failure reasons** are the structured skip-class strings
already defined per endpoint (see `POST /aginfer/migrate` below
for the canonical list: `race:*`, `promote_load_back_declined:*`,
`promote_raised:*`, `write_through_declined:*`, `unknown_tier:*`,
`unsupported_tree_cache:*`).  The same reason taxonomy is used
both in sglang's synchronous response body (for any code path
that does still read it, e.g. cold-start probes) and in the
`APPLY_FAILED` webhook (the fire-and-forget steady-state path).

**Unknown `batch_id` after daemon restart.**  Daemon emits batch X
→ daemon crashes → daemon restarts → sglang's `APPLY_FAILED` for X
arrives at the new daemon, which has no record of X (outbound
queue is volatile per §10).  The daemon drops the webhook silently
and continues; the load-class fault classification (§10) and the
always-fresh + idempotent invariants together guarantee
re-convergence — the next handler entry re-fetches state and any
still-needed action gets re-emitted.  Not a deployment-bug fault;
no forensic dump.

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
| `["DRAM"]` | `["HBM"]` | write-through then evict — the canonical "HBM→DRAM migrate" |
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
  fail, SWA subpool evict short, etc.).  Usually transient.
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
`theta_crit` / `heartbeat_s`.  Two-stage lifecycle:

1. **Bootstrap fetch (sglang side).**  At sglang launch, sglang
   `GET /aginfer/thresholds`.  If the daemon is unreachable,
   sglang halts loudly — this is a deployment-ordering bug
   (daemon must be up before sglang).  No local cache, no
   last-known values: aginfer-managed sglang has one canonical
   thresholds source and that source is the daemon.
2. **Daemon-side update broadcast.**  When the daemon's threshold
   config changes at runtime (operator restart with new defaults,
   reload via signal), the daemon enqueues a `PUT /aginfer/thresholds`
   onto its outbound queue (§6 "Fire-and-forget delivery") — handler
   returns immediately, the outbound worker issues the HTTP.  Sglang
   updates its in-memory thresholds atomically on receipt.  Until
   the broadcast propagates (≪ event interval typically), sglang
   and daemon may transiently disagree by one update; the next
   state-fetch reconciles.  If the apply fails, sglang fires
   `APPLY_FAILED` (§4) and the daemon's next handler re-enqueues.
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

The endpoint takes the FINAL state directly, not a transition verb:

Request:
```json
{"pid": str,
 "state": "REASONING"|"ACTING"|"PAUSED"|"ENDED",
 "pre_pause_state": "REASONING"|"ACTING"|"PAUSED"|"ENDED"|null}
```

The three transitions the daemon emits map onto this schema as
follows:

* **PAUSE** (`state: "PAUSED"` + the prior state in `pre_pause_state`):
  `pre_pause_state` is the state the program was in immediately before
  admission paused it.
  **Source for the value**: the daemon reads
  `state.per_program_usage[p].state` from the state snapshot
  fetched at handler entry (the always-fresh snapshot — not the
  daemon's `program_tracker` cache, which may already reflect a
  prior pause).  sglang on receipt sets `state = PAUSED`,
  `pre_pause_state = payload.pre_pause_state`.  The proxy gate
  (in the daemon's proxy process) starts queueing p's next
  request immediately upon the daemon's `Pause(p)` decision —
  the PUT is the persistence side; the gate is the enforcement
  side.  Both happen on the daemon side of the wire.
* **RESUME**: daemon sends `pre_pause_state = null` (sentinel, not
  the restored state).  sglang reads its **own** stored
  `pre_pause_state` to determine the restored state, then sets
  `state = (the stored pre_pause_state)` and `pre_pause_state =
  null`.  The payload field is intentionally null-only on RESUME
  to make it impossible for the daemon to disagree with sglang's
  authoritative view.  **Proxy-gate release**: in parallel with
  enqueueing the RESUME PUT, the daemon releases p's currently-
  gated request (if any) from the proxy gate so it can flow
  through to sglang.  Resume's two effects (state field update
  via PUT, gated request release via local proxy.gate) are
  intentionally split: sglang owns the persistent state, the
  daemon owns the request-flow control.  No `/aginfer/`
  endpoint releases the gate — it's a daemon-internal operation.
* **END** (`state: "ENDED"`, `pre_pause_state: null`): sglang sets
  `state = ENDED`; `pre_pause_state` is cleared.

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
| TP > 1 | same logical unit's **different head-dim slice** on each rank | sglang's tokenizer-server fans `/aginfer/state` / `migrate` / `hints` out to all rank schedulers; the snapshot returned to the daemon is aggregated across ranks per the rules below | every action is **all-rank atomic by semantic requirement** — see below |
| EP > 1 | prefix KV mirrored across ranks (same as TP); only the MoE expert weights / activations differ per rank | same as TP from the daemon's perspective — expert weights aren't in the daemon's scheduling scope | same as TP |
| DP > 1 | each DP replica has its own independent KV pool serving its own program subset | each DP replica is a **separate sglang endpoint** with its own daemon-sglang pairing; no cross-replica daemon coordination | independent per replica |

### Per-field aggregation rule (TP > 1)

| field | aggregation | reason |
|---|---|---|
| `pool_usage.<tier>.subpools[sp].{used_bytes, cap_bytes, available_bytes, evictable_bytes}` | sum across ranks | each rank holds an independent slice of the unit's bytes; total occupancy is the sum |
| `per_program_usage[p].{hbm,dram}.{committed,inflight}[sp]` | sum across ranks | same reason |
| `link_stats[σ→τ].{peak_bw_bps, recent_throughput_bps}` | sum across ranks | each rank moves its own slice on its own link concurrently; total free bandwidth scales with rank count |
| `link_stats[σ→τ].time_since_last_sample_s` | `max` across ranks | "link has been idle long enough" requires *every* rank's link to have been idle that long |
| `throughput_ema.prefill_bps` | mean across ranks | each rank sees the same wall-clock prefill batch; per-rank EMA is a measurement of the same logical work, so the mean is the unbiased estimator (sum would inflate by N×) |
| `throughput_ema.decode_per_program[<pid>]` | mean across ranks | same reason: same logical decode rate observed N times |
| `tier_holding_cost[τ][sp].h_max_per_byte_sec` | identical across ranks (operator config) | static; sglang halts loudly on cross-rank disagreement |
| `units[i].n_bytes[τ][sp]` | identical across ranks | derived from architecture; cross-rank disagreement is a deployment bug → `fatal()` |
| `pool_usage.<tier>.subpools` keys | identical set across ranks | every rank runs the same architecture, so subpool component names match; mismatch is a deployment bug → `fatal()` |

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

**Open risk.**  If a divergence is ever observed under high hint
churn at TP > 1 (rank 0 evicts a hash rank 1 didn't pick), the
all-rank atomicity invariant is broken and the design needs a
stronger primitive.

## 7. Decision rule

> **Pseudocode conventions.**  Python-flavored pseudocode in §7 and
> §9 assumes the standard `from dataclasses import dataclass, replace`
> and standard typing names (`dict`, `list`, `tuple`).  `replace` is
> used in §9 to derive new instances of frozen `@dataclass(frozen=True)`
> candidates (Migrate / Pause / Resume) without mutation.

### Formal model

The system is a partially-observable MDP that aginfer solves
event-by-event:

* **State `S`** — per `/aginfer/state` snapshot at handler entry
  (always-fresh per §10).  Includes: every unit's `residence ⊆
  {HBM, DRAM, DISK}` plus per-(tier, subpool) bytes; every
  program's `state ∈ {REASONING, ACTING, PAUSED, ENDED}` plus
  `pre_pause_state` and HBM `committed`/`inflight` byte
  footprints; per-tier per-subpool allocator occupancy; per-link
  bandwidth EMA; per-tier per-subpool holding-cost constant.
  Sglang is the authoritative store; the daemon is stateless
  modulo a derived `program_tracker` cache (§11).
* **Action `A`** — joint over the union space `{unit migrate} ∪
  {program pause/resume}`.  Unit migrate = residence-set
  transition `(add_tiers, remove_tiers)` for a single unit (§7
  `migrate_candidates`).  Program pause/resume = state-machine
  transition (§8 `pause_candidates` / `resume_candidates`).
  Action set per event is enumerated by `joint_decide` (§9) and
  the chosen subset is applied via §6 endpoints.
* **Transition** — driven by sglang's allocator / scheduler /
  request stream, **not by aginfer**.  Aginfer's actions
  steer the transition (a Pause re-shapes future
  `inflight_bytes`; a Migrate re-shapes future `residence` for
  the affected unit) but do not by themselves advance time.
  The MDP "tick" is the next workload event.
* **Reward** — negative seconds-cost.  Per-event reward is the
  total net value of the chosen subset under §9's value-maximising
  knapsack (value = −cost; the empty set scores 0, so a no-op is
  never worse than acting).  Long-run reward is total
  inference seconds-saved vs the always-DROP baseline, per the
  paper's reward decomposition.
* **Policy `π`** — `joint_decide(state, event)` from §9.
  Deterministic given state + event + policy parameters
  (`theta_hi`, `theta_lo`, `theta_crit`, `K_MAX`, etc).  Policy
  is **not learned** — it solves an exact 0/1 knapsack over the
  per-event action set, with the V_u formula (§7) as the
  ranking signal.  A learned alternative over the same
  state/action/reward is neither in scope nor ruled out by the
  framework; the V_u rule could serve as a bootstrap if such a
  variant were ever pursued.

The formulation is **partially observable** in two ways: (a)
sglang's per-rank distributed cache state is summed to a
single-rank-equivalent view in `/aginfer/state` (§6 multi-rank);
the daemon never sees per-rank disagreement.  (b) future
workload events (next `LLM_PREFILL` arrival, tool ETA accuracy)
are estimated, not observed.  Estimators feed `p_hat` and
`E[remaining_tokens]` (§7 / §8); the policy is exact w.r.t. its
estimator inputs even when those inputs are noisy.

### Symbols and units

All cost-side quantities are in **seconds** (paper §3 Reward
units: time saved/paid against wall-clock).  All relief / re_use
quantities are in **bytes**.  The joint knapsack (§9) mixes
seconds-cost with bytes-budget; the comparison is well-typed
because the constraint is a one-sided byte threshold.

| symbol | unit | meaning |
|---|---|---|
| `u.n_bytes`, `u.n_tokens` | nested-dict / tokens | `u.n_bytes` is a nested dict `{tier: {subpool: bytes}}` (per §5).  `bytes_at(u, τ)` = bytes u occupies at tier τ (= sum over τ's subpools, returns 0 if τ ∉ residence) |
| `p_hat(u, Δt)` | unitless ∈ [0,1] | conditional reuse probability over horizon Δt |
| `λ` | accesses / sec | per-program-state access rate used in hint-pushed `p_access` estimators (§6 hint table, §7 estimator chain); a Poisson-fit shorthand subsumed by the conditional `p_hat` form |
| `Δt` | seconds | decision look-ahead horizon (§7 inputs) |
| `hold_time` | seconds | expected residence in candidate tier (§7 inputs) |
| `reload_from(u, τ)`, `reload_from_DROP(u)` | seconds | per-paper-§3 reload costs `ρ_τ × n_tokens`, `π_u × n_tokens` |
| `h_(τ, sp)(occ)` | sec / (byte × sec of holding) | per-(tier, subpool) marginal displacement cost at occupancy `occ`; one entry per subpool listed in `pool_usage[τ].subpools`.  Monotone function of occupancy `h_(τ, sp)_max × f(occ)` parameterised by `state.tier_holding_cost[τ][sp]` |
| `occupancy_of(τ, sp, state)` | unitless ∈ [0,1] | `pool_usage[τ].subpools[sp].used_bytes / .cap_bytes` |
| `bw_free(σ, τ)` | bytes/sec | live free bandwidth on σ↔τ link; reads `state.link_stats[σ→τ].recent_throughput_bps` when active, falls through to `peak_bw_bps` when link idle |
| `transfer_bytes(u, σ, τ)` | bytes | `bytes_at(u, σ)` — what the link physically moves when adding τ to residence (source is `authoritative_tier(residence)`) |
| `transfer_time(σ, τ)` | seconds | `transfer_bytes / bw_free` |
| `page_bytes(τ, sp)` | bytes | per-(tier, subpool) DP quantisation granularity, read from `state.pool_usage[τ].subpools[sp].page_bytes`.  §9's multi-axis DP uses each axis's own bucket size — no global LCM/min collapse |
| `cost`, `gain`, `V_u`, `V_u_program` | seconds | net value at the same time-axis |
| `relief`, `re_use`, `acquired` | per-(tier, subpool) bytes-dict | HBM-resource axes (and destination-tier consumption for `acquired`); `relief` carries the pressured-subpool targeting filter, `acquired`/`re_use` the value knapsack's budget consumption; see §9 |
| `forecast(state, event)` | dict[subpool, bytes] | per-HBM-subpool predicted bytes if no scheduling action is taken: `pool_usage.HBM.subpools[sp].used_bytes + forecast_inflight_demand(state)[sp] + event_carried_demand(event)[sp]`.  The third term is demand the event predicts but that is not yet in-flight (e.g. `SUB_DISPATCH_ASYNC`'s K imminent children), 0 for events carrying no prediction.  Compare each entry against `theta_hi × subpools[sp].cap_bytes`.  Horizon = `forecast_horizon(state)`, see §8 |
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
                              # (§8 — each term is itself a per-subpool dict).
                              # Keyed by subpool only (no outer tier key)
                              # because pausing a program never touches
                              # non-HBM tiers.  §9's joint_decide rewrites
                              # this to nested {HBM: relief} before feeding
                              # the DP so all candidate types share one
                              # shape.

@dataclass(frozen=True)
class Resume:
    program_id: str
    gain: float               # seconds (V_u_program_if_active)
    re_use: dict[str, int]    # per-HBM-subpool bytes
                              # = expected_peak_hbm_after_resume (§8).
                              # Subpool-keyed because Resume only affects
                              # the HBM tier.  §9's joint_decide nests this
                              # into {HBM: re_use} before feeding the
                              # headroom DP (symmetric with Pause.relief);
                              # all candidate types share the
                              # nested {tier: {sp: int}} shape inside the
                              # DP code path.
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

# Bytes a unit occupies at tier τ (sum across τ's subpools).
# Caller MUST check `τ in u.residence` first; otherwise a KeyError
# fires loudly — per §10 "Subpool key consistency", missing keys
# are a deployment bug, not a silent zero.
def bytes_at(u, τ):
    assert τ in u.residence, f"bytes_at called for {τ} not in u.residence"
    return sum(u.n_bytes[τ].values())

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

def unavailability_cost(u, add_tiers, remove_tiers, state):
    # Sum across each new tier τ being added: probability of an
    # access landing during u's source→τ transfer × probability that
    # serving from the source fails × reload penalty from the source.
    # Under write-through HiCache the middle factor is 0 in steady
    # state; the term is kept in the formula for non-write-through
    # write policies (zero-copy moves, etc.) without rewriting _value.
    σ = authoritative_tier(u.residence)
    return sum(
        p_hat(u, transfer_time(u, σ, τ, state))
        * P_serve_from_source_fails(σ, write_policy(σ))
        * reload_from(u, σ)
        for τ in add_tiers if τ not in u.residence
    )

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
            cost += unavailability_cost(u, add, remove, state)

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

**Apply-feasibility pre-filter.**  A `Migrate` whose `remove` sglang
is *structurally guaranteed to reject* frees nothing yet costs a
webhook round-trip, so candidate generation never emits one.  Two
classes of reject are pre-filtered, both mirroring sglang's
apply-site guards:

* **Structural leaf guards.**  A remove is rejected unless the unit
  is the relevant leaf: `remove {HBM}` needs a *device leaf* (no
  child holds device-resident KV); `remove {DRAM}` needs a *host
  leaf*; a full DROP needs a *tree leaf* (no children at all —
  stricter than device-leaf, since a node with disk-only children
  is a device leaf yet not a tree leaf).  These flags come straight
  from the state dump.

* **Active-decode holder guard.**  Even a unit that *is* a device
  leaf in the dump is reject-guaranteed for `remove {HBM}` when one
  of its holder programs is **actively decoding** — i.e. has a
  request in the running batch (`per_program_usage[pid].hbm.inflight`
  > 0).  Such a node is a device leaf only in the brief gap between
  that program's forward passes; its `lock_ref` oscillates (locked
  during each pass, released between), and the dump samples it in a
  released instant.  By apply time — the dump is up to ~1 s old under
  load — the holder has re-locked its session tail and sglang rejects
  with `remove_hbm_not_device_leaf`.  The node's instantaneous
  `lock_ref` is the **wrong** signal to gate on (it reads 0 at the
  dump instant — that is *why* the node dumped as a leaf); the
  program-level `inflight` signal is **stable across the per-pass
  oscillation**, so it cleanly excludes these hot tails.  A
  tool-parked program (awaiting a tool result → not in the running
  batch → `inflight` empty) is *not* gated, so its idle session tail
  stays evictable — the demote-during-the-tool-gap that is the
  scheduler's core source of relief.  Absent / cold-start `inflight`
  never suppresses, so the gate cannot strand the policy before the
  signal populates.

  *Bound.*  The holder set comes from the unit's `session_ids`, which
  sglang accumulates add-only over a program's lifetime, so the guard
  is a slight **over**-approximation: a device-leaf on an abandoned
  branch a still-decoding program touched earlier is gated even though
  that program will not re-lock it.  This is deliberately on the safe
  side — it only forgoes a demote that *would* have been valid, never
  proposes an invalid one — and is small in practice because (i) a
  program's ancestors on its live root→tail path are non-leaves (so
  already excluded by the structural device-leaf guard) and (ii)
  temperature-0 agentic decode grows each session as a single chain
  with few abandoned branches.  The exact-but-unstable alternative
  (the node's instantaneous `lock_ref`) is rejected for the reason
  above; a future exact-and-stable signal would have sglang emit each
  active program's current decode frontier.

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

This is option **(C) concurrent-serve-from-source**, and it strictly
dominates the alternatives:

| | this access | next access | bytes wasted |
|---|---|---|---|
| (A) wait for transfer | σ_latency + transfer_time | τ_latency | 0 |
| (B) cancel transfer | σ_latency | σ_latency | partial transfer |
| **(C) concurrent-serve** | σ_latency | τ_latency | 0 |

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
| `TOOL_CALL_START` | caller's EXCLUSIVE tail = demote candidate while idle; promote-ahead is scheduled here too, timed by the tool ETA so the unit lands before the next prefill.  Shared prefix excluded — a per-program event doesn't touch fleet-shared resources |
| `TOOL_CALL_END` | caller's EXCLUSIVE tail = promote-now if ahead-of-time promote didn't catch up; otherwise no-op |
| `SUB_DISPATCH_BLOCKING` | parent's exclusive tail + shared prefix (disjoint union — the tail is exclusive, so no overlap) |
| `SUB_DISPATCH_ASYNC` | shared prefix only |
| `SUB_RETURN` | parent's exclusive tail (promote candidate) + the child's output that just materialised as a new `subagent_ctx` unit (decide initial tier) |
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
    """The caller's EXCLUSIVE tail — units held ONLY by `sid`
    (`session_ids == {sid}`), ordered by last access descending.
    These are the demote candidates while p is ACTING (tool-bound)
    and the promote candidates ahead of p's next prefill.  No fixed-K
    cap — the joint knapsack (§9) picks a subset under its byte budget.

    EXCLUSIVE, not "every unit p touches": a TOOL_CALL is a
    PER-PROGRAM event — only p's PRIVATE units changed value when p
    went idle.  A SHARED prefix's value did not change just because one
    of its many holders went tool-bound (the others still need it), so
    a single program's tool call must NOT nominate a fleet-shared
    resource for demotion.  Shared-prefix residence is driven by the
    events that actually bear on it: `SESSION_ARRIVAL` (preload) and
    `MEMORY_PRESSURE` (global `top_k_by_regret`).  Coincides with
    `session_scoped_units` below (same predicate) — kept as two names
    for the two event contexts.  Implemented as
    `kv_scheduler._units_for_session`."""
    return frozenset(u for u in units if u.session_ids == {sid})

def session_scoped_units(units, sid):
    """Units held ONLY by this session — `session_ids == {sid}`.
    Used at SESSION_END: these units have no other holder, so
    after END they're either evicted (if all V_u<0) or demoted to
    the cheapest tier that still offers nonzero `p_hat` from the
    workload-prior (next program with this prefix shape will
    benefit from a warm DRAM/DISK copy)."""
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
bytes_needed_total(state) =
    Σ_sp max(0, forecast(state)[sp]
                - theta_hi × pool_usage.HBM.subpools[sp].cap_bytes)
                        # summed across HBM subpools; §9 keeps the per-axis
                        # breakdown but top-k sizing only needs the total
mean_unit_bytes(state) =
    if |{u : HBM ∈ u.residence}| == 0:
        return ∞        # short-circuit: empty HBM means pressure-phase
                        # cannot fire (bytes_needed_total would also be
                        # 0 since pool_usage HBM is empty).  Returning
                        # ∞ makes top_k_pressure collapse to K_MIN via
                        # the max() guard; the DP runs trivially.
    else:
        return (Σ_{u : HBM ∈ u.residence} bytes_at(u, HBM))
             / |{u : HBM ∈ u.residence}|
        # arithmetic mean over HBM-resident units only — top_k_pressure
        # sizes the pressure-phase candidate count and only HBM-resident
        # units are plausible pressure-relief candidates.

top_k_pressure(state) =
    min(K_MAX,
        max(K_MIN, bytes_needed_total(state) / mean_unit_bytes(state)
                    × K_SAFETY))
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
scheduler schedules the promote-back as an **action-timeline action**
(§3) for `T_start + tool_ETA - load_back_latency` so the unit is in
HBM exactly when the next prefill arrives.  Waiting until
`TOOL_CALL_END` to start the promote means the prefill races the
in-flight transfer.  Realization, per §3: the action is placed on
the due-action heap and dispatched when the event stream next
advances past its due time, then **re-validated against belief** —
if the program already returned, ended, or its tail was meanwhile
dropped, the promote is an idempotent no-op.  When no reliable ETA
is available (e.g. an untyped `bash` whose duration spans ms–min),
it degrades to **promote-at-`TOOL_CALL_END`** — still earlier than
HiCache's on-access load_back, just without the full pre-stage.

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
3. **Bootstrap (cold-start)**: average inter-event spacing
   (10 ms–1 s range on agent workloads).  Used only at session
   warm-up before per-event-kind statistics accumulate; falls
   away as program_tracker history fills in.

#### `hold_time` — expected duration in the candidate tier

`hold_time` is "how long the unit will actually stay in τ before
the next migrate/evict/drop reconsiders it".  This is the time
the `h_(τ, sp)(occupancy) × bytes × hold_time` holding tax integrates
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
2. **Bootstrap (cold-start)**: average inter-event spacing
   across all events (matches Δt's level-3 cold-start case).
   Used only at session warm-up before per-event-kind statistics
   accumulate.  In steady state on busy agent workloads,
   hold_time ≈ Δt; the two diverge sharply when the unit is held
   by a quiet program (long stretches with no D_t-relevant event).

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
h_(τ, sp)(occ) = h_(τ, sp)_max × occ      # monotone in occ; see below
h_(τ, sp)_max  = state.tier_holding_cost[τ][sp].h_max_per_byte_sec
occ            = occupancy_of(τ, sp, state)
               = pool_usage[τ].subpools[sp].used_bytes
               / pool_usage[τ].subpools[sp].cap_bytes
```

Why per-subpool, not per-tier: in a hybrid Mamba+attention model,
the Mamba snapshot subpool and the attention `full` subpool can
sit at very different occupancies and have very different
displacement-cost curves (a Mamba snapshot is a much larger byte
unit and rarer-to-reuse).  Sharing one `h_(τ, ·)` across both subpools
would mean Mamba pressure shows up as attention demotion or vice
versa.

The shape is **linear in occupancy** — the simplest non-trivial
monotone function: 0 cost when the subpool is empty (can't displace
anything), `h_(τ, sp)_max` cost when full.

The curve `h_(τ, sp)(occ)` is, physically, the relationship between
current subpool occupancy and the V_u of the marginal (lowest-V_u)
resident **within that subpool** at that occupancy.  The linear form
is the spec with operator-calibrated per-subpool `h_max`; a more
physically motivated candidate is hyperbolic `α / (1 - occ)`
(diverges as occ → 1, matching the §9 admission cap), fit per subpool
against logged `(subpool, occ, marginal_V_u)` triples.

#### `bw_free(σ, τ)` — free bandwidth on σ↔τ link

Live bytes-per-second available on the σ↔τ link, read directly
from `state.link_stats`:

```
LINK_IDLE_SECONDS = 1.0    # deployment constant; idle gap classifier

bw_free(σ, τ) =
    let s = state.link_stats["σ->τ"]
    if s.time_since_last_sample_s > LINK_IDLE_SECONDS:
        # Link has genuinely been idle long enough that no other
        # traffic is competing.  Physics: full peak available.
        return s.peak_bw_bps
    else:
        # A recent sample exists; the EMA is the live view.
        return s.recent_throughput_bps
```

Note the structure: there is no "what if `s` itself is missing"
branch.  If `link_stats[σ->τ]` is absent or `peak_bw_bps` is
unset, the daemon halts loudly (§10 invariant "Physical inputs
sourced from sglang").  The two-branch rule above resolves *only*
the idle-vs-active dichotomy on a live link, which is a real
physical distinction (PCIe / NVLink / SSD bus has no contention
when no IO is in flight).

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
* **`time_since_last_sample_s`** is sglang's wall-clock since
  the most recent EMA sample on this link.  Updated on every IO
  completion.

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
because there cannot be one — DRAM/DISK are pure storage
classes, only reachable through a `load_back` round-trip and
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
                       + marginal_pause_cost(p, state),
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
all paused programs would tie at 0 and the value knapsack's
`gain / re_use` ranking would be undefined.

§9 consumes these candidates: relief takes net-positive Migrates from
§7 (the Pause lever is dormant, §9); the resume phase takes net-positive
Resumes.

### Component definitions

* `V_u_program(p, state) = Σ_{h ∈ p.unit_hashes} V_u(state.units[h])`
  — computed against the current state, used as Pause's cost (the
  V_u we'd lose if p stops here).  The conditional p_hat from §7
  is already a holder-product, so shared-prefix attribution is
  built in; no `1/|session_ids|` weight is needed.  The same holds for
  the Resume `gain` below.
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

  **Snapshot relief under the unified cache.** sglang's radix cache is
  UNIFIED: a running request's KV lives IN the tree, so the SAME physical
  bytes are reported by both the tree-walk `committed` and `inflight`.
  The two are measured on DIFFERENT bases — `committed` is
  **shared-aware** (a node held by `n` running reqs reports
  `bytes // n_holders` to each holder's program), whereas `inflight` is
  the **undivided** full running-request KV (no per-holder attribution).
  Combining them mixes the two bases and mis-credits shared prefixes: for
  a 12 MB prefix held by 3 reqs the per-holder committed is 4 MB but
  inflight is 12 MB, so `inflight − committed` would credit a single
  pause with the whole 12 MB even though pausing ONE holder frees only
  its 4 MB share.  Use the shared-aware `committed` ALONE — it is exactly
  the bytes a pause frees:

  ```
  snapshot_relief(p, state)[sp] =
      max(0, committed[sp] − migrate_domain[sp])   # shared-aware radix
                                                   # bytes pause frees that
                                                   # migrate isn't handling;
                                                   # inflight is NOT relief
  ```

  `marginal_pause_cost` reads the FULL undivided `inflight` (the decoded-
  so-far bytes re-prefilled on resume — a COST, not a relief, where the
  undivided footprint is the right basis and the shared-aware split would
  understate the re-prefill work).  The one known gap: bytes that are
  in-flight but NOT yet in the radix tree (an uncached prefilling prompt,
  non-radix Mamba state) are uncredited to the pause — a conservative
  under-free, recovered next event; the shared-aware uncached-inflight
  term needs per-unit holder attribution.

  The `future_inflight_savings` term is what makes Pauses
  trajectory-strong (Migrates only deliver snapshot relief).
  This is *why* PRESSURE_CRITICAL doesn't need a special cost
  twist in §9: under high `forecast`, `future_inflight_savings`
  is large for actively-decoding programs, so Pauses win on
  relief/cost naturally.

  * `snapshot_relief(p, state)[sp]` — read from
    `state.per_program_usage[p].hbm.committed`.  (The sibling
    **inflight[sp]** — p's undivided in-flight decode footprint — is
    NOT part of snapshot_relief under the unified cache; it
    feeds `marginal_pause_cost` only.)
    * **committed[sp]** — p's shared-aware share of subpool-sp radix
      bytes (`node_bytes // n_holders`); if pausing p drops
      `len(session_ids)` of a node to 0 on that subpool, the node
      becomes evictable on that subpool.
      **Disjoint-lever exclusion:** committed
      bytes of p's units that are in THIS event's `D_t` are the
      *migrate* lever's domain, so they are SUBTRACTED here — otherwise
      a `Migrate(u)` and a `Pause(p∋u)` in the same pressure plan both
      count u's bytes, the §9 DP over-estimates relief (additive
      across candidates) and under-frees. The two levers then attack
      physically-disjoint HBM (radix-via-migrate vs in-flight-via-pause).
      A D_t unit the DP does not migrate is simply uncredited to the
      pause too (conservative under-free, recovered next event) — never
      double-counted. (`pause_relief` ∖ `_committed_in_dt`.)
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
      Σ_{h ∈ p.unit_hashes : h ∈ state.units,
                             HBM ∉ state.units[h].residence}
          unit_hbm_subpool_bytes(state.units[h], sp)   # load-back of a
                                                  # SURVIVING non-HBM unit.
                                                  # An HBM-resident unit
                                                  # (kept alive by another
                                                  # holder) consumes no new
                                                  # HBM → counted 0.
    + dropped_reprefill_credit(p, state)[sp]      # see below
    + future_inflight_savings(p, state)[sp]       # post-resume decode
                                                  # growth in subpool sp

  unit_hbm_subpool_bytes(u, sp):
      """How many HBM-subpool-sp bytes the unit would occupy if
      HBM-resident.  This is a physical property of u's token
      content (n_tokens × per-token-subpool-bytes for the
      architecture) — independent of whether HBM is currently in
      residence.  When HBM IS in residence, u.n_bytes[HBM][sp]
      gives this directly; when HBM is not, sglang stores the
      same per-architecture quantity in u.n_bytes[<other tier>][sp]
      because the unit's content shape doesn't change between
      tiers.  Read whichever tier is in residence; assert that at
      least one tier is."""
      assert u.residence, f"unit {u.hash} has empty residence"
      τ = next(iter(u.residence))                  # any resident tier
      return u.n_bytes[τ][sp]

  dropped_reprefill_credit(p, state)[sp]:
      """A hash in p.unit_hashes that is ABSENT from state.units was
      DROPped while p was gated, so the load-back sum above credits it
      0 — yet on resume p re-prefills that prefix straight back into HBM
      (its conversation context needs it), a near-instantaneous burst
      capacity_fits would otherwise miss.  Credit each dropped hash a
      re-prefill estimate sized from p's OWN surviving units (mean
      HBM-equivalent bytes per unit, per subpool):

          mean_sp = (Σ_{h surviving} unit_subpool_bytes(h, sp)) / n_surviving
          credit[sp] = mean_sp × n_dropped

      Applied ONLY when p has surviving units.  A FULLY-dropped program
      (no survivor to size from) keeps credit = 0 so it can still
      un-starve: releasing its proxy gate is free, and its re-prefill is
      future work sglang's allocator admission-controls on the actual
      prefill.  This is exactly the §9 zero-re_use un-starve — only
      PARTIAL drops, which carry real resident context, pay the
      re-prefill reserve.

      The surviving-unit mean is an ESTIMATE forced by missing data: a
      DROPped unit is gone from state.units, so its exact bytes are
      unknown at resume time.  The ideal model instead puts the dropped
      prefix's re-prefill in the FORECAST TRAJECTORY term (a resumed
      program's predicted imminent prefill demand, sized by prefill_bps
      × its known context) rather than a resume-time reserve; the mean
      is the approximation used when that trajectory is unavailable —
      not provably conservative, since the dropped unit was evicted by
      sglang's victim policy (recency/value), which need not correlate
      with size."""
  ```

  Why "not already in HBM" is the correct filter for the load-back sum:
  HBM bytes for a unit already resident are accounted in
  `pool_usage[HBM].subpools[sp].used_bytes` (the first term of
  `forecast`).  `capacity_fits` checks each subpool independently —
  re-adding bytes via `re_use[sp]` would double-count them and
  over-reject Resume candidates whose units are kept alive by other
  holders.

* `capacity_fits(p, state)` — true iff for **every** HBM subpool sp:
  `forecast(state)[sp] + expected_peak_hbm_after_resume(p, state)[sp]
   ≤ theta_hi × pool_usage[HBM].subpools[sp].cap_bytes`.
  Resume candidates failing this gate on any subpool are omitted
  (not in the candidate set).
* `forecast(state)` — per-HBM-subpool dict.  For each subpool sp:
  ```
  forecast(state, event)[sp] =
      state.pool_usage["HBM"].subpools[sp].used_bytes
    + forecast_inflight_demand(state)[sp]
    + event_carried_demand(event)[sp]
  ```

  The **third term** is the prediction the triggering event carries
  about demand that is *about to arrive but is not yet in-flight*
  (so `forecast_inflight_demand`, which filters on `inflight[sp] > 0`,
  cannot see it).  Its load-bearing case is `SUB_DISPATCH_ASYNC`: a
  parent spawning K background sub-agents creates K imminent active
  programs whose first prefills have not landed — `event_carried_demand`
  estimates their near-term HBM footprint (K × the dispatch payload's
  per-child token estimate × `bytes_per_token_in_subpool`) so the
  decision plane can make room *before* the spike, the §3 action-plane
  way, rather than reacting after HBM is already over.  For events that
  carry no demand prediction the term is 0 and `forecast` reduces to
  the in-flight view below — so the bare **`forecast(state)`** written
  elsewhere in this doc denotes that common no-prediction case
  (equivalently `forecast(state, event)` with `event_carried_demand = 0`).

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

  Two properties of this form:

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
      heartbeat_s if recent_event_rate(state) <= 0
      else min(heartbeat_s,                # bounded by webhook re-fire
               1.0 / recent_event_rate(state))  # typical when active
  ```
  `recent_event_rate == 0` is the cold-start case before any event
  has been observed; capping at `heartbeat_s` is correct since that's
  the next guaranteed event arrival (the sglang heartbeat).

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

  **Inputs.** The trajectory product
  (`forecast_inflight_demand` + `pause_relief.future_inflight_savings`)
  is assembled from three state-dump inputs:
  * `bytes_per_token_in_subpool` —
    `pool_usage[*].subpools[sp].decode_bytes_per_token` (attention =
    bytes/token, Mamba = 0).
  * `decode_throughput` = `throughput_ema.decode_per_program[p]`, a
    per-program decode tokens/sec EMA sampled by the scheduler; 0
    (program not currently decoding) ⇒ no contribution.  Pure DECODE
    counts 1 token/req; MIXED batches split out their decode reqs; spec
    decode counts the post-forward `accept_lens` (≥ 1/req).  `prefill_bps`
    and the undivided `per_program_usage[p].hbm.inflight` (both feeding
    `marginal_pause_cost`) are measured the same way.
  * `E[remaining_tokens]` reads the per-program
    `expected_remaining_tokens` field; **absent ⇒ `None`, and the
    program is skipped — NOT bootstrapped to `max_completion_tokens`.**
    Per the `max`-vs-mean argument above, projecting the bootstrap upper
    bound in steady state would over-pause ~5×, so the term is omitted
    until a real conditional estimate exists rather than risk that
    regression.

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
`admission_controller` must read the SAME values; this is launched
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

**The pause lever is withheld until it can be correctly valued.**  The
argument above is why the *architecture* is a single decision over the
union action space rather than two sequential passes — it keeps pause
and migrate commensurable.  But a Pause cannot yet be correctly valued:
its cost misses the paused agent's forgone progress and its OOM-benefit
is unmodelled (§8).  Until both are modelled, `joint_decide` evaluates
{migrate-relief} ∪ {resume}, both value-gated and coexisting in one
event.  The union framing and the one-decision-function structure are
unchanged; only the pause item is held back.

### Joint decide

The §9 decision is a single **value-maximising** choice over the union
action space, and **no-op is always available**.  An action enters the
plan only when its **net value is positive** — the benefit it produces
exceeds its cost.  When nothing pays, the plan is empty and the daemon
does nothing.  This is what makes the daemon **do-no-harm by
construction**: it acts only where acting helps, and otherwise leaves
the state to sglang's own backstop (which is exactly the no-daemon
baseline).

There is **no "cover" target and no must-relieve wall**.  `theta_hi` is
a **signal** that gates whether relief candidates are generated — *not*
a hard limit the daemon must push occupancy back under.  A subpool at
high occupancy is worth relieving only if some relief action is
net-positive; if every relief action costs more than it saves — the
case for an over-subscribed, non-migratable subpool such as attention
`swa`, whose resident units are all in active use (high V_u, not leaves)
— the relief no-ops and the bottleneck stays a bottleneck, exactly as
with no daemon.  A min-cost-**cover** pressure phase that *forced* relief
whenever `forecast > theta_hi` is wrong: against a permanently-pegged,
unrelievable subpool it forces the daemon to pause active programs it
could never usefully resume → agent-stall timeouts worse than baseline.
Value-gating removes the forcing: the same DP that picks beneficial
actions also picks the empty set when none is beneficial.

`PRESSURE_CRITICAL` still takes the same code path as `MEMORY_PRESSURE`;
urgency enters through `forecast(state)` (larger at higher occupancy,
which makes more units net-positive to migrate out), and the
CRITICAL-specific behaviour lives in the event router (§4) — CRITICAL
events preempt the queue, they don't change the decision function.

The decision has two budget-shaped pieces; **both are value-maximising
0/1 knapsacks with the empty set allowed** (multi-axis, exact sparse DP
— machinery below).  They run in the SAME event and their plans
combine — independent levers on different state, **not mutually
exclusive**:

* **Relief** (candidates generated when **any** HBM subpool's
  `forecast[sp]` crosses `theta_hi × cap`): items are Migrate-HBM-out
  units, wrapped as value-items with **value = −cost**.  A Migrate whose
  `V_u(next) − V_u(current) − M_eff > 0` is net-positive — worth moving.
  An item is kept only if (a) it is net-positive (`cost < 0`) AND (b) it
  **relieves a subpool that is actually pressured** AND (c) it is **not a
  remove-HBM on a reuse-imminent tail**.  (a) protects hot units: because
  V_u already nets each unit's per-subpool holding cost and reuse value, a
  COLD unit (low V_u → cheap to move) has `cost < 0` while a hot/active
  unit (high V_u) has `cost > 0` and is dropped.  (b) keeps relief on the
  bottleneck and never churns a subpool that has room.  (c) handles a case
  (a) does NOT catch: at `TOOL_CALL_END` the decision set is the caller's
  session tail, which the §7 table marks a *promote* candidate ("about to
  reuse").  Under HBM pressure evicting it to DRAM is net-POSITIVE (HBM
  holding at high occupancy is dear and DRAM preserves the bytes), so (a)
  does not stop it — yet it is FUTILE: the session resumes and extends the
  tail, the frontier leaf gains a device child by apply time, and sglang
  rejects `remove_hbm_not_device_leaf` (a dump→apply TOCTOU).  So relief
  suppresses remove-HBM for a reuse-imminent (`TOOL_CALL_END`) tail;
  genuine HBM pressure is relieved by the `MEMORY_PRESSURE` events'
  cold-unit top-k, not by evicting the hot frontier.  (A per-hash
  cooldown, §10, remains the reactive backstop for the residual same-tick
  race on the pressure path.)  Together: a subpool full of active windows
  (`swa`) has **no Migrate that both improves V_u and relieves it** →
  relief no-ops (do-no-harm); a subpool full of cold cached prefix yields
  many → the daemon relieves it.  **No per-subpool threshold tuning** —
  the value rule carries the per-unit decision, the pressured-subpool
  filter carries the targeting.  The **Pause lever is withheld** (above):
  `V_u_program` goes negative under the holding tax, so the value gate
  alone would not drop an under-costed Pause — it could look net-positive
  and wrongly stall an active agent.  Until both the progress-cost and the
  OOM-benefit are modelled (§8), the daemon never pauses; Migrate is the
  working relief lever.  Constraint: do not overflow any destination
  `(DRAM|DISK, subpool)` cap.  Goal: **maximise total net value**, take
  only net-positive items, **empty set allowed**.

* **Resume** (candidates evaluated **every event**): items are
  Resume-paused-program candidates whose re_use **fits** —
  `forecast[sp] + re_use[sp] ≤ theta_hi × cap` on every subpool the
  resume *grows*; a resume that adds **0 bytes** to a subpool can never
  make it worse, so it fits even when that subpool is pegged (the free
  un-starve of a program whose units were dropped while gated).
  Constraint: per-HBM-subpool free room at `theta_lo` (hysteresis
  margin, clamped ≥ 0).  Goal: **maximise total V_u gain**, **empty set
  allowed**.

Relief and Resume run together in one event: the daemon may migrate cold
KV out of a pressured subpool AND resume a fitting paused program at
once.  Both are gated on net value; both can do nothing.

**`LLM_PREFILL` runs `joint_decide` like every other event.**  Its
`D_t` is `∅` (§7) so `migrate_candidates` returns `[]`, but the Resume
candidates are still evaluated; typically nothing is net-positive and an
empty plan is returned.  All 13 events trigger `joint_decide`; what
differs is which candidates are eligible.

```python
def capacity_left_bytes(state, τ, sp):
    """Bytes free in subpool sp of destination tier τ."""
    s = state.pool_usage[τ]["subpools"][sp]
    return s["cap_bytes"] - s["used_bytes"]

def joint_decide(state, event):
    hbm_subpools  = state.pool_usage["HBM"]["subpools"]
    dram_subpools = state.pool_usage["DRAM"]["subpools"]
    disk_subpools = state.pool_usage["DISK"]["subpools"]
    plan = []

    # ---- Relief: VALUE-GATED.  Candidates are generated only when some
    #      HBM subpool is pressured, but the DP takes ONLY net-positive
    #      actions and MAY RETURN NOTHING.  There is no cover / no forced
    #      relief: theta_hi gates candidate generation, it is not a wall.
    pressured_sps = {sp for sp in hbm_subpools
                     if forecast(state)[sp] > theta_hi * hbm_subpools[sp]["cap_bytes"]}
    if pressured_sps:
        cands = kv_scheduler.migrate_candidates(state, decision_set(event, state))
        # value = −cost.  Keep a Migrate iff (a) net-positive (cost < 0 ⇒
        # moving a COLD unit out improves total V_u; a HOT unit's cost > 0
        # → dropped) AND (b) it relieves an ACTUALLY-pressured subpool
        # (never churn a healthy subpool that has room).  re_use = the
        # destination bytes it consumes.  NO Pause items (dormant lever, §8).
        items = [ValueItem(gain=-c.cost, re_use=c.acquired, group=c.group, src=c)
                 for c in cands
                 if c.cost < 0
                 and any(sp in pressured_sps for sp in c.relief.get("HBM", {}))]
        # Constraint: do not overflow any destination (DRAM|DISK) cap.
        cap_left = {("DRAM", sp): capacity_left_bytes(state, "DRAM", sp)
                    for sp in dram_subpools} | \
                   {("DISK", sp): capacity_left_bytes(state, "DISK", sp)
                    for sp in disk_subpools}
        bucket_size = {(t, sp): state.pool_usage[t]["subpools"][sp]["page_bytes"]
                       for (t, sp) in cap_left}
        chosen = knapsack_max_value_multi(items, budget=cap_left,
                                          bucket_size=bucket_size)
        plan += [it.src for it in chosen]            # [] if nothing net-positive

    # ---- Resume: VALUE-GATED, runs EVERY event, independent of relief
    #      (no longer mutually exclusive — a resume that fits never makes
    #      a pressured subpool worse, so it can coexist with relief).
    free_room = {("HBM", sp): max(0, theta_lo * hbm_subpools[sp]["cap_bytes"]
                                     - forecast(state)[sp])
                 for sp in hbm_subpools}
    rcands = admission.resume_candidates(state)       # already capacity_fits-gated
    rcands = [replace(c, re_use={"HBM": c.re_use}) for c in rcands]
    bucket_size = {(t, sp): state.pool_usage[t]["subpools"][sp]["page_bytes"]
                   for (t, sp) in free_room}
    plan += knapsack_max_value_multi(rcands, budget=free_room,
                                     bucket_size=bucket_size)

    return plan                                       # MAY BE EMPTY — no-op
```

A workload with a Mamba snapshot subpool at 95 % occupancy and an
attention `full` subpool at 60 % has `pressured_sps == {mamba}` — the
relief filter keeps only Migrates of cold Mamba-leaf units (to
DRAM/DISK) that relieve `mamba`, and drops any migrate that would only
touch the healthy `full` subpool.  If every Mamba-resident unit is hot
(high V_u → `cost > 0`), no item survives and relief no-ops; attention-
resident units stay put either way.

Both phases are **exact 0/1 knapsack** at K ≈ tens of items
(K_MAX = 256 per §7 top-k cap).  Resource-axis count is
`|HBM subpools| + |DRAM subpools| + |DISK subpools|`, varying
per architecture (see §12 Scenarios for concrete counts).

Dense table size grows exponentially in axis count; the DP runs
**sparse** over a `dp` dict (cells reachable from any subset-take).
Reachable-cell count is bounded by `K × ∏_axis (1 + r_axis)` where
`r_axis` is the typical bucket-delta each candidate contributes
along that axis.  Worked example: K = 30 candidates, 5 axes, each
candidate touches 1–2 axes non-trivially (most Migrate candidates
free bytes on one HBM subpool and consume on one DRAM/DISK
subpool; Pauses touch only HBM subpools).  With median r ≈ 2 on
touched axes and r = 0 on untouched axes, the upper-bound product
is `30 × 3² ≈ 270` *new* cells per item, capped at `K × ∏ buckets`
along touched axes per item — empirically ≤ 10⁵ reachable cells on
agent workloads.  At K_MAX = 256 this scales to ~10⁶ in the worst case.

**Cost is NOT "microseconds" at the upper end.**
Python materialises ~50 k cells/sec, so the 10⁶-cell worst case is
≈ 20 s — a real event-loop stall, reached when candidates carry
large, DISTINCT relief/acquire bucket-deltas across many axes (the
regime the worked example assumes away).  The DP guards
with a ``max_dp_cells`` ceiling (default 10×10⁵): past it the DP
FAILS LOUD (``KnapsackBudgetExceededError`` → ``fatal(
"joint_decide_dp_blowup")``, crash-only) rather than stalling — a
blow-up means the candidate set / quantisation is misconfigured.

```python
def _bk(n, bucket_size):                            # round-down quantisation
    return n // bucket_size

def _bk_up(n, bucket_size):                         # round-up quantisation
    return (n + bucket_size - 1) // bucket_size

def knapsack_max_value_multi(items, budget, bucket_size):
    """0/1 knapsack: subset S maximising Σ gain(s∈S) subject to
    for each (tier, sp) axis a:
        Σ_{s∈S} re_use[tier][sp](s)   <= budget[a]
    `bucket_size` keyed by axis (tier, sp); each axis quantises at
    its own page granularity.  Quantises re_use round-UP (safe for
    <=).  This is the ONE primitive the value-gated joint_decide uses:
    relief feeds it Migrate value-items (gain = −cost, re_use =
    destination `acquired`, budget = DRAM/DISK free room); resume feeds
    it Resume candidates (gain = V_u, re_use = {HBM: re_use}, budget =
    per-HBM-subpool free room).  Both may select the empty set."""
    axes = list(budget.keys())                       # list of (HBM, sp)
    W    = {a: _bk(budget[a], bucket_size[a]) for a in axes}
    K    = len(items)
    NEG  = float("-inf")

    dp   = {tuple(0 for _ in axes): 0.0}
    take = {}
    for k, c in enumerate(items, start=1):
        d = tuple(_bk_up(c.re_use.get(tier, {}).get(sp, 0),
                         bucket_size[(tier, sp)])
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
            s_pick = tuple(s_pick[i] - _bk_up(
                                c.re_use.get(axes[i][0], {}).get(axes[i][1], 0),
                                bucket_size[axes[i]])
                           for i in range(len(axes)))
        k -= 1
    return chosen
```

`Migrate.acquired` is itself a per-(tier, subpool) dict — sglang's
apply path reads `acquired.keys()` directly to decide which
destination subpools absorb which bytes of the unit, without
needing a separate `target_subpool` field.

**Reconstruction.** The pseudocode above subtracts each taken item's
quantised delta to recover the predecessor state.  The exact form
instead records the **predecessor state per improving transition**
(`parent[(gi, s_new)] = (s, member)`) and follows those pointers.  This
is required once the DP runs over **multiple-choice groups** (point 1
below): the subtract-by-item-index readout cannot tell which member of a
group was taken, and nothing guarantees the per-axis subtraction lands
back on the exact predecessor cell.  Parent pointers make the
chosen-subset readout exact.

**Three refinements to the pseudocode above:**

1. **Multiple-choice, not plain 0/1, over a unit's transitions.**
   `migrate_candidates` emits several transitions per unit (evict /
   spill / DROP).  A plain 0/1 knapsack can take TWO of them, which is
   physically incoherent — the relief double-counts the unit's bytes
   and the costs aren't additive (each is scored as a marginal change
   from the unit's ORIGINAL residence).  The DP treats candidates
   sharing a `group` key (= unit hash) as **at-most-one**
   (multiple-choice); Resume is ungrouped (one per program).
2. **`budget` (destination room) clamps to `max(0, cap − used)`.**  A
   destination tier can be over-subscribed (`cap − used < 0`).  A
   negative budget makes the DP reject even zero-consume candidates (the
   accumulator starts at 0, already `>` a negative bound).  No room is 0
   room, never less than zero.  The value-gated DP needs no
   infeasibility / best-effort handling at all: every phase is value-
   maximising with the empty set reachable, so "no subset relieves the
   pressure" is simply the empty plan (no-op), never a crash — the
   in-flight-dominated pressure case (a non-migratable pegged subpool)
   is handled by construction, not by a `best_effort` fallback.
3. **The budget covers EVERY axis a candidate consumes, not just the
   configured destination subpools.**  The DP reads consumption only on
   axes present in the `budget` dict; an axis a candidate consumes but
   that is absent would be silently treated as 0 bytes (free), letting
   the DP over-subscribe a destination subpool that isn't mirrored across
   tiers (§10 subpool-key consistency assumes it is — this is the
   fail-safe for when it isn't).  So `joint_decide` extends `cap_left` /
   `free_room` with every `(tier, subpool)` appearing in any candidate's
   `acquired` / `re_use`, defaulting an UNCONFIGURED subpool to **0 room**
   (a unit cannot be written to a subpool that does not exist → any
   consumption there rounds up to ≥1 bucket > 0 → the candidate is
   rejected).  Consistent with `page_bytes`'s fail-loud stance: a missing
   destination axis is never silently free.

**SESSION_END migration is pressure-gated.**  Migrate candidates are
emitted only in the relief phase, so SESSION_END demotes the ended
program's exclusive units **only when HBM is pressured AND the demotion
is net-positive** — otherwise they stay resident.  This is correct under
§9's value-gated model and benign under the §3 superset framing: the
ended units carry the lowest V_u (hint-table broadcast), so they are the
FIRST evicted under any pressure, and sglang's own inline eviction
reclaims them when it needs the space regardless of the daemon.  The
control-plane path (gate release, ENDED transition, `PUT {ENDED}`) runs
independently.

#### Why exact DP, not greedy

LP-relaxation greedy (sort by `value/byte`, take densest-first until
the budget fills) is the standard 0/1 knapsack approximation.
Worst-case it's 2× off — e.g. budget 10, items A=(value 6, weight 6),
B=(value 5, weight 5), C=(value 5, weight 5): greedy takes A first
(density 1.0) then can fit neither B nor C (4 left), total value 6;
optimal is B+C at value 10.

That worst case happens whenever the densest item crowds out a pair of
slightly-less-dense items that together pay more.  Not adversarially
constructed — common when a few cold units share a destination subpool's
limited room and one bulky migrate would block two cheaper ones.

At K ≈ 30 and bucketised W ≈ 100, exact DP is microseconds — the
same order as greedy.  There is **no efficiency reason** to use
the approximation.  The exact form is the design.

#### Properties this satisfies

1. **Optimal net value** under the knapsack formulation.  No
   1/2-approximation gap; the empty set is always a candidate, so a
   negative-value plan is never chosen.
2. **Per-tier sub-budgets.**  The destination `budget[(τ, sp)]` tracks
   how many bytes remain in each subpool; the DP rejects any subset that
   would overflow τ.  Without this the plan could schedule 50 HBM→DRAM
   moves into a DRAM that's already 95 % full.
3. **Relief and Resume coexist, both value-gated.**  They are NOT
   disjoint phases — relief (cold-unit migrate out of a pegged subpool)
   and Resume (un-starve a fitting paused program) run in the same event
   and their plans combine.  Each is its own value-maximising knapsack
   that may pick the empty set, so neither forces the other and neither
   forces action: a resume that adds 0 bytes to a pegged subpool fits
   even under pressure, and a pegged subpool with no net-positive migrate
   simply yields no relief.

#### Always-fresh state at the inter-event boundary

Exact DP solves one knapsack against a single snapshot of state.
Within that DP, candidates' `cost` / `relief` are evaluated **on
the snapshot fetched at event entry** — the formulation is
exact w.r.t. that snapshot, so no per-pick re-ranking is needed
(unlike a greedy loop which mutates state mid-pass).  The
always-fresh invariant (§10) is satisfied at the event boundary:
the next event's joint_decide will refetch state and re-solve
its knapsack from scratch.

### What collapses out

* **Trade gate**: the value-maximising DP includes an item only if some
  net-positive subset contains it; any item that doesn't pay for itself
  (or is dominated by a cheaper alternative achieving the same) is
  rejected by construction — no explicit "is this move worth it?" guard
  is needed, the `value = −cost > 0` gate and the empty-set option carry
  it at the DP level.
* **Composition order**: the DP enumerates subsets implicitly across the
  union; there's no "run kv_scheduler first then admission".  A relief
  Migrate and a Resume can both appear in the combined plan, or only one,
  or neither — whichever maximises total net value.
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
| **Daemon is a single asyncio process**: one OS process hosts the event_router consumer task, the proxy's request-forwarding tasks, and the outbound action worker task.  They share memory directly (no IPC).  On crash all tasks die together; "the daemon" and "the proxy" never desynchronise.  This makes the volatile-queue and proxy-gate invariants below well-defined as a single failure domain | daemon launch script + asyncio runtime |
| **Two fault classes + sustained-escalation tier**: per-event, faults split into **deployment-bug** (schema mismatch, missing required state fields, joint_decide DP blow-up, `peak_bw_bps ≤ 0`, mode-switch attempt, hash collision — `fatal(...)`) and **load** (apply_failed race, sglang briefly slow, transient outbound queue depth — log + absorb).  The two never blur per-event.  In aggregate, sustained load failure escalates: when consecutive outbound POST failures (accumulated across worker wakes, reset only on a 2xx) + the dispatched batch's queue-wait age (= the oldest `enqueue_ts` among the coalesced burst being dispatched) BOTH cross operator-tunable thresholds on the same dispatch, the daemon `fatal('sglang_sustained_unreachable', ...)` exits and the supervisor (systemd/k8s) restarts — crash-only software pattern.  `fatal()` uses `os._exit(1)` (NOT `sys.exit(1)`) so the crash path does not depend on asyncio Task exception propagation routing the `SystemExit` past uvicorn / `gather` / `shield` wrappers.  Running degraded forever (queue growing while sglang stays dead) is NOT a valid state.  Operator alerts off the `outbound_queue_depth` / `outbound_oldest_age_ms` quantiles; for live snapshots, `/health` exposes `outbound_consecutive_failures` and `outbound_oldest_age_ms` — the age field is computed LIVE from the current in-queue head (decays to 0 when the queue drains), not cached from the last pop, so k8s readiness probes do not get stuck after sglang heals.  fatal-escalate is the hard backstop | daemon code review + outbound queue observability |
| **Fatal halts emit forensic state dump**: every `fatal(reason, **context)` call writes a structured JSON file to `<daemon-data>/forensic/<reason>_<ts>.json` containing the event that triggered the handler, the full `/aginfer/state` snapshot fetched at handler entry, the candidate sets produced upstream, all DP inputs (`budget`, `bucket_size`, axes, etc.), the failure reason string, and the Python traceback — then logs a fatal-level line pointing at the file path and `os._exit(1)` (NOT `sys.exit(1)`, per the fault-class row above — the crash path must not depend on asyncio `SystemExit` propagation).  Supervisor restart policy is deployment-controlled; the forensic file survives the restart for post-hoc analysis | daemon `fatal()` helper |
| **Policy mode is launch-time, never runtime-switched**: sglang launches in exactly one mode — either with the aginfer daemon attached (full policy via hint table), or with the default policy module (LRU-equivalent V_u, baseline ablation).  A daemon configured for "aginfer full" that loses its daemon mid-run halts loudly; it does not degrade to the default module.  Mode is a deployment choice, not a runtime fallback | sglang launch flags + daemon liveness check |
| **Single-worker event loop + belief-event-sourcing** (§3): handlers serialised, no concurrent migrate races.  **Belief** — kv_scheduler, admission, forecast, program_tracker — recomputes only on event arrival, **never from elapsed wall-clock** (no "looks idle/ended" time-inference; SESSION_END is client-signalled, §4).  The **action-timeline** plane MAY dispatch an *already-decided* action at a predicted due time, clocked by the event stream's advance (with the `heartbeat_s` re-fire as a floor) and **re-validated against belief at fire** (stale ⇒ idempotent no-op); this paces execution, not state | asyncio queue + single consumer + due-action heap |
| **Unit hashes are content-derived and collision-detected on migrate build**: `u.hash` is sglang's existing SHA-256 page-chained content hash (`compute_node_hash_values` over the unit's token prefix; computed lazily on KV-event emission and on migrate-action processing).  Collision probability is negligible at any tree size aginfer encounters (≤ 10⁷ live units, birthday-paradox bound < 10⁻²²) but is verified, not assumed: every `apply_aginfer_migrations` call builds a `{hash → node}` lookup via a single DFS over the radix tree.  At that point sglang checks whether the same hash key already maps to a different node; if so, it fires the `HASH_COLLISION` webhook (§4) with both nodes' summaries and the daemon `fatal()`s with a forensic dump.  Cost is amortised free — the DFS already runs O(N) per migrate batch; adding the collision check is one extra comparison per node | sglang `apply_aginfer_migrations` |
| **Always-fresh state**: every handler entry re-fetches `/aginfer/state`; event payload's snapshot is never trusted | daemon `KvScheduler.handle()` |
| **Idempotent daemon→sglang actions**: every endpoint accepts re-application — migrate (a no-op `applied=0` with a `race:*` skip when the target tier already matches), pause/resume (200 with `applied:false` when the program is already in the requested state), hint PUT (overwrite-by-stamp).  Same reasoning across all of them: the daemon may emit the same action twice across consecutive event handlers because state-dump propagation lags the daemon's just-emitted action; the sglang side must absorb this without error.  Idempotency is the correctness backstop.  For resume specifically, the daemon ALSO avoids the redundant re-fire using its EXISTING program-state authority — the `program_tracker` (a derived cache, this section) records a resume it has ISSUED and reconciles that record against the fresh dump each event: while the dump still lags (shows PAUSED) the resume is not re-proposed; once the dump reflects the clear the record drops; if the dump never reflects it within a bounded window the clear is treated as LOST and the resume re-fires (recovery — the outbound queue does not retry).  This is NOT a violation of "No daemon-side hint cache" below: that invariant forbids an UNBOUNDED per-unit shadow map with no authority role; resume-in-flight is bounded by the paused-program count and is part of the tracker's program-lifecycle truth | sglang endpoints + program_tracker reconciliation |
| **Outbound queue is volatile**: the daemon's outbound action queue lives in memory only.  On daemon crash, pending actions are lost — and that's correct.  After restart, the daemon's first `GET /aginfer/state` reads the live state; if the lost action was needed, `joint_decide` re-issues it.  No disk WAL exists for outbound actions, by the same first-principles argument that rules out a webhook persistence WAL (§11): every authoritative quantity already lives in sglang's state, the daemon is just a decision function over it | daemon outbound worker |
| **No daemon-side hint cache**: the daemon does not maintain a shadow `{hash: last_pushed_value}` map.  Per-event, the daemon re-scores the units in `D_t` (small set, ≤ K_MAX = 256) and pushes their hints unconditionally; sglang's hint table is overwrite-by-stamp (§6) and dedupes on its side.  Eliminates an unbounded daemon-side data structure; the trade is some redundant PUTs that overwrite identical values — negligible at D_t cardinality | daemon kv_scheduler hint emitter |
| **Proxy gate releases on client disconnect**: a request held in the proxy gate awaits BOTH the gate condition AND `request.is_disconnected()`; whichever fires first wins.  TCP disconnect deterministically signals the client gave up — no timer, no fallback.  On disconnect the proxy releases the gated request locally, the daemon's `program_tracker.client_disconnected(p)` enqueues `PUT /aginfer/program_paused {transition: END, ...}` onto the outbound queue, and `p`'s residence is reaped at the next state-dump | daemon proxy gate |
| **Observability for state-dump cost**: every `GET /aginfer/state` records its wall-clock latency on the sglang side, and the daemon logs (a) latency-per-fetch and (b) event-queue depth at handler entry.  No backpressure mechanism is wired (drop-on-full / coalescing) — these are kept off the spec until measured evidence shows the queue grows unboundedly.  The logs are the first-class signal for when to revisit | sglang dump path + daemon event_router metrics |
| **Atomic unit visibility**: units appear in `/aginfer/state.units` **only after sglang commits the chunk to the radix tree** (page-aligned commit boundary).  Partial-prefill chunks under chunked prefill do not appear as units; the daemon does not observe in-progress prefill state, only the post-commit snapshot.  This eliminates a class of "what's the p_hat of a half-written unit" questions by construction — half-written units don't exist in the spec's data model | sglang radix-tree commit path |
| **Subpool key consistency**: for every unit `u` and every tier `τ ∈ u.residence`, `u.n_bytes[τ].keys() ⊆ state.pool_usage[τ].subpools.keys()`.  Sglang's UnifiedRadixCache component registry guarantees this on commit; §7's `_value` holding-cost loop iterates `u.n_bytes[τ]` and looks up `pool_usage[τ].subpools[sp]` directly without any defensive `.get(...)` — a missing key is a deployment bug | sglang component registry |
| **Preemption transparency**: sglang's continuous-batching preempt-and-resume of in-flight requests changes `per_program_usage[p].hbm.inflight` between events without any daemon action.  The daemon does not track inflight state across events (cf. always-fresh invariant); each handler re-fetches state, so a preempted-then-resumed program is indistinguishable from one that never preempted.  Forecast / relief budget / pause_relief are computed on the live snapshot, so they always reflect post-preemption truth | always-fresh state + §5 per_program_usage |
| **State-dump internal consistency**: a single `/aginfer/state` response is a snapshot taken under a single read-side lock on sglang's tree cache + allocator + per-program tables.  `units[*]`, `per_program_usage[*].unit_hashes`, and `pool_usage` always refer to the same logical timestamp; `state.units[h]` is safe to dereference for every `h ∈ per_program_usage[p].unit_hashes` | sglang `dump_aginfer_state` |
| **Hint clear ordering**: when sglang evicts unit u, the order is (1) inline scorer finishes its current heap-iteration read (using the hint that was live when u was picked); (2) eviction commits, allocator reclaims u's bytes; (3) hint entry for u is cleared.  This rules out a "scorer reads cleared hint" race; the scorer never sees a missing entry for a unit that's still in the heap | sglang inline scorer + allocator commit path |
| **Webhook mandatory**: every sglang launch passes `--aginfer-notify-url`; the admission trigger path is not optional | launch script per-deployment |
| **Pool-truth admission, per subpool**: admission's `forecast(state)` returns a dict over HBM subpools, each reading `state.pool_usage.HBM.subpools[sp].used_bytes` and `cap_bytes` (byte-denominated, since `forecast_inflight_demand` is byte-scaled and §9 budgets are bytes).  Pressure fires when **any** subpool's forecast crosses `theta_hi`; this prevents the failure mode where a Mamba snapshot subpool at 95 % is hidden by an attention subpool at 60 %.  If the snapshot lacks `pool_usage.HBM.subpools` the daemon halts loudly | daemon `admission_controller.forecast` |
| **Physical inputs sourced from sglang**: `bw_free(σ, τ)` reads `state.link_stats[σ→τ]`, never an in-daemon estimate; `h_(τ, sp)(occ)` reads `state.tier_holding_cost[τ][sp].h_max_per_byte_sec`, never a constant baked into the daemon; `prefill_throughput(state)` reads `state.throughput_ema.prefill_bps`; `decode_throughput(p)` reads `state.throughput_ema.decode_per_program[p]`.  Sglang owns the measurements (HiCache + Mooncake instrumentation hooks expose link throughput; the prefill / decode loops update their per-step throughput EMA; operator config sets per-(tier, subpool) `h_max`); the daemon consumes them.  Required positivity: `peak_bw_bps > 0`, `h_max_per_byte_sec > 0`, `prefill_bps > 0` (once any prefill has run), and `page_bytes > 0` for every configured subpool (the §9 DP quantises by it — a 0 would divide-by-zero).  Missing fields or non-positive values are deployment bugs → `fatal()` | sglang instrumentation + operator config |
| **Pool-truth V_u**: V_u's `h_(τ, sp)(occ)` term reads `pool_usage[τ].subpools[sp]` (the per-subpool allocator-truth view) so holding cost reflects real pressure, not an inferred radix-tree slice | daemon `OursGreedyPolicy._value` |
| **All traffic through daemon proxy**: every chat-completion to sglang arrives via the daemon's `/v1/chat/completions` proxy; direct-to-sglang clients are out of scope and would render admission's program-pause unenforceable | deployment topology |
| **Hint table covers every live unit**: sglang seeds a "fresh access just happened" entry on unit birth (`p_hat ≈ 1` for the next event horizon); the daemon refines via `PUT /aginfer/hints`; eviction never falls back to LRU on absent hints | sglang allocator hook + daemon hint pusher |
| **Hint atomicity**: the inline scorer's read of a hint entry and the daemon's `PUT` of a new entry are atomic per-key (read-modify-write would race a daemon update against an in-flight eviction).  Per-key seqlock or compare-and-swap suffices; full RW lock is overkill at 10²/s | sglang hint-table data structure |
| **Layer enable flags**: HiCache, kv_scheduler, and admission each have an independent enable flag.  Admission can only fire when kv_scheduler is also on (admission's pre-pause migrate path requires the daemon's V_u machinery).  HiCache is independent of both | daemon CLI + sglang flags |
| **Threshold parity**: sglang and the daemon use the SAME `theta_hi` / `theta_lo` / `theta_crit` / `heartbeat_s` values.  The daemon is the canonical source; sglang halts at bootstrap if the daemon is unreachable.  Runtime changes flow daemon → sglang via `PUT /aginfer/thresholds` (§6).  Aginfer-managed sglang requires the daemon to be up first; no cache, no last-known-values fallback | daemon `GET /aginfer/thresholds` + daemon→sglang update broadcast |

## 11. Recovery (daemon restart / sglang restart)

`per_program_usage` (in sglang's `/aginfer/state`) is the
authoritative store for every program-level state field that
survives across crashes — `state`, `pre_pause_state`,
`unit_hashes`, `hbm`/`dram` footprint.  The daemon's in-process
`program_tracker` is a **derived cache**, rebuilt from
`/aginfer/state` on every restart; it never holds information
not also persisted in sglang.  The daemon's *truly* volatile
state is narrower: the proxy's pause-gate suspended request set
and the `migrate` retry queue.

### Daemon restart

1. `GET /aginfer/state` rebuilds the `program_tracker` cache by
   walking `per_program_usage`: each entry's `state` becomes the
   tracker's REASONING / ACTING / PAUSED / ENDED label.  No field
   is lost; `pre_pause_state` (consumed by the §8 Resume gain
   counterfactual) is read straight from the snapshot, no
   in-process daemon memory required.
2. **Released pauses**: the proxy gate is empty after restart, so
   the request a PAUSED program was holding is lost.  Any program
   with `per_program_usage[p].state == PAUSED` at restart is
   re-registered as PAUSED and the proxy gates its next arrival;
   the client hits the gate when it retries.  Silently releasing
   all paused requests would be wrong (user-visible inconsistency
   vs the intended pause).  This relies on retry-on-timeout
   clients — mandated by the "All traffic through daemon proxy"
   invariant (§10); non-retrying clients are out of scope.
3. **In-flight migrate POSTs** are not retried — idempotent (§10),
   so re-issuing on the next event is harmless.

### sglang restart

The daemon detects restart via `/aginfer/state` failure modes
(connection refused, then 200 with reset counters).  On the first
200 after a failure it clears its hint-side tracking (sglang's
hint table is empty at startup) and re-pushes hints lazily as
units reappear in subsequent snapshots; `program_tracker` state
is preserved daemon-side and the proxy gate re-applies to incoming
requests.

### SESSION_END normal path (REASONING / ACTING / ENDED programs)

Harbor signals `SESSION_END` for a program not held in the proxy
gate:

```python
def on_session_end(p, event):
    if program_tracker.state(p) == PAUSED:
        return on_session_end_paused(p, event)     # see below
    # joint_decide uses D_t = session_scoped_units(p) (§7), so
    # migrate_candidates yields demote/drop candidates for p's
    # exclusive units.  Admission sees p with state==ENDED, filtered
    # out of both pause_candidates and resume_candidates (§8).
    plan = joint_decide(state, event)
    program_tracker.set_state(p, ENDED)
    outbound.enqueue(plan)                          # migrate batch
    outbound.enqueue(PUT("/aginfer/program_paused",
        {"pid": p, "state": "ENDED", "pre_pause_state": null}))
```

`session_scoped_units(p)` are the units only p holds
(`session_ids == {p}`) — pure-eviction candidates after p ends.
Units p shared with other programs survive p.

### SESSION_END for a PAUSED program

If `SESSION_END` arrives while p is PAUSED (its next request
sitting in the proxy gate), the gated request is cancelled — the
client said they're done with the session:

```python
def on_session_end(p, event):
    if program_tracker.state(p) == PAUSED:
        gated = proxy.gate.release(p)
        if gated is not None:
            gated.respond(status=499, body=b"")   # client closed request
    program_tracker.set_state(p, ENDED)
    outbound.enqueue(PUT("/aginfer/program_paused",
        {"pid": p, "state": "ENDED", "pre_pause_state": null}))
```

The PUT clears `per_program_usage[p].state` to ENDED and
`pre_pause_state` to null on the next dump; `session_scoped_units(p)`
becomes the §7 D_t for the event, demoting/dropping any units only
p owned.

### Lost webhook is correct by construction

If sglang fires a webhook while the daemon is down, the event is
**lost** — and that is correct, not a trade-off.  An event is the
*derivative of state*, and state is authoritative in sglang (§5);
every handler re-fetches `/aginfer/state` at entry (always-fresh,
§10), so its input is the live state regardless of whether any
specific payload was delivered.  A lost webhook just means the
daemon never woke for that transition; the next event of any kind
wakes it and the re-fetched state already reflects the missed
transition.  sglang's `heartbeat_s` re-fire under HIGH/CRITICAL
guarantees a wake within ≈ 5 s, and any chat-completion through
the proxy wakes it sub-second at 10²/s event rates.  A persistent
queue or WAL would be a duplicate authoritative store for state
sglang already holds; at-least-once delivery adds no information
to a recipient that re-reads the source every handler.

### Burst pressure: harmless duplicate plans

Two `MEMORY_PRESSURE` events in quick succession (an up-crossing
plus a heartbeat while the daemon is still draining event 1's
outbound queue): event 2's handler re-fetches state, sees whatever
of plan A has landed, and computes plan B against it.  If plan A
fully applied, plan B sees relieved pressure and is empty/minimal.
If plan A is mid-flight, plan B may re-issue the same `Migrate`;
sglang's idempotency (§10) makes that a 200 `applied=0` `race:*`
skip, and any resulting APPLY_FAILED webhook is an ordinary event.
Harmless by construction — always-fresh + idempotent absorb the
duplication; the only cost is wasted outbound HTTP and APPLY_FAILED
chatter, cured upstream (lower the HIGH heartbeat rate) if it ever
dominates, never by daemon-side action de-dup.

### Threshold tuning guidance (operator notes)

CLI defaults:

* `theta_hi = 0.85` — pressure trigger.  Lower if APPLY_FAILED
  rate is high; raise toward 0.90 only on very smooth occupancy
  curves.
* `theta_lo = 0.70` — pressure-resolved threshold.  Should sit
  ≥ 10 % below `theta_hi` so the hysteresis band absorbs a decode
  cycle's growth without re-triggering.
* `theta_crit = 0.95` — escalation watermark.  Don't tune; hitting
  it regularly means `theta_hi` is too high.
* `heartbeat_s = 5` — webhook re-fire interval under HIGH/CRITICAL.
  Lower (1–2 s) where pressure resolves fast; raise (10–15 s) where
  it persists and the chatter is wasted.

Start at the defaults and adjust on state-dump observability
telemetry: queue-depth growth + APPLY_FAILED rate are the leading
indicators that thresholds need re-tuning.

## 12. Scenarios

The framework above is architecture-agnostic: it operates on
named-subpool dicts (`pool_usage.<tier>.subpools`, `u.n_bytes`,
`tier_holding_cost`), instantiated per the model's attention /
state-space component mix.  Each scenario lists which keys appear
and how unit bytes split across them; the §7 / §8 / §9 machinery
is unchanged.

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

forecast / capacity_fits / the relief budget each have a single
`attn` key; the multi-axis sparse DP reduces to the 3-axis case.

### S2 — SWA-hybrid attention (e.g. Mistral, Gemma)

Two attention components: full-attention layers and sliding-window
layers, with separate subpools because SWA layers retain only the
last N tokens per sequence.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"full": {...}, "swa": {...}}` |
| `pool_usage.DRAM.subpools` | `{"full": {...}, "swa": {...}}` |
| `pool_usage.DISK.subpools` | `{"full": {...}, "swa": {...}}` |
| `units[i].n_bytes` | `{"full": F, "swa": S}` where `S > 0` only when the unit's tokens are still inside the SWA window; `S == 0` for aged-out units |
| §9 DP axis count | 5 (full + swa across HBM relief, DRAM cap, DISK cap; plus the same across DRAM and DISK destinations) |

Architecture-specific behaviour: when the SWA window slides past
a unit's tokens, sglang sets the unit's `n_bytes.swa` from `S` to
`0` at the next state-dump.  The unit stays in the radix tree (its
`full` bytes are still valuable for prefix reuse); the next
`joint_decide` event sees the new `n_bytes` shape and re-scores
naturally — no special daemon handler.

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

Architecture-specific behaviour: only leaf nodes carry mamba bytes
(one snapshot per leaf).  When the tree extends a leaf, the ex-leaf
becomes intermediate and its mamba bytes move to the new leaf or
are freed — sglang manages this transparently; the daemon observes
the updated `n_bytes` shape on the next dump.  In-flight Mamba
state grows only at snapshot boundaries, so
`forecast_inflight_demand[mamba]` is 0 between snapshots (§8).

### S4 — Mamba+SWA+full (future hybrid)

Three subpool keys in HBM; not yet observed in production models
but the framework supports it without modification.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | `{"full": {...}, "swa": {...}, "mamba": {...}}` |
| `units[i].n_bytes` | `{"full": F, "swa": S, "mamba": M}` with `S = 0` when aged out of the SWA window and `M = 0` when the unit isn't a Mamba leaf |
| §9 DP axis count | 9 (3 HBM-relief axes + 3 DRAM-cap axes + 3 DISK-cap axes — one per (tier, subpool) pair) |

Sparse DP cell count remains the binding factor — the dense table
is intractable at this axis count, but the reachable-cell count
under agent workloads (most candidates touch only 1–2 axes
nontrivially) stays in the 10⁵ envelope.

### S5 — Speculative decoding (orthogonal to S1–S4)

Speculative decoding adds a `draft` subpool whose bytes are
**per-request transient**: allocated by the verifier, lived
across one verify step, then discarded or promoted into the
attention subpool.  Draft units are **not radix-tree-cached** —
they never appear in `state.units`.  Their bytes show up only
in `state.per_program_usage[p].hbm.inflight["draft"]`.

| field | shape |
|---|---|
| `pool_usage.HBM.subpools` | adds `"draft"` alongside the base scenario's keys (`"attn"` or `"full"/"swa"` or `"full"/"mamba"`) |
| `state.units` | no change — draft KV is not radix-tracked |
| `per_program_usage[p].hbm.inflight["draft"]` | bytes currently in the draft buffer for p |
| `n_bytes` for tree units | no `draft` key (drafts are never tree units) |
| §9 DP axes | +1 HBM axis (`("HBM", "draft")`) |

`forecast_inflight_demand[draft]` matches the attention case:
`inflight["draft"]` grows proportional to `decode_throughput(p) ×
draft_factor` (speculative expansion, typically 4–8× for Medusa /
Eagle) and saturates at a fixed per-request maximum (the draft
window).  Pause-relief for a spec-decoding program includes the
draft subpool (pausing frees its discarded `inflight["draft"]`
bytes).  No Migrate candidates over draft units — a discarded
draft has no prefix-reuse value; the only action over draft bytes
is pausing the owning program.

### Composition of scenarios

The scenarios are orthogonal in the subpool axis; real deployments
combine them by union of subpool keys — Mistral (S2) + spec-decode
(S5) has `{"full", "swa", "draft"}`; Jamba (S3) + spec-decode has
`{"full", "mamba", "draft"}`.  The §9 axis count grows linearly
with subpool count, and the assertion below holds for any
composition the engine exposes.

### Out of scope (framework-compatible)

**Multi-LoRA serving**.  Adapter activations live per-request in
HBM, much like draft KV, and fit the named-subpool abstraction
(declare an `"adapter"` subpool, route adapter bytes via
`per_program_usage[p].hbm.inflight["adapter"]`).  No
multi-LoRA-specific schema or decision-rule machinery is required;
the spec's primitives suffice but do not instantiate it.

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

Across S1 – S5 the daemon code path is identical: read
`pool_usage.HBM.subpools.keys()` at state-dump entry, iterate
over those keys for every per-subpool quantity (`forecast`,
`cap_left`, `pause_relief`, `re_use`, `relief`,
`tier_holding_cost`).  No scenario-specific branch lives in the
daemon — the keys come from the state-dump, the rest follows.
