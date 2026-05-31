# aginfer — implementation & verification plan

Companion document to `DESIGN.md`.  Captures the **forward-looking
work** (empirical calibration, implementation gaps, scenario
verification) that the design references but does not itself
contain.  Each section is keyed back to the `DESIGN.md` section
that motivates it.

## 1. Empirical calibration tasks

Tasks that produce **data** the design depends on but doesn't
prescribe a specific value for.  Each lands a small per-task
write-up under `verify/<task>/README.md` per the per-task RESULTS
doc convention.

### T11 — p_hat estimator (DESIGN §7)

Today `p_hat(u, Δt)` is computed from a session-state-conditional
estimator with the holder product `1 - Π(1 - p_access(u, s, Δt))`.
The per-holder `p_access` table is currently rule-based on
program state (REASONING / ACTING / PAUSED / ENDED).

Goal: replace the rule with a **workload-agnostic estimator**:

- Histogram-based / per-program-bucket / Hawkes-fit candidates
  (existing `MEMORY/todo-empirical-phat.md` for context)
- Must be **session-state-aware** but not session-state-branched —
  the program state is a feature input, not a switch (per
  `feedback-workload-agnostic-phat.md`)
- Evaluation: replay T9 / T11a traces with each candidate
  estimator, score by `Σ |observed_access - predicted_p_hat|` on
  held-out segments

Open output: `verify/t11/README.md`.

### T12 — `h_(τ, sp)(occ)` shape calibration (DESIGN §7)

Spec uses linear placeholder `h_(τ, sp)_max × occ`.  Real shape
is unknown; candidate functional forms:

- linear `α × occ`
- power `α × occ^γ` for γ > 1 (right-tail-heavy)
- hyperbolic `α / (1 - occ)` (diverges as occ → 1, matches §9
  admission cap)

Method:

1. At every event during T9 / T11 runs, log `(tier, subpool, occ,
   marginal_V_u)` quadruples — `marginal_V_u` is the V_u of the
   lowest-V_u resident in that subpool at that occupancy
2. Fit the three candidate shapes per subpool, report residuals
3. Pick the one that minimises residual; falsify the linear
   placeholder if its residual is materially worse

Open output: `verify/t12/README.md`.

### T13 — `bw_free` EMA validation (DESIGN §5 link_stats, §7 bw_free)

Sglang instruments `recent_throughput_bps` on each direction
(HiCache write_backup / load_back for HBM↔DRAM, Mooncake put / get
for DRAM↔DISK).  Validate that the EMA tracks reality:

- Compare `recent_throughput_bps` against ground-truth wall-clock
  per migrate, under both idle-link and contended-link conditions
- Probe the fallback path (link idle → `peak_bw_bps` reported)
- Pin `samples_in_window > 0` consistency across consecutive
  state-dumps

Open output: `verify/t13/README.md`.

## 2. Observability instrumentation

The design pulls some failure-mode policies out of scope, replacing
them with metrics so we can spot the problem empirically before
designing the fix.

### State-dump cost (DESIGN §10 "Observability for state-dump cost")

- Sglang side: per-call wall-clock for `dump_aginfer_state`,
  emitted as a metric (histogram).  Trigger condition: p99 > 50 ms.
- Daemon side: queue depth at `event_router` entry, time-in-queue
  per event.  Trigger condition: queue depth > 64 sustained, or
  time-in-queue > 100 ms p99.
- Either trigger flips us back into a F3-revisit task to decide
  drop-on-full vs coalesce vs incremental-state.

### Hint table cross-rank divergence (DESIGN §6 "Hint consistency", round-8 H3)

The eventual-consistent hint table is justified by the
"eviction is the cross-rank sync point" argument.  The argument
fails if two ranks ever evict **different** units near-simultaneously
under stale hints.  Test plan:

- TP > 1 deployment under high hint churn (forced by aggressive
  workload mix that pushes scorer decisions repeatedly)
- Log per-rank evicted-hash set per state-dump window
- Compare sets; any divergence breaks the §6 invariant

Open output: `verify/hint_cross_rank/README.md`.

### bytes_at-doesn't-double-count (DESIGN §8 capacity_fits / re_use)

Verify post-fix B1: a paused program whose units are still
HBM-resident (kept alive by other holders) has `re_use[sp] == 0`
for those units.  Round-9 part 1 fixed the formula; this verifies
the actual daemon code agrees.

## 3. sglang implementation work

Concrete sglang patches needed to align the code with the design.
Order roughly by dependency.

### Schema-side

1. **State-dump schema upgrade** (DESIGN §5):
   - `pool_usage.<tier>.subpools` dict (currently `pool_usage.HBM`
     is flat; SWA fields exist as optional)
   - `pool_usage.<tier>.subpools[sp].page_bytes` per subpool
   - `per_program_usage[p].hbm.committed` / `inflight` as
     per-subpool dicts
   - `per_program_usage[p].state` and `pre_pause_state`
     authoritative (round-6 H2)
   - `per_program_usage[p].unit_hashes` materialised list
   - `units[i].residence: list[Tier]` (round-9 part 4b)
   - `units[i].n_bytes: {tier: {subpool: bytes}}` nested
   - `link_stats` with peak / recent / samples per direction
   - `tier_holding_cost` per-(tier, subpool)
   - Drop legacy top-level `page_size` field

2. **State-dump internal consistency** (DESIGN §10 R1):
   - single-snapshot under one read-lock so `units[*]` and
     `per_program_usage[*].unit_hashes` and `pool_usage` refer to
     the same logical timestamp

3. **Atomic unit visibility** (DESIGN §10 D4):
   - units appear in `/aginfer/state.units` only after
     page-aligned chunk commit; partial-prefill chunks not exposed

### Endpoints

4. **`POST /aginfer/migrate` payload** (DESIGN §6):
   - accept `{add_tiers, remove_tiers}` per action
   - implement residence-set transitions
   - add `action_id` opaque correlator
   - skip-reason classes extended with `write_through_declined:*`

5. **`PUT /aginfer/program_paused`** (DESIGN §6 round-6 H2):
   - new endpoint
   - writes `state` and `pre_pause_state` into `per_program_usage`

6. **`GET /aginfer/thresholds` + `PUT /aginfer/thresholds`**
   (DESIGN §6 round-6 H3):
   - bootstrap fetch returns canonical thresholds
   - `PUT` from daemon → sglang updates local cache file atomically
   - sglang halts on first-launch-without-cache-or-daemon

7. **`APPLY_FAILED` webhook** (DESIGN §4 round-9 B4):
   - on action apply failure, sglang fires this webhook back to
     daemon with `{endpoint, action_id, reason}`

8. **All action endpoints idempotent** (DESIGN §10 R2):
   - re-applying the same action returns 200 with `applied=0`
   - migrate, pause/resume, hint PUT, threshold PUT

### Instrumentation hooks

9. **HiCache + Mooncake throughput EMA** (DESIGN §7 bw_free, M4a):
   - HBM↔DRAM: CUDA-event bracket each `write_backup` / `load_back`
     pair; maintain EMA per direction
   - DRAM↔DISK: wall-clock bracket each Mooncake `put` / `get`;
     maintain EMA per direction
   - Expose via `state.link_stats`

10. **Hint clear ordering** (DESIGN §10 R3):
    - scorer's heap-iteration read happens-before eviction commit
      happens-before hint clear

11. **`should_write_through(node)` plugin point** (DESIGN §3
    superset framing):
    - factor out the `hit_count >= write_through_threshold` check
      into a pluggable hook
    - default implementation preserves current behaviour
    - aginfer registers a V_u-aware version when daemon is attached

12. **`SGLANG_KV_POLICY_MODULE` eviction scorer plugin**
    (DESIGN §3 superset framing — partially exists):
    - default module: LRU-equivalent V_u (last_access as p_hat
      surrogate) so baseline runs match historical behaviour
    - aginfer registers its hint-table-aware V_u

13. **Proxy gate disconnect awareness** (DESIGN §10 F1):
    - daemon's proxy gate awaits BOTH gate condition AND
      `request.is_disconnected()`
    - on disconnect: release gate, respond 499, transition program
      to ENDED via outbound queue

14. **`harbor /aginfer/session_end` endpoint** (DESIGN §4):
    - out-of-band signal channel the client uses to declare
      "session done"
    - lives outside sglang, on the harbor / agent client side

### Multi-rank correctness

15. **All-rank atomic actions** (DESIGN §6 multi-rank): every
    `migrate` / `program_paused` / `hints` action is applied
    atomically across TP/EP ranks via the existing
    tokenizer-server fanout

## 4. Daemon implementation work

Concrete daemon patches needed.  Most of the core decision rule is
already implemented in some form; the big work is the residence-set
generalization.

1. **Residence-set candidate generator** (DESIGN §7):
   - `migrate_candidates(state, D_t)` emits `(add_tiers,
     remove_tiers)` tuples
   - the 6 meaningful transitions per unit per §7

2. **Multi-axis sparse DP** (DESIGN §9):
   - `knapsack_min_cost_multi` with per-axis `bucket_size`
   - `knapsack_max_value_multi` for headroom phase
   - dict-keyed by `(tier, subpool)` tuple
   - feasibility-fallback to max-relief-bucket when full target
     unreachable under destination caps

3. **`authoritative_tier(residence)`** (DESIGN §7):
   - HBM if present else DRAM else DISK

4. **Outbound action queue + worker** (DESIGN §6 B4):
   - in-memory `asyncio.Queue[Action]`
   - dedicated worker task drains via `httpx.AsyncClient`
   - handler enqueues and returns immediately
   - `action_id` UUID assigned at enqueue time

5. **`APPLY_FAILED` event handler** (DESIGN §4 B4):
   - daemon registers a handler for the new event kind
   - default behaviour: log + let next `joint_decide` re-evaluate
   - failure-class metrics emitted

6. **Default-policy module** (DESIGN §3 superset framing):
   - LRU-equivalent V_u (uses `last_access_time` as p_hat
     surrogate, `hit_count`-aware)
   - matches sglang's historical scorer exactly when daemon is
     absent or hints are uninitialized

7. **F1 proxy-gate disconnect handler** (DESIGN §10 F1):
   - `program_tracker.client_disconnected(p)` API
   - enqueue `PUT /aginfer/program_paused {END}` on disconnect

8. **F2 hint emitter** (DESIGN §10 F2):
   - re-score `D_t` units per event, push all to sglang
     unconditionally
   - **no shadow `{hash: last_pushed_value}` map**

9. **F5 SESSION_END-for-PAUSED handler** (DESIGN §11 F5):
   - on SESSION_END for PAUSED program: release gate with HTTP
     499, transition ENDED, enqueue PUT

10. **Observability logging** (DESIGN §10 F3):
    - state-fetch latency p50 / p95 / p99
    - event-queue depth at handler entry
    - time-in-queue per event
    - failure-class counters for APPLY_FAILED breakdown

## 5. Scenario-specific verification

For each `DESIGN §12` scenario, a small verification run that
confirms the named subpools actually appear in `/aginfer/state`
and that the decision rule reacts as the scenario claims.

### S1 — Single-stack attention (current benchmark)

Model: DeepSeek-V4-Flash (MLA).

- 1 HBM subpool (`"attn"`) ✓ already running
- 3-axis §9 DP ✓ baseline by construction
- T9 / T11a / T12 / T13 all land on this scenario today

### S2 — SWA-hybrid

Model: Mistral / Gemma (any sglang-supported SWA hybrid).

- Verify `pool_usage.HBM.subpools` has `{"full", "swa"}`
- Verify `units[i].n_bytes["HBM"].swa` transitions to 0 when the
  unit's tokens age out of the sliding window
- Verify pressure on SWA subpool fires `joint_decide` even when
  full-attention subpool is at < theta_hi

### S3 — Mamba+attention hybrid

Model: Jamba / Zamba (any sglang-supported Mamba+attn hybrid).

- Verify only leaf radix-tree nodes carry `n_bytes["HBM"].mamba > 0`
- Verify Mamba snapshot Migrate transitions (DRAM archive, load_back)
  follow the same residence-set semantics as attention units
- Verify `forecast_inflight_demand["mamba"] == 0` between snapshot
  boundaries

### S5 — Speculative decoding

Model: Medusa / Eagle on any base model.

- Verify `pool_usage.HBM.subpools` adds `"draft"`
- Verify draft KV does NOT appear in `state.units`
- Verify `per_program_usage[p].hbm.inflight["draft"]` tracks draft
  buffer occupancy per program
- Verify `pause_relief[draft]` for a spec-decoding program is the
  full discarded draft buffer

### Multi-LoRA (out of scope)

No verification task in the current plan.  The §12 "Out of scope"
note covers the framework's extensibility — instantiate an
`"adapter"` subpool and route per-request adapter bytes through
`per_program_usage[p].hbm.inflight["adapter"]` when a scenario
lights this up.

## 6. Revisit triggers

Conditions that flip an out-of-scope item back into the work plan.

- **F3 state-dump backpressure**: triggered by either of
  - sglang-side `dump_aginfer_state` p99 > 50 ms, or
  - daemon-side event-queue depth > 64 sustained, or
  - time-in-queue > 100 ms p99
- **Hint cross-rank divergence**: triggered if the test in
  Section 2 ever observes per-rank evicted-hash divergence under
  TP > 1
- **Linear `h_(τ, sp)(occ)` placeholder**: replaced by whatever
  T12 calibration picks; if T12 finds linear *is* the best fit,
  the placeholder graduates from "Planned" to spec
- **Multi-LoRA scenario**: triggered when the workload mix
  includes per-request LoRA adapter activations occupying material
  HBM bytes
