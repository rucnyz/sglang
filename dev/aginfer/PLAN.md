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
- Evaluation: replay traces from existing `scenarios/` cycle runs
  with each candidate estimator, score by `Σ |observed_access -
  predicted_p_hat|` on held-out segments

When this lands, the T12 follow-on (collect `(tier, subpool, occ,
marginal_V_u)` quadruples from the same scenario cycle runs +
falsify the linear placeholder) becomes unblocked — see task
#173 + PLAN §1 T12 status.

Open output: `verify/t11/README.md`.

### T12 — `h_(τ, sp)(occ)` shape calibration (DESIGN §7)

Spec uses linear placeholder `h_(τ, sp)_max × occ`.  Real shape
is unknown; candidate functional forms:

- linear `α × occ`
- power `α × occ^γ` for γ > 1 (right-tail-heavy)
- hyperbolic `α / (1 - occ)` (diverges as occ → 1, matches §9
  admission cap)

**Status (#173 — 2026-06-01)**:
* **DONE** in `verify/t12/`:
  * `fitter.py` — `fit_all` / `best_by_aic` over the 3 candidate
    shapes (least-squares + AIC picker, simpler model wins on ties)
  * `parse_t12_log_lines` — parser for the structured
    `aginfer_metric event=t12_calibration tier=… subpool=… occ=…
    marginal_v_u=…` log format (consumed by the calibration script
    that lands with T11's scenario-run data path)
  * verify: clean-data recovery for each shape, robustness under
    5% Gaussian noise, malformed-line drop, AIC tie-break
* **DEFERRED until T11's data path lands** (T11 owns the scenario
  cycle runs that emit `(tier, subpool, occ, marginal_V_u)`):
  * wire the daemon log line into the kv_scheduler hot path
    (gated by env var so prod runs aren't spammed)
  * collect data from a real scenario cycle (S1 at minimum)
  * pick the best shape per `(tier, subpool)`; falsify the linear
    placeholder if its residual is materially worse
  * if linear loses, swap the placeholder in
    `baselines/costs.py:holding_unit_cost` (or thread the picked
    shape into the policy)

Output: `verify/t12/README.md`.

### T13 — `bw_free` EMA validation (DESIGN §5 link_stats, §7 bw_free)

Sglang instruments `recent_throughput_bps` on each direction
(HiCache write_backup / load_back for HBM↔DRAM, Mooncake put / get
for DRAM↔DISK).  Validate that the EMA tracks reality.

**Status (#172 — 2026-06-01)**:
* **DONE** in `verify/t13/`:
  * sglang emission contract (4 directions × 3 keys; cold-start
    `recent_throughput_bps == 0`; `time_since_last_sample_s > LINK_IDLE_SECONDS`)
  * daemon `bw_free` branch logic: idle / busy / saturated /
    `peak <= 0` fatal / threshold boundary
* **DEFERRED to T26** (HiCache + Mooncake instrumentation must
  land first — no ground-truth signal to compare against):
  * compare `recent_throughput_bps` vs ground-truth wall-clock per
    migrate under idle + contended conditions
  * pin `time_since_last_sample_s` monotonicity across
    consecutive state-dumps on a quiet link
  * EMA decay-rate calibration

Output: `verify/t13/README.md`.

## 2. Observability instrumentation

The design pulls some failure-mode policies out of scope, replacing
them with metrics so we can spot the problem empirically before
designing the fix.

### T14 — State-dump cost (DESIGN §10 "Observability for state-dump cost")

- Sglang side: per-call wall-clock for `dump_aginfer_state`,
  emitted as a metric (histogram).  Trigger condition: p99 > 50 ms.
- Daemon side: queue depth at `event_router` entry, time-in-queue
  per event.  Trigger condition: queue depth > 64 sustained, or
  time-in-queue > 100 ms p99.
- Either trigger flips us back into a F3-revisit task to decide
  drop-on-full vs coalesce vs incremental-state.

**F3-revisit status (#160 — 2026-06-01)**:
* **Sglang trigger FIRED** on 2026-05-31 with peak p99 = 321.94 ms
  (originally reproduced N=3 today as 343.90 ± 0.65 ms).
* **Fix landed**: HTTP-layer cache + 50 ms-cadence background
  refresh in `python/sglang/srt/entrypoints/http_server.py`.
  This is the "coalesce + background refresh" option from the
  three F3 candidates.  Daemon's HTTP-observed p99 dropped from
  343 ms → **12.4 ms** (27× lower).
* **Remaining gap**: the SCHEDULER-INTERNAL `state_dump_metrics.
  p99_ms` still reads ~336 ms.  The cache hides this from the
  daemon, but the scheduler still spends real wall-clock on slow
  dumps when GIL-contended with prefill/decode.  Reducing the
  scheduler-side compute is a separate sglang architectural
  change (dedicated thread / lock-free walk) — tracked as #179.
* **Trigger re-scoping**: PLAN T14's `p99 > 50 ms` clause should
  be split into:
    1. **Daemon-facing latency** (HTTP-observed) — the trigger
       the F3-revisit options were designed to fix.  Currently
       **12.4 ms**; under threshold.
    2. **Scheduler-internal compute** (`state_dump_metrics.p99_ms`
       from inside `_dump_aginfer_state_impl`) — separate concern;
       reducing it is the #179 follow-on.

Output: `verify/t14/README.md`.

### T15 — Hint table cross-rank divergence (DESIGN §6 "Hint consistency", round-8 H3)

The eventual-consistent hint table is justified by the
"eviction is the cross-rank sync point" argument.  The argument
fails if two ranks ever evict **different** units near-simultaneously
under stale hints.  Test plan:

- TP > 1 deployment under high hint churn (forced by aggressive
  workload mix that pushes scorer decisions repeatedly)
- Log per-rank evicted-hash set per state-dump window
- Compare sets; any divergence breaks the §6 invariant

**Status (#174 — 2026-06-01)**:
* **DONE** in `verify/t15/`:
  * `detector.py` — `detect_divergence` (window-diff over time-series
    of per-rank state dumps) + `summarise` formatter
  * `verify.py` — 11 synthetic stages (single-rank no-op, identical
    eviction no-op, distinct-rank divergence, partial overlap, 3-rank
    2v1, sustained 4-window divergence, rank-set-change ValueError,
    time_counter propagation, summary smoke)
  * `run_dp2_real.py` — real DP=2 sglang launcher + churn driver +
    detector run.  Demonstrated 102 real per_rank JSON snapshots
    parsed cleanly (parser-integrity contract green).  The
    divergence count under DP=2 is NOT a §6 signal — without
    hicache content-hash mode active, unit hashes are per-process
    counters (`node-N`) and the comparison is meaningless; driver
    now WARNs on counter-format dominance (audit #175-round2)
* **BLOCKED on sglang patch (#174)** — the §6 invariant the probe
  is actually meant to catch is cross-TP-rank divergence (NOT
  cross-DP-rank).  But `/aginfer/state` aggregates across TP ranks
  inside the tokenizer-server fanout; the per-TP-rank pre-aggregation
  view is not exposed as any endpoint today.  To finish the spec:
  * patch sglang to expose `/aginfer/state?per_tp_rank=1` (or a
    debug endpoint) that returns each TP rank's local view BEFORE
    the multi-rank aggregator in `http_server.py` runs
  * re-run the detector against TP > 1 sglang under churn
  * any non-empty report = §6 invariant break

Output: `verify/t15/README.md`.

### T16 — Coresidence budget `re_use` no-double-count (DESIGN §8 capacity_fits / re_use)

Verify post-fix B1: a paused program whose units are still
HBM-resident (kept alive by other holders) has `re_use[sp] == 0`
for those units.  Round-9 part 1 fixed the formula; this verifies
the actual daemon code agrees.

Open output: `verify/t16/README.md`.

## 3. sglang implementation work

Concrete sglang patches needed to align the code with the design.
Order roughly by dependency.

### Schema-side

1. **T17 — State-dump schema upgrade** (DESIGN §5):
   - `pool_usage.<tier>.subpools` dict — replaces the current flat
     pool_usage.HBM fields and the current SWA optional fields
     (the upgrade *replaces*, does not add-alongside; the daemon
     halts loudly if the new schema isn't present)
   - `pool_usage.<tier>.subpools[sp].page_bytes` per subpool
   - `per_program_usage[p].hbm.committed` / `inflight` as
     per-subpool dicts
   - `per_program_usage[p].state` and `pre_pause_state`
     authoritative (round-6 H2)
   - `per_program_usage[p].unit_hashes` materialised list
   - `units[i].residence: list[Tier]` (round-9 part 4b)
   - `units[i].n_bytes: {tier: {subpool: bytes}}` nested
   - `link_stats` per direction: peak_bw_bps,
     recent_throughput_bps, time_since_last_sample_s
     (no samples_in_window — the daemon needs the idle gap, not
     a sample count, per round-14 F2)
   - `tier_holding_cost` per-(tier, subpool)
   - `throughput_ema.prefill_bps`, `throughput_ema.decode_per_program[<pid>]`
   - Drop the current top-level `page_size` field
   - Drop the current SWA `swa_*` optional fields (subsumed by
     subpools dict)

2. **T18 — State-dump internal consistency** (DESIGN §10 R1):
   - single-snapshot under one read-lock so `units[*]` and
     `per_program_usage[*].unit_hashes` and `pool_usage` refer to
     the same logical timestamp

3. **T19 — Atomic unit visibility** (DESIGN §10 D4):
   - units appear in `/aginfer/state.units` only after
     page-aligned chunk commit; partial-prefill chunks not exposed

### Endpoints

4. **T20 — `POST /aginfer/migrate` payload** (DESIGN §6):
   - accept `{add_tiers, remove_tiers}` per action
   - implement residence-set transitions
   - add `action_id` opaque correlator
   - skip-reason classes extended with `write_through_declined:*`

5. **T21 — `PUT /aginfer/program_paused`** (DESIGN §6 round-6 H2):
   - new endpoint
   - writes `state` and `pre_pause_state` into `per_program_usage`

6. **T22 — `GET /aginfer/thresholds` + `PUT /aginfer/thresholds`**
   (DESIGN §6 round-6 H3):
   - bootstrap fetch returns canonical thresholds
   - `PUT` from daemon → sglang updates its in-memory thresholds
     atomically (no local cache file — round-14 dropped the cache;
     sglang halts at bootstrap if the daemon is unreachable)
   - sglang halts on first-launch-without-cache-or-daemon

7. **T23 — `APPLY_FAILED` webhook** (DESIGN §4 round-9 B4):
   - on action apply failure, sglang fires this webhook back to
     daemon with `{endpoint, action_id, reason}`

8. **T24 — `HASH_COLLISION` webhook + detection in `apply_aginfer_migrations`**
   (DESIGN §4 + §10 round-15/16/17):
   - `apply_aginfer_migrations` already builds a `{hash → node}` map
     via a single DFS over the radix tree (`unified_radix_cache.py`
     around line 2280) — extend it to detect collision at that point
   - on collision: fire the `HASH_COLLISION` webhook with payload
     `{hash, node_a_summary, node_b_summary}`
   - daemon `fatal()`s on receipt — deployment-bug class
   - cost is one extra comparison per node in an already-O(N) DFS;
     hash computation (`compute_node_hash_values`) is already done
     lazily upstream when KV-event emission or migrate-action
     processing requires it

9. **T25 — All action endpoints idempotent** (DESIGN §10 R2):
   - re-applying the same action returns 200 with `applied=0`
   - migrate, pause/resume, hint PUT, threshold PUT

### Instrumentation hooks

10. **T26 — HiCache + Mooncake throughput EMA** (DESIGN §7 bw_free, M4a):
   - HBM↔DRAM: CUDA-event bracket each `write_backup` / `load_back`
     pair; maintain EMA per direction
   - DRAM↔DISK: wall-clock bracket each Mooncake `put` / `get`;
     maintain EMA per direction
   - Expose via `state.link_stats`
   - When this lands, T13 ground-truth-vs-EMA comparison + monotonicity
     pins (deferred from T13) must be wired into `verify/t13/` (or a
     follow-on stage in `verify/t26/`).

11. **T27 — Hint clear ordering** (DESIGN §10 R3):
    - scorer's heap-iteration read happens-before eviction commit
      happens-before hint clear

12. **T28 — `should_write_through(node)` plugin point** (DESIGN §3
    superset framing):
    - factor out the `hit_count >= write_through_threshold` check
      into a pluggable hook
    - default implementation preserves current behaviour
    - aginfer registers a V_u-aware version when daemon is attached

13. **T29 — `SGLANG_KV_POLICY_MODULE` eviction scorer plugin**
    (DESIGN §3 superset framing — partially exists):
    - default module: LRU-equivalent V_u (last_access as p_hat
      surrogate) so baseline runs match historical behaviour
    - aginfer registers its hint-table-aware V_u

14. **T30 — Proxy gate disconnect awareness** (DESIGN §10 F1):
    - daemon's proxy gate awaits BOTH gate condition AND
      `request.is_disconnected()`
    - on disconnect: release gate, respond 499, transition program
      to ENDED via outbound queue

15. **T31 — `harbor /aginfer/session_end` endpoint** (DESIGN §4):
    - out-of-band signal channel the client uses to declare
      "session done"
    - lives outside sglang, on the harbor / agent client side

### Multi-rank correctness

16. **T32 — All-rank atomic actions** (DESIGN §6 multi-rank): every
    `migrate` / `program_paused` / `hints` action is applied
    atomically across TP/EP ranks via the existing
    tokenizer-server fanout

## 4. Daemon implementation work

Concrete daemon patches needed.  Most of the core decision rule is
already implemented in some form; the big work is the residence-set
generalization.

1. **T33 — Residence-set candidate generator** (DESIGN §7):
   - `migrate_candidates(state, D_t)` emits `(add_tiers,
     remove_tiers)` tuples
   - the 6 meaningful transitions per unit per §7

2. **T34 — Multi-axis sparse DP** (DESIGN §9):
   - `knapsack_min_cost_multi` with per-axis `bucket_size`
   - `knapsack_max_value_multi` for headroom phase
   - dict-keyed by `(tier, subpool)` tuple
   - infeasibility (no subset hits every relief target under
     destination caps) calls `fatal()` with a forensic dump per
     §10; this is an algorithm bug — DROP + Pause are always
     feasible so reaching here means top-k under-sized or a
     filter dropped a candidate it shouldn't have

3. **T35 — `authoritative_tier(residence)`** (DESIGN §7):
   - HBM if present else DRAM else DISK

4. **T36 — Outbound action queue + worker** (DESIGN §6 B4):
   - in-memory `asyncio.Queue[Action]`
   - dedicated worker task drains via `httpx.AsyncClient`
   - handler enqueues and returns immediately
   - `action_id` UUID assigned at enqueue time

5. **T37 — `APPLY_FAILED` event handler** (DESIGN §4 B4):
   - daemon registers a handler for the new event kind
   - default behaviour: log + let next `joint_decide` re-evaluate
   - failure-class metrics emitted

6. **T38 — Default-policy module** (DESIGN §3 superset framing):
   - LRU-equivalent V_u (uses `last_access_time` as p_hat
     surrogate, `hit_count`-aware)
   - matches sglang's historical scorer exactly when daemon is
     absent or hints are uninitialized

   **Status (#169 + #176 audit closure — 2026-06-01)**:
   * **DONE** in `verify/t38/`: callable
     `baselines.sglang_adapter:default_policy_score(node, layer)`
     = `last_access_time + hit_count × 2^-50`; pluggable via
     `SGLANG_KV_POLICY_MODULE`; 9 verify stages (A0/A1 shape, B0–
     B4 ordering invariants incl. unbounded `hit_count`, C0/C1
     plugin resolution).
   * **DEFERRED — follow-on tasks**:
     * **#177 — Wire `default_policy_score` as the in-process
       default** in `unified_radix_cache.py:_load_eviction_scorer`
       (currently falls back to `_default_eviction_score`).  This
       realises DESIGN §3's "one code path" claim.  Needs an
       ablation regression check vs stock sglang to confirm no
       behavioral change on the no-tied-age path.
     * **#178 — `should_write_through(node)` plugin point**.
       DESIGN §3 names this as the OTHER half of the default-
       policy module.  Mirrors the eviction-scorer plugin point
       but for write-through decisions.

7. **T39 — F1 proxy-gate disconnect handler** (DESIGN §10 F1):
   - `program_tracker.client_disconnected(p)` API
   - enqueue `PUT /aginfer/program_paused {END}` on disconnect

8. **T40 — F2 hint emitter** (DESIGN §10 F2):
   - re-score `D_t` units per event, push all to sglang
     unconditionally
   - **no shadow `{hash: last_pushed_value}` map**

9. **T41 — F5 SESSION_END-for-PAUSED handler** (DESIGN §11 F5):
   - on SESSION_END for PAUSED program: release gate with HTTP
     499, transition ENDED, enqueue PUT

10. **T42 — Observability logging** (DESIGN §10 F3):
    - state-fetch latency p50 / p95 / p99
    - event-queue depth at handler entry
    - time-in-queue per event
    - failure-class counters for APPLY_FAILED breakdown

11. **T43 — `fatal(reason, **context)` helper** (DESIGN §10):
    - shared entry point for every deployment-bug-class halt
    - serialises `(event, state, candidates, dp inputs, reason,
      traceback)` to `<daemon-data>/forensic/<reason>_<ts>.json`
    - logs fatal-level line pointing at the file path
    - `sys.exit(1)`; supervisor decides restart policy
    - call sites: `joint_decide` infeasibility; `bytes_at` τ-not-in-
      residence assertion; missing `link_stats` / `tier_holding_cost`
      / `throughput_ema` fields; cross-rank subpool-key disagreement;
      `peak_bw_bps ≤ 0`; daemon-attached mode losing daemon mid-run

## 5. Scenario-specific verification

For each `DESIGN §12` scenario, a small verification run that
confirms the named subpools actually appear in `/aginfer/state`
and that the decision rule reacts as the scenario claims.

### Methodology (applies to every scenario)

- **N ≥ 3 cycles per arm**, B/O alternation order across cycles
  (Ours/Baseline/Ours/Baseline/...) to neutralise GPU-thermal /
  docker-cache / time-of-day drift.
- **Each cycle from clean slate**: `pkill sglang daemon
  mooncake_master` → drain zombies → fresh start.
- **Fixed knobs** within a matrix: `--ak temperature=0.0` (greedy
  decoding), `--random-seed 42` on sglang, identical
  `harbor -l N -n N -k 1`, identical `sglang HEAD` (verify
  `git rev-parse HEAD` matches at every cycle start).
- **Headline metric**: per-trial wall (`finished_at − started_at`
  from `harbor_jobs/<run_id>/instance_*/result.json`).  Aggregate:
  mean ± std per cycle, then across-cycle mean ± std per arm.
- **Fine metric**: per-turn TTFT from sglang `Prefill batch` log
  line (prompt_tokens, cached_tokens, ttft_ms).
- **Acceptance**: `ours_mean + ours_std < baseline_mean −
  baseline_std` at N=3 → claim significant; if not, do N=6 before
  abandoning.

### T44 — Verify DESIGN §12 S1 (single-stack attention, current benchmark)

Model: DeepSeek-V4-Flash (MLA).

- 1 HBM subpool (`"attn"`) ✓ already running
- 3-axis §9 DP ✓ baseline by construction
- T12 / T13 calibration runs land on this scenario

Open output: `verify/t44/README.md`.  Specifically verifies the
named-subpool framework **correctly degenerates to a single
subpool** (`pool_usage.HBM.subpools` has exactly one entry, no
hardcoded single-subpool assumption survives in the daemon).
T11 / T12 / T13 run on this scenario but do not check the
degeneracy explicitly — T44 does.

### T45 — Verify DESIGN §12 S2 (SWA-hybrid)

Model: Mistral / Gemma (any sglang-supported SWA hybrid).

- Verify `pool_usage.HBM.subpools` has `{"full", "swa"}`
- Verify `units[i].n_bytes["HBM"].swa` transitions to 0 when the
  unit's tokens age out of the sliding window
- Verify pressure on SWA subpool fires `joint_decide` even when
  full-attention subpool is at < theta_hi

Open output: `verify/t45/README.md`.

### T46 — Verify DESIGN §12 S3 (Mamba+attention hybrid)

Model: Jamba / Zamba (any sglang-supported Mamba+attn hybrid).

- Verify only leaf radix-tree nodes carry `n_bytes["HBM"].mamba > 0`
- Verify Mamba snapshot Migrate transitions (DRAM archive, load_back)
  follow the same residence-set semantics as attention units
- Verify `forecast_inflight_demand["mamba"] == 0` between snapshot
  boundaries

Open output: `verify/t46/README.md`.

### T47 — Verify DESIGN §12 S5 (speculative decoding)

Model: Medusa / Eagle on any base model.

- Verify `pool_usage.HBM.subpools` adds `"draft"`
- Verify draft KV does NOT appear in `state.units`
- Verify `per_program_usage[p].hbm.inflight["draft"]` tracks draft
  buffer occupancy per program
- Verify `pause_relief[draft]` for a spec-decoding program is the
  full discarded draft buffer

Open output: `verify/t47/README.md`.

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
- **Hint cross-rank divergence**: triggered if T15 ever observes
  per-rank evicted-hash divergence under TP > 1
- **Linear `h_(τ, sp)(occ)` placeholder**: replaced by whatever
  T12 calibration picks; if T12 finds linear *is* the best fit,
  the placeholder graduates from "Planned" to spec
- **Multi-LoRA scenario**: triggered when the workload mix
  includes per-request LoRA adapter activations occupying material
  HBM bytes
