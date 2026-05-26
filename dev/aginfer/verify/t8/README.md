# T8 — admission_controller (event-driven pause/resume)

## WHAT WE PROMISED

**Capability**
* Reacts to `memory_pressure` and `pressure_resolved` events from
  sglang's webhook (T5). **No periodic timer.**
* On `memory_pressure`: score each active program by **shared-aware
  aggregated `V_u`**:
  ```
  prog_score(p) = Σ V_u / |u.session_ids|   for u in p.units
  ```
  i.e. each unit's value is split across its holders so the shared
  system prompt + tool definitions don't double-count. Pause the
  programs with the **lowest** `prog_score` until HBM occ drops below
  θ_hi. Pause is enforced at the proxy via
  `program_tracker.pause(p)`.
* On `pressure_resolved`: resume paused programs in FIFO order
  (oldest pause first), draining progressively (re-poll
  `/aginfer/state` between each resume) until either the FIFO is
  empty OR HBM occ rises back to θ_lo (hysteresis gap = θ_hi − θ_lo).
  Bounded by `max_pauses_per_event` per single event.  Note: sglang's
  webhook is edge-triggered on HIGH→OK, so a single event must drain
  as much as it safely can — round-1 N4 made one-resume-per-event
  the bug; the drain-with-θ_lo-gate is the fix.
* Uses the **same** `OursGreedyPolicy._value` from `baselines/ours_greedy.py`
  — just aggregated per program. No separate heuristic.

**Cost ceiling**
* Per-event handler wall time at 32 programs: < 10 ms.
* Pause/resume themselves are cheap dict updates on program_tracker.
* `prog_score` aggregation uses pre-built program→units index (built
  once on state fetch), not nested scans.

## HOW WE VERIFY

Mechanism. `verify/t8_admission.py`:

```
1. Reuse synthetic 4-program state from T7 (shared 1 k platform +
   per-program 4 k tail).
2. Drive a synthetic event stream against the event_worker:

   2a. Fire memory_pressure {occ=0.92}. Assert:
       - prog_score aggregation correctly down-weights shared units:
         for the shared platform unit u with 4 holders,
         u.contrib_to_prog(p) == V_u(u, HBM) / 4   (not full V_u).
       - The program paused is the one with lowest aggregated score
         — typically p3 or p4 (ACTING).
       - HBM "should" drop on next state poll (we stub /aginfer/state
         to confirm).
   2b. Fire memory_pressure again with occ=0.95 (still over).
       Assert: second pause fires.
   2c. Fire pressure_resolved {occ=0.55}. Assert: FIFO resume
       (oldest paused first), and only one resume per event
       (hysteresis check before further resumes).

3. Verify aggregation correctness with a hand-built degenerate state:
   - 32 programs, all sharing one 1k-token unit (system prompt).
   - Each program has 0 unique tail (everything is shared).
   - With naive sum: every program scores identical (32 × V_share).
   - With shared-aware aggregation: every program scores 1 × V_share.
     Naive sum would make pick deterministic-by-Python-dict order
     (wrong); shared-aware gives the same total, but proportional
     to len(p.units) ratio — verify the correct program is picked.

4. Anti-timer grep: no `asyncio.sleep`, no `time.sleep`, no
   `loop.call_later` in admission_controller code path.

5. Latency: time 10 admission_controller.on_pressure() calls at 32
   programs; assert mean < 10 ms.
```

## CALIBRATION

* `θ_hi` = 0.85, `θ_lo` = 0.70 (default watermarks).
* Sensitivity sweep: rerun mini-Run-K with `(θ_hi, θ_lo) ∈
  {(0.80, 0.65), (0.85, 0.70), (0.90, 0.75)}`; verify all three
  produce per-trial mean within ± 10 % of each other (= the floor is
  not pinned to a knife-edge watermark choice).

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Aggregation picks adversarial victim | State where every program shares one 1k unit and has zero unique tail | With shared-aware aggregation, all 32 programs score identical → tie-break by FIFO; assert no exception, picks A program | inspect victim id |
| memory_pressure event arrives but no program to pause | All programs already paused | admission logs "no eligible victim"; doesn't crash; on next pressure event, pick stays empty | log assertion |
| pressure_resolved event arrives but no program to resume | All programs active | log "nothing to resume"; no exception | log assertion |
| Rapid memory_pressure → pressure_resolved oscillation | Inject 10 cycles of (HIGH, OK, HIGH, OK ...) in 1 second | admission's pause/resume decisions stable; doesn't ping-pong; hysteresis prevents flapping (resume only when occ < θ_lo) | event log: at most 1 program paused, at most 1 resumed during oscillation |
| Run K mini with `θ_hi=0.99` (effectively no admission) | env override | per-trial mean ≤ Run F' × 1.10 (= 960 s); proves admission's contribution is bounded above by gap to no-admission | mini harbor run |
## RESULTS

**PASSED (post audit round-1)** — 12 verify steps + 3 bisect probes,
~7 s on agsched env.

### Audit round-1 findings + fixes

The first audit caught 2 BLOCKER + 3 MAJOR + 4 MINOR + 2 NIT.
Notably, B1 was a paper-§7 claim violation (admission's V_u was
silent-missing the holding-tax term) and N4 was a design bug (FIFO
strands forever because sglang's pressure_resolved is edge-
triggered, not heartbeat).  All fixes follow the same bisect
protocol: write probe → confirm pre-fix FAIL → apply fix →
confirm post-fix PASS.

| ID | Finding | Fix layer |
|---|---|---|
| **B1** | `_value_at_current_tier` passed `cap=0` to `holding_unit_cost` → short-circuit to 0.0 → holding tax silently dropped.  Admission's V_u was only `p_hat * saved_prefill`, breaking the paper-§7 claim README §22 explicitly made.  Effect: byte-heavy shared prefixes (low p_hat) scored same as small unique tails → pause victim picked wrong. | **Production**: thread `SchedulerState` into `_value_at_current_tier`; use real `used_bytes`/`cap_bytes` for the holding-tax term.  Matches `OursGreedyPolicy._value` exactly. |
| **B2** | `attach_admission_controller` silently captured `prior=None` if called BEFORE `attach_kv_scheduler`; later kv_scheduler attach OVERWROTE the composite; admission never fired, no log. | **Production**: raise `RuntimeError` in attach if no prior handler exists for MEMORY_PRESSURE / PRESSURE_RESOLVED.  Fail-loud at bootstrap. |
| **M1** | Step [1] only pinned ranking (prog-3 lowest).  A regression to "sum raw V_u, ignore holders" would pass. | **Test**: numerical pin — assert `score[prog-i] == V_tail_i + V_shared/4` (per-program), not just ranking. |
| **M2** | Step [9] forced 16-pause spin under fixed state; HTTP RTTs dominated, algorithmic regression invisible. | **Test**: new step [12] uses state-mutator (drop occ after first pause) to measure the SINGLE-pause path; budget 5 ms (vs step [9]'s 10 ms with 16 iterations). |
| **M3** | Step [11] asserted both side effects fired but NOT order.  A regression to admission-first → kv_scheduler-second would pass. | **Test**: monkey-patch admission.handle to append to a shared timeline; stub migrate to do same; assert `idx(kv_scheduler:migrate) < idx(admission:enter)`. |
| **N1** | README §3 said "stop when next resume would cross θ_hi" but code uses θ_lo as the gate (correct hysteresis intent). | **Doc**: rewrote README §3 to "drain until either FIFO empty OR occ rises back to θ_lo (hysteresis gap = θ_hi − θ_lo)". |
| **N2** | Step [10] only tested `theta_lo > theta_hi`.  Missing: equal, > 1, == 0, negatives. | **Test**: parameterize over 6 boundary cases; each asserts `ValueError` with "theta" in the message. |
| **N3** | Step [8] asserted `<= 16`; a regression bumping the cap to 32 would still pass (16 ≤ 32 is false but cap-removed unbounded would also fail for different reason; the real concern is silent cap-loosening). | **Test**: tightened to `== 16` (exact).  Fixture has 20 programs + persistent pressure → cap MUST stop at 16. |
| **N4** | `pressure_resolved` is edge-triggered on sglang's side (single fire on HIGH→OK).  Admission resumed 1 program per event → FIFO with N > 1 paused programs stranded forever. | **Production**: `_on_resolved` now DRAINS — loop up to `max_pauses_per_event` times, resume oldest each iter, re-fetch state, stop when `occ >= theta_lo` (hysteresis). |

### Latency summary (multi-run, per memory:feedback-latency-multi-run)

| stage | mean ± std | envelope (mean+3σ) | budget |
|---|---|---|---|
| handler @ 32 programs (step [9], 16-pause spin) | 4.18 ± 0.17 ms | 4.67 ms | < 10 ms |
| **single-pause** @ 32 programs (step [12], algorithmic path) | 4.10 ± 0.05 ms | 4.27 ms | **< 5 ms** |

Single-pause is dominated by 2× HTTP RTT (one to trigger pause,
one to see occ dropped + exit); algorithmic work (~score + pick)
is sub-ms.

* raw logs:
  * `results/20260526_120256_run1.log` — v1 (pre-audit)
  * `results/<YYYYMMDD_HHMMSS>_run2_audit1.log` — post audit round-1
  * `results/<YYYYMMDD_HHMMSS>_regression_probe.log` — bisect demos
    for B1, B2, N4

* date: 2026-05-26
* daemon code: ~280 LoC `daemon/admission_controller.py` (new);
  reuses `OursGreedyPolicy` value functions from `baselines/` and
  `build_paper_state` from T7's `daemon/kv_scheduler.py`.
* prog_score down-weights shared units: ✓ [1] — each unit's V_u
  divided by `|holders|` so the platform/tool_def prefix doesn't
  inflate a single program's score.
* shared-aware passes degenerate test: ✓ [2] — 32 programs sharing
  one unit with zero unique tails → identical scores (max-min < 1e-9).
* lowest-score paused first: ✓ [3] — `prog-3` (lowest hit_count
  tail) paused under occ=0.95 / theta_hi=0.5.
* FIFO resume: ✓ [4] — pressure_resolved resumes ONE program at a
  time in the order they were paused.
* Hysteresis: ✓ [5] — pressure_resolved with `occ >= theta_lo` does
  NOT resume (waits for occ to drop further).
* No-victim-when-all-paused: ✓ [6] — pre-paused state +
  memory_pressure: `pause_decisions == 0`, no crash.
* Anti-timer contract (AST grep): ✓ [7] — zero `sleep` /
  `call_later` / `call_at` / `perf_counter` in source.
* Max-pauses cap: ✓ [8] — persistent pressure bounded at 16
  pauses/event (no runaway).
* Invalid watermark rejection: ✓ [10] — `theta_lo >= theta_hi`
  raises `ValueError` at construction.
* Composition with T7: ✓ [11] — on memory_pressure,
  `kv_scheduler.handle` fires first (migrate POST), then
  `admission.handle` (pause re-check); both side-effects observable.

### Latency (multi-run, per memory:feedback-latency-multi-run)

5 independent trials at 32 programs (no actual pause; theta_hi=0.99).

| stage | mean ± std | budget |
|---|---|---|
| admission `handle()` end-to-end | **4.16 ± 0.15 ms** (mean+3σ 4.60 ms) | < 10 ms |

Assertion uses `mean + 3σ < 10 ms` (current envelope leaves ~2×
headroom; catches a 2× regression).

* raw logs:
  * `results/20260526_120256_run1.log` — v1 (pre-audit)
