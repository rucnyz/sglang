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
  (oldest pause first) one at a time, ack each via re-poll of
  `/aginfer/state`, stop when the next resume would cross θ_hi again
  (hysteresis).
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

**PASSED (v1, pre-audit)** — all 11 verify steps, ~5 s on agsched env.

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
