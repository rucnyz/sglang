# T9 — Performance gaps & untested diagnostics (after N=3 matrix)

Companion to `N3_matrix_SUMMARY.md` + `N3_ROOT_CAUSE.md`.  Catalogs
**what was promised**, **what we actually measured**, and
**what we never measured**, so we don't accidentally claim the
gaps closed when they haven't been.

## 1. Measured-and-failed expectations

Each row: paper / DESIGN.md promised → N=3 matrix measured →
delta status.

### 1.1 kv_scheduler V_u beats LRU

* **Promised** (`DESIGN.md` Done section, historical Run H' vs F'):
  paper §7 value-based eviction reduces p99 −28% and stdev −19 %
  vs LRU at the same backend config.
* **Measured (N=3, same-session)**:
  * ours_full (kv_scheduler+admission ON, ours_greedy inline): 1344.0 ± 54.6 s
  * baseline kv_off (ours_greedy inline only, daemon scheduling OFF): 1389.3 ± 39.7 s
  * Δ = −45.3 s, SE ≈ 39.0 s, z = −1.16 → **not statistically
    significant**.
  * Per-cycle stdev (within-trial spread) similar across configs
    (838–923 s); no p99 improvement signal in N=3.
* **Verdict**: the V_u-vs-LRU mean/p99/stdev advantage that Run H'
  showed under default sampling has **not been re-demonstrated**
  under the current `temperature=0.0 seed=42` settings.
* **Compounding**: we don't even have a fresh LRU arm in this
  matrix — 4-arm matrix would give it (cycles 1, 3, 5 in the
  4-arm run = LRU; cycles 2, 4, 6 = ThunderAgent).  Only 3 / 6
  cycles done (paused for shared-GPU contention).

### 1.2 T11a daemon-side `program-alive` rule

* **Promised** (`t11/README.md` original, now patched): conditional
  `p_hat = 1.0` for units whose program is alive (replaces
  `hits/age` proxy) should beat the proxy on multi-turn agent
  workloads.
* **Measured (N=3 matrix, daemon-side T11a applied; commit
  `888ea822`)**: same as 1.1 — Δ = −45.3 s, not significant.
* **Verdict**: rule doesn't measurably help.  Most likely because
  the workload's prefix-reuse is already saturated (see §3.1) so
  there's nothing left to gain from a better p_hat.
* **Out-of-scope inference**: this does NOT disprove the rule
  *theoretically*; it just shows that on this workload at this
  hit-rate ceiling, the marginal value is below noise.

### 1.3 admission_controller pause/resume

* **Promised** (`DESIGN.md` §"admission_controller"): under HBM
  pressure, pause programs in BFD-bin-packing order to free
  capacity; resume FIFO when occ < theta_lo.  Expected to add
  value vs no admission control under high concurrency.
* **Measured (N=3 matrix)**:
  * K-full (admission ON) vs K-a-equivalent (admission OFF)
    comparison: the matrix `baseline = kv_off` had BOTH off,
    `ours = full` had BOTH on.  We don't have a clean
    admission-only-OFF arm in N=3.
  * Single-shot K-full 1559 s vs K-a 1549 s (Δ = +10 s) is
    inside noise.
* **Verdict**: **admission's marginal contribution is not isolated
  in N=3**.  Even if it had a real effect, we couldn't see it
  because we collapsed kv_scheduler + admission into one config.

## 2. Promised-and-NEVER-measured (instrumentation gaps)

These are claims in DESIGN.md / paper drafts that don't have any
N=3 evidence one way or the other.  Listed in priority order for
follow-up.

| # | Promise | What's missing | How to measure |
|---|---|---|---|
| **G1** | `admission_controller` pauses programs when occ ≥ theta_hi (0.85) | **# of pauses per cycle** — no instrumentation in daemon log | daemon already logs `program_tracker: paused %s`; add a per-cycle aggregate to parse_matrix.py |
| **G2** | sglang fires `memory_pressure` events when occ ≥ theta_hi (0.7) | **# of pressure events per cycle** — and what occ was at fire time | sglang heartbeat log line; parse and tally |
| **G3** | kv_scheduler issues `migrate` POSTs in response to events | **# of migrate actions per cycle** + their accept/reject mix at sglang | daemon log already has these; just need parser |
| **G4** | Per-tier hit rate composition (HBM vs DRAM vs DISK) | **No per-tier stats** — we only have aggregate cache hit ratio (~95.5 %) | sglang `Prefill batch` log line already has #cached-token but not per-tier; would need a sglang patch to bucket by tier |
| **G5** | HBM occupancy trajectory over a cycle | **No occ time series** — we don't know if HBM ever crossed theta_hi | parse sglang `Prefill batch` log line `full token usage:` field as time series |
| **G6** | shared-aware aggregation protects system-prompt prefix | **No measurement** — hit rate is symmetric across configs so we can't tell if `shared_aware_prog_scores` actually changed behavior | parse migration trace + cross-ref with system-prompt unit hash |
| **G7** | Inline-side T11a (`sglang_adapter.py:_node_to_unit` swap) helps when daemon proxy is bypassed | **Never implemented** — deliberately deferred because daemon-side N=3 showed no signal; would need a separate matrix |
| **G8** | Run J — daemon without HiCache (paper §9 deployment claim) | **Never run** — was scoped in original T9 plan, dropped during noise discovery | run_J variant of run_matrix.sh + N=3 cycles |
| **G9** | sglang webhook and daemon admission thresholds are consistent | **Implementation mismatch unverified**: sglang fires `memory_pressure` at occ ≥ `theta_hi=0.7` (and `theta_crit=0.9`); daemon's `admission_controller` only pauses at occ ≥ `theta_hi=0.85`.  Events in occ ∈ [0.7, 0.85) → daemon fetches `/aginfer/state` but takes no action ⇒ wasted RTT.  Noted during single-shot K-full inspection (2026-05-26).  Whether this hurts wall time or just adds harmless RTT was never measured. | (a) align both to one theta in launch script; OR (b) suppress sglang fire when daemon's own theta would not act.  Measure via G2 + per-event "no-op" count. |

## 3. Why the expected wins probably don't exist on this workload

These are conjectures explaining the gap.  None of them are
proven in N=3 data; they would each need a dedicated probe.

### 3.1 Prefix-reuse ceiling (~ 95.5 %)

The cache hit ratio across all configs is ≈ 0.955.  This is
inline `ours_greedy_score` alone, with no daemon scheduling.  If
the inline scorer already keeps the right things in HBM, the
daemon's V_u-based migration has very little to add.

Practical implication: any scheduler improvement is bounded above
by (1 − 0.955) = 4.5 % of LLM-bound time as the maximum possible
prefill saving.  Even a perfect scheduler at this hit rate could
only save a few percent — well below the 80 % runaway tail.

### 3.2 Workload not in the pressure regime

Mooncake L3 contributes ~200 GB DRAM per TP rank; with TP=2 that's
~400 GB.  Plus HiCache's host pool (1.5 × device pool).  Total host-
side capacity dwarfs the device working set.

If HBM never crosses theta_hi (admission's pause threshold), the
admission_controller is by definition a no-op.  We have not
verified whether HBM crosses theta_hi or not (G5 above).  If it
doesn't, admission's mechanism is completely untested by this
workload regardless of the matrix outcome.

### 3.3 Runaway dominance (already in `N3_ROOT_CAUSE.md`)

1 % of LLM requests generate 60 k completion tokens → 80 % of
e2e_latency.  These are pure decode workload, holding their own
KV cache — no migration / eviction / admission decision applies
to them.  So 80 % of trial wall is structurally untouched by the
daemon's three layers.

The remaining 20 % is where scheduler quality could matter, but
within that 20 % the prefix-reuse ceiling (§3.1) caps it further.

Net: under current settings, the daemon's theoretical maximum
improvement is bounded above by 20 % × 4.5 % ≈ **0.9 % of trial
wall**, which is well below the N=3 SEM (≈ 39 s on a 1389 s
mean = 2.8 %).  We're below our own detection threshold.

## 4. What "closing the gap" would mean

To make the daemon's contribution visible, both of the following
need to happen:

1. **Suppress runaway generation** so trial time isn't 80 % decode-
   bound.  Cheapest: cap `max_completion_tokens` at ~ 4 k via
   harbor `--ak`.
2. **Push workload into HBM-pressure regime** so admission and
   migration actually fire.  Either:
   * shrink `MAX_TOTAL_TOKENS` (current `--max-total-tokens`
     unset = ~ 10 M default), or
   * disable HiCache (Run J ablation) so eviction goes to DROP
     instead of DRAM-spill — forces tier-aware decisions.

Both have to be done in **the same matrix** with N ≥ 3 and proper
B / O alternation, otherwise we'll be back in the setting-drift
trap.

## 5. Open instrumentation work (sub-tasks for paper finalization)

Each of these is < 1 day work, mostly log parsing:

* **G1+G2+G3 → unified daemon_events_summary.py**: per cycle,
  count pause/resume, pressure/resolved, migrate POSTs (issued vs
  accepted at sglang), per `EventKind`.  Add to `parse_matrix.py`.
* **G4 → sglang per-tier hit-rate patch**: small sglang patch to
  bucket `#cached-token` by tier of origin.  Behind an env var so
  it's no-op in the default config.
* **G5 → HBM-occ time series**: trivial parse of existing
  `Prefill batch` log lines.  Plot `full token usage` over time
  per cycle.
* **G6 → migration-target audit**: scan daemon migrate POST bodies
  for system-prompt prefix unit hash; verify it's never demoted /
  dropped.
* Workload-pressure runs (§4 above) — orthogonal to instrumentation
  work; would consume another ~5 h of GPU time per matrix.

## 6. Pointer to other docs

* `N3_matrix_SUMMARY.md` — what we measured (the headline numbers)
* `N3_ROOT_CAUSE.md` — why historical Run H' 885 s isn't reachable
* `N3_ttft_analysis.md` — per-request distribution (incl. cache
  hit ratio)
* `methodology.md` — N=3 protocol (settings, sample-size math)
* `RUN_K_*_SUMMARY.md` — SUPERSEDED N=1 single-shot files
* `verify/t11/README.md` — why T11a inline-side was NOT done
