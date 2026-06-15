# aginfer — experiment plan (`EXP_PLAN.md`)

The **experiment** side (the implementation side is `Impl_PLAN.md`). This is the thin
roadmap that ties together: *which* scenarios we win on, *how* we prove the value-aware
scorer, *how* we measure, and *where* the runnable packages live. It **references**
the detailed scenario catalogue in `wherewewin/` rather than duplicating it.

## The claim + the three competitors
**Thesis:** *proactive + value-aware + program-aware* KV scheduling beats *reactive +
recency + program-blind*. The two baselines we measure against:
- **B** = vanilla sglang + HiCache + **LRU** (recency, program-blind). The do-no-harm floor.
- **TA** = **ThunderAgent** router — cache-blind, router-side admission (pause/admit) only;
  never migrates/promotes/evicts.
- **Ours** = event-driven per-unit value `V_u` (tier + holder-count + reuse-prob), 4-tier
  residence-set migration `{HBM,DRAM,DISK,DROP}`, value-gated pause/resume, predictive promote.

Full competitor analysis + measurement discipline: **`wherewewin/README.md`** (the source).

## The scenario set → see `wherewewin/`
`wherewewin/` is the **catalogue of all scenarios** (one driver each; S8 = capstone).
Do not re-derive them here — this is the index + current home/status:

| # | Scenario (driver) | Lever | Canonical package | Status |
|---|---|---|---|---|
| **S1** | tool-call predictability | demote + **predictive promote** | `reproduce/RQ1/scenarios/s1-predictive-promote/` | **DONE** (paper artifact #242); redo on Dynamo #245 |
| **S2** | fleet-shared prefix vs churn | **value eviction** (holder-count) | `dev/dynamo/` (agentreplay) → fold into `reproduce/RQ1/scenarios/s2-*` | **IN PROGRESS** (#243; blocker below) |
| **S3** | drop-on-death (compaction/end) | drop-on-death | `wherewewin/s3-drop-on-death/` | planned |
| **S5** | value-gated pause under overload | admission pause | `wherewewin/s5-overload-pause/` | planned (needs live overload) |
| **S6** | blocking sub-agent | reuses S1 + drop | `wherewewin/s6-blocking-subagent/` | planned |
| **S7** | background fan-out | proactive demote/pause | `wherewewin/s7-background-fanout/` | planned |
| **S8** | comprehensive (all at once) | full joint_decide | `wherewewin/s8-comprehensive/` | capstone, run last |

(Numbering keeps the wherewewin convention; there is no standalone S4.)

## How we PROVE the scorer — the 3-config factorial (the new core)
For the **eviction/migration axis** (the `V_u` scorer), prove usefulness in three settings
of increasing strength. This is a `{router} × {eviction: LRU, ours}` factorial, applied
first to **S2** (holder-count, the cleanest eviction case) and reusable for S1/S3:

- **① default Dynamo path (no ThunderAgent) + ours-scorer vs LRU.** Cleanest isolation of
  the scorer — no admission interference. *Closest to done; ≈ current setup (TA pausing
  disabled = a plain router). Gate: the moderate-pressure regime (see blocker).*
- **② ThunderAgent ON (pausing enabled) + ours-scorer vs LRU.** Proves the scorer is
  **orthogonal / additive** — it helps even under someone else's admission control →
  universality. *Risk: TA pausing can remove the very pressure the scorer needs → may come
  back inconclusive (which is NOT "scorer useless"). Design the regime so pressure survives.*
- **③ our router + ours-scorer (full aginfer) vs the full baseline (TA + LRU).** The
  complete system (value-aware admission + value-aware eviction) — the headline. *Biggest
  lift: our admission (pause/resume) on Dynamo is unit-tested but not yet live-A/B'd (#247).
  If our pause works it may also relieve pressure and dodge the crash below.*

Recommended order: **① → ③ → ②** (prove the scorer clean; then the full stack; then the
trickiest universality claim last).

Honesty: ③ "should be best" is a hypothesis — it only wins if our admission actually adds
value beyond the scorer in this workload; it may merely tie ①.

## Methodology (the short version; full discipline in `wherewewin/README.md`)
- **Token-exact replay** via **agentreplay** (`/generate` + `forced_output_ids`), real CC
  traces, byte-identical prompts across arms → do-no-harm is meaningful. NOT a live agent.
- **Metric ≠ cache-hit** (HiCache makes DRAM-hit == HBM-hit). Measure re-prefill
  (`#new-token`), TTFT, load_back bytes, makespan/goodput, p99.
- **Genuine 4-tier pressure** so DROP actually occurs (where value-aware vs recency diverge).
  Gate first on **baseline B's logs** that DROP/recompute is really happening.
- **N ≥ 3**, report mean ± std; do-no-harm = ours ≤ B in every paired unit.

## OPEN BLOCKER (gates ①/②/③ on Dynamo right now)
The V4-Flash worker **crashes under heavy oversubscription** — `Scheduler watchdog timeout
(300s) → SIGQUIT` (eviction retry-storm at occ≈0.98). Both open-loop and closed-loop replay
trip it. **Plan A (recommended): moderate-pressure regime** (tune churn so occ peaks ≈0.90).
**Plan B: root-cause the evict-storm.** Full diagnosis + the exact Dynamo bring-up + tuning
knobs are in **`dev/dynamo/S2_RESULTS.md`** ("RESUME HERE").

## Execution — where to run
- **Canonical paper packages:** `sglang/reproduce/RQ1/scenarios/sN-*/` (scripts + results;
  S1 lives here today).
- **Current Dynamo working dir (S2):** `dev/dynamo/` — `build_s2_trace.py` (real-data trace),
  `s2_replay_ab.py` (agentreplay orchestrator), `S2_RESULTS.md` (results + RESUME procedure).
  Fold into `reproduce/RQ1/scenarios/s2-*` once it produces a number.
- **Stack startup:** `RUNBOOK.md` (Dynamo bring-up).

## Secondary: per-architecture mechanism coverage (was Impl PLAN §5 T44–T47)
Confirm the state surface + decision rule bind correctly per attention architecture
(DESIGN §12): **S1** single-stack (done, V4-Flash MLA), **S2-arch** SWA-hybrid, **S3-arch**
Mamba+attn, **S5-arch** speculative decode. Low priority vs the win experiments; verify
opportunistically when a matching model is up. (Distinct from the win scenarios S1–S8 above
— this is correctness-across-architectures, not a benchmark.)
