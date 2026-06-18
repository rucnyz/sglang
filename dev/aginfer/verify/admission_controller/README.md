# admission_controller — DESIGN §8 program-level candidate generator

Admission is **not** an event-driven pause/resume loop anymore.  #194
(DESIGN §9) replaced the sequential "run kv_scheduler, then run
admission" decompose with ONE `joint_decide` over the union action
space.  Admission is now the **program-level candidate generator** that
`joint_decide` consumes, alongside §7's unit-level Migrate candidates.

The old `AdmissionController` class (FIFO of paused programs,
`_on_pressure` / `_on_resolved` loops, `attach_admission_controller`
composite) was **removed** in #194 — its job is subsumed by:

* the §8 generator functions in `daemon/admission_controller.py`
  (`pause_candidates`, `resume_candidates`, `forecast`,
  `marginal_pause_cost`, `pause_relief`, `capacity_fits`,
  `shared_aware_prog_scores`);
* `joint_decide` (`daemon/joint_decide.py`) — phase selection + DP;
* `KvScheduler._dispatch_plan` — executes the chosen Pauses/Resumes
  (`tracker.pause`/`resume` + `PUT /aginfer/program_paused`).

Thresholds (`theta_hi`/`theta_lo`) live on the EventRouter (the single
source of truth, T22/§10); the paused set is read from sglang's
`per_program_usage` state each event — no daemon FIFO, so a restart
loses no admission bookkeeping.

## What this verify pins (post-#194)

Pure functions over a post-T17 `SchedulerState`; no server, no event
loop (the old Layer-A uvicorn harness is gone with the loop it tested):

```
scoring       shared_aware_prog_scores — V_u split across holders
forecast      per-HBM-subpool used_bytes (+ inflight term, 0 under the
              T26/T11 placeholders); forecast_horizon → heartbeat_s
pause-cost    marginal_pause_cost (0 while prefill_bps=0) + pause_relief
              (shared-aware committed only; #205)
pause-cands   one Pause per REASONING/ACTING program (PAUSED+ENDED skipped)
resume-cands  one Resume per PAUSED program; capacity_fits gates overflow
trajectory    #199: the §8 forecast trajectory term — assembled from
              decode_bytes_per_token (now exposed) × synthetic
              decode_throughput × E[remaining]; proves the full product
              + the per-input T26/T11/Mamba gating (0 when any absent)
disjoint      holistic-review #2: a unit's committed (radix) bytes that
              are in THIS event's D_t are excluded from pause_relief —
              so Migrate(u) and Pause(p∋u) free disjoint HBM and the §9
              DP can't double-count (under MEMORY_PRESSURE u∈D_t →
              excluded; under LLM_PREFILL D_t=∅ → committed kept)
overlap       #205: sglang's UNIFIED cache reports a running req's KV in
              BOTH committed (shared-aware bytes//n_holders) AND inflight
              (undivided) — pause_relief uses committed ONLY.  Pins the
              two audit-gap regimes: B5 shared-prefix (12MB/3 holders →
              4MB relief, not 12MB) and A3 disjoint same-subpool
```

The §9 phase selection + live mixed-plan dispatch are pinned by
`verify/joint_decide/` (5 stages) and `verify/integration_stress/`
(flavors A/D/G on the real B300 stack).

## Honest degradation (gated on T26 / T11 — #199)

`forecast_inflight_demand` and `pause_relief`'s `future_inflight_savings`
term are 0 until `decode_throughput` (T26), `E[remaining_tokens]`
(T11/#126), and `bytes_per_token_in_subpool` (architecture constant)
are wired.  So `forecast[sp] == used_bytes[sp]` and `pause_relief =
shared-aware committed` (snapshot only; #205) today — the §9 triggers
reduce exactly to allocator-truth HBM occupancy (behaviour-preserving),
and become trajectory-aware once those inputs are measured.

## Historical note (G1 / G9)

* **G1** (number of pauses per real harbor cycle never measured) — the
  pause decision now lives in `joint_decide`; `_dispatch_plan` emits an
  `admission_pause` metric per dispatched Pause.
* **G9** (theta mismatch between sglang's webhook fire and the daemon's
  pause threshold) — closed by T22 (`GET`/`PUT /aginfer/thresholds`
  broadcast from one source); both sides read the router's thresholds.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/admission_controller/verify.py
```

## RESULTS

**PASSED** — all 8 §8-generator stages (scoring, forecast, pause-cost,
pause-cands, resume-cands, trajectory, disjoint, overlap).

* date: 2026-06-04 (#194 rewrite — was stale pre-T33)
* date: 2026-06-05 (#205 audit closure) — `pause_relief` is now
  shared-aware `committed` ONLY (was `committed + max(0, inflight −
  committed)`).  The audit found that mixing the undivided `inflight`
  with the per-holder `committed` over-counts shared prefixes (B5: a
  12MB/3-holder prefix credited the whole 12MB to a single pause) and
  under-counts disjoint same-subpool bytes (A3).  `stage_overlap` now
  pins both regimes; `inflight` is read only by `marginal_pause_cost`.
