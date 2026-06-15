# RQ1 — Per-scenario performance vs. default HiCache and ThunderAgent

**Research question.** Across distinct agentic-serving workload scenarios, how does
**Ours** (event-driven, value-aware, program-aware 4-tier KV scheduling) compare against
the two strong baselines a production agent stack would otherwise use?

This folder is a **self-contained, paper-facing reproduce package**: scripts, exact
commands, and the measured numbers for each scenario.

## The three arms

| Arm | What it is | Key limitation |
|---|---|---|
| **B** (default) | vanilla sglang + HiCache + LRU | reactive (demotes only under pressure), recency-keyed eviction, on-access best-effort load-back, **program-blind** (manages bytes, not programs) |
| **TA** (ThunderAgent) | router-side admission control (TR mode) in front of sglang | "pause" makes **zero backend calls** — pure router-side withholding of a program's next prefill, keyed to HBM-only `max_total_num_tokens`; **HiCache-unaware**, never migrates / promotes / evicts |
| **Ours** | sglang + the aginfer daemon | per-unit value `V_u` (tier + holder-count + reuse-prob), residence-set migration across `{HBM,DRAM,DISK,DROP}`, value-gated pause/resume, and **predictive (ETA-timed) promote** |

All three arms run **with HiCache on** so the comparison isolates the *scheduling policy*,
not the cache tier itself.

## Scenarios

| ID | Scenario | Distinctive driver | Status |
|---|---|---|---|
| **S1** | [Program-aware KV scheduling across tool gaps](scenarios/s1-predictive-promote/) | a program parks in a tool call → its idle prefix is evicted → it resumes the *same* prefix | ✅ **measured on real CC traces, N=3**: program-aware eviction = **71.6 % vs 55.8 % cache-hit (−42 % re-prefill)** at moderate concurrency — a **goodput** win (NOT the old predictive-promote *latency* claim, which is a microbench only); ≈ LRU at the heavy fleet. TA ≈ B (never promotes). See [README](scenarios/s1-predictive-promote/) + [FLEET_FINDINGS](scenarios/s1-predictive-promote/FLEET_FINDINGS.md) |
| **S2** | [Shared-prefix retention under scratch churn](scenarios/s2-shared-prefix-retention/) | a fleet-shared system prefix LRU ages out by recency under scratch churn | 🔄 **moved to Dynamo + token-exact agentreplay (2026-06-15).** The "dead V4 multi-tier store" blocker was a config artifact (tier is live post-sync); holder-count had 4 stacked bugs, all fixed + the lever now fires (`n_holders=8`). Chat-harness clean re-run showed **ours −23% re-prefill vs LRU, do-no-harm strict (N=3)**; the token-exact agentreplay redo is in progress (open V4-Flash watchdog blocker). **Current state of record: `dev/dynamo/S2_RESULTS.md`** + `dev/aginfer/EXP_PLAN.md`. (The narrow/TIE finding in this folder's README is the SUPERSEDED earlier result.) |
| S3… | (future scenarios) | — | planned |

> RQ1 is structured to grow: each scenario is a self-contained sub-folder with its own
> README + scripts + results. The 3-arm framing (B / TA / Ours) is shared.

## Prerequisites (the system under test)

This package contains the **workload drivers + analysis**; the *system* (sglang build +
the aginfer daemon) is the dependency:

- **Code**: the aginfer fork of sglang with the daemon under `dev/aginfer/`. Point
  `AGINFER_ROOT` at that `dev/aginfer` directory. (Canonical checkout: branch
  `aginfer-synced` at `/scratch/yuzhou/projects/sglang` — the old `sglang-sync` worktree was
  consolidated away.)
- **Python env**: conda env `agsched-rebase` (torch 2.11+cu130, sglang dev, mooncake,
  sgl_kernel). Model: DeepSeek **V4-Flash**, TP=2.
- **Hardware (as measured)**: 8× B300 (275 GB); GPUs **5,6** free. KV ≈ 1.17 KB/token.
- **Tiers**: HBM (the sglang KV pool) + DRAM (HiCache, `--hicache-ratio`) + DISK (mooncake).
  `MAX_TOTAL_TOKENS` sets HBM-pool pressure; the DRAM/DISK sizes control *where an evicted
  prefix lands* (and therefore the win magnitude — see the S1 results).

> ⚠️ **V4-Flash min-pool**: do NOT shrink `MAX_TOTAL_TOKENS` below the model's minimum
> (a single 12K prefill deadlocks at pool=28K). Use the default pool and tune the
> *workload*, not the pool.

## How to reproduce (S1)

```bash
export AGINFER_ROOT=/path/to/sglang/dev/aginfer   # the aginfer code (branch aginfer-synced)
conda activate agsched-rebase
cd scenarios/s1-predictive-promote

bash scripts/stack_up.sh          # mooncake -> daemon -> sglang ; waits for "[s1-stack] READY"
bash scripts/run_live_ab.sh       # HEADLINE: clean live A/B (Ours vs B), N=3
bash scripts/run_controlled.sh    # controlled full-eviction win (the 91% magnitude)
bash scripts/run_microbench.sh    # per-resume B-vs-prestaged microbench
python scripts/link_characterize.py   # offline tier-link bandwidths (no stack needed; CUDA_VISIBLE_DEVICES=5)
```

See [`scenarios/s1-predictive-promote/README.md`](scenarios/s1-predictive-promote/README.md)
for the workload, the claim, the exact knobs, and the measured numbers.

## Headline result (S1, this hardware)

- **Clean live win** (6 concurrent programs, establish→park→resume, N=3): **Ours 1251 ± 71 ms
  vs B 2109 ± 314 ms → 41 % faster TTFT**, every cycle Ours wins; Ours cached ≈ 29881/30000
  (full hit) vs B 0 (recompute) every cycle.
- **Controlled magnitude** (fully-evicted 50K prefix, daemon-driven warm, N=3): **274 ms vs
  3094 ms = 91 % / 2.82 s**.
- **The win is gated on the GPU-idle premise** (real tool-parking → idle GPU the predictive
  promote uses for free). A compute-saturated synthetic regime hides it (documented caveat).
