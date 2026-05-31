# T9 Run K — measurement methodology (post-T11a daemon-side)

Defined 2026-05-26 after the K-full / K-a / kv_off single-shot
diagnostic round.  102 s differences between configs were inside
the within-run SEM (≈ 113 s), so no claim could be made.  This
protocol fixes that.

## Goal

Decide whether the T11a daemon-side fix (commit 888ea822, program-
alive rule replacing `hits/age` in `build_paper_state`) produces a
**statistically meaningful** per-trial latency reduction over a
matched baseline.

## Discriminator design (one variable at a time)

Both configs use the SAME daemon binary (today's HEAD, T11a daemon
side applied).  The discriminator is just two daemon flags:

| config | daemon flags | scope of T11a rule |
|---|---|---|
| **baseline** | `--kv-scheduler=disabled --admission-controller=disabled` | inline scorer only (hits/age, not yet fixed) |
| **ours** | `--kv-scheduler=enabled --admission-controller=enabled` | inline scorer (hits/age) + daemon V_u (program-alive rule) |

Inline scorer is identical in both → any latency delta is
attributable to the daemon-side rule.

Inline-side T11a is **out of scope for this round** by user
direction ("先看看 daemon 问题是否解决再说").  If this round shows
a meaningful win, we proceed to inline-side T11a; if not, we look
at the H' ↔ kv_off 570 s gap first.

## Replication (N=3 per config, alternating)

Six runs total, alternating to neutralise time-of-day / GPU drift /
docker-cache warming:

```
cycle 1: baseline
cycle 2: ours
cycle 3: baseline
cycle 4: ours
cycle 5: baseline
cycle 6: ours
```

Each cycle starts from a **clean slate**:
* pkill sglang / daemon / mooncake_master
* drain zombies, GPU memory pre-flight
* fresh mooncake_master, sglang, daemon

Per cycle wall: ~50 min.  Six cycles: ~5 h.

## Fixed variables (reduce within-run noise)

* **LLM sampling**: `--ak temperature=0.0` on harbor (greedy
  decoding; agent decision path deterministic conditional on
  identical KV cache state).
* **sglang seed**: `--random-seed 42` on sglang launch (covers
  any internal scheduling randomness).
* **Instance set**: `harbor -l 32 -n 32 -k 1` — runs all 32
  swebenchpro task directories (alphabetical, deterministic),
  fully parallel, 1 attempt each.  **Beware**: harbor `-n` is
  `--n-concurrent` (parallelism), NOT total trial count; `-l` is
  `--n-tasks` (total cap); `-k` is `--n-attempts` (retries).  The
  earlier K runs used `-n 32` only and ran the same 32 because
  the dataset happens to have exactly 32 tasks.  We now pin all
  three to remove ambiguity.
* **Task cap**: `--ak max_turns=200` (unchanged from K full).
* **Hardware**: GPUs 4, 7 (B300 SXM6).  TP=2 EP=2.
* **HiCache**: enabled, `--hicache-ratio 1.5`, write-through-
  selective.  Mooncake L3 enabled.
* **sglang build**: must not change mid-experiment.  Verify
  `git rev-parse HEAD` is identical at the start of each cycle.

## Metrics (coarse + fine)

### Coarse: per-trial wall time

From `harbor_jobs/<run_id>/instance_*/result.json`:
* `started_at` / `finished_at` → duration_s

Aggregate per cycle: mean, p50, p99, min, max, stdev.
Aggregate across cycles per config: across-run mean ± std of the
per-cycle mean.

Sample size for the comparison:
* N=3 cycles × 32 trials = 96 trials per config
* Within-run stdev ~ 642 s (K full) → SEM(96) ≈ 65 s
* So a 100 s delta is now ~1.5σ; 200 s is ~3σ.  Beats single-shot.

### Fine: per-turn TTFT from sglang log

`sglang.log` emits a line per prefill with prompt length / cached
prefix length / TTFT.  6400 turns per cycle × 3 cycles = 19 200
samples per config.  This is what we use for the **clean signal**;
trial wall time is confounded by docker exec / agent decision
branches / harbor concurrency.

Parser: `verify/t9/parse_ttft.py` (TBD).  Emits per-turn:
`(trial_id, turn_idx, prompt_tokens, cached_tokens, ttft_ms)`.
Aggregate: TTFT mean / p50 / p99 per config; cache hit ratio
mean.

## Output layout

```
verify/t9/results/n3_matrix_<run_date>/
  cycle_1_baseline/
    harbor_jobs/...
    sglang.log
    daemon.log
    mooncake_master.log
    ttft.csv          # parsed
    trial_stats.json  # per-trial duration_s
  cycle_2_ours/
    ...
  ...
  cycle_6_ours/
  SUMMARY.md          # aggregated mean ± std table
```

## Stopping conditions (interrupt allowed)

After cycles 2, 4, 6 (each config has equal N), evaluate:
* If `ours_mean + ours_std < baseline_mean - baseline_std` →
  T11a daemon-side wins, stop early; proceed to inline-side.
* If `|ours_mean − baseline_mean| < (baseline_std + ours_std)/2`
  after 6 cycles → no signal; investigate H' ↔ kv_off 570 s gap
  before inline-side work.

## Orchestrator

`verify/t9/run_matrix.sh` — runs the N=3×2 matrix, alternating
configs.  Resumable: each cycle gets its own subdir and is skipped
if the result.json is present.

## Known uncontrolled variables

* Docker exec time: `docker compose exec main bash -c /tests/test.sh`
  has nondeterministic wall (filesystem cache, kernel scheduling).
  Hopefully averaged out by 32 × 3 = 96 trials per config.
* Other users' GPU jobs on 0–3, 5, 6: shouldn't touch 4 or 7 (we
  pre-flight check) but can affect host scheduling / IO.
* mooncake_master cold cache vs warm: every cycle resets it, so
  equal treatment.
* Network jitter on litellm → 127.0.0.1: localhost, negligible.
