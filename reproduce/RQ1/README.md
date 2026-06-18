# RQ1 — End-to-end comparison (\sys{} vs. baselines)

RQ1 evaluates \sys{} on **three real agent workloads, all constructed from the
Claude Code (cc) trace** — same data source, different workload *shapes* that
bind different pools. Together they show \sys{} adapts to whichever pool is the
bottleneck (and stays zero-cost when neither is). All three run on the
**token-exact agentreplay harness** (replays real CC traces deterministically via
`forced_output_ids`); agentreplay is invoked only as a replay TOOL.

## Cases

| # | folder | workload | binds | direction |
|---|---|---|---|---|
| 1 | `case1/` | a few super-long-horizon sessions overflow KV while their O(1) recurrent states leave mamba idle | KV | **m2k** — grow KV from idle mamba |
| 2 | `case2/` | a short-prompt swarm: `max_running` binds while KV sits idle | mamba | **k2m** — grow mamba from idle KV |
| 3 | `case3/` | dynamic A->B flip (long KV-bound phase, then the case2 swarm) on one boot | shifts over time | both |

Each `caseN/` holds its token-exact runner plus a `data/` with that case's cc
trace (100s of MB, gitignored — see `caseN/data/README.md` to populate). Results
+ the full investigation journey are in `FINDINGS.md`.

## Shared (RQ1 root)

- `run_arm.sh` — boots ONE arm (`base`|`sys`), runs N agentreplay reps on the
  same server (`--flush` between reps), tears down. The only non-default sglang
  flags: `--reasoning-parser`, `--enable-cache-report`, and
  `--mamba-scheduler-strategy extra_buffer` (the one knob that keeps overlap ON
  for hybrid mamba + radix cache). mem-frac / context / pool split = sglang
  defaults — the A/B measures the cross-pool change, not tuning.
- `case_default_build.py` — builds the case2/case3 traces from case1's trace
  (swarm source) + the agentreplay corpus long source.
- `ablations/` — eviction axis (`ab*_262k`: base / inter / full), tick cadence
  (`ablate_tick`), decode profile (`profile_decode`), case2 sweeps (`ablate_case2*`).
- `FINDINGS.md` — investigation journal + headline results.

## Prerequisites

- agentreplay cloned at `/scratch/yuzhou/projects/agentreplay` (the `$AR` /
  `PYTHONPATH` in `run_arm.sh`), corpus pulled into `$AR/data/traces/`.
- sglang built into `.venv` (the `VENV` in `run_arm.sh`). GPU 7 by default
  (`GPU=` env to override). Populate each `caseN/data/` first (see its README).

## Experiment standard (applies to ALL cases)

Every case must achieve:
1. **No-regression**: sys ≥ base on every metric. sys < base*0.98 on ANY metric = our bug, must fix before proceeding.
2. **Large win**: find a workload shape (from the 2.3G corpus) where sys wins significantly. A flat result means the workload doesn't exercise the mechanism, not that the mechanism is broken.
3. **Record**: the winning workload shape + N=3 A/B become that case's official result.

During the search, any regression found on ANY workload is a bug, not noise. Fix it, then continue.

## Run

```bash
bash reproduce/RQ1/case1/case1_kvbound_win.sh    # base N=3 then sys N=3
```

Outputs land in `caseN/runs/` (gitignored).

## Related

- `waste/` — the KV-waste motivation figures.

The earlier genai-bench *synthetic* benches (the old `table1/2/3`) were removed
(task #328): the token-exact agentreplay cases above are the sole RQ1 harness.
