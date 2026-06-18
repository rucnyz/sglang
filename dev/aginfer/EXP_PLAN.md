# aginfer — experiment plan

## Architecture (post-cleanup 2026-06-18)

- **Platform**: NVIDIA Dynamo (container `aginfer_dyn`, GPUs 5,6)
- **Engine backend**: sglang fork (branch `aginfer-synced` on latest upstream)
- **Replay harness**: agentreplay (`convert` real CC traces → `replay`/`replay-dynamo`)
- **In-engine scheduler**: `SGLANG_AGINFER_IN_ENGINE=1` (no external daemon)
- **Teacher-forcing**: overlap-compatible GPU scatter (`forced_tokens.py`)
- **Traces**: real Claude-Code trajectories via `agentreplay convert` (no synthetic workloads)

## The claim + competitors

**Thesis:** *proactive + value-aware + program-aware* KV scheduling beats *reactive +
recency + program-blind*.

| Arm | What | How enabled |
|---|---|---|
| **B (baseline)** | Dynamo default (LRU eviction, program-blind) | default sglang, no aginfer env |
| **TA** | ThunderAgent router (cache-blind admission only) | Dynamo ThunderAgent router |
| **Ours** | value-aware eviction + program-aware scheduling | `SGLANG_AGINFER_IN_ENGINE=1` |

The ONLY difference between B and Ours at the sglang level = one env var.
Dynamo/router configuration is identical across arms.

## Methodology

- **Token-exact replay** via agentreplay: `/generate` + `forced_output_ids` (overlap-compatible),
  real CC traces, byte-identical prompts across arms.
- **Metrics**: re-prefill (`#new-token` = tokens NOT cached), TTFT, makespan, per-program e2e,
  `cached_tokens` (from `meta_info`). agentreplay `report` computes mean±std per arm.
- **N ≥ 3**, paired measurements; do-no-harm = ours ≤ B in every metric.
- **Traces**: `agentreplay convert --tokenizer <target-model> --max-turns N --max-prompt-tokens M`
  from real `~/.claude/projects/` data. No synthetic workloads, no gap-scale manipulation.

## Scenario set (from `wherewewin/`)

| # | Scenario | Lever | Status |
|---|---|---|---|
| **S1** | tool-call predictability | predictive promote | needs Dynamo redo |
| **S2** | shared-prefix retention under churn | value eviction (holder-count) | needs clean run |
| **S3** | drop-on-death (session end) | session-scoped eviction | planned |
| **S5** | overload pause | admission control | planned |
| **S6** | blocking sub-agent | S1 + drop | planned |
| **S7** | background fan-out | proactive demote | planned |
| **S8** | comprehensive | full joint_decide | capstone, last |

## Execution

1. `agentreplay convert` to produce a trace for the target model (matching vocab)
2. Start Dynamo with sglang backend (our fork)
3. For each arm: set env (ours: `SGLANG_AGINFER_IN_ENGINE=1`), restart worker, run
   `agentreplay replay-dynamo --trace <trace.jsonl> --label <arm>`
4. `agentreplay report --ours ours.json --base base.json` for the verdict

## Open blocker

V4-Flash worker crashes under extreme oversubscription (scheduler watchdog timeout at
occ≈0.98). Moderate-pressure traces (occ peaks ~0.90) work. Use `--max-turns` /
`--max-prompt-tokens` in convert to control trace intensity.
