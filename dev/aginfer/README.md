# aginfer

Workspace for the paper **"Multi-Agent KV Cache Scheduling as an MDP"**
(source: `~/projects/aginfer_paper/main.tex`).

## Architecture (post-cleanup 2026-06-18)

- **Platform**: Dynamo (no standalone sglang experiments)
- **Scheduling**: in-engine driver (`SGLANG_AGINFER_IN_ENGINE=1`), no external daemon
- **Lifecycle cleanup**: Dynamo forwards a terminal session signal to
  `POST /aginfer/session_end`; SGLang waits for in-flight ownership to drain,
  synchronizes TP/CP ranks, and immediately frees exclusive HBM/DRAM KV
- **Replay**: agentreplay (`convert` → `replay-dynamo`), real CC traces only
- **Teacher-forcing**: overlap-compatible GPU scatter (`managers/forced_tokens.py`)

## Where to look

| Read this | When |
|---|---|
| [`DESIGN.md`](DESIGN.md) | The spec: state/action surface, decision rule, joint MDP |
| [`Impl_PLAN.md`](Impl_PLAN.md) | Implementation history (note: daemon refs are historical) |
| [`EXP_PLAN.md`](EXP_PLAN.md) | Experiment roadmap: scenarios, methodology, execution |

## Subdirs

| Path | What |
|---|---|
| [`wherewewin/`](wherewewin) | Scenario catalogue (S1–S8) |
| [`verify/`](verify) | Server-free correctness tests |
| [`baselines/`](baselines) | Core policy library (shared types, cost model, knapsack DP, sglang adapter) |

## Conventions

- **GPUs**: 5, 6
- **Conda env**: `agsched-rebase` (`~/miniconda3/envs/agsched-rebase`)
- **CUDA**: 13.2 (`/usr/local/cuda-13.2`)
- **sgl_kernel**: 0.4.4 (built from local src for B300/sm_103)
