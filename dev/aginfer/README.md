# aginfer

Workspace for the paper **"Multi-Agent KV Cache Scheduling as an MDP"**
(source: `~/projects/aginfer_paper/main.tex`).

## Where to look first

| Read this | When |
|---|---|
| [`DESIGN.md`](DESIGN.md)   | What aginfer is, the state/action surface, decision rule, joint MDP — **the spec, source of truth** |
| [`Impl_PLAN.md`](Impl_PLAN.md) | **Implementation** work (calibration, observability, sglang+daemon; layering §7, refactor §8), keyed to DESIGN |
| [`EXP_PLAN.md`](EXP_PLAN.md) | **Experiment** roadmap: the 3-config scorer factorial + the `wherewewin/` scenarios; execution → `reproduce/RQ1/` |
| [`RUNBOOK.md`](RUNBOOK.md) | **Dynamo** stack startup (worker + router + frontend + daemon + bridge) + the operational gotchas |

## What's in each subdir

| Path | What | Read its own README for details |
|---|---|---|
| [`wherewewin/`](wherewewin) | **Scenario catalogue** — the distinct agentic scenarios (S1–S8) where ours should beat B/TA, with each one's win + metric | [`wherewewin/README.md`](wherewewin/README.md) |
| [`daemon/`](daemon)         | The aginfer daemon: proxy, event router, program tracker, admission controller, kv scheduler | — |
| [`verify/`](verify)         | Per-item correctness + performance tests, aligned with `Impl_PLAN.md` sections | [`verify/README.md`](verify/README.md) |
| [`baselines/`](baselines)   | **Live core policy library** (despite the name): shared data types (`Tier`/`Scope`/`ReuseUnit`/`Action`), cost model, the OursGreedy policy, the knapsack DP, the sglang adapter — imported by the daemon + verify. (paper §8 baseline policies also live here) | — |
| [`scripts/`](scripts)       | Launch + sanity scripts | — |
| [`results/`](results) / [`logs/`](logs) | Raw run output / server stdout (`*.log.prev` = previous run) | — |
| [`archive/`](archive)       | Legacy, do-not-use: old **harbor** scenario tree + the §2.4 `workload/` Agent-DAG model — superseded by Dynamo (`../dynamo/`) + `reproduce/RQ1/` | [`archive/README.md`](archive/README.md) |

> **Live experiments now run on Dynamo** — see [`../dynamo/`](../dynamo) (platform + S2),
> [`EXP_PLAN.md`](EXP_PLAN.md) (what to run), and `../../reproduce/RQ1/` (paper packages).

## Conventions

- **GPUs**: default 5, 6 (`AGINFER_GPUS` in `.env`)
- **Conda env**: `agsched` at `~/miniconda3/envs/agsched`
- **CUDA**: 13.2 at `/usr/local/cuda-13.2`
- **HF cache**: default `~/.cache/huggingface` (do NOT override)
- **Logs**: `logs/*.log`, previous run preserved as `*.log.prev`

## External forks (all `aginfer` branch)

| Repo | Why |
|---|---|
| [rucnyz/sglang](https://github.com/rucnyz/sglang/tree/aginfer)            | 4 patches on PR #26062 (V4 sidecar pool support, anchor-buffer skips, lock_host kwarg) |
| [rucnyz/Mooncake](https://github.com/rucnyz/Mooncake/tree/aginfer)        | Cherry-picks upstream PR #2174 (TCP UAF in `ClientSession::writeBody`) |
| [rucnyz/FlashMLA](https://github.com/rucnyz/FlashMLA/tree/aginfer)        | Diagnostic patch on `get_decoding_sched_meta.cu` (smem cap surfacing) |
| [rucnyz/ThunderAgent](https://github.com/rucnyz/ThunderAgent/tree/aginfer) | Lazy-router startup fix so CLI backends flags actually take effect |
| [rucnyz/harbor](https://github.com/rucnyz/harbor/tree/aginfer)            | `lite_llm.py` mirrors `extra_body.session_id` as `program_id` with UUID fallback |

Build steps for each fork are in [`RUNBOOK.md`](RUNBOOK.md) §0.
