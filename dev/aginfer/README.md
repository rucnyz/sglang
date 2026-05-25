# aginfer experiments

> Reproducible workspace for the paper **"Multi-Agent KV Cache Scheduling
> as an MDP"**. Everything in this directory is self-contained — clone the
> 3 git forks listed below, follow [`RUNBOOK.md`](RUNBOOK.md), and the
> numbers in [`results/SUMMARY.md`](results/SUMMARY.md) +
> [`results/ALGO_BASELINES.md`](results/ALGO_BASELINES.md) come back the same.

Paper source: `~/projects/aginfer_paper/main.tex`.

## What's in here

| Path | What |
|---|---|
| [`NOTES.md`](NOTES.md)         | Working log — environment, model selection, all known patches, every gotcha hit along the way |
| [`RUNBOOK.md`](RUNBOOK.md)     | One-shot reproducer: install → start services → run benchmark → cleanup |
| [`scripts/`](scripts)          | Launch + sanity scripts (sourced by RUNBOOK). Read `.env` for the few real knobs |
| [`baselines/`](baselines)      | Paper §8 policy implementations + simulation harness |
| [`workload/`](workload)        | Agent-DAG data model (paper §2.4) shared by policies and the workload driver |
| [`results/`](results)          | Numbers + per-trial harbor outputs |
| [`logs/`](logs)                | Server / build / harbor stdout (rotated, `*.log.prev` is the previous run) |

## Two parallel evaluations

### 1. End-to-end serving (DeepSeek-V4-Flash + sglang + Mooncake)

Six SWE-bench-Pro 32-task runs across three KV-pool pressure regimes
(loose / moderate / tight) × HiCache {on, off}. The matrix:

- Wall-clock runtime
- Peak `#running-req`
- Max SWA token usage
- Peak input throughput

[`results/SUMMARY.md`](results/SUMMARY.md) — full table + per-regime
discussion + the precise root cause of Run D's FlashMLA crash (sm_100
per-block dynamic-shmem cap, see [§FlashMLA crash](#flashmla-crash) below).

Reproduce: cap is the env var `MAX_TOTAL_TOKENS`; HiCache on/off is which
launch script (`launch_sglang_v4flash.sh` vs `..._nohicache.sh`). See
[`RUNBOOK.md §2`](RUNBOOK.md).

### 2. Algorithmic policy comparison (paper §8 baselines)

All five paper-§8 baselines + our greedy specialization, scored on a
synthetic agent-DAG event stream:

| Policy | File |
|---|---|
| LRU              | [`baselines/lru.py`](baselines/lru.py) |
| ThunderAgent     | [`baselines/thunder_agent.py`](baselines/thunder_agent.py) |
| InferCept        | [`baselines/infercept.py`](baselines/infercept.py) |
| Continuum        | [`baselines/continuum.py`](baselines/continuum.py) |
| KVFlow           | [`baselines/kvflow.py`](baselines/kvflow.py) |
| Ours (greedy)    | [`baselines/ours_greedy.py`](baselines/ours_greedy.py) |
| harness          | [`baselines/compare.py`](baselines/compare.py), [`baselines/sweep_seeds.py`](baselines/sweep_seeds.py) |

Scored by the paper's reward decomposition `r1 (saved prefill) − r2
(migration) − r3 (holding)` and by wall-clock-equivalent metrics
(`total_runtime_s`, `throughput_tok_per_s`) — same units as
SUMMARY.md so the two evaluations line up.

[`results/ALGO_BASELINES.md`](results/ALGO_BASELINES.md) — single-seed
table, 8-seed sensitivity, per-baseline interpretation.

Reproduce:
```bash
cd /scratch/yuzhou/projects/sglang/dev/aginfer
source scripts/env.sh
python -m baselines.compare        # one deterministic seed
python -m baselines.sweep_seeds    # 8-seed mean/std
```

## Stack

Forked repos (`aginfer` branch on each):

| Repo | Why |
|---|---|
| [rucnyz/sglang](https://github.com/rucnyz/sglang/tree/aginfer)        | 4 patches on top of PR #26062's `support_unified_tree_l3_main`: V4 sidecar pool support in mooncake_store + hybrid_cache_controller. See [NOTES §8](NOTES.md#8-v4-flash--4-tier-hicache-的-6-个-patchstatus-working) |
| [rucnyz/Mooncake](https://github.com/rucnyz/Mooncake/tree/aginfer)    | Cherry-picks upstream PR #2174 — TCP UAF in `ClientSession::writeBody`. Without it, batched store crashes within seconds |
| [rucnyz/FlashMLA](https://github.com/rucnyz/FlashMLA/tree/aginfer)    | Diagnostic patch on `get_decoding_sched_meta.cu` — see below |
| [rucnyz/ThunderAgent](https://github.com/rucnyz/ThunderAgent/tree/aginfer) | Lazy-router fix so `--backends` / `--backend-type` are not silently ignored at startup (the package's `__init__.py` eagerly imports `.app`, which used to build the router with default config before `set_config()` ran). Used as the paper §8 ThunderAgent baseline in Run G |
| [rucnyz/harbor](https://github.com/rucnyz/harbor/tree/aginfer)        | `lite_llm.py` mirrors `extra_body.session_id` as `extra_body.program_id` (with UUID fallback when caller passes `session_id=None`) so router proxies — e.g. ThunderAgent — can key off it for per-program scheduling |

## FlashMLA crash

Run D (cap 256K + HiCache OFF) deterministically crashes ~3-13 min into
the workload. The original error is opaque (`CUDA error: invalid
argument`); with our debug-instrumented patch the failure prints the
exact culprit:

```
b=13065 smem=261304 capture_status=0 → CUDA error: invalid argument
```

sglang's dsv4 NSA decode path passes `q.shape[0] ≈ 13K` for transient
mixed prefill+decode batches under tight KV pressure. The metadata
kernel allocates `4*(b*5+1)` B of dynamic shmem; b=13065 asks for ~255 KB
which exceeds sm_100 (B300)'s 228 KB per-block cap → the kernel launch
itself fails. Not a graph-capture issue, not a `cudaFuncSetAttribute`
issue — straight smem overflow.

Structural fix is upstream FlashMLA work (multi-block metadata kernel)
or sglang chunking; both out of scope for this paper. We patched
FlashMLA to surface a precise diagnostic in place of the generic CUDA
error and document the regime in SUMMARY.md.

## Conventions

- **GPUs**: default 5, 6 (configurable via `$AGINFER_GPUS` in
  [`.env`](.env))
- **HF cache**: default `~/.cache/huggingface` — do NOT override
- **Logs**: `logs/*.log`, previous run preserved as `*.log.prev`
- **CUDA**: 13.2 at `/usr/local/cuda-13.2` (set by `scripts/env.sh`)
- **Conda env**: `agsched` at `~/miniconda3/envs/agsched` (see
  [NOTES §1](NOTES.md#1-环境))

## What's NOT in here

- Paper LaTeX source (lives at `~/projects/aginfer_paper/main.tex`)
- Model weights (`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash`, 149 GB FP8)
- Workload datasets (`~/projects/harbor/datasets/{aime,swebenchpro}` — generated by `harbor`'s adapters, not by anything here)
