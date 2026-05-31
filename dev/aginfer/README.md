# aginfer

Workspace for the paper **"Multi-Agent KV Cache Scheduling as an MDP"**
(source: `~/projects/aginfer_paper/main.tex`).

## Where to look first

| Read this | When |
|---|---|
| [`DESIGN.md`](DESIGN.md)   | What aginfer is, the state surface, action surface, decision rule, joint MDP — **the spec, source of truth** |
| [`PLAN.md`](PLAN.md)       | Open implementation work, ordered by dependency.  Each item points at the DESIGN section it implements |
| [`RUNBOOK.md`](RUNBOOK.md) | One-shot reproducer: install → start services → run a benchmark → cleanup |

## What's in each subdir

| Path | What | Read its own README for details |
|---|---|---|
| [`baselines/`](baselines)   | Paper §8 policy implementations (LRU / TA / InferCept / Continuum / KVFlow / Ours-greedy) + simulation harness | — |
| [`daemon/`](daemon)         | The aginfer daemon: proxy, event router, program tracker, admission controller, kv scheduler | — |
| [`workload/`](workload)     | Agent-DAG data model used by baseline policies | — |
| [`scripts/`](scripts)       | Launch + sanity scripts sourced by RUNBOOK | — |
| [`verify/`](verify)         | Per-item correctness + performance tests, aligned with PLAN.md sections | [`verify/README.md`](verify/README.md) |
| [`scenarios/`](scenarios)   | End-to-end workload × arm comparisons (e.g. swebench_default × {LRU, TA, ours_inline, ours_full}) | [`scenarios/README.md`](scenarios/README.md) |
| [`results/`](results)       | Raw harbor / scenario output, one dir per labelled run | — |
| [`logs/`](logs)             | Server / build stdout, rotated (`*.log.prev` = previous run) | — |

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
