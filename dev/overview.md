# Hybrid LLM Memory Hierarchy — implementation track

This directory documents the prototype work behind the paper's two-layer hybrid memory hierarchy:

```
Layer 1 (within-pool):     state-recoverable prefix cache for hybrid LLMs
                           — big-page + small-page + SSM snapshots
                           — input-token-only indexing
                           — host-DRAM tier

Layer 2 (cross-pool):      VMM-based budgeter that physically reallocates
                           HBM across paged KV / live SSM / LoRA / prefix-cache
                           pools at phase timescale, driven by observed V_σ pressure
```

The two layers compose: Layer 1 makes prefix cache valuable on hybrid models (steeper $V_{\text{prefix}}$); Layer 2 reallocates bytes across pools as workload mix shifts. Without Layer 1 the budgeter has nothing meaningful to arbitrate around (Phase 0 Sweep 3 confirmed this: $V_{\text{prefix}}$ is flat on hybrid + naive RadixCache); without Layer 2 Layer 1's cache is sized once at startup and can't follow workload phase shifts.

We implement on SGLang because (1) named per-pool classes match the paper's 4-pool framework directly, (2) `UnifiedRadixCache` already does cross-component eviction, (3) per-pool stats are already exported in Prometheus, and (4) it ships a CUDA-VMM-aware allocator (`--enable-memory-saver`) we can extend to per-chunk granularity for Layer 2.

## Phases

| Phase | Goal | Code touched | Output |
|---|---|---|---|
| **0** ([`0.md`](0.md)) | Static config sweeps to characterize per-pool utility curves $V_\sigma(m_\sigma)$ | None — config flags only | Empirical $V_\sigma$ curves; static-default-is-wrong evidence |
| **1** ([`1.md`](1.md)) | Read-only joint pressure dashboard | ~300 LOC, instrumentation | Signal contract validated; 24 named gauges → JSONL → 6-panel PNG |
| **2** ([`2.md`](2.md)) | Cross-pool VMM budgeter (= "Layer 2" / paper's A) | ~2 k LOC | Physical HBM reallocation across pools; the demo |
| **3** ([`3.md`](3.md)) | Layer 1 implementation: state-recoverable prefix cache | ~600 LOC | Input-token-only indexing, page-aligned SSM snapshots, snapshot-aware eviction, host-DRAM tiering |

Phase 3 is the Layer 1 implementation track on SGLang.

## Conventions

- **Working dir**: `/scratch/yuzhou/projects/sglang`. All commands assume this is `pwd` unless noted.
- **Python env**: `./.venv` (uv-managed, Python 3.12). Activate with `source .venv/bin/activate`. Never use system Python or any other env.
- **Env vars**: live in `./.env` at the project root. Source with `set -a; source .env; set +a` before SGLang commands. Don't export to shell rc files.
- **GPU selection**: use idle GPUs (the box has 8× H200). Default to GPU 2 unless a phase doc says otherwise. Run `nvidia-smi` first to verify.
- **Artifacts**: each phase has a sibling directory (`dev/0/`, `dev/1/`, ...) holding raw logs, scripts, and plots. Markdown files reference them by relative path.
- **Reproducibility**: every result reported in a phase doc must include the exact command that produced it, the SGLang commit, and the GPU model.

## Hardware (snapshot 2026-04-29)

```
8× NVIDIA H200 (143 GiB each), driver 595.58.03
CUDA 13.2, V13.2.51
```

## Status

| Phase | State | Last update |
|---|---|---|
| 0 | **complete** — Sweeps 1 (KV↔SSM), 2 (KV↔LoRA), 3 (KV↔prefix); see [`0.md`](0.md) | 2026-04-29 |
| 1 | **complete** — sampling client + dashboard + 3-phase validation; see [`1.md`](1.md) | 2026-04-29 |
| 2 | **2a/2b + 2e.1–2e.4.c** done — full VMM stack proven end-to-end; SGLang KV pool serves real requests with arena-backed tensors. 2e.4.d (set_capacity + budgeter wiring) pending. See [`2e/`](2e/). | 2026-04-30 |
| 3 | not started | — |
