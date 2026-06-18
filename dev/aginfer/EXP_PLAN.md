# aginfer — experiment plan

## Architecture

- **Platform**: NVIDIA Dynamo (container `aginfer_dyn`, GPUs 5,6)
- **Backend**: sglang fork (`aginfer-synced`, latest upstream)
- **Model**: DeepSeek-V4-Flash, tp=2
- **Replay**: agentreplay (`convert` → `replay-dynamo`), real CC traces only
- **In-engine scheduler**: `SGLANG_AGINFER_IN_ENGINE=1` (no daemon)
- **Teacher-forcing**: overlap-compatible GPU scatter (`forced_tokens.py`)

## Launch command (both arms identical except env var)

```bash
# baseline (B)
python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-storage-backend nixl    # full 4-tier: HBM → DRAM → SSD → DROP

# ours (only difference = one env var)
SGLANG_AGINFER_IN_ENGINE=1 python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-storage-backend nixl
```

No other flags. Dynamo router + frontend use defaults.

## Experiment matrix

### Axis 1: tier configuration

| Config | Flags | Tiers | Purpose |
|---|---|---|---|
| **4-tier (primary)** | `--enable-hierarchical-cache --hicache-ratio 2 --hicache-storage-backend nixl` | HBM → DRAM → SSD → DROP | Main result: full win |
| **2-tier** | `--enable-hierarchical-cache --hicache-ratio 2` (no backend) | HBM → DRAM → DROP | Ablation: fewer tiers, must be no-regression |
| **mooncake** | `--enable-hierarchical-cache --hicache-ratio 2 --hicache-storage-backend mooncake` | HBM → DRAM → SSD → DROP | Extensibility: different transport backend |

### Axis 2: router (ThunderAgent)

| Config | Router | What it tests |
|---|---|---|
| **default router** | Dynamo default (round-robin / KV-aware) | Pure isolation of our eviction scorer |
| **ThunderAgent** | `dynamo.thunderagent_router` | Our scorer + TA admission; proves orthogonality |

### Arms (same 4 arms in every E)

| Arm | Router | Engine eviction | What it is |
|---|---|---|---|
| **B** | default | LRU | Pure baseline |
| **TA** | ThunderAgent | LRU | Dynamo's existing best (cache-blind admission) |
| **Ours-evict** | default | value (in-engine) | Our eviction only, no admission |
| **Ours-full** | our router | value (in-engine) | Full system (admission + eviction, superset of TA) |

Key comparisons: Ours-full vs TA (headline), Ours-evict vs B (eviction value),
Ours-full vs Ours-evict (admission marginal value), TA vs B (TA value).

### Full matrix

| # | Tiers | Arms | What we show |
|---|---|---|---|
| **E1** | 4-tier nixl | B, TA, Ours-evict, Ours-full | **Headline**: full system on full tier stack |
| **E2** | 2-tier (no SSD) | B, TA, Ours-evict, Ours-full | No-regression with fewer tiers |
| **E3** | mooncake | B, TA, Ours-evict, Ours-full | Extensibility to different transport backend |

Recommended order: **E1 → E2 → E3**.

## Future experiments (not current scope, recorded for planning)

| # | What | Prerequisite | Why |
|---|---|---|---|
| **F1** | **KVBM on vLLM** (Dynamo-native KV manager, no engine fork) | vLLM backend + KVBM eviction policy plugin (Dynamo PR only, zero vLLM change) | KVBM supports vLLM/TRT-LLM (sglang ❌). Our value-aware eviction replaces KVBM's frequency-based policy at the Dynamo layer. SSD lifespan filter (freq≥2, default ON) modeled in our cost function (DISK cost = DROP cost when freq<2) — no need to disable it. |
| **F2** | **KVBM + PD disaggregation** | F1 + multi-GPU (≥4), NIXL KV transfer | KVBM on prefill worker + disagg decode. KV lifetime changes: prefill produces, NIXL transfers, decode consumes. NCCL replicated mode (`DYN_KVBM_NCCL_MLA_MODE`) for MLA models (DeepSeek) — only rank 0 loads back, broadcast to others. |
| **F3** | **KV-aware routing** (`--router-mode kv` + agent hints) | Multi-worker setup (≥2 workers) | Router steers requests to workers with cached prefixes; agentreplay emits `nvext.session_metadata` (session_id/trajectory_id) for session pinning. Proves our intra-worker eviction composes with inter-worker routing. |

E1 ablation: **NCCL MLA mode** on/off (single data point in the primary 4-tier experiment, measures load-back cost difference).

## Methodology

- **Token-exact replay** via agentreplay: real CC traces, `forced_output_ids`, faithful tool gaps
- **Traces**: `agentreplay convert --tokenizer deepseek-ai/DeepSeek-V4-Flash --max-turns N`
- **Metrics**: re-prefill (`#new-token`), TTFT, makespan, `cached_tokens`, per-program e2e
- **N ≥ 3** paired runs per cell; `agentreplay report` for mean±std + do-no-harm verdict
- **Do-no-harm**: ours ≤ B in every metric for every config (not just the win config)

## Scenario set (from `wherewewin/`)

| # | Scenario | Lever | Priority |
|---|---|---|---|
| **S2** | shared-prefix retention under churn | value eviction (holder-count) | first (cleanest) |
| **S1** | tool-call predictability | predictive promote | second |
| **S3** | drop-on-death (session end) | session-scoped eviction | third |
| **S5** | overload pause | admission control | needs ThunderAgent comparison |
| **S8** | comprehensive | full joint_decide | capstone |

## Open blocker

V4-Flash worker crashes under extreme oversubscription (occ ≈ 0.98). Use moderate-pressure
traces (`--max-turns` / `--max-prompt-tokens` in convert) to keep occ ≤ 0.90.
