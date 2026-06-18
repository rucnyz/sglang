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

### Full matrix

| # | Tiers | Router | Arms | What we show |
|---|---|---|---|---|
| **E1** | 4-tier nixl | default | B, **TA**, Ours | **Headline win** (all three compared on full tier stack) |
| **E2** | 4-tier nixl | ThunderAgent | B(=TA+LRU), Ours(=TA+value) | Orthogonality (ours helps even under TA admission) |
| **E3** | 2-tier (no SSD) | default | B, **TA**, Ours | No-regression with fewer tiers |
| **E4** | mooncake | default | B, **TA**, Ours | Extensibility to different transport backend |

Recommended order: **E1 → E3 → E2 → E4**.

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
