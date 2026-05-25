# aginfer reproduce runbook

> One-shot reproducer for V4-Flash + 4-tier HiCache + concurrent harbor agent workload.
> All paths are absolute on `di-bm-vykpmg-145`. Replace with your own when porting.

---

## 0. Prerequisites (one-time install)

```bash
# Conda env with torch, sglang, mooncake, sgl_kernel, flash_mla
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched
```

If `agsched` doesn't exist, see [[agsched-env]] notes (~2 hours to rebuild). The current install
already has all five pieces below; the build steps are listed only for porting to another machine.

### 0a. FlashMLA from source (sglang dsv4 backend hard-imports it)

Use our patched fork — the upstream kernel crashes under tight KV pool + HiCache OFF.

```bash
# First-time only: clone the fork.
git clone git@github.com:rucnyz/FlashMLA.git /scratch/yuzhou/projects/FlashMLA
cd /scratch/yuzhou/projects/FlashMLA && git checkout aginfer

# Build + install (rebuild after pulling new commits).
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/build_flash_mla.sh
```

`aginfer` branch carries one commit on top of upstream `main`:

```
487d509  fix(get_decoding_sched_meta): skip kernel when batch is 0
```

### 0b. Our patched sglang (rucnyz/sglang `aginfer`)

`/scratch/yuzhou/projects/sglang` is already on `aginfer` (4 commits on top of upstream PR #26062).
Editable install via the existing `pip install -e python/`.

```
ab4acdeda  fix(mooncake_store): skip anchor buffer registration when kv_buffer is None
8ae7eb271  fix(swa_component): accept lock_host kwarg in acquire/release_component_lock
c4bdf1ba1  feat(mooncake_store): generate per-pool keys for DeepSeek-V4 sidecar pools
4138a93d8  fix(hybrid_cache_controller): skip anchor super() when kv_buffer is None
```

### 0c. Our patched Mooncake (rucnyz/Mooncake `aginfer`)

Cherry-picks upstream PR #2174 (TCP UAF in `ClientSession::writeBody`). Rebuild:

```bash
cd /scratch/yuzhou/projects/Mooncake && git checkout aginfer
mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python) -DPYTHON_EXECUTABLE=$(which python) ..
make -j 16

# Install client .so + master binary
SITE=/scratch/yuzhou/miniconda3/envs/agsched/lib/python3.12/site-packages/mooncake
cp mooncake-integration/engine.cpython-312-x86_64-linux-gnu.so $SITE/
cp mooncake-integration/store.cpython-312-x86_64-linux-gnu.so  $SITE/
cp mooncake-common/libasio.so $SITE/
mkdir -p ~/.local/bin && cp mooncake-store/src/mooncake_master ~/.local/bin/
```

### 0d. V4-Flash weights (149 GB FP8)

```bash
hf download deepseek-ai/DeepSeek-V4-Flash --max-workers 8
# Lands in ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash
```

### 0e. Harbor

```bash
cd /scratch/yuzhou/projects/harbor && pip install -e .
```

---

## 1. Start the serving stack

Each step in its own shell. All scripts source `aginfer/scripts/env.sh` which loads `.env`
and activates `agsched`.

### 1a. Mooncake master (RPC 50051, metrics 9053)

```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/start_mooncake_master.sh
```

Look for `Master service started on port 50051` in `logs/mooncake_master.log`. The binary used
is `~/.local/bin/mooncake_master` (our patched version — `env.sh` prepends it to PATH).

### 1b. SGLang V4-Flash (GPUs 5,6 — see [[gpu-layout]])

```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/launch_sglang_v4flash.sh
```

Wait for `Uvicorn running on http://0.0.0.0:30000` in `logs/sglang_v4flash.log`. Total ~5 min
(weight load 60s + cuda graph 90s + host pool alloc 200s + Mooncake warmup).

### 1c. Sanity (optional)

```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/sanity_request.sh
# repeat the same call: second one should report #cached-token > 0 in server log
```

---

## 2. Run an agent benchmark

### 2a. Generate the dataset

```bash
# AIME math (60 tasks, light)
cd /scratch/yuzhou/projects/harbor/adapters/aime && uv run aime
# datasets land at /scratch/yuzhou/projects/harbor/datasets/aime

# SWE-bench Pro python subset (limit 32, much heavier)
cd /scratch/yuzhou/projects/harbor/adapters/swebenchpro && uv run swebenchpro --language python --limit 32
# datasets land at /scratch/yuzhou/projects/harbor/datasets/swebenchpro
```

### 2b. Run harbor with terminus-2 against V4-Flash

The labelled runs reported in `results/SUMMARY.md` (A–G) all use
`swebenchpro --limit 32 -n 32 --ak max_turns=200`. Pick a `--jobs-dir`
under `results/run_X_…` so the SUMMARY columns stay consistent.

```bash
cd /scratch/yuzhou/projects/harbor

OPENAI_API_KEY=sk-fake-do-not-check \
  harbor run \
    -p datasets/swebenchpro \
    -a terminus-2 \
    -m openai/deepseek-ai/DeepSeek-V4-Flash \
    --ak api_base=http://172.17.0.1:30000/v1 \
    --ak max_turns=200 \
    -n 32 \
    --jobs-dir /scratch/yuzhou/projects/sglang/dev/aginfer/results/run_<X>
```

`http://172.17.0.1:30000` is the docker bridge gateway IP — every harbor agent container can
reach the host's `0.0.0.0:30000` through it. (V4-Flash must be launched with `HOST=0.0.0.0`,
which `launch_sglang_v4flash.sh` already does.)

### 2b'. Run G — through the ThunderAgent router

Same harbor command as 2b, but point `api_base` at the router (port 9100
on the host) instead of sglang directly. Requires
[`rucnyz/ThunderAgent@aginfer`](https://github.com/rucnyz/ThunderAgent/tree/aginfer)
installed in the `agsched` env and
[`rucnyz/harbor@aginfer`](https://github.com/rucnyz/harbor/tree/aginfer)
which mirrors `session_id` into `extra_body.program_id` (with UUID
fallback). See [NOTES §9](NOTES.md#9-thunderagent-run-g-集成) for details.

```bash
# 1. backend (cap 512 K + HiCache ON, identical to Run F)
MAX_TOTAL_TOKENS=524288 \
  bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/launch_sglang_v4flash.sh

# 2. router (port 9100; host port 9000 is taken on di-bm-vykpmg-145)
TA_PORT=9100 \
  bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/launch_thunderagent.sh

# 3. harbor → router → sglang
OPENAI_API_KEY=sk-fake-do-not-check \
  harbor run \
    -p datasets/swebenchpro \
    -a terminus-2 \
    -m openai/deepseek-ai/DeepSeek-V4-Flash \
    --ak api_base=http://172.17.0.1:9100/v1 \
    --ak max_turns=200 -n 32 \
    --jobs-dir /scratch/yuzhou/projects/sglang/dev/aginfer/results/run_G_thunderagent
```

Verify routing is real (not pass-through): `curl http://127.0.0.1:9100/programs`
should list N UUID program ids, not a single `"default"`.

Result layout per job:
```
<jobs-dir>/<timestamp>/
├── job_metadata.json
├── trials/<trial_id>/{logs,traj,result}.json
└── ...
```

### 2c. Watch what's happening

```bash
# Active harbor containers (each one = one trial)
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'aime|swebenchpro'

# Server's prefill batches with cache-hit counters
tail -F /scratch/yuzhou/projects/sglang/dev/aginfer/logs/sglang_v4flash.log | grep "Prefill batch"
# look for #cached-token > 0  and input throughput numbers
```

---

## 3. Resume / cleanup

```bash
# Resume a job that got interrupted
cd /scratch/yuzhou/projects/harbor
harbor job resume -p /scratch/yuzhou/projects/sglang/dev/aginfer/results/run_<X>/<timestamp>

# Kill all harbor docker leftovers
docker ps --format '{{.Names}}' | grep -E 'instance_|swebenchpro' | xargs -r docker kill
```

---

## 4. Default knobs (changeable in `scripts/launch_sglang_v4flash.sh`)

| Knob | Current | Rationale |
|---|---|---|
| `--tp 2 --ep 2` | 2 GPUs, TP + EP | Fits V4-Flash on GPUs 5,6 |
| `--mem-fraction-static 0.85` | 85% HBM | Leaves room for DeepGEMM 6 GB activations; 0.95 OOMs |
| `--hicache-ratio 1.5` | Host pool = 1.5× device pool | Auto-sized; do not write `--hicache-size N` |
| `--hicache-write-policy write_through_selective` | Async backup of hot pages | `write_through` synchronous blocks prefill |
| `--context-length 65536` | 64K | Plenty for AIME / SWE-bench Pro |
| `--moe-a2a-backend none` | Triton fused MoE | `deepep` not installed, `mooncake` a2a slower for our load |

Mooncake L3 config in `MOONCAKE_EXTRA` JSON inside the launch script:
- `global_segment_size: 200gb` — DRAM each TP rank contributes
- `protocol: tcp` + `metadata_server: P2PHANDSHAKE` — single-node, no RDMA
- SSD spill currently **off** (4-tier reduced to tier 3 HBM + tier 2 DRAM + Mooncake DRAM
  pool). To re-enable disk spill, add `enable_ssd_offload: true, ssd_offload_path:
  /scratch/yuzhou/mooncake_ssd` and restart `mooncake_master` with `--enable_offload=true`.
