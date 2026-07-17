# dev/

Working area for the inter-pool memory layer on SGLang.

**Read first**: [`PRINCIPLES.md`](PRINCIPLES.md) — engineering rules
for the code in this directory and the production modules it
exercises under `python/sglang/srt/` (fail-fast over defensive,
TDD on every bug, no `getattr(self, "_x", None)` for state, comments
describe current behavior not session history, etc.). New
contributions are expected to follow these.

Two layers split into two directories:

- **`intralayer/`** — per-pool eviction policy (LPB, replaces recency
  LRU). Bench drivers for LRU-vs-LPB comparison live here.
- **`interlayer/`** — cross-pool capacity reallocation (move HBM
  between KV and mamba). Design + validation + planner correctness
  experiments live here.

Code lives at:

- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — LPB scoring
  on the radix tree (intralayer).
- `python/sglang/srt/budgeter/` — interlayer planner + agent. Uses
  `xpool_planner.py` (design.md §340 Budgeter) + `fire_planner.py`
  (decision → FirePlan) + `scheduler_owner_provider.py` (scheduler
  state → OwnerMap). `agent.py` wires these into the scheduler tick.
- `python/sglang/srt/arena/` — interlayer cuMem*-based actuator:
  `chunk_arena.py`, `shared_pool.py`, `multi_tensor_arena.py`,
  `xpool_actuator.py` + per-pool `kv_actuator.py` / `mamba_actuator.py`.

## Active layout

```
dev/
├── README.md                  this file
├── PRINCIPLES.md              engineering rules for code + tests here
├── intralayer/                LPB / per-pool eviction (bench drivers)
├── interlayer/                cross-pool reallocation
│   ├── design.md              full first-principles design + journey
│   ├── verify/                cost-conjecture validation (C1-C5)
│   └── planner_validate/      slack-harvest planner correctness
├── archive_path_a/            historical docs from VMM-remap era
├── eval/                      paper-eval run scripts
├── figures/                   plot scripts (regenerated from eval/runs)
├── parallel_gpu.sh            shared GPU-allocation helper
├── debug_fused_moe.py         one-off MoE routing debug
└── probe_moe_routing_entropy.py  MoE routing characterization
```

## How a benchmark cell runs

`dev/eval/*.sh` drives a server with budgeter flags set, runs a
genai-bench client at a fixed traffic-scenario, and dumps:

- `*_server.log`        SGLang server log (`grep execute[seq=` for fires)
- `*_client.log`        genai-bench client log
- `*_budgeter.jsonl`    per-tick budgeter snapshot
- `genai_results/`      per-request TTFT/TPOT/throughput

`*_budgeter.jsonl` field reference (set at `python/sglang/srt/budgeter/agent.py:_snapshot`):
- `xpool_plan_direction`  `"none"` / `"kv_to_mamba"` / `"mamba_to_kv"`
- `xpool_plan_reason`     why the gate fired or skipped
- `xpool_plan_executed`   True if a fire was actually attempted
- `xpool_plan_seq`        monotonic plan id (set on real fires)
- `xpool_aborted`         True if execute() bailed mid-fire
- `xpool_unmapped_total`  pages physically `cuMemUnmap`'d
- `xpool_granted_total`   pages physically `cuMemMap`'d
- `xpool_fire_total_us`   wall time for the whole fire

## Working dir

`/data/yuzhou/projects/sglang`. All paths in scripts assume this is
`pwd`.

## Local sgl-kernel build (after rebasing onto newer upstream)

A rebase forward often bumps the minimum `sgl-kernel` (e.g. upstream HEAD wants
`0.4.2.post2`); the boot then asserts out before pool init.
Build it from the in-tree `sgl-kernel/` so the version matches HEAD exactly,
rather than pulling a wheel.

Two gotchas, both non-obvious:
- CUDA 13 moved the CCCL headers to `targets/<arch>/include/cccl/`, so host g++
  (compiling mscclpp etc.) cannot find `<cuda/atomic>`. Put that dir on the host
  include path.
- Install NON-editable. The editable split-layout serves python from the source
  tree (no compiled `.so`) and the arch-subdir `common_ops` import fails.

```bash
CCCL=/usr/local/cuda/targets/x86_64-linux/include/cccl
export CPLUS_INCLUDE_PATH="$CCCL${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$CCCL${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export NVCC_PREPEND_FLAGS="-I$CCCL"
# H200: full Hopper, skip the archs we never run on (faster build).
CMAKE_ARGS="-DSGL_KERNEL_ENABLE_SM90A=ON -DENABLE_BELOW_SM90=OFF -DSGL_KERNEL_ENABLE_SM100A=OFF" \
  uv pip install ./sgl-kernel --no-build-isolation --python .venv/bin/python
# verify: .venv/bin/python -c "import sgl_kernel; print(sgl_kernel.__version__)"
```

If a fresh boot still reports the old kernel version, suspect a stale
`server.log` (check its mtime) or a launch command that `pkill -f`'s a pattern
matching its own command line, not a real version mismatch.
