# aginfer 实验工作日志

> 多 agent KV cache scheduling MDP 实验。Paper: `~/projects/aginfer_paper/main.tex`。
> 工作日志 + reproduce 命令。要接手靠这个文件 + `scripts/` 里的脚本。

---

## 1. 环境

| 项 | 值 |
|---|---|
| Host | `di-bm-vykpmg-145` |
| GPU | 8× NVIDIA B300 SXM6 AC，275 GB HBM each，**sm_100** |
| 默认用卡 | **GPU 5, 6**（其它常被别人占；`AGINFER_GPUS=5,6` 由 `env.sh` 设） |
| Mem | 3.0 TB |
| Disk | `/scratch` 16 TB 可用 |
| CUDA toolkit | **13.2**（`/usr/local/cuda-13.2`，`nvcc V13.2.51`） |
| Conda env | `agsched` @ `/scratch/yuzhou/miniconda3/envs/agsched`：torch 2.11.0+cu130、sglang 0.5.13.dev64+g229cadec0、mooncake、sgl_kernel、flash_mla 1.0.0+9241ae3 |
| HF cache | 默认 `~/.cache/huggingface`（**不要 export HF_HOME**） |
| Mooncake master 二进制 | `/usr/local/bin/mooncake_master` |

加载 env：
```bash
source /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/env.sh
```

---

## 2. 模型

**`deepseek-ai/DeepSeek-V4-Flash`**

| 字段 | 值 |
|---|---|
| 权重 | FP8 e4m3，46 个 safetensors，**149 GB** |
| Arch | `DeepseekV4ForCausalLM`，43 层，hidden 4096 |
| MoE | 256 routed experts + 1 shared, top-6 |
| Attention | MLA（q/o lora rank 1024, head_dim 512）+ NSA index（index_topk=512） |
| Sliding window | 128 |
| Max context | 1 M（yarn factor 16，base 65536） |
| Quant | FP8 block 128×128；`expert_dtype: fp4` |

下载：
```bash
hf download deepseek-ai/DeepSeek-V4-Flash --max-workers 8
```

---

## 3. 端口与服务

| 端口 | 服务 |
|---|---|
| 50051 | `mooncake_master` RPC |
| 9053 | `mooncake_master` 指标 HTTP |
| 30000 | sglang server (V4-Flash) |
| 30001 | sglang smoke (Qwen3-0.6B) |

单节点：mooncake 用 `protocol=tcp` + `metadata_server=P2PHANDSHAKE`，不需要 RDMA、不需要独立 metadata service。

---

## 4. 选型决策

- **Engine**: SGLang + Mooncake HiCache，**不做 PD 分离**
  - HiCache 给 paper 的 tier 3 → 2 → 1 物理层
  - `--enable-hierarchical-cache --hicache-storage-backend mooncake`
- **并行**: `--tp 2 --ep 2`
- **MoE backend**: `--moe-a2a-backend none --moe-runner-backend deep_gemm`（Triton fused MoE 路径，稳定优先；`deepep` 未装，后期上）
- **Attention**: `dsv4`（auto-detected），依赖 `flash_mla`
- **HiCache size**: `--hicache-ratio 1.5`（host = 1.5 × device pool），**不要写死 `--hicache-size`**——device pool 跟 `mem-fraction-static` 走，不一致会触发 `host >= device` 断言失败
- **HiCache write policy**: `write_through_selective`（默认热点写穿到 DRAM）
- **SSD spill**: `/scratch/yuzhou/mooncake_ssd`

---

## 5. Reproduce

### 5.1 一次性安装（已完成）

```bash
# FlashMLA（dsv4 backend 强依赖；sgl_kernel 还没 vendor 它）
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/build_flash_mla.sh
```

### 5.2 起服务

终端 A — mooncake master：
```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/start_mooncake_master.sh
```

终端 B — sglang V4-Flash（GPU 5,6，TP=2 EP=2，HiCache + mooncake + SSD spill）：
```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/launch_sglang_v4flash.sh
```

终端 C — sanity request：
```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/sanity_request.sh
```

smoke server（Qwen3-0.6B，GPU 5）：
```bash
bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/launch_sglang_smoke.sh
PORT=30001 MODEL=Qwen/Qwen3-0.6B bash /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/sanity_request.sh
```

日志在 `logs/`，每次启动 `rotate_log` 把上一次挪成 `*.log.prev`。

---

## 6. 进度

- [x] V4-Flash 权重下载（149 GB → `~/.cache/huggingface/hub`）
- [x] mooncake_master 起来
- [x] Qwen3-0.6B + HiCache + Mooncake smoke：sanity request 通
- [x] FlashMLA 从官方源码本地编译装好（`/scratch/yuzhou/projects/FlashMLA/`）
- [x] Baseline 框架骨架（`baselines/{base,lru,ours_greedy,costs}.py` + `workload/agent_dag.py`）
- [x] **V4-Flash + 完整 4-tier HiCache 跑通**（3 连发请求 + cache hit，详见 §8 修复说明）
- [x] **Baseline 端到端 benchmark**：6-run swebenchpro matrix → `results/SUMMARY.md`
- [x] **paper §8 算法 baseline 实现 + 模拟比较**（LRU / ThunderAgent / InferCept / Continuum / KVFlow / Ours）→ `results/ALGO_BASELINES.md`
- [x] **FlashMLA b=0 crash 修了**：fork rucnyz/FlashMLA aginfer 分支，patch + 重 build
- [x] **Real ThunderAgent baseline (Run G)**：rucnyz/ThunderAgent aginfer + rucnyz/harbor aginfer patch，跑 swebenchpro × 32，per-trial mean 666 s vs Run F 981 s（-32%）。详见 [§9](#9-thunderagent-run-g-集成)
- [ ] Workload DAG driver 接 SGLang client，跑真请求（不只是 swebenchpro adapter）

---

## 7. 已知坑

- **dsv4 backend 强依赖 `flash_mla`**：sglang 的 commit `621dfb888 Import flash_mla from sgl-kernel` 只迁了 `flashmla_backend.py` + `nsa_backend.py`，dsv4 backend 还是 `import flash_mla` 外部包。所以必须本地装 flash_mla。
- **CUDA 13 编译 FlashMLA**：`pybind.cpp` 被纯 g++ 编译，不像 nvcc 那样自动加 `cccl` isystem 路径，会 `#include <cuda/std/utility>` 找不到。`scripts/build_flash_mla.sh` 已经 export `CPATH=/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl:$CPATH`。
- **FlashMLA 源**：用官方 `https://github.com/deepseek-ai/FlashMLA`，**不要**用 `/scratch/yuzhou/projects/eb-vllm/.deps/flashmla-src/`（vllm-project fork，setup.py 引用不存在的 `csrc/sm90/decode/dense_fp8/`）。
- **DeepSeek-V4-Flash NSA**：`index_topk=512`，sglang NSA 路径在 commit `cadfa2d02 Support piecewise CUDA graph with NSA` 之后才稳。出问题先 `--disable-cuda-graph` 排查。

---

## 8. V4-Flash + 4-tier HiCache 的 6 个 patch（status: working）

6 个独立修复，缺一不可：sglang 4 个、Mooncake 1 个、FlashMLA 1 个。前 4 个在 `rucnyz/sglang aginfer`，Mooncake 在 `rucnyz/Mooncake aginfer`，FlashMLA 在 `rucnyz/FlashMLA aginfer`。

### FlashMLA 侧（`rucnyz/FlashMLA` aginfer 分支）

Fork：**https://github.com/rucnyz/FlashMLA/tree/aginfer**

| commit | 说明 |
|---|---|
| `99cc18b` | smem 超 sm_100 228 KB 上限时直接报详细信息（b、smem、cap），把 opaque CUDA error 变成可读 |
| `56af982` | (中间版本) `cudaFuncSetAttribute` 一次性 init 尝试；事后证明不是 root cause |
| `487d509` | (中间版本) `params.b<=0` 早返；保留为额外 guard |

**根因（debug 出来的）**：sglang 的 dsv4 NSA decode 路径在 tight KV pressure 下偶发把 `q.shape[0]` (= FlashMLA `params.b`) 推到 ~13K（推测：transient prefill+decode 混合 batch）。`get_decoding_sched_meta` 是单 block kernel，要 `4 * (b*5 + 1)` 字节 dynamic smem。b=13065 ⇒ smem=255 KB ⇒ 超过 **B300 (sm_100) 单 block dynamic smem 的 228 KB cap** ⇒ kernel launch 直接 `invalid argument`。

不是 `cudaFuncSetAttribute` 的问题（早期 hypothesis 错），不是 CUDA graph capture 的问题（`cudaStreamIsCapturing` 返回 `cudaStreamCaptureStatusNone`），就是物理 smem 容量越界。

不能在 FlashMLA 这个单 block 设计里修；要么 (a) 让 sglang dsv4 不要把 b 撑这么大（mixed batch chunking）、要么 (b) FlashMLA 重写成多 block。两个都超 paper 范围。

实际表现：sglang V4-Flash + cap 256K + HiCache OFF + harbor swebenchpro `-n 32`，起来跑 ~3-13 分钟 100% 复现。HiCache ON（Run C）同 cap 跑 38 分钟不崩——因为 HiCache 吸 eviction，让 scheduler 不用 retract+retry，b 始终在合理范围。

调试 commit 路径：v1 `487d509`（b≤0 guard）→ v2 `56af982`（怀疑 cudaFuncSetAttribute graph capture）→ debug 版（打印 b、smem、stream、capture state）→ 看到 `b=13065 smem=261304 capture_status=0` → 定位 smem cap → v3 `99cc18b`（友好报错 + 保留前面的 guard）。

直接重 build：
```bash
cd /scratch/yuzhou/projects/FlashMLA
git remote add rucnyz git@github.com:rucnyz/FlashMLA.git  # 首次
git fetch rucnyz && git checkout aginfer
source /scratch/yuzhou/projects/sglang/dev/aginfer/scripts/env.sh
TORCH_CUDA_ARCH_LIST="10.0" FLASH_MLA_DISABLE_SM90=1 \
  CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}" \
  pip install --no-build-isolation -v .
```

### sglang 侧（aginfer branch 4 commits on top of upstream PR #26062）

### sglang 侧（aginfer branch 4 commits on top of upstream PR #26062）

| commit | 说明 |
|---|---|
| `ab4acdeda` | `mooncake_store`: anchor 无 buffer (LogicalHostPool) 时 register 路径早返 |
| `8ae7eb271` | `swa_component`: `acquire/release_component_lock` 加 `lock_host=False` kwarg |
| `c4bdf1ba1` | `mooncake_store`: 给 V4 6 个 sidecar PoolName 加 key 后缀 |
| `4138a93d8` | `hybrid_cache_controller`: anchor 无 buffer 时跳过 super()._page_backup/_page_transfer |

### Mooncake 侧（rucnyz/Mooncake `aginfer` 分支）

Fork：**https://github.com/rucnyz/Mooncake/tree/aginfer**
- 基于 `upstream/main`，cherry-pick 上游 PR #2174 `[TE] Fix TCP connection pool SIGSEGV by deferring cleanup with asio::post`
- 起源：`ClientSession::writeBody` 的 use-after-free。上游 Issue #2145 已经报过（vLLM 用户报的），跟我们 V4 + 4-tier 的崩点签名完全一致

```bash
# 拿 fork（首次）
cd /scratch/yuzhou/projects
git clone --recurse-submodules git@github.com:rucnyz/Mooncake.git -b aginfer

# 用 agsched env 的 Python 重 build
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd Mooncake && mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python) -DPYTHON_EXECUTABLE=$(which python) ..
make -j 16

# 装到 agsched env 和 ~/.local/bin
SITE=/scratch/yuzhou/miniconda3/envs/agsched/lib/python3.12/site-packages/mooncake
cp $SITE/engine.cpython-312-x86_64-linux-gnu.so $SITE/engine.cpython-312-x86_64-linux-gnu.so.bak
cp $SITE/store.cpython-312-x86_64-linux-gnu.so  $SITE/store.cpython-312-x86_64-linux-gnu.so.bak
cp $SITE/libasio.so $SITE/libasio.so.bak
cp build/mooncake-integration/engine.cpython-312-x86_64-linux-gnu.so $SITE/
cp build/mooncake-integration/store.cpython-312-x86_64-linux-gnu.so  $SITE/
cp build/mooncake-common/libasio.so $SITE/

mkdir -p ~/.local/bin
cp build/mooncake-store/src/mooncake_master ~/.local/bin/
```

`env.sh` 把 `~/.local/bin` 加在 PATH 前面，`start_mooncake_master.sh` 自动用到 patched master。

**怎么验证 patch 生效**：起来后跑 2 个相同长 prompt（>page_size），第二个看 `#cached-token > 0` 且 `input throughput` 比第一个高 10x+。崩点是 `__memmove_avx512_unaligned_erms` in `ClientSession::writeBody` 异步回调，无 patch 时 backup_thread 第一次 batch_set 就会触发。

---

## 9. ThunderAgent (Run G) 集成

ThunderAgent 是一个 program-aware HTTP proxy/router，坐在 harbor 和 sglang 中间，按 `program_id` 跟踪每个 agentic program 的 KV-cache 用量，必要时 pause/resume 整个 program。我们用它作 paper §8 的 real-serving baseline (Run G in [`results/SUMMARY.md`](results/SUMMARY.md))。

### 架构

```
harbor docker container (per trial)
  → litellm.acompletion with extra_body={session_id: <uuid>, program_id: <uuid>}
    → POST http://172.17.0.1:9100/v1/chat/completions   [ThunderAgent TR mode]
      → forward to http://127.0.0.1:30000/v1/chat/completions   [sglang V4-Flash]
```

### 两个上游 bug

| Repo / fork | commit | 问题 | 修复 |
|---|---|---|---|
| `rucnyz/ThunderAgent aginfer` | `7bfb07e` | `ThunderAgent/__init__.py` eager `from .app import ...`，导致 `app.py` 顶层 `router = _create_router()` 在 `__main__.set_config()` **之前**就执行；CLI `--backends` / `--backend-type` 永远被吞 | router 创建延迟到 FastAPI startup hook |
| `rucnyz/harbor aginfer`       | `1c5a47b`, `7721898c` | (1) `lite_llm` 只把 `session_id` 塞进 `extra_body`，proxy 拿不到 `program_id`；(2) `terminus_2` 把 `session_id=None` 提前传给 LLM init（早于自身 UUID fallback 行），导致大部分调用 `self._session_id is None`，patch 完全不触发 | (1) 镜像 `session_id` 为 `extra_body.program_id`；(2) lite_llm 自己 mint UUID 兜底 |

### Run 起来

```bash
# 1. sglang V4-Flash 起来 (用 launch_sglang_v4flash.sh，需要 --enable-metrics)
MAX_TOTAL_TOKENS=524288 bash scripts/launch_sglang_v4flash.sh

# 2. ThunderAgent 起来 (port 9100，因为 host 上 9000 被别的服务占了)
TA_PORT=9100 bash scripts/launch_thunderagent.sh

# 3. harbor 打到 TA 而非 sglang (172.17.0.1:9100 = docker bridge 上的 host)
OPENAI_API_KEY=sk-fake-do-not-check harbor run \
  -p datasets/swebenchpro -a terminus-2 \
  -m openai/deepseek-ai/DeepSeek-V4-Flash \
  --ak api_base=http://172.17.0.1:9100/v1 --ak max_turns=200 -n 32
```

### 验证 router 真的 routing

`curl http://127.0.0.1:9100/programs` 应该看到 N 个 UUID program（非 `default`）。Run G 跑完看到 17 个 acting program，对应那 17 个成功的 trial。如果只看到 `default` 一个，说明 harbor 的 program_id 注入没生效，跑的等于裸 sglang + HTTP 转发（round 1 就是这个 bug）。

### Trial-count 偏差

Run F 29/32、Run G 17/32 都因为 swebenchpro docker compose 偶发失败 (不同的 trial set)。**wall-clock 比对受偏差污染**，per-trial mean (981 s vs 666 s) 才是干净的。详见 SUMMARY.md regime 2 部分。
