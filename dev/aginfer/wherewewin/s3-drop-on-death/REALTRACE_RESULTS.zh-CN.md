# S3 SESSION_END：真实 Claude Code AgentReplay 结果

实验日期：2026-08-24

## 摘要

在单节点 4×GB300、TP4、HBM+DRAM HiCache 上，我们用真实 Claude Code transcript 的固定子集，对 Qwen3.8-27B 和 DeepSeek-V4-Flash 各运行了两组配对 A/B：

- Baseline：默认 LRU，不发送生命周期结束信号；
- Ours：相同 LRU，在 program 的最后一轮完成后发送 `SESSION_END`。

共完成 8 个 run、288 个推理请求。288/288 请求成功，teacher-forced token 与长度均 100% 精确匹配；Ours 的 36/36 个 `SESSION_END` 最终成功。

最明确的结果是内存回收：

| 模型 | Baseline 最终 Dead HBM / DRAM | Ours | Sampled peak HBM | Sampled peak DRAM |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 575.14 / 575.14 MiB | 0 / 0 | -6.93% | -6.78% |
| DeepSeek-V4-Flash | 21.24 / 21.24 MiB | 0 / 0 | -7.97% | -7.38% |

因此可以声称：**真实 trace 上的 Dead KV 最终被全部清除，sampled peak KV-pool 占用下降约 7%，且没有观察到 cache-hit 损失。**

不能声称稳定的请求性能提升。Qwen 两个 pair 的吞吐变化方向相反（+22.45%、-3.28%）；DeepSeek 第一对受冷态/JIT 噪声影响，第二对吞吐持平、TTFT mean 改善 2.18%。每个模型只有两对实验，不足以建立统计显著性。

## Trace 来源和转换

Trace 来自 gated private Hugging Face dataset `UCSB-SURFI/claude-code-traces`：

- 固定 revision `5b9f36fb6ef1e31ba7db4d60e9184cb19574fe71`；
- 对应 dataset PR #8 合并版本；
- 只使用 `data/contrib/cc_swe_bench/**`；
- 确定性选择 8 个 DAG-clean root session，并纳入四轮前缀内 spawn 的后代；
- 最终为 8 个 root program、1 个 linked sub-agent、36 个请求。

Qwen 和 DeepSeek 分别用对应 serving checkpoint 的 tokenizer/renderer 生成独立 token trace。两份 trace 共享逻辑 DAG、turn 边界和时序，但 token IDs 与长度是 model-specific 的，不能跨模型混用。

| Trace | Prompt tokens | Output tokens | 最大单请求 | SHA256 |
|---|---:|---:|---:|---|
| Qwen3.8-27B | 642,924 | 3,145 | 23,295 | `45f5cc7202f23ae8307e67256f48b5b8f69079b9ebd477ce3a4403ce7c3894c6` |
| DeepSeek-V4-Flash | 644,544 | 3,874 | 22,958 | `926b29478aea3fd4b7229e4d6c5dffa91afa106267d80e955c48e3a9a92c2ddd` |

两份 trace 的 topology SHA256 均为 `3535296e79b9620e171b07105ae9a036314cf6f73e6e3793645f023391fc8472`，每份 27 个跨轮 prefix-continuity 检查全部通过。

## 实验控制

- 硬件：单节点 4×NVIDIA GB300，TP=4、DP=1、PP=1；
- SGLang commit：`e45f330690f2c93aa534a140e539d157e5bb5cc4`；
- context length 131,072；`max-total-tokens=262144`；`max-running-requests=16`；
- HBM+DRAM HiCache ratio 1.2，`page_first_direct`，direct I/O，write-through；
- replay concurrency 4，root stagger 0.5 s；
- tool gap 按 0.01 倍压缩并封顶 2 s；
- 每个 arm 使用 fresh server 并 flush cache；
- 两个 arm 均关闭 aginfer in-engine policy，启动日志确认使用 `default_lru`；
- 同一个 pair 的 trace、salt、请求顺序和资源限制完全相同；唯一 treatment 是 Ours 发送 `SESSION_END`；
- END 使用独立 control semaphore，不占用推理并发槽。

Baseline 是同一份 patched binary 上的 lifecycle-signal ablation，而不是不同 binary 的比较。

## 内存与 cache hit

下列 byte 数来自 `/aginfer/state` 的 TP-rank logical scheduler view，不能直接乘以 4 当作四卡物理总量。Peak 由 1 秒状态采样得到，可能低估瞬时峰值。KV pool 是预分配的，因此 block 回收后 NVML 进程显存不一定下降。

| 模型 | 指标 | Baseline | Ours | 变化 |
|---|---|---:|---:|---:|
| Qwen | Final dead HBM | 575.14 MiB | 0 | -100% |
| Qwen | Final dead DRAM | 575.14 MiB | 0 | -100% |
| Qwen | Sampled peak HBM | 577.14 MiB | 537.13 MiB | -6.93% |
| Qwen | Sampled peak DRAM | 575.14 MiB | 536.13 MiB | -6.78% |
| Qwen | Cache hit | 91.58% | 91.55% | -0.03 pp |
| DeepSeek | Final dead HBM | 21.24 MiB | 0 | -100% |
| DeepSeek | Final dead DRAM | 21.24 MiB | 0 | -100% |
| DeepSeek | Sampled peak HBM | 35.79 MiB | 32.94 MiB | -7.97% |
| DeepSeek | Sampled peak DRAM | 21.24 MiB | 19.68 MiB | -7.38% |
| DeepSeek | Cache hit | 94.05% | 94.05% | 0 pp |

Baseline 在 workload 完成后的 10 秒观察窗口中仍保留这些 Dead KV，因此它的回收延迟只能记作 right-censored ≥10 秒，不能解释为永远不会回收。

## 延迟和吞吐

| 模型 / Pair | Baseline TTFT mean | Ours TTFT mean | Baseline tok/s | Ours tok/s |
|---|---:|---:|---:|---:|
| Qwen r1 | 1,373.31 ms | 611.27 ms | 24.50 | 30.00 |
| Qwen r2 | 484.51 ms | 522.34 ms | 30.50 | 29.50 |
| DeepSeek r1 | 4,240.65 ms | 1,010.00 ms | 10.30 | 14.20 |
| DeepSeek r2 | 983.25 ms | 961.85 ms | 14.40 | 14.40 |

Qwen r1 和 DeepSeek r1 的数值偏向 Ours，但各自的 r2 并未复现大幅收益。特别是 DeepSeek r1 明显受首次运行/JIT 或系统冷态影响。当前结果支持内存收益，不支持把两对均值当作稳定 TTFT 或吞吐收益。

## SESSION_END 延迟

| 模型 | 最终成功 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 18/18 | 7.89 ms | 9.84 s | 15.76 s |
| DeepSeek-V4-Flash | 18/18 | 14.47 ms | 18.71 s | 23.08 s |

36 次 END 共发生 5 次 retry attempt，最终没有 deferred、error 或 remaining node。p50 很低，但 p90/p99 有明显秒级长尾。这里测量的是 control RPC 到全 rank 确认回收完成的 ACK 延迟，不是纯 allocator free 时间；长尾是生产化前需要继续优化的主要问题。

## 限制

- 这是每个 program 最多 4 个逻辑 turn 的真实会话前缀，不是完整 session EOF；
- 原始数据不含完整 system prompt/tool schema，转换器使用简化 header 并填充至 16K system tokens；
- 数据没有可恢复的 hidden thinking tokens；
- teacher forcing 不执行工具、补丁或测试，不衡量自由生成质量与 SWE-bench 成功率；
- 四轮边界的 `SESSION_END` 是具有完美结束知识的实验 oracle，不代表生产系统能自动识别 session death；
- tool gap 被压缩，因此不是原始 wall-clock agent workload；
- 每模型只有两对实验，且存在冷态噪声；
- 本次只验证单节点 TP4、PP1、HBM+DRAM，不覆盖 SSD/NIXL per-key deletion、PP>1、decode async offload 或多节点 P/D。

## 当前判断

**Dead-KV 方案的内存正确性与收益已被真实 trace 验证；性能影响目前总体中性且方差较大，SESSION_END 秒级长尾仍需优化，暂不宣称吞吐或延迟胜出。**

下一步应在一致预热后运行至少 5 组 ABBA，并增加 live-prefix eviction pressure，验证提前回收能否稳定转化为 cache hit、live TTFT 和 goodput 收益。

原始 transcript、派生 token trace、token IDs、program/session identifiers、模型权重、凭据和原始日志均未提交。
