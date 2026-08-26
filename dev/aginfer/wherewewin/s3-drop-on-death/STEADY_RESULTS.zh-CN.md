# Dead KV：稳态与饱和吞吐实验

更新时间：2026-08-25（America/Los_Angeles）

## 结论摘要

本轮实验回答两个不同问题：

1. 在持续有新 session 到达、旧 session 结束、live session 再次访问的稳态
   workload 中，主动 `SESSION_END` 是否能持续回收 Dead KV；
2. 当请求队列持续有工作时，节省的 KV 与 prefill 重算能否转化为最大吞吐。

主要结论：

- Dead-KV 回收的功能与内存收益稳定成立。C8 三组正式配对中，HBM 和 DRAM
  dead-byte AUC 分别下降 `96.06%` 和 `98.29%`。
- 在固定到达率的 open-loop 实验中，Ours 显著提高 live cache hit 并降低 TTFT，
  但两臂完成相同的输出 token，goodput 相同。这说明低于容量上限时，吞吐主要由
  offered load 决定。
- 在固定 playlist 的 closed-loop C8 容量实验中，Ours 三组吞吐变化均为正，
  平均 inference throughput 提升 `2.85%`，END-inclusive pipeline throughput
  提升 `2.76%`。
- 在 C16 单组校准中，Ours 反而回退 `5.12%`。此时 Baseline 与 Ours 的 live
  cache hit 仅差 `0.13` 个百分点，说明瓶颈已经转向模型计算/decode，主动回收
  的边际收益不足以覆盖控制与缓存维护成本。

因此，Dead-KV 不是“并发越高越快”。目前测到的最佳区域是：请求队列有足够
工作、dead KV 会挤压可复用 live KV，但 GPU 尚未进入纯计算饱和的中等并发。

## 实验配置

- 机器：单机 4×NVIDIA GB300
- 模型：Qwen3.8-27B
- 推理：TP4，PP1，direct SGLang `/generate`
- 缓存：Unified Radix Cache，HBM+DRAM HiCache，write-through
- `max-total-tokens=40960`
- `max-running-requests=16`
- page size：64
- Baseline：原生 LRU，不在 workload 内发送 `SESSION_END`
- Ours：program 完成后异步发送 `SESSION_END`
- 所有请求使用 teacher forcing，并要求请求成功、输出长度和 token 精确匹配

输入来自私有 Claude Code trace 的 Qwen token 化版本。稳态生成器保留真实 turn
前缀，并在公共 16K-token 前缀之后插入确定性 identity token，使不同 replica
产生不同的物理 KV，同时仍共享真实公共前缀。原始 transcript、token IDs、
program IDs 和逐请求日志不进入版本库。

## 稳态运行发现的正确性问题

最初的稳态 Ours 运行中，`SESSION_END` 会在持续流量下超时：结束 program 与
其他 live program 共享的 16K 前缀一直处于 path-lock 或 write-through 状态，
旧的 preflight 把这些共享节点也视为物理删除 blocker。实际上，结束一个 holder
时共享节点只需删除 holder 元数据，不会释放其 KV allocation。

修复后，preflight 只对最后一个 holder 所拥有的独占节点检查 lock 和异步缓存
操作；共享节点可立即解绑。独占节点的 HBM/DRAM 释放安全检查、TP/CP all-rank
barrier 和幂等语义保持不变。

修复验证：

- 相关 SESSION_END/FanOut 测试：29 passed，另有 9 个 subtests passed；
- 修复前 C1/C4 稳态 canary 均会长时间无 ACK；
- 修复后相同 canary 中 39/39 END 成功、0 retry，平均约 206 ms，最大约
  1.19 s，最终 backlog 和 dead bytes 均为 0。

对应修复提交为 `5721eea`（`fix(aginfer): ignore shared nodes in end preflight`）。

## Open-loop 稳态结果

该实验按固定墙钟时间持续引入 session，live program 每 180 秒重访。每个 arm
包含 420 个请求；B/O 各一轮均成功且 exact。

| 指标 | Baseline | Ours | 变化 |
|---|---:|---:|---:|
| Measurement goodput | 23.612 tok/s | 23.612 tok/s | 0% |
| Live cache hit | 77.17% | 97.89% | +20.72 pp |
| Live TTFT mean | 892 ms | 707 ms | -20.7% |
| Dead HBM AUC | 138,911 MiB·s | 1,177 MiB·s | -99.15% |
| Dead DRAM AUC | 247,809 MiB·s | 1,177 MiB·s | -99.53% |

两臂在 900 秒测量窗内都完成了 222 个请求、21,251 个输出 token，因此
`21,251 / 900 = 23.612 tok/s`。这不是服务器最大容量，而是 workload 的固定
到达率。Ours 节省的 prefill 反映在 cache hit 和 TTFT 上，但当两边都能跟上
输入速率时，不会自动产生更高 goodput。

## Closed-loop C8 正式结果

容量实验使用固定 420-request playlist。请求不再等待墙钟 arrival time；最多
8 个 inference worker 从 ready queue 取请求，一个完成后立即补位。Live turn
按照固定的 churn-program completion distance 重新变为 eligible。每个 arm
使用 fresh server，并先执行一个不计分的预热请求和 cache flush。

三组顺序为 B/O、O/B、B/O。6/6 arms 均 valid；每个 arm 420/420 请求成功且
forced-token exact。

| Pair | Baseline inference | Ours inference | Ours 变化 |
|---|---:|---:|---:|
| 1 | 62.790 tok/s | 65.394 tok/s | +4.15% |
| 2 | 65.462 tok/s | 65.583 tok/s | +0.18% |
| 3 | 65.334 tok/s | 68.083 tok/s | +4.21% |
| 配对均值 | 64.529 tok/s | 66.353 tok/s | +2.85% |

| 指标（3 对均值） | Baseline | Ours | 变化 |
|---|---:|---:|---:|
| Inference throughput | 64.529 tok/s | 66.353 tok/s | +2.85% |
| Pipeline throughput（含 END） | 64.490 tok/s | 66.260 tok/s | +2.76% |
| Inference makespan | 618.40 s | 601.38 s | -2.75% |
| Live cache hit | 85.85% | 87.73% | +1.87 pp |
| Live TTFT mean | 815.7 ms | 780.0 ms | -4.38% |
| Dead HBM AUC | 45,634 MiB·s | 1,797 MiB·s | -96.06% |
| Dead DRAM AUC | 105,104 MiB·s | 1,799 MiB·s | -98.29% |
| Mean HBM subpool utilization | 93.74% | 88.94% | -4.80 pp |
| Mean DRAM subpool utilization | 96.67% | 90.65% | -6.02 pp |

三组 inference throughput 的方向全部为正。以三个 pair 为独立样本计算，平均
提升为 `2.85%`，中位数为 `4.15%`；小样本 paired t 95% CI 约为
`[-2.88%, +8.57%]`，因此尚不能称为统计显著结论。该结果也略低于预先设定的
`>=3%` 强结论门槛，应该表述为“小幅、可复现的正向证据”。

Live TTFT 的跨 pair 方向并不完全一致；均值下降约 4.4%，但不应像阶段式高压
probe 那样宣称稳定的大幅下降。

Ours 共完成 315/315 个 `SESSION_END`，无最终失败；3 次瞬时失败均在重试后
成功。END 平均延迟约 282 ms，最大约 2.07 s，最终 backlog 和 cache 均清空。

## Closed-loop C16 校准

C16 使用同一个固定 playlist，只将客户端 inference concurrency 从 8 改为 16；
当前只有一组配对，因此只用于寻找并发拐点。

| 指标 | Baseline | Ours | 变化 |
|---|---:|---:|---:|
| Inference throughput | 92.385 tok/s | 87.659 tok/s | -5.12% |
| Pipeline throughput | 92.021 tok/s | 87.469 tok/s | -4.95% |
| Inference makespan | 431.78 s | 455.06 s | +5.39% |
| Live cache hit | 87.260% | 87.393% | +0.13 pp |
| Dead HBM AUC | 6,320 MiB·s | 1,817 MiB·s | -71.3% |
| Dead DRAM AUC | 37,329 MiB·s | 2,870 MiB·s | -92.3% |

C16 时两臂都接近计算饱和，Baseline 的高内存压力也会主动触发 LRU，因而留下的
dead HBM 已明显少于 C8。两臂 live cache hit 几乎相同，主动回收没有省下足够
prefill；此时控制 RPC、树更新和 allocator 操作的成本可能超过收益。因为只有
一对，`-5.12%` 仍需交错复验，不能推广为所有高并发配置都会回退。

## 什么条件下有吞吐收益

当前结果支持以下因果条件：

1. 请求队列必须持续有工作。低于容量时，完成吞吐等于 offered load，节省的
   计算只改善延迟。
2. Baseline 中必须积累足够多的 dead KV，且这些 KV 会挤掉仍会重访的 live
   prefix。仅增加占用、但 live prefix 仍全部命中时，不会产生性能收益。
3. Prompt/prefill 成本必须占有可观比例。Dead-KV 不减少既定输出 token 的
   decode FLOPs。
4. 并发不能高到纯计算饱和。当前配置下 C8 是已测到的 sweet spot；C16 已转为
   compute/decode-bound。
5. `SESSION_END` 服务率必须跟得上结束速率。共享前缀 blocker 修复后，C8 正式
   workload 的 END backlog 可排空，pipeline 吞吐仍保持正向。

## 限制与下一步

- C8 只有 3 个 pair，置信区间仍较宽；建议扩展到至少 6 对。
- C16 只有 1 个 pair；需要反向顺序复验才能确认并发拐点。
- Closed-loop workload 是从真实 trace 构造的固定容量 playlist，保留真实 token
  前缀和输出，但不再复现原始 wall-clock 到达时间或完整 root/sub-agent 阻塞。
- 实验使用 direct SGLang；Dynamo full-pipeline 已做功能测试，但尚未做同规模的
  steady capacity benchmark。
- d3 的 VT snapshot 每两分钟检查一次。正式 C8 中 B2 和 O3 各重叠一次约
  650 MiB home 打包，其余检查约 1 秒；干扰大致平衡，但仍应在复现实验中消除。
- 当前覆盖 HBM+DRAM；SSD/NIXL per-key delete 和 `CONTEXT_COMPACTED` 仍不在
  本实验范围。

可对外使用的谨慎结论是：

> 在 4×GB300、Qwen3.8-27B、TP4、HBM+DRAM、固定 C8 closed-loop workload
> 下，主动 SESSION_END 将 dead-KV AUC 降低约 96%–98%，并在三组配对中均
> 提高吞吐，平均约 2.85%。当并发提高到 C16 时该收益没有保持，表明 Dead-KV
> 的吞吐收益来自缓解 KV/prefill 瓶颈，而不是普遍加速 decode。
