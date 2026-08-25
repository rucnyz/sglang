# Dead KV：真实 AgentReplay 高压力 A/B 实验

更新时间：2026-08-24（America/Los_Angeles）

## 实验目标

上一轮真实 trace 验证了 `SESSION_END` 能清零 Dead KV，但 KV pool
压力较低，无法证明它会改善后续仍存活 program 的 cache hit、TTFT 或
吞吐。本轮固定两个会继续访问的 live programs，并在它们的下一轮请求前
运行 terminal churn：

1. `live-seed`：两个 live program 运行前三轮，不发送 END；
2. `terminal-churn`：另外七个 program 完成四轮；Baseline 不发送
   END，Ours 发送 END；
3. 固定 30 秒 barrier；
4. `live-probe`：两个 live program 运行第四轮，测 cache hit、TTFT 和吞吐。

两臂使用相同模型、trace、请求顺序、salt、并发和缓存容量。所有输出继续
使用 teacher forcing，并要求请求成功率、输出长度和 token 精确匹配均为
100%。

## Qwen3.8-27B

配置：TP4，`max-total-tokens=40960`，3 组配对，顺序为 B/O、O/B、B/O。

| 指标（3 对均值） | Baseline | Ours | 变化 |
|---|---:|---:|---:|
| Probe 前 Dead HBM | 217.33 MiB | 0 | -100% |
| Probe 前 Dead DRAM | 217.33 MiB | 0 | -100% |
| Probe 前最紧张 HBM 子池 | 88.59% | 54.69% | -33.91 pp |
| Live-probe cache hit | 98.67% | 98.67% | 0 pp |
| Live-probe TTFT mean | 338.75 ms | 310.88 ms | -8.23% |
| Terminal inference throughput | 24.93 tok/s | 25.93 tok/s | +4.01% |

TTFT 三组的相对变化分别为 `+13.38%`、`+37.53%`、`-43.20%`（负数表示
Ours 更快）；平均 paired delta 为 `-27.87 ms`，95% bootstrap CI 为
`[-217.83 ms, +102.09 ms]`，因此不能声称稳定改善。Terminal throughput 三组均
偏向 Ours（`+2.78%`、`+1.16%`、`+8.40%`），平均 paired delta 为
`+1.00 tok/s`，样本内 95% CI 为 `[+0.30, +2.00] tok/s`。但只有 3 对，
且第三个 Baseline 较慢，证据仍弱。

Qwen 的 terminal `SESSION_END` 也是 21/21 成功、无重试；三组 mean 为
`3.38–4.15 s`，p90 约为 `8.25–8.41 s`。

Qwen 的关键结论是：压力与回收效果稳定，但 Baseline 在 88.59% 占用下仍
保留了两个 live prefix，因此 cache hit 没有分叉；没有 cache miss 差异时，
TTFT 只能反映运行噪声和较小的调度差异。

## DeepSeek-V4-Flash

配置：TP4，`max-total-tokens=64000`，`SWA_FULL_TOKENS_RATIO=0.7`，3 组配对，
顺序为 B/O、O/B、B/O。DeepSeek 是 FULL+SWA 混合缓存，压力以各 HBM
子池中最高的 `used/cap` 为准。

| 指标（3 对均值） | Baseline | Ours | 变化 |
|---|---:|---:|---:|
| Probe 前 Dead HBM | 8.41 MiB | 0 | -100% |
| Probe 前 Dead DRAM | 8.41 MiB | 0 | -100% |
| Probe 前最紧张 HBM 子池 | 84.57% | 50.86% | -33.71 pp |
| Live-probe cache hit | 99.13% | 99.13% | 0 pp |
| Live-probe TTFT mean | 453.59 ms | 451.09 ms | -0.55% |
| Terminal inference throughput | 12.93 tok/s | 12.93 tok/s | 0.00% |

TTFT 三组的相对变化分别为 `-0.81%`、`+0.48%`、`-1.33%`（负数表示
Ours 更快）；平均 paired delta 为 `-2.51 ms`，95% bootstrap CI 为
`[-6.07 ms, +2.19 ms]`。Terminal throughput 的平均 paired delta 为
`0.00 tok/s`，95% CI 为 `[-0.10, +0.20] tok/s`。这些都是性能持平，不是
稳定改善。

Ours 的 terminal `SESSION_END` 21/21 最终成功，但控制 RPC 长尾明显：3 组
mean 为 `8.51–11.39 s`，p90 为 `15.49–18.68 s`，共触发 4 次重试。这些
END 与其他推理并行，没有造成明显的 terminal throughput 回退，但仍是生产化
前需要优化的控制面长尾。

## 当前结论

这轮完成了 12 个正式 arm（每模型 3 对），共 432 个正式请求。请求成功率、
输出长度匹配和 forced-token 精确匹配均为 100%，最终 cache 均清空，服务已
停止。

能够确定的收益是内存回收：

- Qwen 的 Dead HBM/DRAM 从各 217.33 MiB 降为 0，最紧张 HBM 子池从
  88.59% 降为 54.69%；
- DeepSeek 的 Dead HBM/DRAM 从各 8.41 MiB 降为 0，最紧张 HBM/SWA
  子池从 84.57% 降为 50.86%；
- 两个模型的内存结果在 3 对实验中完全重现。

但这一 workload 下不能声称 cache hit、TTFT 或吞吐改善。Qwen 和 DeepSeek
的 live-probe cache hit 分别始终为 98.67% 和 99.13%，说明 Baseline LRU 在
84%–89% 压力下仍保住了这两个最近访问的 live prefix。既然没有额外
cache miss，Ours 提前释放的空间只是 headroom，不会自动转化成更快的 probe。

下一个性能验证应改变 churn 时序，而不是继续单纯增大占用：使 live programs
在 terminal churn 前更早暂停，再用多波新 session 把 Baseline 的 live prefix 真正挤出
快层。如果那时 Ours 仍保持命中，才能验证 TTFT/吞吐收益。

## 隐私与口径

- 报告只包含聚合数值、模型名和 trace hash，不包含原始 transcript、token IDs、
  program/session ID 或逐请求文本。
- HBM/DRAM 数值来自 `/aginfer/state` 的逻辑 scheduler view，不等同于 NVML
  进程显存。
- `sampled peak` 和子池利用率来自周期采样，可能低估瞬时峰值。
- 该 workload 使用真实 transcript 的四轮前缀和实验 oracle END，不代表线上可以自动
  从 transcript EOF 推断永久结束。
