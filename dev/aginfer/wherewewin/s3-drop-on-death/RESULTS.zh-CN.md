# S3 SESSION_END 实验结果 — 2026-08-23

[English version](RESULTS.md)

## 结论

在受控的合成 Dead-KV workload 中，显式发送 `SESSION_END` 能回收所有测得的 Dead KV，并保护仍然存活的 working set。默认 LRU 在观察窗口结束时仍保留大量已经不会再被访问的 KV。

| Trial-level 中位数 | 默认 LRU | LRU + `SESSION_END` | 配对变化 |
|---|---:|---:|---:|
| 观察结束时的 Dead KV | 479.72 MiB | 0 | 清除 100% |
| 观察结束时 HBM used | 208.11 MiB | 0 | -208.11 MiB |
| 观察结束时 DRAM used | 271.61 MiB | 0 | -271.61 MiB |
| Peak HBM used | 218.70 MiB | 169.31 MiB | -22.58% |
| Live-probe cached tokens | 0 / 768 | 704 / 768 | +91.67 个百分点 |
| Live-probe TTFT | 1.2029 s | 0.1438 s | -88.06% |
| 所有请求 TTFT p50 | 0.9690 s | 0.9369 s | -3.95% |
| 所有请求 TTFT p95 | 1.9033 s | 1.8439 s | 约 -2.86% |
| 推理吞吐 | 1.8037 req/s | 1.9777 req/s | +9.88% |
| 包含 END 的完整 pipeline 吞吐 | 1.8037 req/s | 1.9211 req/s | +6.44% |

五组配对实验的 paired-mean bootstrap 95% 置信区间为：

- live-probe TTFT 改善 87.96%–88.15%；
- 推理吞吐变化 +9.34%–+12.01%；
- 包含 END RPC 的完整 pipeline 吞吐变化 +5.59%–+9.54%；
- HBM Dead-byte AUC 降低 89.40%–92.14%；
- DRAM Dead-byte AUC 降低 91.40%–93.59%。

实验组 final 请求的 HTTP 延迟中位数约为 9.89 ms。状态轮询观察到每一波 KV 完全回收的中位时间约为 0.299 s，但轮询间隔为 250 ms，因此这个数值受采样分辨率限制，不能当作 allocator 的实际执行时间。

## 实验控制

- Run ID：`20260824T014610Z-f277a942`。
- SGLang 基线：`497dc27b7f42dbce233223345a2ccd350c058980`，加上本次实验使用的 Dead-KV patch。
- 实验时 source archive SHA-256：`b7f60a578d05ff6bae3ae8141323740794d4e11057ab324589f569b6ec9c88b4`。
- 实验 harness SHA-256：`32f22e805e7487b9525a2fae00e402e9f0ef0a901f36289b6f1e842e1bd4d318`。
- 硬件：单节点 4× NVIDIA GB300，tensor parallelism 为 4。
- 模型：Qwen3-0.6B。
- Cache：HBM + DRAM HiCache；HBM state-view 容量为 236,716,032 bytes。
- Workload：每个 arm 包含 4 个长期 live session 和 40 个 terminal session；并发度为 4；每个请求输入 768 tokens，最多输出 8 tokens。
- 重复次数：5 组 paired comparison，共 10 个 arms，顺序按 ABBA 交替。
- 两组使用同一个 patched SGLang 进程、模型、allocator 容量、请求顺序、prompt、采样配置和显存限制。
- 两组均关闭 in-engine scheduling policy，并取消自定义 policy module；启动日志确认使用默认 LRU 和默认 write-through policy。
- Baseline 不发送生命周期结束信号；实验组发送 `SESSION_END`，其他运行参数不变。

因此，这里的 baseline 是同一个二进制上的 lifecycle-signal ablation，并不是与另一个 upstream build 比较。这样可以隔离 terminal lifecycle knowledge 的影响，避免二进制或配置差异成为混淆变量。

## 低压力对照

另一组低压力实验使用 2 个 live session 和 4 个 terminal session，Peak HBM 约为容量的 56%。

- 默认 LRU 在 10.02 s 空闲观察窗口结束时仍保留 253.97 MiB Dead HBM+DRAM KV。
- `SESSION_END` 将测得的 Dead KV 降为 0。
- 两组均保留 704 cached tokens。
- Live TTFT 分别为 106.8 ms 和 107.1 ms，基本相同。

这说明在没有 eviction contention 的情况下，本次实验没有观察到显式回收带来的明显 TTFT 回退。Baseline 的回收时间属于 right-censored：只能说明它在 10.02 s 内没有回收，不能解释为永远不会回收。

## 原理解释

默认 LRU 只能根据“最近是否访问”判断淘汰顺序，不知道一个 session 是否已经永久结束。因此，刚完成的 terminal session 即使永远不会复用，其 KV 仍可能因为刚被访问而长时间留在 HBM 或 DRAM 中，并把真正会复用的 live prefix 挤出快速层。

`SESSION_END` 提供了确定的生命周期信息：

1. 从 radix tree 中移除已结束 program 的 holder；
2. 保留仍由其他 live program 引用的共享 prefix；
3. 对没有任何 live holder 的独占节点，从叶到根释放 HBM/DRAM KV；
4. 等所有相关 rank 完成后再返回 ACK。

在本 workload 中，这个机制释放了 Dead KV，保住了 live working set，因而同时改善 cache hit、live TTFT 和吞吐；即使计入 END RPC 开销，完整 pipeline 吞吐仍然提高。

## 指标口径

`/aginfer/state` 当前报告的是一个 TP-rank logical view。479.72 MiB 是该视图中 208.11 MiB HBM 与 271.61 MiB DRAM 的总和，不能直接乘以 4，也不能表述为四张 GPU 的精确物理总量。

“Dead KV 清除 100%”仅指观察结束时可证明已经死亡的 KV，不代表整个模型进程的显存占用下降 100%。SGLang 会预分配 KV pool，因此 NVML 所见的进程显存通常不会随 block 回收而同步下降。

704 / 768 是 live-probe 的缓存命中，不是所有请求的全局 cache hit rate。

## 局限与后续工作

当前结果是一个小模型、一个合成 workload、一个节点和五组 paired trials 上的方向性证据，不是对所有模型服务 workload 的普遍性能结论。

尚未覆盖：

- SSD/NIXL per-key deletion；
- PP>1；
- decode-side asynchronous KV offload；
- DP/CP 和多节点 P/D disaggregation；
- Router 重启后的 lifecycle mapping 恢复；
- DeepSeek-V4 checkpoint；
- 真实 AgentReplay/生产 workload；
- 更长时间的稳定性与泄漏测试。

原始运行日志和大体积 state artifacts 保存在源码仓库之外，没有提交模型、凭据、PID、runtime archive 或机器专用启动脚本。
