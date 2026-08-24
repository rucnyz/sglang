# S3 SESSION_END result — 2026-08-23

## Result

On a controlled synthetic Dead-KV workload, explicit `SESSION_END` reclaimed
all measured dead KV and protected the live working set. The default-LRU arm
retained dead bytes through the observation window.

| Trial-level median | Default LRU | LRU + `SESSION_END` | Paired change |
|---|---:|---:|---:|
| Dead KV at observation end | 479.72 MiB | 0 | 100% removed |
| HBM used at observation end | 208.11 MiB | 0 | -208.11 MiB |
| DRAM used at observation end | 271.61 MiB | 0 | -271.61 MiB |
| Peak HBM used | 218.70 MiB | 169.31 MiB | -22.58% |
| Live-probe cached tokens | 0 / 768 | 704 / 768 | +91.67 pp |
| Live-probe TTFT | 1.2029 s | 0.1438 s | -88.06% |
| All-request TTFT p50 | 0.9690 s | 0.9369 s | -3.95% |
| All-request TTFT p95 | 1.9033 s | 1.8439 s | approximately -2.86% |
| Inference throughput | 1.8037 req/s | 1.9777 req/s | +9.88% |
| Pipeline throughput, including END | 1.8037 req/s | 1.9211 req/s | +6.44% |

The mean paired bootstrap 95% intervals across five pairs were:

- live-probe TTFT improvement: 87.96% to 88.15%;
- inference throughput change: +9.34% to +12.01%;
- pipeline throughput change: +5.59% to +9.54%;
- HBM dead-byte AUC reduction: 89.40% to 92.14%;
- DRAM dead-byte AUC reduction: 91.40% to 93.59%.

The treatment arm's median final-request HTTP latency was approximately
9.89 ms. State polling observed each wave fully reclaimed after approximately
0.299 s median, but the poll interval was 250 ms, so that value is an upper
resolution bound rather than allocator execution time.

## Experimental controls

- Run ID: `20260824T014610Z-f277a942`.
- SGLang base: `497dc27b7f42dbce233223345a2ccd350c058980` plus the
  Dead-KV working-tree patch used for this experiment.
- Source archive SHA-256: `b7f60a578d05ff6bae3ae8141323740794d4e11057ab324589f569b6ec9c88b4`.
- Experiment harness SHA-256: `32f22e805e7487b9525a2fae00e402e9f0ef0a901f36289b6f1e842e1bd4d318`.
- Hardware: one node with four NVIDIA GB300 GPUs, tensor parallelism 4.
- Model: Qwen3-0.6B.
- Cache: HBM and DRAM HiCache; HBM state-view capacity 236,716,032 bytes.
- Workload: four live sessions plus forty terminal sessions per arm, four-way
  request concurrency, 768 input tokens and at most eight output tokens per
  request.
- Repetitions: five paired comparisons, ten arms total, with alternating ABBA
  order.
- Both arms used the same patched SGLang process, model, allocator, request
  order, prompts, sampling configuration, and memory limit.
- The in-engine scheduling policy was disabled, custom policy modules were
  unset, and startup reported the default LRU and default write-through policy.
- The baseline omitted the lifecycle signal. The treatment sent
  `SESSION_END`; no other runtime setting differed.

The baseline is therefore a lifecycle-signal ablation on the same binary, not
an unrelated upstream build. This isolates the effect of terminal knowledge and
avoids binary/configuration drift.

## Low-pressure control

A separate two-live/four-terminal run peaked at roughly 56% HBM capacity.
Default LRU still retained 253.97 MiB of dead HBM+DRAM bytes after a 10.02 s
idle observation window, while `SESSION_END` reduced the measured dead bytes to
zero. Both conditions preserved 704 cached tokens and live TTFT was effectively
unchanged: 106.8 ms versus 107.1 ms. This run found no visible low-pressure TTFT
regression from explicit reclamation.

## Interpretation and limits

The result supports the intended mechanism: lifecycle knowledge can reclaim KV
before recency-based eviction, leaving fast-tier capacity for live sessions.
Under this workload, the internal reduction translated into higher cache reuse,
lower live-request TTFT, and higher throughput even after END overhead.

`/aginfer/state` reports one TP-rank logical view. The 479.72 MiB value is the
sum of 208.11 MiB HBM and 271.61 MiB DRAM in that view; it must not be multiplied
by four or presented as an exact cross-GPU physical total.

This is directional evidence from one small model, one synthetic trace, one
node, and five pairs. It is not a production-wide performance claim. Remaining
validation includes SSD/NIXL deletion, PP>1, DP/CP and multi-node deployments,
DeepSeek-V4, and a production AgentReplay trace. Raw run artifacts are retained
outside the source repository and are intentionally not committed.
