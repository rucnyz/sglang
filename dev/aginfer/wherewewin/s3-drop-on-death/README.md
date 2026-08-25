# S3 — Drop-on-death (compaction + program/sub-agent end)

**Distinctive driver:** KV that is provably dead. The system learns that a
session will never reuse its cache from a lifecycle event instead of inferring
death from recency.

Two triggers share this mechanism:

- **Context compaction:** an agent summarizes old turns and discards the source
  span. This still needs a `CONTEXT_COMPACTED` hook.
- **Program or sub-agent end:** a task emits `SESSION_END` or `SUB_RETURN` and
  its session-scoped KV becomes dead. The `SESSION_END` full pipeline is
  implemented and benchmarked.

## Implemented lifecycle path

On `SESSION_END`, the Dynamo router sends `POST /aginfer/session_end` to every
worker that served the program. Each worker fans the request to every DP rank;
the scheduler then waits for in-flight requests, native SGLang session locks,
and asynchronous HiCache work to drain across the TP/CP cache group. It removes
the ending program from shared nodes and deletes exclusively-held leaves
bottom-up, releasing their HBM and DRAM allocations immediately instead of
waiting for memory pressure or an LRU sweep.

The operation is idempotent and acknowledges completion only after every rank
has completed the same lifecycle barrier. `SUB_RETURN` can use this path when a
child is represented as its own terminal session. `CONTEXT_COMPACTED` still
needs a range-aware lifecycle signal and is not implemented.

## Why explicit reclamation can win

Vanilla SGLang with HiCache has no signal that a session is permanently dead.
LRU eventually evicts its KV under pressure, but until then those bytes can
displace a live, reusable prefix into a slower tier or out of the cache. The
explicit lifecycle path removes session ownership immediately and frees only
nodes with no remaining live holders. Shared prefixes remain intact.

ThunderAgent's program release affects router bookkeeping only; it does not
issue backend KV deletion. Dead-KV reclamation therefore remains reactive in
that baseline.

## Reproducible A/B benchmark

`deadkv_ab.py` is a Python-standard-library-only paired benchmark. It uses one
running binary for both conditions so that the treatment differs by one action:

- **Baseline:** complete the same sessions without sending `SESSION_END`; the
  native pressure/LRU path decides when to evict their KV.
- **Ours:** send `SESSION_END` after every terminal session.

The default order is ABBA. Each arm uses the same model process, allocator
capacity, prompt IDs, request order, concurrency, and sampling parameters. The
benchmark flushes the cache and verifies that it is empty before each arm.

The default workload keeps four live programs resident while four epochs each
create ten terminal programs. Direct mode uses page-aligned `input_ids` with no
globally shared pages by default. This isolates lifecycle reclamation without
making the baseline accumulate ever-growing holder sets. Use `--shared-pages`
for a separate shared-prefix safety experiment.

Run it only against a dedicated benchmark deployment because it calls
`flush_cache`:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/deadkv_ab.py \
  --backend direct \
  --server-url http://127.0.0.1:30001 \
  --artifact-dir /tmp/deadkv-ab-results \
  --confirm-dedicated-server
```

Run a small smoke workload first when validating a new server build:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/deadkv_ab.py \
  --backend direct \
  --server-url http://127.0.0.1:30001 \
  --repeats 1 --live-sessions 2 --epochs 1 --dead-per-epoch 2 \
  --shared-pages 0 --tail-pages 2 --max-tokens 1 \
  --retention-seconds 1 \
  --confirm-dedicated-server
```

`--repeats 1` produces AB rather than ABBA and is suitable only as a smoke
test. Use at least three paired comparisons for reported performance results.

An optional Dynamo mode validates the full external path:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/deadkv_ab.py \
  --backend dynamo \
  --frontend-url http://127.0.0.1:8000 \
  --worker-url http://127.0.0.1:8081 \
  --model deadkv-e2e \
  --prompt-mode text \
  --confirm-dedicated-server
```

Do not use Dynamo-mode performance as the primary isolated comparison: router
tracking and admission can react to the intentionally accumulating baseline
sessions. Its purpose is to cross-check the complete control path.

## Metrics and artifacts

Every run writes `summary.json`, one JSON file per arm, and a compact
`report.md`. With `--save-raw-states`, it also stores unit-level state snapshots.

- **Dead bytes:** physical bytes whose complete holder set consists only of
  programs declared terminal by the workload.
- **Dead-byte AUC:** time-integrated dead bytes, separately for HBM, DRAM, and
  DISK.
- **Occupancy:** allocator `pool_usage` bytes, not just radix-node accounting.
- **Reclaim latency:** time from END dispatch until state no longer contains
  the ended programs. A retained baseline is reported as right-censored.
- **Cache hit:** response `cached_tokens`, with state `hit_count` retained as
  supporting evidence.
- **TTFT:** the first output-token SSE event.
- **Throughput:** both inference-only throughput and pipeline throughput that
  includes END RPC time but excludes measurement polling.

Capacity fingerprints must be identical across all arms or the run fails.
Generate paired bootstrap intervals after a completed run with:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/analyze_results.py \
  /tmp/deadkv-ab-results/<run-id>/summary.json
```

The lightweight mock test exercises ABBA execution, reporting, dead-byte
accounting, bootstrap analysis, and the direct E2E verifier without GPUs:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/test_deadkv_ab.py
```

## AgentReplay high-pressure paired experiment

The completed d5 study and its Chinese interpretation are in
[`PRESSURE_RESULTS.zh-CN.md`](PRESSURE_RESULTS.zh-CN.md).

The synthetic A/B harness above is the first acceptance gate. The
AgentReplay tools in this directory provide a separate real-trace experiment
that asks whether early reclamation preserves a reusable live prefix under
allocator pressure.

This workflow requires an external AgentReplay checkout and a dedicated direct
SGLang deployment exposing `/generate`, `/aginfer/state`,
`/aginfer/session_end`, and `/flush_cache`. Configure the deployment separately
so that the baseline arm reaches roughly 80%--90% HBM-pool utilization before
the live probe. Machine-specific model launchers are intentionally not included.
Pool utilization is the maximum `used_bytes / cap_bytes` across ranks and
subpools, which prevents a constrained subpool from being hidden by aggregate
free capacity. Backend-reported per-tier `token_usage` is retained as supporting
telemetry.

First split a private four-step token trace into reusable live roots and one or
more terminal churn waves. A wave owns complete, distinct root/descendant
closures; records are never copied between waves:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/split_agentreplay_pressure_trace.py \
  --trace /path/to/private/source.jsonl \
  --out-dir /path/to/private/phases \
  --live-roots 8 \
  --terminal-waves 3
```

The generated JSONL files contain token and program IDs and must remain private.
With multiple waves, terminal files are named
`terminal-churn-wave-001.jsonl`, etc. The generated `manifest.json` contains
only counts, lengths, and hashes.

Do not create extra waves by replaying the same terminal JSONL with a different
salt or renamed program IDs. Radix entries are keyed by token sequence, so that
only adds holder metadata and does not add unique physical KV pressure. The
runner rejects repeated paths, overlapping program IDs, and byte-identical token
payloads. The splitter assigns distinct real root closures to each wave.

Run both arms with the same phase files, server configuration, and salt. The
baseline retains terminal sessions through the fixed barrier; Ours emits
`SESSION_END` for them after each wave. Both arms then probe the still-live
step-4 turns. In a multi-wave run, every terminal wave uses a distinct derived
session namespace, and each wave records both full-workload and live-only state. This shows the
first wave at which the baseline reduces live HBM/DRAM bytes. The accompanying
program count means only “has at least one holder”; it must not be interpreted as
complete-prefix retention. The runner flushes the cache during cleanup, so do
not point it at a shared deployment.

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_pressure_arm.py \
  --mode baseline \
  --live-seed /path/to/private/phases/live-seed.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-001.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-002.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-003.jsonl \
  --live-probe /path/to/private/phases/live-probe.jsonl \
  --out-dir /path/to/private/runs/deployment-a/baseline-r1 \
  --salt pair-01 --label baseline-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --url http://127.0.0.1:30001/generate \
  --seed-concurrency 4 --terminal-concurrency 8 --probe-concurrency 4 \
  --barrier-seconds 30

python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_pressure_arm.py \
  --mode ours \
  --live-seed /path/to/private/phases/live-seed.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-001.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-002.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn-wave-003.jsonl \
  --live-probe /path/to/private/phases/live-probe.jsonl \
  --out-dir /path/to/private/runs/deployment-a/ours-r1 \
  --salt pair-01 --label ours-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --url http://127.0.0.1:30001/generate \
  --seed-concurrency 4 --terminal-concurrency 8 --probe-concurrency 4 \
  --barrier-seconds 30
```

For a concurrency sweep, keep seed/probe concurrency fixed and vary only
`--terminal-concurrency`, placing each level in a separate model directory such
as `deployment-c4`, `deployment-c8`, and `deployment-c16`. A concurrency value
above the number of independently runnable roots in a wave does not create more
parallelism; each wave should therefore contain at least as many roots as the
largest tested concurrency. Keep each wave's marginal unique-KV budget similar
when comparing concurrency levels.

Use at least three pairs and alternate arm order, for example AB, BA, AB. Keep
each pair's salt identical between arms but use a fresh salt for the next pair.
The per-arm directories contain child logs and replay artifacts and therefore
remain private. `summary.json` is aggregate-only and stores a salt fingerprint,
not the salt or program IDs.

Generate a shareable aggregate report from the per-arm summaries:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/analyze_agentreplay_pressure.py \
  /path/to/private/runs/deployment-a \
  --out-dir /path/to/shareable/pressure-analysis
```

The analyzer opens only `baseline-rN/summary.json` and
`ours-rN/summary.json`. It validates exact replay completion, phase hashes,
configuration equality, cleanup, and paired salts; reports dead HBM/DRAM,
pool utilization, live-probe cache hit and TTFT, inference-only and full-pipeline
terminal throughput, and END latency; and emits bootstrap intervals once at
least three comparable pairs are available. Multi-wave summaries additionally
expose live holder presence plus HBM/DRAM byte retention after every wave. The
aggregate END mean is weighted by completed calls; cross-wave p50/p90 are not
fabricated, and the report instead labels the maximum observed per-wave p50/p90.

If AgentReplay emits a complete `cached_tokens_details` breakdown, the telemetry
sanitizer and analyzer retain aggregate device/host/storage hit ratios. The
current AgentReplay deployment must be configured or extended to request and
aggregate that field; without it, these columns remain blank and total cache hit
cannot distinguish HBM hits from DRAM hits. Partial breakdown coverage is
rejected rather than silently treated as zero.

The standard-library-only mock tests do not require a GPU or real trace:

```bash
python3 -m unittest \
  dev/aginfer/wherewewin/s3-drop-on-death/test_run_agentreplay_with_telemetry.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_split_agentreplay_pressure_trace.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_run_agentreplay_pressure_arm.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_analyze_agentreplay_pressure.py
```

## AgentReplay steady-state experiment

The phased pressure experiment proves that terminal KV can evict a later live
probe, but its two probe requests are too sparse to establish sustained
throughput.  The steady-state harness continuously admits new root sessions
through warmup, measurement, and cooldown.  Short sessions create terminal
churn; long-lived sessions revisit a still-live prefix at a fixed think time.
The same generated trace, salt, arrival schedule, and concurrency limit are used
for Baseline and Ours.

Generate the private steady trace once.  Cloned sessions receive a deterministic
token-space identity immediately after the source trace's real common prefix.
This prevents repeated templates from aliasing the same physical radix entries
while retaining genuine shared-prefix behavior:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/build_agentreplay_steady_trace.py \
  --source-trace /path/to/private/source.jsonl \
  --out-trace /path/to/private/steady.jsonl \
  --manifest /path/to/private/steady-manifest.json \
  --arrival-rate <sessions-per-second> \
  --warmup-seconds 300 --measurement-seconds 1800 --cooldown-seconds 300 \
  --live-fraction 0.25 --live-steps 4 --live-revisit-seconds 60 \
  --churn-gap-seconds 0 --identity-insert-offset 16384 \
  --max-request-tokens 64000 --seed 1
```

Churn replicas preserve each selected root's complete sub-agent closure and
parent/child blocking topology.  Live templates are childless roots so their
delayed turns unambiguously represent one reusable session.  Omit
`--identity-insert-offset` to use the exact common-prefix length detected from
the source; when the model trace has a known fixed system prefix, an explicit
offset (for example 16K tokens) makes the intended sharing boundary auditable.

Run paired arms on a dedicated direct-SGLang deployment.  The root-session
arrival process is open-loop; `--max-concurrency` is the common inference
admission cap.  A live session's later turns remain closed-loop after the prior
response plus its configured think time, matching agent behavior.  END control
requests have their own semaphore and never consume an inference slot.

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_steady_arm.py \
  --mode baseline \
  --trace /path/to/private/steady.jsonl \
  --manifest /path/to/private/steady-manifest.json \
  --out-dir /path/to/private/runs/baseline-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --salt pair-01 --label baseline-r1 \
  --url http://127.0.0.1:30001/generate \
  --max-concurrency 8 --session-end-max-concurrency 1 \
  --confirm-dedicated-server

python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_steady_arm.py \
  --mode ours \
  --trace /path/to/private/steady.jsonl \
  --manifest /path/to/private/steady-manifest.json \
  --out-dir /path/to/private/runs/ours-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --salt pair-01 --label ours-r1 \
  --url http://127.0.0.1:30001/generate \
  --max-concurrency 8 --session-end-max-concurrency 1 \
  --confirm-dedicated-server
```

Use a pilot to choose an offered rate that holds Baseline near the intended
80%--90% KV-pool pressure without unbounded admission delay, then freeze that
rate for every pair.  The generated manifest reports both scheduled request
rate and forced-output-token rate to make calibration explicit.  Use at least
three alternating pairs.

`summary.json` reports the fixed measurement-window metrics:

- completion-accounted output goodput and request rate, using the configured
  window rather than total drain time;
- start-cohort cache hit, TTFT, and E2E, separately for live-initial,
  live-revisit, and churn traffic;
- time-weighted Dead-HBM/DRAM bytes, dead-byte-seconds, allocator occupancy, and
  maximum subpool utilization sampled throughout the window;
- root-session admission delay, which reveals overload or a growing queue; and
- SESSION_END latency, control-queue delay, retry count, and sampled backlog.

Output tokens are charged at request completion.  The measurement window should
therefore be much longer than p99 request latency; warmup and cooldown provide
boundary guard bands.  The runner stores no prompt or token IDs, but the input
trace itself remains private.  It flushes the cache before and after every arm.

The pure scheduling/accounting tests do not need AgentReplay, a model, or GPUs:

```bash
python3 -m unittest \
  dev/aginfer/wherewewin/s3-drop-on-death/test_agentreplay_steady.py
```

## Direct full-pipeline acceptance test

`verify_dead_kv_e2e.py` is a stricter black-box acceptance test for a live
SGLang deployment. It verifies that:

- completing a response does not itself end the program's KV lifetime;
- two programs can share a prefix while retaining distinct tails;
- `SESSION_END(A)` receives an all-rank ACK and physically releases A-only
  HBM/DRAM allocations without damaging B's shared or exclusive units;
- a repeated END is an idempotent `already_absent` no-op;
- B still gets a cache hit after A is reclaimed; and
- generation and health checks continue to work after full cleanup.

By default the verifier calls `flush_cache`. Run it only on a dedicated server
and acknowledge that operation explicitly:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/verify_dead_kv_e2e.py \
  --base-url http://127.0.0.1:30001 \
  --artifact-dir /tmp/deadkv-e2e \
  --confirm-dedicated-server
```

`--skip-flush` is available for controlled environments where the caller has
already isolated cache state; the verifier still creates and ends its own
unique program IDs.

See [RESULTS.md](RESULTS.md) or the
[Chinese report](RESULTS.zh-CN.md) for the controlled synthetic GB300 result.
The follow-up [real Claude Code AgentReplay report](REALTRACE_RESULTS.zh-CN.md)
covers Qwen3.8-27B and DeepSeek-V4-Flash.
Raw logs, model weights, runtime archives, credentials, PID files, and
machine-specific launch scripts are intentionally not versioned here.

## Full 4-tier follow-up

The current physical lifecycle deletion covers radix-tree allocations in HBM
and DRAM. A complete four-tier experiment must also enable and pressure DISK so
that an unreclaimed corpse can push reusable live KV across the DISK-to-DROP
frontier. Per-key deletion from an external HiCache storage backend is a
separate follow-up; until that exists, do not claim that `SESSION_END` deletes
backing-store bytes.

## Honest status and falsification

The current results validate the `SESSION_END` trigger end-to-end for a
single-node TP4, PP=1, HBM+DRAM deployment, including a controlled workload and
a bounded real Claude Code trace. The implementation includes TP/CP
synchronization, shared-prefix preservation, deferred in-flight cleanup, and
duplicate delivery. It deliberately fails closed for PP>1 and decode-side
asynchronous KV offload. It does not yet validate `CONTEXT_COMPACTED`, SSD/NIXL
per-key deletion, multi-node disaggregation, or a production trace.

The claimed win is falsified for a workload if native LRU reclaims dead KV at
roughly the same time, or if earlier reclamation does not preserve a live cache
entry or improve a user-visible metric under controlled pressure. Low-pressure
runs should be reported too: they establish whether explicit END introduces a
regression when there is no eviction contention.
