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

First split a private four-step token trace into two reusable live roots and a
terminal churn set:

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/split_agentreplay_pressure_trace.py \
  --trace /path/to/private/source.jsonl \
  --out-dir /path/to/private/phases
```

The three generated JSONL files contain token and program IDs and must remain
private. The generated `manifest.json` contains only counts, lengths, and
hashes.

Run both arms with the same phase files, server configuration, and salt. The
baseline retains terminal sessions through the fixed barrier; Ours emits
`SESSION_END` for them. Both arms then probe the two still-live step-4 turns.
The runner flushes the cache during cleanup, so do not point it at a shared
deployment.

```bash
python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_pressure_arm.py \
  --mode baseline \
  --live-seed /path/to/private/phases/live-seed.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn.jsonl \
  --live-probe /path/to/private/phases/live-probe.jsonl \
  --out-dir /path/to/private/runs/deployment-a/baseline-r1 \
  --salt pair-01 --label baseline-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --url http://127.0.0.1:30001/generate \
  --barrier-seconds 30

python3 dev/aginfer/wherewewin/s3-drop-on-death/run_agentreplay_pressure_arm.py \
  --mode ours \
  --live-seed /path/to/private/phases/live-seed.jsonl \
  --terminal-churn /path/to/private/phases/terminal-churn.jsonl \
  --live-probe /path/to/private/phases/live-probe.jsonl \
  --out-dir /path/to/private/runs/deployment-a/ours-r1 \
  --salt pair-01 --label ours-r1 \
  --agentreplay-root /path/to/AgentReplay \
  --url http://127.0.0.1:30001/generate \
  --barrier-seconds 30
```

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
pool utilization, live-probe cache hit and TTFT, terminal throughput, and END
latency; and emits bootstrap intervals once at least three comparable pairs are
available.

The standard-library-only mock tests do not require a GPU or real trace:

```bash
python3 -m unittest \
  dev/aginfer/wherewewin/s3-drop-on-death/test_run_agentreplay_with_telemetry.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_split_agentreplay_pressure_trace.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_run_agentreplay_pressure_arm.py \
  dev/aginfer/wherewewin/s3-drop-on-death/test_analyze_agentreplay_pressure.py
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
