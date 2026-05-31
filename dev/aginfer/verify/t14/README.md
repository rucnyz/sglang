# T14 — State-dump cost observability (PLAN §2, sglang side)

PLAN.md §2 first item.  Per-call wall-clock for `dump_aginfer_state`,
emitted as a metric (histogram).  Sglang owns the metric (the
tree-cache lives in the scheduler process); the daemon and any human
monitor read it via `/aginfer/state`.

**Trigger condition** (PLAN T14): `p99 > 50 ms` on the state-dump
latency flips the F3-revisit task back to active — meaning the
state-dump path got too expensive and we need to revisit drop-on-full
vs coalesce vs incremental-state design.

## WHAT WE PROMISED

**On-the-wire piggyback.**  Every `/aginfer/state` response carries a
top-level `state_dump_metrics` field with the contract shape:

```json
{
  "state_dump_metrics": {
    "n_samples":         <int, current ring-buffer size>,
    "n_recorded_total":  <int, total dumps ever — grows by 1 per poll>,
    "capacity":          <int, ring buffer cap; default 1024>,
    "window_seconds":    <float, since first record() call>,
    "p50_ms":  <float>,
    "p95_ms":  <float>,
    "p99_ms":  <float>,
    "max_ms":  <float>,
    "mean_ms": <float>,
    "last_dump_ms":      <float, latency of the PREVIOUS dump>,
    "last_dump_bytes":   <int, serialised size; -1 sentinel for
                         dict-path callers — bytes-path is the HTTP
                         hot path and always reports a positive int>
  }
}
```

**Why piggyback** (not a separate endpoint).  The daemon already polls
`/aginfer/state` per-event; the metrics ride free.  Adding a separate
`/aginfer/metrics` would need a parallel IPC roundtrip
(`tokenizer_manager → scheduler → tree_cache`) for what is just a
small auxiliary dict.  PLAN's "~50 LoC instrumentation hook" envelope
fits the piggyback design; a separate-endpoint design would be ~150
LoC of req/response scaffolding.  Daemon ignores the new field
(its `build_paper_state` only consumes the keys it cares about; extras
are passed through harmlessly).

**Chicken-and-egg note.**  The summary embedded in dump-N reflects
samples 1..N-1 — dump-N's own latency lands in the ring buffer only
AFTER the response is built.  This means: `n_recorded_total` grows by
exactly 1 per `/aginfer/state` poll (verifiable by deltaing two
sequential polls).  Last_dump_ms in dump-N is dump-(N-1)'s latency.

**Ring buffer capacity** = 1024.  At the daemon's typical ~3.4 Hz
poll rate that's about 5 minutes of recent history — deep enough for
a stable p99, shallow enough that one stale outlier ages out within
the window so the F3-revisit trigger doesn't stick on a single past
spike.

**Hot-path overhead.**  ~50 ns per dump (one `perf_counter_ns()` pair,
one list append, one `len > cap` check, occasional `del [0]` shift on
wrap).  Summary computation is O(n log n) sort over the ring (1024
entries → ~50 μs Python).  Both negligible against the 50 ms
threshold the metric guards.

## WORST CASE

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Cold-start `/aginfer/state` (no dumps yet) | first poll on a fresh launch | `n_samples=0`, all quantile fields = 0.0, `last_dump_bytes=-1` | A0 |
| Ring buffer wrap | record 2000 samples cap=512 | window holds last 512; `n_recorded_total=2000` | A2 |
| Single slow outlier among many fast | record 99 × 1 ms + 1 × 1000 ms | p50≤p95≤p99≤max; max = 1000 ms; p99 captures the outlier | A3 |
| Mixed dict + bytes path | alternate `dump_bytes={-1, +N}` | mean aggregates BOTH latencies; last_dump_bytes = latest | A4 |
| Empty `link_stats` summary read | call `summary()` before any record | returns contract dict with 0.0/-1 sentinels, NEVER raises | A0 |
| HTTP `/aginfer/state` against live sglang | curl 30 times | each poll bumps `n_recorded_total` by exactly 1; bytes-path always positive | B1, B3 |

## HOW WE VERIFY

`verify/t14/verify.py` runs in two phases:

**Phase A** (in-process, no sglang needed): imports `_StateDumpMetrics`
directly from the sglang source tree and exercises the ring-buffer
contract.  Fast (<100 ms total) and isolated from CUDA / model load.

```
A0  empty summary returns contract keys, zero quantiles, -1 bytes
A1  record 10 known latencies; mean/max/last echo correctly
A2  record 2000 samples cap=512; n_samples=512, n_recorded_total=2000
A3  99 fast + 1 slow → p50≤p95≤p99≤max; max=outlier
A4  dict-path -1 sentinel + bytes-path mix; mean aggregates both
```

**Phase B** (live sglang; opt-in via `$AGINFER_VERIFY_BASE`): hits the
real `/aginfer/state` and verifies the piggyback contract end-to-end.

```
B0  GET /aginfer/state contains the state_dump_metrics key with the
    contract field set
B1  three sequential polls; n_recorded_total grows by exactly 1 per
    poll (validates the "summary in dump-N reflects N-1 samples"
    semantics)
B2  fire 40 polls; assert quantile monotonicity on real measurements
B3  bytes-path /aginfer/state must report last_dump_bytes > 0 (never
    the -1 sentinel)
```

Phase B is skipped (soft-pass) if the env var is unset.

## REPRODUCING

Phase A only (no sglang):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t14/verify.py
```

Phase A + B (live sglang):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang

# 1. sglang
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=6 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30002 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
  > /tmp/sglang_t14.log 2>&1 &
until grep -q "fired up" /tmp/sglang_t14.log; do sleep 5; done; sleep 8

# 2. verify
AGINFER_VERIFY_BASE=http://127.0.0.1:30002 \
  python dev/aginfer/verify/t14/verify.py

# 3. shutdown
pkill -9 -f sglang.launch_server
```

## RESULTS

**PASSED** — all 9 stages (5 Phase A + 4 Phase B against live sglang
on Qwen3-0.6B / GPU 6 / max-total-tokens 65 536 / HiCache write_through).

* date: 2026-05-31
* lines: ~85 in new `_StateDumpMetrics` class + 8 LoC threading the
  summary into the dict/bytes paths + 5 LoC `__init__` member — total
  ~100 LoC.  Matches PLAN's "~50 LoC instrumentation hook" envelope
  (the ~50 LoC overshoot is the standalone class definition for
  testability; the integration hook itself is ~13 LoC).

| Stage | Result |
|---|---|
| A0 empty summary | PASS — contract field set verified, all zeros + -1 bytes sentinel |
| A1 record + summary | PASS — mean=5.5ms, max=10.0ms, last echoes |
| A2 ring buffer wrap | PASS — n_samples=512 cap, n_recorded_total=2000, window holds the tail |
| A3 quantile monotonicity | PASS — outlier (1000ms) captured at max + p99 |
| A4 dict-path -1 sentinel | PASS — mixed +/- bytes; mean aggregates both paths |
| B0 piggyback field present | PASS — state_dump_metrics key on live /aginfer/state |
| B1 n_recorded_total deltas | PASS — exactly +1 per poll for both deltas |
| B2 live quantile monotonicity | PASS — n_samples≥20 after 40 polls; ordering holds |
| B3 bytes-path positive | PASS — last_dump_bytes > 0 (live: 1894 bytes for empty tree) |

**Live measurement** on cold tree (no chat traffic, idle scheduler):

```json
{
  "n_samples": 77,
  "n_recorded_total": 77,
  "capacity": 1024,
  "window_seconds": 2.190653544,
  "p50_ms": 0.058837,
  "p95_ms": 0.083008,
  "p99_ms": 0.197226,
  "max_ms": 0.623616,
  "mean_ms": 0.07040248,
  "last_dump_ms": 0.067121,
  "last_dump_bytes": 1894
}
```

Cold-tree p99 = **0.20 ms** — three orders of magnitude under PLAN's
50 ms F3-revisit threshold.  Live-traffic numbers will be higher
(more units to serialise) but the threshold is generous.

* raw log: `results/20260531_t14_initial_pass.log` (includes the
  live state_dump_metrics snapshot above)
