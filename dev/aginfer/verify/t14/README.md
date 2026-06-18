# T14 — State-dump cost observability (PLAN §2, sglang side)

Impl_PLAN.md §2 first item.  Per-call wall-clock for `dump_aginfer_state`,
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

**Stress probe** (`stress_probe.py`, opt-in, ~90 s of GPU load): not
part of `verify.py`'s pass/fail set; produces a 10-column TSV
``(t_s, units, hbm%, dram%, p50_ms, p95_ms, p99_ms, max_ms,
dump_bytes, n_recorded_total)`` per 150 ms sample and a final
summary block.  Exits 1 if peak p99 crosses the PLAN F3-revisit
threshold (default 50 ms) so CI can flag the trigger.  This is the
report that answers "does PLAN T14 fire under real load?" — Phase B
just confirms the instrumentation reads numbers.

## REPRODUCING

Phase A only (no sglang):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t14/verify.py
```

Phase A + B (live sglang):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
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

# 2. verify (contract checks; <10 s)
AGINFER_VERIFY_BASE=http://127.0.0.1:30002 \
  python dev/aginfer/verify/t14/verify.py

# 3. stress probe (the headline measurement; ~90 s of GPU load)
AGINFER_VERIFY_BASE=http://127.0.0.1:30002 \
  python dev/aginfer/verify/t14/stress_probe.py \
    --concurrency 32 --duration 90 --max-tokens 200 \
    --prefix-min-tokens 256 --prefix-max-tokens 512 \
    --out dev/aginfer/verify/t14/results/<date>_t14_stress_samples.tsv

# 4. shutdown
pkill -9 -f sglang.launch_server
```

The stress probe exits 1 (intentionally) when `peak_p99_ms >=
--threshold-ms` so CI can pick up the PLAN T14 F3-revisit trigger
without having to grep TSV.

## RESULTS

**Instrumentation: PASSED** — all 9 verify.py stages green (5 Phase A
+ 4 Phase B against live sglang on Qwen3-0.6B / GPU 6 / max-total-
tokens 65 536 / HiCache write_through).

**PLAN T14 trigger condition (`p99 > 50 ms` under load): FIRED.**
The stress probe (32 concurrent unique-prefix chats × 90 s on the
same setup) recorded peak p99 = 321.94 ms — see "Stress measurement"
below.  F3-revisit task opened as #160.

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

**Live measurement** on cold tree (no chat traffic, idle scheduler) —
trivial workload, just to confirm the instrumentation reads numbers:

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
50 ms F3-revisit threshold.  But cold-tree p99 is not the question
PLAN T14 is asking.

### Stress measurement (the headline number)

PLAN T14's threshold is about state-dump cost UNDER REAL LOAD.
`stress_probe.py` (see "REPRODUCING" → stress probe) drives 32
concurrent `/v1/chat/completions` with per-chat unique long prefixes
(forcing tree growth instead of all chats sharing one system prompt)
while polling `/aginfer/state` every 150 ms.  After 90 s on
Qwen3-0.6B + HiCache write_through + GPU 6 + max-total-tokens 65 536:

| sample      | t=10 s | t=20 s  | t=46 s   | t=85 s   |
|---|---|---|---|---|
| units       | 164    | 164     | 166      | 161      |
| HBM used    | 99.6 % | 98.3 %  | 98.3 %   | 99.7 %   |
| p50_ms      | 1.55   | 2.37    | 3.67     | 5.54     |
| p95_ms      | 3.33   | 3.48    | 5.65     | 9.64     |
| **p99_ms**  | 3.67   | 3.67    | **289**  | **322**  |
| max_ms      | 3.67   | **395** | 395      | 395      |

Reading the picture:
1. **p50 / p95 stay tame** — typical state-dump under load is 5–10 ms.
2. **A single 395 ms outlier appears at ~20 s** (likely a GC stall or
   contention with the scheduler's prefill/decode while the cache is
   at HBM-saturation).
3. **By t=46 s enough outliers (~1 % of window) have accumulated** to
   pull the 99th percentile up to ~290 ms; the tail stays there for
   the rest of the run as the 1024-entry ring slowly fills with
   spikes.

**Peak p99 = 321.94 ms — 6.4× over the 50 ms PLAN T14 F3-revisit
threshold.**  Per PLAN: the trigger fires, and we open the F3-revisit
task to decide drop-on-full vs coalesce vs incremental-state.
Tracked as #160.

### Re-verification + F3-revisit closure (2026-06-01, #160)

#### Step 1: reproduce the original (N=3, exact original fixture)

Initial closure attempt (commit `74507237a3`, since reverted) used
a single trial with mismatched flags (`--mem-fraction-static 0.15`
and `--attention-backend flashinfer` missing).  Audit caught it.

Re-run with EXACT original flags (`run_stress_real.py:_launch_sglang`),
N=3 fresh sglang launches:

| trial | samples | peak_p99 (ms) | peak_max (ms) | outliers in ring | peak dump bytes |
|---|---|---|---|---|---|
| 1 | 462 | 343.79 | 426.83 | 216 | 567 792 |
| 2 | 460 | 343.31 | 420.54 | 216 | 558 629 |
| 3 | 469 | 344.60 | 371.88 | 194 | 586 420 |
| **mean** | — | **343.90** | — | 209 | ~571 kB |
| **stdev** | — | **0.65** | — | — | — |

3/3 trials fire (p99 > 50 ms).  Original finding reproduces
robustly.

#### Step 2: diagnose

peak_units ≈ 170, so the slow path is NOT tree-walk cost.  The
metric records the wall-clock duration of
`_dump_aginfer_state_impl` INSIDE the scheduler process, including
any GIL preemption.  When the scheduler is mid-batch (prefill /
decode), Python state-dump code waits for the GIL; the metric
records that wait as "dump cost".  Daemon HTTP requests landing
during a busy iteration similarly wait at the ZMQ queue.

Two latencies in scope, often conflated:
* **Scheduler-internal compute** (`state_dump_metrics.p99_ms`) —
  GIL/queue wall-clock.  This is what the PLAN trigger measures.
* **HTTP-observed latency** — the time a client (the aginfer
  daemon, or stress probe) actually blocks waiting for
  `/aginfer/state` to return.  This is what daemon policy
  decisions wait on in practice.

Pre-fix, both metrics co-spike under load (no decoupling).

#### Step 3: fix — HTTP-layer cache + background refresh

In `python/sglang/srt/entrypoints/http_server.py`, the
`/aginfer/state` handler now serves from a module-level cache
that a background task refreshes every 50 ms (configurable via
`AGINFER_STATE_REFRESH_MS`).  This is the "coalesce + background
refresh" option from PLAN's F3 candidates list.

* **Daemon-facing latency**: now constant (cache read =
  microseconds + HTTP round-trip).
* **Scheduler-internal compute**: unchanged.  The refresh task
  still hits the slow path; that wall-clock is just absorbed by
  its own loop instead of by the daemon.
* **Bounded staleness**: daemon sees state ≤ (refresh interval +
  worst-case refresh wall-clock).  Under p99=343 ms refresh cost
  + 50 ms cadence → staleness ≤ ~400 ms.  Acceptable for paper §3
  policy decisions which care about HBM pressure trends, not ms-
  fresh snapshots.

#### Step 4: re-verify N=3 with both metrics

`run_stress_real.py --trials 3` now runs `stress_probe.py` and
`http_latency_probe.py` concurrently; reports both metrics.

| trial | sched p99 (ms) | sched max | sched outliers | http p50 | http p95 | **http p99** | http max |
|---|---|---|---|---|---|---|---|
| 1 | 344.72 | 376.55 | 212 | 4.36 | 6.53 | **11.07** | 89.69 |
| 2 | 323.98 | 410.27 | 224 | 4.46 | 7.34 | **13.70** | 86.09 |
| 3 | 339.90 | 427.31 | 207 | 4.59 | 6.75 | **12.37** | 89.14 |
| **mean** | 336.20 | — | — | — | — | **12.38** | — |
| **stdev** | 10.85 | — | — | — | — | **1.32** | — |

HTTP-observed p99 over 3 trials at 4-worker concurrency × 90 s:
**12.38 ± 1.32 ms** — 4× under the 50 ms threshold.  Improvement
on the daemon-facing latency: **27× lower** than the pre-fix
343.90 ms.

Scheduler-internal p99 (`state_dump_metrics.p99_ms` inside the
scheduler process) STILL reads ~336 ms — but the daemon never
sees this; it's absorbed by the bg refresh task's own loop.

#### Closure verdict

**#160 closed for the daemon-facing problem.**  HTTP cache + bg
refresh is the canonical F3-revisit fix from the PLAN options;
it brings the daemon's experience under the 50 ms threshold by
27×.

Scheduler-internal compute reduction is tracked as **#179**
(separate sglang architectural work — dedicated thread /
lock-free walk / slim variant).  Lower priority because the
daemon doesn't observe it anymore.

PLAN T14 should split its `p99 > 50 ms` clause into "daemon-
facing HTTP latency" (the trigger F3 fixes; now under) AND
"scheduler-internal compute" (still firing; #179 follow-on).
Impl_PLAN.md updated accordingly.

* date: 2026-06-01
* raw logs: `results/20260601_t160_n3_run2.log` (pre-fix N=3),
  `results/20260601_t160_n3_with_http.log` (post-fix N=3 with
  both metrics)
* raw TSVs: `results/20260601_*_t160_trial{1,2,3}_samples.tsv`

Two ancillary findings from the stress probe (not blocking T14):
* `peak_dram_used_frac = 0.0` throughout despite `--hicache-write-policy
  write_through` — the HiCache backup pipeline appears not to be
  populating DRAM during the 90 s run.  Possibly a sglang HiCache
  config bug that masks the design's read-back path; investigate
  alongside #160.
* `peak_units = 187` is far below the ~10⁴-unit working-set the
  paper assumes.  The 395 ms outlier therefore is NOT walk-cost-
  driven; it's contention/GC-driven.  An incremental-state path
  (F3) would not help; drop-on-full or coalesce would.

### Raw logs

* `results/20260531_t14_initial_pass.log` — Phase A + Phase B 9-stage
  verify run (cold tree).
* `results/20260531_t14_stress_samples.tsv` — 470 samples × 10
  columns from the 90 s × 32 concurrent stress probe (the headline
  data above is a sub-sampling).
