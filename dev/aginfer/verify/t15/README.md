# T15 — Hint-table cross-rank divergence (PLAN §2, DESIGN §6)

The eventual-consistent hint table (DESIGN §6 "Hint consistency")
is justified by "eviction is the cross-rank sync point": two ranks
may read different hint versions during PUT propagation, but the
NEXT eviction acts as a synchronisation barrier — both ranks
observe the eviction and rescore.  The argument FAILS if two
ranks ever evict DIFFERENT units in the same window.

## STATE OF THE WORLD (2026-06-01)

T15 has two halves; the second is genuinely gated on a sglang
implementation gap (not on "just run it").

| half | needs | status |
|---|---|---|
| (1) Detector tool + synthetic-data verify | nothing | **DONE — this verify (11 stages)** |
| (2) Real TP > 1 churn run against detector | sglang patch — expose per-TP-rank state pre-aggregation (the existing endpoint aggregates inside `http_server.py`) | **BLOCKED on sglang patch (task #174)** |
| (intermediate) Real DP > 1 churn run | nothing | **DONE — `run_dp2_real.py` (102 snapshots parsed without raising)** |

The intermediate DP run exists for two reasons:
1. **Detector contract green against real JSON** — proves the
   parser handles `per_rank` shape from a live sglang, not just
   synthetic dicts.
2. **Sanity-check the eviction trajectory** — under sustained
   unique-prompt churn, 8,948 peak units → 8,926 final units →
   eviction is actively triggering.

This run does NOT validate the §6 invariant: see RESULTS below.
The driver now warns when all sampled hashes are per-process
counters (audit #175) so the "divergence count" can't be mis-read
as a §6 signal.

## SCOPE

### Detector (`detector.py`)

```python
DivergenceReport = (window_idx, time_counter_prev, time_counter_curr,
                    per_rank_evicted: Dict[int, FrozenSet[str]])

detect_divergence(state_dumps) -> List[DivergenceReport]
summarise(reports) -> str
```

Window-based diff: for consecutive `S(t), S(t+1)`, per rank
`evicted_r = hashes_in(S(t), r) − hashes_in(S(t+1), r)`.  A window
is divergent iff any two ranks have non-identical eviction sets.

Single-rank dumps and identical-eviction windows produce zero
reports.  Rank-count change between windows raises ValueError
(scale-up/down is a deployment-bug signal, not a workload signal).

### Verify (`verify.py`, 11 stages)

| stage | scenario | expected |
|---|---|---|
| A0 | single-rank, single snapshot | no reports |
| A1 | single-rank, multi-snapshot | no reports (no cross-rank comparison) |
| B0 | multi-rank, no eviction | no reports |
| B1 | multi-rank, identical eviction | no reports |
| C0 | rank-0 evicts u1, rank-1 evicts u2 | 1 report |
| C1 | partial overlap (rank-1 evicts strictly more) | 1 report |
| C2 | 3 ranks: 2 agree + 1 diverges | 1 report |
| C3 | sustained divergence across 4 windows | 4 reports |
| D0 | rank-set changes between windows | ValueError |
| D1 | time_counter propagated to report | asserted |
| D2 | summarise() smoke (text contains hashes) | non-empty |

### Real-run driver (`run_dp2_real.py`)

Launches `--dp 2` sglang on GPUs 5,6 with
`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` (required — without it
sglang falls back to HiRadixCache which returns
`unsupported_tree_cache` for `/aginfer/state`).

Drives 24 concurrent prompt workers for 60 s with ~1.2 k-token
padded prompts (forces enough page commits for the radix tree
to actually populate).  Polls `/aginfer/state` at 2 Hz; pipes
the snapshot time-series through the detector.

## STATE WORTH RECORDING

Caught during T15 wire-up — **NEEDED FOR ANY FUTURE AGINFER RUN**:

1. **`UnifiedRadixCache` is gated behind `SGLANG_ENABLE_UNIFIED_
   RADIX_TREE=1`**.  Without it, sglang falls back to `HiRadixCache`
   which doesn't expose `dump_aginfer_state_bytes`.  The HTTP
   endpoint returns `{"unsupported_tree_cache": "HiRadixCache"}`
   per rank, and the daemon would `fatal()` on receipt.

2. **DP > 1 `/aginfer/state` exposes `per_rank`**; TP > 1 does NOT.
   The communicator returns one response per DP rank (each DP has
   its own scheduler), and the http_server aggregator only emits
   `per_rank` when `len(responses) > 1`.  Inside a single DP
   group, TP ranks share a logical scheduler and the state is
   already aggregated by the time the communicator returns.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang

# Synthetic verify (1s, no GPU)
python dev/aginfer/verify/t15/verify.py

# Real DP=2 run (3-5 min including startup; needs GPUs 5,6)
python dev/aginfer/verify/t15/run_dp2_real.py --duration 60 --workers 24
```

## RESULTS

### Synthetic verify

**PASSED** — all 11 stages.

* date: 2026-06-01
* raw log: see commit log; runs in ~1 s

### Real DP=2 run

**Detector + parser PROVEN against real per_rank JSON.**  102
snapshots captured over 60 s of 24-worker churn; 13,620 requests
served; 8,948 peak units in the radix tree; no parser exception.

**Divergence count is NOT a §6 signal under this configuration**
(audit #175): without hicache content hashing active, sglang
emits ``f"node-{node.id}"`` as the unit hash, where ``node.id`` is
a per-process counter.  Across DP=2, two different programs in
the two replicas can land on the same counter ID and look
"identical" to a string-equality detector; conversely two
content-equivalent units with different counter values look
"divergent".  The driver now WARNs when all sampled hashes are
counter-format.

What this run does prove:
* `/aginfer/state` `per_rank` wire format matches the detector's
  parser assumptions.
* `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` is the env var required
  for the aginfer state schema to be emitted at all (without it,
  every rank returns `{"unsupported_tree_cache": ...}`).
* The detector survives 100+ real-world snapshots without raising.

* date: 2026-06-01
* raw log: `results/20260601_t15_real_dp2_unified.log`

**§6 invariant verification (TP > 1):** BLOCKED on sglang patch
#174 — needs (a) per-TP-rank state endpoint AND (b) content-hash
mode active (hicache storage backend or T26 wiring).  PLAN §2 T15
status block has the full deferred list.

## WHEN #174 LANDS

1. Re-run `run_dp2_real.py` (or a sibling launcher) with `--tp 2`
   AND a hicache storage backend (or another path that activates
   `compute_node_hash_values`).
2. Hit the new per-TP-rank debug endpoint exposed by #174.
3. With content hashes active, detector should report ZERO
   divergence over any reasonable churn duration — TP's all-rank-
   atomic eviction protocol guarantees identical eviction sets
   across ranks per window.
4. Any non-zero report = §6 invariant break.  The detector's
   per-window evicted-hash listing identifies which migrate/evict
   pair leaked.
