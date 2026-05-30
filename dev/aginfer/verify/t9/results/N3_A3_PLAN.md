# T9 A3 — push workload into HBM-pressure regime

## Why this experiment exists

The 3-cycle instrumented chain (`N3_INSTRUMENT_FINDINGS.md`)
proved with hard counters that under the current matrix /
H'_now / 4-arm workload, the daemon's three layers are
**mathematically correct to do nothing**:

* HBM occupancy never exceeds 0.07 (sglang's own batch log)
* admission_controller's pressure trigger (theta_hi = 0.85)
  is never even close to being reached
* kv_scheduler's V_u says "leave everything where it is" because
  with HBM ~empty, the holding cost `h_HBM` is approximately 0
  → no demote is positive-value
* No `migrate_post` issued across ~18k events per cycle, all
  three configs (OURS_full / K-a / Run J)

This is **correct behavior**, not a bug.  But it means the
4-arm matrix's 8.7 % OURS_full-vs-LRU spread is entirely
attributable to the **inline scorer** (`ours_greedy_score` in
sglang's eviction heap) — the daemon's three layers contribute
0.  Paper §8 needs a workload where the daemon DOES fire to
have any credible story for the daemon layer's value.

## Hypothesis to test

If we **simultaneously**:
1. cap `max_completion_tokens = 4096` to kill the runaway
   60 k-token decode tail (1 % of requests = 80 % of LLM time
   per `N3_ROOT_CAUSE.md`)
2. shrink `--max-total-tokens` from default ~10 M to **256 K**
   (Run C historical regime — max SWA 0.97, i.e. firm pressure)

then HBM occ should regularly cross theta_hi = 0.85, sglang
should fire `memory_pressure`, admission should pause programs,
and kv_scheduler should issue real migrate POSTs.

**Both knobs are needed** — alone, either is insufficient:

| knob | alone effect | why insufficient |
|---|---|---|
| cap completion only | mean drops, but pool still empty | scheduler still has no work |
| shrink pool only | 32 × 60 k runaway = 1.9 M tokens; OOM | sglang scheduler can't even start |

## Configuration

| param | value | source |
|---|---|---|
| daemon | kv_scheduler ON + admission ON | matches `run_k.sh full` |
| HiCache | ON | matches `run_k.sh full` |
| max_completion_tokens | **4096** | new — `--ak llm_call_kwargs='{"max_tokens":4096}'` |
| --max-total-tokens | **262144** (= 256 K) | new — `export MAX_TOTAL_TOKENS=262144` |
| theta_hi (admission) | 0.85 | unchanged |
| theta_lo (admission) | 0.70 | unchanged |
| temperature | 0.0 | unchanged |
| seed | 42 | unchanged |
| -l / -n / -k | 32 / 32 / 1 | unchanged |
| max_turns | 200 | unchanged |
| GPUs | 0,1 | only free pair |

## Expected daemon counters (vs OURS_full instrumented baseline)

| metric | OURS_full (default pool) | A3 expected |
|---|---|---|
| events_received | 18 039 | ≈ 18 000 |
| kv_decisions | 12 017 | ≈ 12 000 |
| **HBM occ peak (daemon)** | **0.000** | **> 0.85 (the test!)** |
| **memory_pressure events** | **0** | **≥ several dozen** |
| **admission pauses** | **0** | **> 0** |
| **migrate_post count** | **0** | **> 0** |
| per-trial mean (s) | ~1369 | likely **lower** (runaway gone) |

If the daemon **still** produces 0 migrate / 0 pause under A3
settings, that points to a different bug — likely the G10
divergence (radix-tree-walk vs allocator counter) being severe
enough that the daemon never sees pressure even when pool is
saturated.  In that case we have to also do **G10 fix** before
A3 can prove anything.

## Risk: A3 might fail to launch

256K pool with 32 concurrent agents averaging ~3 k prefix tokens
each = 96 K prefix.  Plus 32 × 4 k completion = 128 K decode
working set.  Total ~224 K — just fits in 256 K.

If sglang reports CUDA OOM on launch or under load, **fall back
to 512 K** (`MAX_TOTAL_TOKENS=524288`).  This was the historical
Run F regime that still showed firm pressure (max SWA 0.76).

## Smoke test (do first, costs ~10 min)

Fire `SMOKE_N_TASKS=2 SMOKE_N_CONCURRENT=2 SMOKE_MAX_TURNS=20
bash run_k.sh a3` to:
1. verify `llm_call_kwargs` propagates (check sglang's
   `--log-requests` JSON for `max_tokens=4096`)
2. verify 256 K pool boots (no OOM at startup)
3. verify daemon emits any non-zero counter

Only after smoke passes, fire the full 32-trial cycle.

## Files

* runner: `verify/t9/run_k.sh` (gets new "a3" variant)
* expected output:
  `results/run_K_a3_instrument_<TS>/{daemon.log,sglang_v4flash.log,harbor_jobs/}`
* parser: `verify/t9/parse_daemon_events.py` (already supports
  all the counters we need)
* this plan: `verify/t9/results/N3_A3_PLAN.md`
