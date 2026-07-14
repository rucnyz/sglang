# case1 finding: the "−22% regression" was a self-inflicted base crash (#343), not a HiMA regression

## Symptom

`case1_kvbound_win.sh` (longkv, conc 48) showed \sys ~22% below base and a decode
batch that collapsed to a single request for ~45% of steps.

## Root cause (#343, a bug introduced by #342 THIS session)

#342 added per-victim LPB-loss telemetry to `MambaRadixCache.evict_mamba` /
`evict_full`, opening each with an unconditional `get_cost_curves()`.
`get_cost_curves()` fail-closes (raises) when no cost model is calibrated.
Base serving of a hybrid model (`--radix-eviction-policy lru`, no `SGLANG_CSIGMA_*`)
still uses MambaRadixCache, so the base scheduler SIGQUIT on its first eviction
(`evict_mamba -> get_cost_curves -> RuntimeError: HiMA requires a calibrated cost
model`).
Base then completed only 501/800 (the 299 hardest requests errored out), so its
"536 tps" was measured on half the work while \sys did all 800.
The "−22%" was base-drops-hard-requests, not a real regression.
The batch-collapse and extra-prefill observations were the same contamination
(base crashed ~9 min in; its 8-9 batching was its short pre-crash window).

## Diagnosis path

1. 3-arm isolation (base / sys-full / sys-zerofire via new `SGLANG_HIMA_NO_ADMITTER`):
   sys-full ≡ sys-zerofire (only 1 fire, fires contribute 0) => not fires.
2. cache_hit 0.9725 (\sys) >= 0.9644 (base) => not cache locality.
3. result-JSON `n_error`: base 299 vs \sys 0 => base crashed.
4. base server log traceback => the `get_cost_curves()` fail-close in eviction.

## Fix (#343)

`cost_model.has_cost_curves()` (non-raising probe: resolves + caches the singleton,
returns bool). Gate the LPB-loss pricing in both `evict_mamba` and `evict_full` on
`track_loss = has_cost_curves()`; skip it when uncalibrated (the telemetry is
Budgeter-only, dead on a base server). Calibrated (\sys) path unchanged.
Reproducing test:
`dev/interlayer/2_admitter/test_mamba_real_pool.py::test_8_evict_without_calibration_must_not_crash`.

## Result with a valid base (both 800/800, cache_hit 0.9725)

| arm | tps |
|-----|-----|
| base (fixed) | 416.1 |
| \sys | 415.5 |

No regression (−0.14%, noise). Every pre-#343 case1 / m2k base number is invalid.

## Why case1 is not a WIN here (open, for the win-hunt)

longkv at conc 48 does NOT bind KV: only ~4 sessions are concurrently active (the
52 roots are multi-turn with per-turn dependencies), so KV usage stays ~0.42 and
`#queue-req` is 0 even after shrinking the pool to 656k tokens (MEMFRAC=0.4).
The m2k mechanism (drain idle mamba -> grow KV -> admit more) only pays off when KV
actually binds, which needs a high-concurrency long-context arrival pattern (a burst
of independent long prompts), not this dependency-serialized trace.
Building that workload is the next step for the case1 throughput headline.
