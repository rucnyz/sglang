# decode_wall — decode-stream wall + fail-fast regression for #205

Production-code regression for design.md §"Threading model"
property A1 + §"Transfer protocol" Stage 3 + §"Decode-stream wall
budget". Counterpart to the synthetic-harness physics proof in
[`../step1_stream_isolated_unmap/`](../step1_stream_isolated_unmap/).

After #205, production fire path runs **no defensive sync, no
`cuMemSetAccess(prot=NONE)` revoke** before `cuMemUnmap`. The sole
safety mechanism is the layer-0 invariant (`fire_planner` picks
only free pages no in-flight req's `state_indices` references). This
folder's three tests together verify:

1. The no-defense path is safe under the layer-0-invariant-satisfied
   case (the production happy path).
2. The fail-fast safety story is real (layer-0 violation surfaces
   loudly as `cudaErrorIllegalAddress`, not silently corrupted).
3. The actual multi-sub-pool batched unmap loop edited by #205 fits
   the §fire_wall+decode_wall decode-stream wall budget.

## Tests

| file | what it pins | design.md ref |
|---|---|---|
| [`test_no_crash.py`](test_no_crash.py) | `arena.shrink_explicit(...)` worker-thread unmap concurrent with captured Triton-graph replay does not crash when layer-0 invariant is satisfied (disjoint VA) | §"Threading model" A1 + §"Transfer protocol" Stage 3 |
| [`test_failfast.py`](test_failfast.py) | Layer-0 violation (kernel reads chunk 150, worker unmaps it) raises `cudaErrorIllegalAddress` on the next replay — design's fail-fast safety claim is real, not paper-only. Subprocess-isolated because faults poison CUDA context | §"Threading model" + §"Transfer protocol" Stage 3 fail-fast clause |
| [`test_decode_wall.py`](test_decode_wall.py) | Multi-sub-pool batched unmap loop (mirrors `xpool_actuator._execute_async_locked`'s `for name in src_names: shrink_explicit(...)`) on a worker thread keeps decode-stream wall ≤ 0.10 ms (n=20 trials, ~3–5× headroom over the stable 0.00–0.03 ms median) while 4096×4096 GEMMs run on the decode stream | §"fire_wall_curve + decode_wall" (decode-stream-wall half) + §"Threading model" A1 |

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python \
    dev/interlayer/0_page_state_machine/decode_wall/test_no_crash.py
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python \
    dev/interlayer/0_page_state_machine/decode_wall/test_failfast.py
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python \
    dev/interlayer/0_page_state_machine/decode_wall/test_decode_wall.py
```

## Status

| test | post-#205 result |
|---|---|
| `test_no_crash.py` | **PASS** — 3 trials × 200 captured-Triton replays + 100-chunk worker unmap, no crash |
| `test_failfast.py` | **PASS** — subprocess subprocess: `AcceleratorError: CUDA error: an illegal memory access was encountered` on replay-after-unmap-of-overlapping-chunk |
| `test_decode_wall.py` | **PASS** — 4-sub-pool × 100-chunk worker unmap loop, decode-wall delta median 0.00–0.03 ms across 3 independent n=20 runs (budget ≤ 0.10 ms) |

If any of these regresses:
- `test_no_crash.py` fail → layer-0 invariant + `cuMemUnmap` atomicity is not sufficient on the platform. Revisit design.md §"Transfer protocol" Stage 3 fail-fast posture; either reinstate `cuMemSetAccess(NONE)` revoke (A1 step 1.7 verified the revoke produces real fault on subsequent reads; worker-thread `cuMemUnmap` decode-wall cost measured at step 1.6 = +0.11 ms wall) or accept defensive sync.
- `test_failfast.py` fail (NO_FAULT) → the `cudaErrorIllegalAddress` we rely on for fail-fast diagnosis is being swallowed somewhere. Design's whole safety story collapses; a layer-0 bug in `fire_planner` would corrupt silently. Defensive layer must be re-added.
- `test_decode_wall.py` fail → the worker-thread unmap loop is stalling the decode stream beyond §fire_wall+decode_wall budget. Investigate whether per-iteration host overhead grew or a new sync was reintroduced into the loop.

## What this folder does NOT cover (yet)

- Full `XPoolActuator._execute_async_locked` end-to-end (which also runs cap-barrier, dst.grow, dst cap-bump). Those are scheduler-thread phases with separate budgets covered in 1_dyn_admission_cap/ and 2_admitter/.
- Larger fire scales (production fires up to 1152 chunks at p50 82 ms host wall, per `bench/bench_cumem_costs.py`). Our test_decode_wall.py exercises 400 chunks across 4 sub-pools — proportional to a small hybrid model but not the largest production fire.
