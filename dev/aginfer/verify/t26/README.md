# T26 (#200) — sglang throughput / in-flight measurement

Replaces the three `/aginfer/state` placeholders the daemon's DESIGN §8
admission math reads — they shipped as `0.0` / `{}` ("T26 wires actual
measurement"):

| field | was | now (measured) |
|-------|-----|----------------|
| `throughput_ema.decode_per_program[pid]` | `{}` | per-program decode tokens/sec EMA |
| `throughput_ema.prefill_bps` | `0.0` | prefill bytes/sec EMA |
| `per_program_usage[pid].hbm.inflight[sp]` | `{}` | running request's current HBM bytes |

## Architecture

The scheduler owns the running batch + forward timing; the radix cache
only stores + serialises.  So the scheduler MEASURES (per forward) and
PUSHES onto the cache before each dump — the same pattern as the
daemon's `_aginfer_program_states` push.

* **`aginfer_metrics.py`** (pure, unit-tested here): `ema_update`,
  `inflight_bytes_by_program`, `decode_tokens_by_program`,
  `running_program_view`.
* **`scheduler._aginfer_record_throughput(batch)`** — called from
  **`run_batch`** (NOT `process_batch_result`): under overlap
  scheduling the forward consumes the batch's `extend_num_tokens` /
  `input_ids`, so they're gone by result-processing time — the prefill
  token count is only available on the FRESH pre-forward batch.  Decode
  tokens come from `batch.reqs` (1/req), which persist.
* **`scheduler._aginfer_push_runtime_metrics()`** — called from
  `get_aginfer_state` (cold, per dump): assembles `decode_per_program`
  (projected onto currently-running programs), `prefill_bps`, and
  `inflight_bytes_by_program(running_batch.reqs)`, prunes stale decode
  EMAs, and pushes via `cache.set_aginfer_runtime_metrics`.
* **`unified_radix_cache`** — stores the pushed dict; `_aginfer_
  throughput_ema` returns it; `_aginfer_overlay_program_states` fills
  `hbm.inflight`.

## Two corrections found by real-stack debugging

1. **inflight = `kv_allocated_len × bpt`, not `allocated − committed`.**
   sglang commits each decoded token immediately (`kv_committed_len`
   tracks `kv_allocated_len`), so `allocated − committed ≡ 0`.  The
   in-flight footprint is the running request's CURRENT total KV — which
   is exactly what `marginal_pause_cost` ("decoded-so-far bytes
   re-prefilled on resume") wants.
2. **Prefill measured at `run_batch`, not `process_batch_result`.**  At
   result time (overlap scheduling) the extend batch's token fields are
   all `None`; the fresh pre-forward batch has them.

## Stages (pure)

```
ema            seed / blend / alpha / malformed-input guard
inflight       current-KV per program; skip untagged/empty; sum; bpt guard
decode-counts  1 token/req + accumulation (spec-decode)
running-view   project decode EMA onto live programs
```

The live wiring is verified end-to-end by `verify/integration_stress`
**stage T26**: under program-tagged decode load, the real sglang
`/aginfer/state` shows `prefill_bps > 0`, `decode_per_program[pid] > 0`
for all tagged programs, and `hbm.inflight` populated.

## What this activates (daemon side)

`marginal_pause_cost` (prefill_bps) and `pause_relief`'s in-flight
snapshot (inflight) become real NOW.  The §8 forecast **trajectory**
term stays 0 until T11 (#126) populates `expected_remaining_tokens` —
`_program_inflight_growth` gates on a real `E[remaining]`, not the
bootstrap (the §8 over-pause anti-pattern).

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t26/verify.py                       # pure helpers
python dev/aginfer/verify/integration_stress/verify.py        # stage T26 (real stack)
```

## RESULTS

**PASSED** — 4 pure stages + integration_stress stage T26 on the real
B300 stack (`prefill_bps≈9.2e8`, `decode≈700 tok/s × 4`, inflight
populated in 81/93 polls).

* date: 2026-06-05
