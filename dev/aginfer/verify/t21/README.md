# T21 — PUT /aginfer/program_paused (#181, DESIGN §6 round-6 H2)

The daemon owns the program-state transition (REASONING / ACTING /
PAUSED / ENDED).  This endpoint lets the daemon push its view into
sglang as a passthrough — sglang stores the latest state per pid
and echoes it back in the next `/aginfer/state` dump's
`per_program_usage[pid].state` + `.pre_pause_state` fields.

## WIRE

```
PUT /aginfer/program_paused
Body: {
  "pid": str,
  "state": "REASONING" | "ACTING" | "PAUSED" | "ENDED",
  "pre_pause_state": str | null   (same set or null; conventionally
                                   the prior state on PAUSED)
}

Response: {"ok": true, "ranks": int, "applied": int}
  - applied = sum across ranks: 0 = idempotent no-op,
              ≥1 = state changed
```

## CONTRACT

* **Idempotent** (DESIGN §10 R2): same (state, pre_pause_state)
  re-applied → `applied=0`.
* **Persistent across dumps**: stored in
  `UnifiedRadixCache._aginfer_program_states` (dict[pid] →
  {state, pre_pause_state}).  Survives until overwritten.
* **Unit-less programs still visible**: a pid with no live units
  (all KV evicted) still appears in
  `per_program_usage` if the daemon previously pushed its state.
  Lets the daemon read its own PAUSED-with-no-residue bookkeeping.
* **Validation server-side**: state / pre_pause_state must be in
  `{REASONING, ACTING, PAUSED, ENDED}`; bad value → 400.
* **Tree-cache requirement**: only `UnifiedRadixCache` supports
  `set_aginfer_program_state`.  Legacy caches reject with a
  reason naming the cache class.

## STAGES (10)

```
A. Setter unit tests (cache-level)
  A0  valid state stores; ok=True, applied=1
  A1  idempotent re-apply; ok=True, applied=0
  A2  invalid state ("HUNGRY") → ok=False, applied=0
  A3  invalid pre_pause_state ("DREAMING") → ok=False, applied=0
  A4  empty / None pid → ok=False
  A5  None pre_pause_state is valid (default)

B. Dump-path echo
  B0  dict-path: stored state OVERLAID onto unit-aggregated entry;
      hbm/dram/unit_hashes preserved
  B1  bytes-path: same overlay formula
  B2  pid with no live units still appears in the dump

C. Scheduler handler
  C0  tree-cache without set_aginfer_program_state → ok=False with
      reason naming the cache type
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t21/verify.py
```

Pure-Python; ~0.5 s.

## RESULTS

**PASSED** — all 10 stages.

* date: 2026-06-01
* raw log: `results/20260601_t21_initial_pass.log`

## REGRESSION SANITY

* T22 (thresholds PUT, the mirror pattern): 8/8 green
* T24 (HASH_COLLISION webhook): 8/8 green
* T36 (outbound queue): 8/8 green
* T164 (sustained-escalate): 11/11 green

## IMPLEMENTATION

| layer | change |
|---|---|
| `io_struct.UpdateAginferProgramPausedReq/Output` | mirrors thresholds req/output dataclasses |
| `unified_radix_cache.set_aginfer_program_state` | validates + stores in `_aginfer_program_states` dict |
| `unified_radix_cache._dump_aginfer_state_dict` + `_bytes` | overlay loop after per-program unit aggregation |
| `scheduler.update_aginfer_program_paused` | dispatcher → cache setter; rejects legacy tree caches |
| `tokenizer_control_mixin.update_aginfer_program_paused` + `_communicator` | fan-out per rank |
| `http_server.PUT /aginfer/program_paused` | parse body → req → aggregate per-rank response |

## UNBLOCKS

* **#183** (T30+T39 proxy disconnect): enqueues PUT
  `/aginfer/program_paused {END}` on client disconnect.
* **#185** (T41 SESSION_END-for-PAUSED): releases gate with HTTP
  499, transitions ENDED, enqueues PUT to inform sglang.
