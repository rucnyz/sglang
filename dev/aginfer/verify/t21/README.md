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
  {state, pre_pause_state}).  Survives until overwritten OR GC'd
  (see ENDED-GC below).
* **Unit-less programs still visible**: a pid with no live units
  (all KV evicted) still appears in `per_program_usage` if the
  daemon previously pushed a NON-terminal state (PAUSED / ACTING /
  REASONING).  Lets the daemon read its own PAUSED-with-no-residue
  bookkeeping.
* **ENDED-GC** (#186 audit — unbounded-growth fix): an ENDED
  program with NO live units is dropped from
  `_aginfer_program_states` during the dump and NOT echoed — a
  terminal program with no residue needs no daemon bookkeeping.
  ENDED programs that STILL have residual units ARE echoed (the
  daemon needs the terminal state while cleanup completes).
  Without this GC, every program ever PUT would accumulate forever
  and pollute every dump.
* **Single overlay code path** (#186 audit): both
  `_dump_aginfer_state_dict` and `_dump_aginfer_state_bytes` call
  the same `_aginfer_overlay_program_states` helper — the two
  paths cannot diverge by construction.  `applied` is reported as
  an integer sum across ranks but should be treated as a BOOLEAN
  by the daemon (>0 = state changed, 0 = idempotent no-op); under
  DP>1 the exact integer conflates idempotency with rank count.
* **Validation server-side**: state / pre_pause_state must be in
  `{REASONING, ACTING, PAUSED, ENDED}`; bad value → 400.
* **Tree-cache requirement**: only `UnifiedRadixCache` supports
  `set_aginfer_program_state`.  Legacy caches reject with a
  reason naming the cache class.

## STAGES (13)

```
A. Setter unit tests (cache-level)
  A0  valid state stores; ok=True, applied=1
  A1  idempotent re-apply; ok=True, applied=0
  A2  invalid state ("HUNGRY") → ok=False, applied=0
  A3  invalid pre_pause_state ("DREAMING") → ok=False, applied=0
  A4  empty / None pid → ok=False
  A5  None pre_pause_state is valid (default)

B. Overlay helper (the REAL production code, post-#186 refactor)
  B0  _aginfer_overlay_program_states overlays state onto an
      existing (unit-derived) entry; hbm/dram/unit_hashes preserved
  B1  BOTH dump paths call the shared helper (source-inspection pin
      against future re-inline + divergence)
  B2  pid with no live units (PAUSED) still appears
  B3  ENDED + no units → GC'd from storage, NOT echoed
  B4  ENDED + residual units → kept + echoed (cleanup ongoing)

C. Scheduler handler
  C0  tree-cache without set_aginfer_program_state → ok=False with
      reason naming the cache type

D. HTTP body validation (#186 coercion-bypass fix)
  D0  _validate_program_paused_body: accepts well-formed;
      REJECTS null/numeric/empty pid + state + numeric pre_pause
      (the previous str() coercion turned null→"None" etc.,
      bypassing the empty-pid guard)
```

**#186 audit note**: the original B0/B1/B2 tested a hand-copied
*replica* of the overlay loop, not the production methods.  The
overlay is now a single `_aginfer_overlay_program_states` method
shared by both dump paths; the verify calls THAT method directly,
so it exercises real production code.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t21/verify.py
```

Pure-Python; ~0.5 s.

## RESULTS

**PASSED** — all 13 stages (10 initial + 3 from #186 audit).

* date: 2026-06-02 (post-#186 audit closure)
* raw logs: `results/20260601_t21_initial_pass.log` (10 stages),
  `results/20260602_t21_post_186_pass.log` (13 stages)

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
