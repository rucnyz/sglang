# byte_transfer — end-to-end byte transfer + working-set invariant

What it tests (design.md §byte_transfer): under a workload where the planner
fires, src allocator reports substantively-lower live capacity, dst
reports substantively-higher live capacity, and `shared_pool.free_count`
remains constant (handles re-bound, not lost).

This is the **first integration test** in the stack — it exercises
the full chain `XPoolPlanner → XPoolFirePlanner → XPoolActuator →
cuMemUnmap/cuMemMap → KVArenaActuator/MambaArenaActuator` on a live
sglang server. If anything in the wire-up is broken, byte_transfer catches it.

## Driver + validator

- `run_byte_transfer.sh` — boots Qwen3.5-9B with `SGLANG_HIMA=1`,
  drives 120s random workload (RPS=32, output=1024) sustained against
  a forcibly small mamba pool (`--max-mamba-cache-size 100`),
  captures `budgeter.jsonl` + `server.log`
- `validate_byte_transfer.py` — parses both, asserts:
  - (a) ≥1 non-aborted fire emitted
  - (b) per-fire: `unmapped_pages == granted_pages > 0` AND
        `shared_pool.free_count` delta = 0 (paper §911 invariant)
  - (c) no engine ERROR / OOM / CUDA / Traceback / leak / SIGQUIT
        (filters known-benign cutlass module-load chatter)
  - (d) policy-correct: each fire's snapshot at fire time shows
        `usage_dst_active ≥ 0.50` — proves fires acted on real
        running-req pressure, not phantom radix-cache LRU saturation
        (added 2026-05-26 after active-fix v2 exposed phantom fires)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
GPU=<gpu> PORT=30077 OUT_DIR=/tmp/d7_run WORKLOAD_S=120 \
    bash dev/interlayer/4_e2e/byte_transfer/run_byte_transfer.sh
```

Wall ~4 min (50s boot + 120s workload + teardown).

## Workload choice — post active-fix v2 design

Two changes vs the original v1 workload:

1. **Forced small mamba pool** (`--max-mamba-cache-size 100`).
   The original v1 used `mem-fraction-static=0.35` to squeeze
   pools, but that didn't isolate the mamba dimension: KV pool
   shrunk too, and sglang's max_running auto-cap dropped to ~33.
   v2 uses `mem-fraction=0.70` for a big KV budget (max_running
   200+) AND forces mamba pool to exactly 100 slots, so even
   modest concurrency floods mamba.

2. **`SGLANG_XPOOL_MAMBA_HIGH=0.50` test override.** Active-fix
   v2 made the planner correctly ignore radix-cache LRU pressure
   (only persist on `usage_mamba_active`). But under byte_transfer's workload
   the active counter peaks at ~0.66 sustained (not 0.80 default)
   for two reasons:
     - sglang's `num_queue_reqs.total` reflects only the
       scheduler's `waiting_queue`, not the API-layer queue where
       the actual backlog lives → planner can't see queue pressure
     - active counter has a phantom component (slots between
       req-completion and cache-installation briefly count as
       "active") on top of true running-req usage
   Lowering high-water to 0.50 in byte_transfer lets persist consec accumulate
   at sustained 0.66; (d) check enforces dst_active ≥ 0.50 so
   fires still need real pressure. This is a **test-specific env
   override**, NOT a planner default change.

The 9B model is fastest to boot (~50s); larger models would take
longer wall but the fire mechanism is identical.

## Result

ALL PASS: 28 non-aborted fires, every `unmapped_pages ==
granted_pages == 64`, `shared_pool.free_count` Δ = 0 (paper §911
invariant holds), zero engine errors over 120s, all 28 fires happened
while `usage_mamba_active ≥ 0.50` (peak m_active=0.66 sustained,
peak waiting_queue=2842).

Drove out two real wire-up bugs during integration:
1. **Off-by-one in cap-barrier slot translation** (`kv_actuator.py`
   + `mamba_actuator.py`'s `expand_pages_to_token_slots`) — leaked
   slot `p*tps` of every unmapped page back into engine free list →
   `cudaErrorIllegalAddress` 6s post-fire
2. **`live_size` ignored `_capped_pages`** in `allocator.py` →
   pool-leak-detector false-positive SIGQUIT after first fire

Both fixed in commit `e5f6d34421`. Unit tests for both bugs land
in this folder as **test-first regression guards** (each was first
shown to FAIL against pre-fix code, then PASS against post-fix —
proves the fix is reachable AND necessary):

- `test_chunk_slot_unit.py` — 5/5 sub-tests; locks in `expand_pages_to_token_slots` half-open bounds for chunk N > 0, plus the post-#226 raise-on-page-0 contract (chunk 0 carries padded slot 0 per design.md §"Per-unit sizes"; tests 1+4 assert the loud ValueError)
- `test_live_size_unit.py` — 5/5 sub-tests; locks in
  `live_size = size − _capped_pages.numel()` formula across both
  cap mechanisms (set_capacity_pages + mark_pages_capped)

Both pure-Python, run with `.venv/bin/python <file>` — no GPU/boot.

Commit: `e5f6d34421` (byte_transfer + fixes), `0e4051b988` (wire-up + multi-source NB)
