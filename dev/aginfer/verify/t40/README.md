# T40 — F2 hint emitter (#184, DESIGN §6 `PUT /aginfer/hints` + §10)

Per event the daemon re-scores the units in `D_t` and pushes ALL of
them to sglang via a fire-and-forget `PUT /aginfer/hints`.  The
inline scorer fires on sglang's allocation-decision callsite and
cannot wait for an HTTP round-trip to fetch fresh V_u inputs, so the
daemon keeps the hint table warm by pushing every event.

Two hard DESIGN §10 invariants:

* **No daemon-side hint cache** — the daemon keeps NO shadow
  `{hash: last_pushed_value}` map.  Every D_t unit is pushed
  unconditionally every event; sglang dedupes on its side.
* **Overwrite-by-stamp** — a PUT whose `stamp` is strictly newer
  wins; an equal stamp is an idempotent no-op (§10 R2); an older
  (stale, out-of-order) stamp is dropped so it can't clobber a newer
  value.

This task is the FULL round-trip: the daemon emitter AND the sglang
receiving side (storage + overwrite contract).

## WHAT THIS ADDS

**Daemon** (`daemon/`):
* `kv_scheduler.hints_from_state(sched_state)` — pure: one
  `{hash, p_hat, lambda, stamp}` per unit in `D_t`, carrying the
  EXACT `p_hat` / `lambda_rate` the scorer computed.
* `kv_scheduler.KvScheduler._dispatch_hints` + the `handle()` wiring:
  dispatch hints BEFORE and independent of the migrate decision,
  on every event with a non-empty `D_t`.
* `outbound.OutboundQueue.enqueue_hints` — `PUT /aginfer/hints`,
  body `{hints:[...], batch_id}`.

**sglang** (full chain, mirrors the program_paused plumbing):
* `http_server._validate_hints_body` (pure) + `PUT /aginfer/hints`.
* `io_struct.UpdateAginferHintsReq` / `…ReqOutput`.
* `tokenizer_control_mixin.update_aginfer_hints` (rank fan-out).
* `scheduler.update_aginfer_hints` handler.
* `UnifiedRadixCache.set_aginfer_hints` (overwrite-by-stamp on
  `_aginfer_hints`) + `get_aginfer_hint` / `clear_aginfer_hint`;
  `/aginfer/state` echoes `n_aginfer_hints` (count, both dump paths).

## THE STAMP

`stamp = int(sched_state.t)` — sglang's own `time_counter` from the
state dump.  This is monotonic, survives a daemon restart (the table
is rebuilt with sglang's clock as the ordering authority), and is
read from the snapshot rather than a `time.*` call — so it respects
the "no wall-clock in the daemon's policy/transition path"
invariant.  If `time_counter` stalls between two events, the equal
stamp makes the re-push an idempotent no-op (the daemon scored
against the same sglang state → same values), which is exactly the
intended dedupe.

## SCOPE BOUNDARY (deferred — separate tasks)

This task is the emitter + storage + overwrite contract.  NOT here:

* the inline scorer CONSUMING the hint table for eviction order
  (T28 #178 / #177);
* unit-birth seeding (`p_hat ≈ 1` on node creation);
* eviction-time hint-clear ORDERING (T27 — `clear_aginfer_hint`
  is the primitive; the scorer-read→evict-commit→clear ordering at
  the eviction callsite is the open work);
* cross-rank hint fan-out atomicity (#174 / probed in T15).

## STAGES (14)

```
A. daemon outbound
  A0 enqueue_hints → {hints:[...], batch_id}, endpoint=hints, PUT
B. daemon kv_scheduler emitter (drives handle())
  B0 non-empty D_t → one hint per unit, EXACT p_hat/lambda +
     stamp == time_counter
  B1 push UNCONDITIONAL: policy declines migrate → hints still pushed
  B1b hints pushed alongside a migrate (both enqueued, independent)
  B2 empty D_t (LLM_PREFILL) → no hints, no migrate
  B3 NO shadow cache: 2 events re-push the SAME unit (no "unchanged"
     suppression), 2nd stamp strictly newer
C. sglang validator
  C0 _validate_hints_body accepts well-formed, returns normalized
  C1 rejects: not-dict / missing hints / hint-not-dict / missing or
     empty hash / non-numeric p_hat·lambda / non-int·negative stamp /
     p_hat out of [0,1] / negative lambda
D. sglang storage — set_aginfer_hints overwrite-by-stamp
  D0 first push applies (applied == n), values readable
  D1 idempotent re-push (same stamp) → applied 0
  D2 newer stamp overwrites → applied counts, value updated
  D3 stale (older) stamp rejected → applied 0, value unchanged
E. wire round-trip
  E0 the EXACT body the daemon emits passes sglang's
     _validate_hints_body AND set_aginfer_hints (catches a
     "lambda" vs "lambda_rate" / "stamp" vs "seq" field mismatch)
F. e2e (env-gated AGINFER_VERIFY_BASE)
  F0 LIVE PUT /aginfer/hints against a real sglang → applied≥2;
     idempotent re-apply → applied 0; newer stamp → applied 2;
     /aginfer/state n_aginfer_hints ≥ 2 (unique per-run stamp+hashes
     so repeated runs against a persistent server stay independent)
```

## REPRODUCING

Pure-Python stages A–E (no GPU, ~1 s):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t40/verify.py        # F0 SKIPs without a server
```

With the LIVE e2e stage F0 (GPU):

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=5 \
  python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 --trust-remote-code \
    --attention-backend flashinfer > /tmp/sglang_t40.log 2>&1 &
until grep -q "Uvicorn running" /tmp/sglang_t40.log; do sleep 6; done; sleep 12
AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
  python dev/aginfer/verify/t40/verify.py
```

## RESULTS

**PASSED** — all 14 stages (incl. F0 live e2e against real sglang).

* date: 2026-06-02
* raw log: `results/20260602_t40_initial_pass.log`
* F0 ran twice back-to-back (cross-run idempotency) — green both.

## REGRESSION SANITY

* kv_scheduler_value_rule: PASS (19) — handle() hint dispatch is
  additive; migrate path unchanged
* T36 outbound: PASS (8) — enqueue_hints additive, POST/PUT dispatch
* T41 SESSION_END: PASS (17) — shares scheduler/io_struct/cache files
* T6 program_tracker: PASS
* integration_stress: PASS (full-stack sglang+daemon, GPUs 5,6)
