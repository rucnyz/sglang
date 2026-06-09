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

## Three corrections (real-stack debugging + audit)

1. **inflight = `kv_allocated_len × bpt`, not `allocated − committed`.**
   sglang commits each decoded token immediately (`kv_committed_len`
   tracks `kv_allocated_len`), so `allocated − committed ≡ 0`.  The
   in-flight footprint is the running request's CURRENT total KV — which
   is exactly what `marginal_pause_cost` ("decoded-so-far bytes
   re-prefilled on resume") wants.
2. **Prefill measured at `run_batch`, not `process_batch_result`.**  At
   result time (overlap scheduling) the extend batch's token fields are
   all `None`; the fresh pre-forward batch has them.
3. **Pure-mode classification, then full MIXED + spec accounting (#200
   audit → #206).**  `is_extend()` is True for `MIXED` / `TARGET_VERIFY`
   / `DRAFT_EXTEND` too, so the original broad check counted spec verify/
   draft tokens — and a MIXED batch's decode tokens — as PREFILL,
   polluting `prefill_bps`.  #200 narrowed to pure `DECODE`→decode, pure
   `EXTEND`→prefill (conservative under spec/chunked).  **#206 measures
   the rest:**
   * **MIXED** (chunked prefill + running decode): split via
     `batch.decoding_reqs` (sglang's own metrics discriminator) — prefill
     reqs' `extend_input_len` → `prefill_bps`, decode reqs → decode EMA.
     `extend_num_tokens` spans the whole batch (1 per decode req) so it is
     NOT usable as the prefill count.
   * **spec-decode**: a verify step commits `accept_lens[i]` tokens/req
     (bonus + accepted drafts), not 1.  The count is exposed only on the
     **spec-v2** (overlap-on) path, as `result.num_correct_drafts_per_req_cpu`
     (= `accept_lens − 1`; +1 = accepted), resolved POST-forward.  So for a
     **`is_spec_v2`** batch the pre-forward `DECODE` branch SKIPS and
     `_aginfer_record_spec_decode` (from `process_batch_result`, raise-safe)
     attributes the accepted count per `program_id`.  **spec-v1** (overlap
     OFF — e.g. ngram, which forces it) does NOT expose `accept_lens`, so it
     KEEPS the conservative 1/req pre-forward count — never blanked, no
     regression.  The gate is `batch.is_spec_v2` on BOTH sides.
   * **`DRAFT_EXTEND`** is excluded — draft proposals aren't committed
     output (the accepted set is already counted at `TARGET_VERIFY`).
   The per-forward hooks stay wrapped in a blanket `except` (a
   scheduler-loop crash is catastrophic; losing a sample is harmless).

## Stages (pure)

```
ema            seed / blend / alpha / prev=0-blends / malformed guard
inflight       current-KV per program; skip untagged/empty; sum; bpt guard
decode-counts  1 token/req + accumulation + per_req_tokens (spec accept_lens)
running-view   project decode EMA onto live programs
routing        pre-forward hook: DECODE→decode, spec-v2 DECODE→skip
               (post-fwd), spec-v1 DECODE→1/req, EXTEND→prefill; raise-safe
mixed          MIXED split via decoding_reqs: prefill→bps, decode→EMA
               (always 1/req here, even spec-v2 — the post-forward hook
               can't reach MIXED); extend_num_tokens NOT used
spec-decode    post-forward hook: accept_lens → per-program decode EMA
               (≠1/req); no-op on missing counts (v1); raise-safe
```

The live wiring is verified end-to-end on the real B300 stack by:
* `verify/integration_stress` **stage T26** — pure DECODE/EXTEND load:
  `prefill_bps > 0`, `decode_per_program[pid] > 0` for all tagged
  programs, `hbm.inflight` populated.
* `verify/t26/realstack_modes.py` (#206) — two extra stacks:
  **MIXED** (`--enable-mixed-chunk --chunked-prefill-size 256`) drives the
  split live (prefill_bps + per-program decode both populate); **spec**
  (`--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1`,
  which is **spec-v1** — ngram forces overlap off) is the v1 REGRESSION
  GUARD: the `is_spec_v2` gating must NOT blank decode (1/req pre-forward
  stays live).  Both assert the sglang log carries no
  `aginfer ... measurement raised` line (the only symptom of a hook
  excepting on a real batch/result).  The **spec-v2** accept_lens →
  per-program EMA path is pinned by the pure `stage_spec_decode` — no
  Qwen3-0.6B EAGLE draft checkpoint is available to drive v2 live.

## What this activates (daemon side)

`marginal_pause_cost` (prefill_bps **and** the undivided `inflight` —
the resume re-prefill cost) becomes real NOW.  `pause_relief` itself
uses the shared-aware `committed` snapshot, NOT inflight (#205 — the
unified cache reports a running req's KV in both, on different bases).
The §8 forecast **trajectory** term stays 0 until T11 (#126) populates
`expected_remaining_tokens` —
`_program_inflight_growth` gates on a real `E[remaining]`, not the
bootstrap (the §8 over-pause anti-pattern).

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t26/verify.py                       # pure helpers
python dev/aginfer/verify/integration_stress/verify.py        # stage T26 (pure-mode real stack)
python dev/aginfer/verify/t26/realstack_modes.py              # MIXED + spec-v1 real stack (#206)
```

## RESULTS

**PASSED** — 7 pure stages + integration_stress stage T26 (pure DECODE/
EXTEND) on the real B300 stack.  The integration stage requires ALL
tagged programs to be measured (not just one) for both decode and
inflight (#200 audit).

* date: 2026-06-05 (#200)
* date: 2026-06-05 (#206) — MIXED + spec-decode accounting.  Pure suite
  7/7 (added `mixed`, `spec-decode` stages; `routing` covers spec-v1 vs
  v2).  Real-stack `realstack_modes.py`: **MIXED**
  (`--enable-mixed-chunk --chunked-prefill-size 256`) drove the split
  live — `prefill_bps≈7.3e8`, decode 4/4 programs, inflight 4/4, no
  suppressed warning; **spec-v1** (`--speculative-algorithm NGRAM`,
  overlap off) decode 3/3, no suppressed warning (the `is_spec_v2`
  gating keeps v1 decode at 1/req).  Audit closed F1 (raise-safety test
  bound the inner method) + F2 (spec-v2 MIXED decode counts 1/req here,
  not dropped — the post-forward hook can't reach MIXED).
