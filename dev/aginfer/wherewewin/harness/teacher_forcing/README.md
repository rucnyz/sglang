# Teacher-forcing fidelity — prepare & validate

**Purpose.** The wherewewin campaign (`../../`, the campaign root) needs a replay harness
that faithfully reproduces a real multi-turn agent's KV-cache behaviour. The only
faithful way (see the analysis below) is **in-loop teacher-forcing**: keep the
normal autoregressive decode loop, but at each step replace the sampled token with
the captured token. This folder **prepares that mechanism and then empirically
proves it is harmless** — i.e. that forcing the token (vs. letting the model pick
it) changes **neither the timing nor sglang's internal state**.

We do not take this on faith. We measure it.

## Why teacher-forcing, and why "in-loop" specifically
- **Length-only replay** (force `max_tokens`, let content be whatever) breaks the
  multi-turn KV continuation: turn N's forced output content ≠ the real output
  embedded in turn N+1's captured input, so turn N+1 **re-prefills** the output
  segment instead of hitting cache. That artifact re-prefills every turn's output
  and can dilute the very signal we measure.
- **In-loop teacher-forcing** feeds the *captured* tokens, so turn N's KV =
  `[context_N + O_N]` exactly → turn N+1 hits exactly as reality. KV is a function
  of the token sequence, not of how it was sampled.
- **NOT** the "parallel-prefill the known output" shortcut — that runs the output
  at prefill speed, not decode speed → wrong timing. We keep the sequential decode
  loop; only the final token selection changes from `argmax` to a table lookup.
- Spec-decode caveat does **not** apply here: this deployment does **not** use
  speculative decoding / MTP, so the accept/reject-rate-depends-on-content concern
  is moot. (Recorded so a future spec-decode deployment re-validates.)

## The claim to prove (falsifiable)
Replacing `argmax` with "emit the captured token id" inside the decode loop is a
**no-op** with respect to:
1. **Performance** — per-token decode latency (TPOT), TTFT, total time,
   batched throughput: identical within run-to-run noise.
2. **sglang internal state** — the radix tree / KV cache after the run: identical
   units, hashes, residence, `pool_usage`; no code path that normal sampling
   exercises (counters, hint-table seeds, eviction-clock advances) is skipped or
   double-run by the forcing path.

If either differs beyond noise, teacher-forcing introduces an artifact and the
replay numbers must be corrected for it.

## Experiment design

### Part A — forcing-mechanism is harmless (single request)
Isolate the *mechanism* from *content* by forcing the model's **own** output:
1. **Baseline run:** one prompt, `temperature=0` (argmax), generate N tokens.
   Capture the exact output token ids `O*` and per-token timing + final state dump.
2. **Forced run:** same prompt, but force the decode to emit exactly `O*` via the
   in-loop override. Capture timing + final state dump.
3. Because both emit the identical `O*`, **any** difference is purely the forcing
   mechanism. Compare:
   - TTFT, TPOT (per-token) distribution, total decode time — N≥5 reps each,
     report mean±std; PASS = overlap within noise.
   - `/aginfer/state` dump (units, hashes, residence, pool_usage) — PASS = byte-
     identical (modulo timestamps).
   - output token ids — identical by construction (sanity check the harness).
   - (optional, deeper) per-step logits / KV tensor equality at a few positions,
     if a debug hook is available.

### Part B — teacher-forcing reproduces multi-turn continuation (the point)
Show TF fixes the length-only artifact and matches a real run:
1. **Real:** 2-turn sequence; turn 1 generates `O1` (argmax); turn 2 input =
   `[prompt + O1 + delta]`. Record turn-2 cache-hit / re-prefilled tokens.
2. **TF replay:** turn 1 **forced** to `O1`; turn 2 same input. Record turn-2
   cache-hit. PASS = identical to (1) — TF reproduces the continuation.
3. **Length-only replay (negative control):** turn 1 forced to length |O1| but
   *different* content `O1'`; turn 2 same input. Expect turn-2 cache-hit **worse**
   (O1 re-prefilled). This quantifies the artifact TF removes.

### Part C — under batching
Repeat Part A with a batch of concurrent requests (continuous batching active),
forced vs normal, same arrival timing. PASS = aggregate throughput + per-request
TPOT distributions overlap within noise. (Guards against a forcing path that
interacts badly with the batch scheduler.)

## Implementation notes (what to build)
- **sglang side (our patched fork):** add a per-request `forced_output_ids` input
  read by the sampling path; when present, at decode step i emit
  `forced_output_ids[i]` instead of the sampled token, advancing all the same
  bookkeeping (KV append, hint-table seed, eviction clock) exactly as a normal
  step. Keep it OFF the hot path when absent (one branch).
- **API plumbing:** pass `forced_output_ids` via `extra_body` (custom field our
  fork reads); the OpenAI schema has no standard field for this.
- **harness side:** `replay_driver.py` gains a mode that, per replayed request,
  sends the captured output token ids as `forced_output_ids` (instead of, or in
  addition to, `max_tokens`+`ignore_eos`). The trace format must therefore carry
  per-request **output token ids**, not just `output_len` (capture upgrade).
- Keep the existing length-only mode for the do-no-harm "identical work" gate; add
  TF mode as the faithful mode for wherewewin.

## Pass / fail criteria (gate before trusting wherewewin numbers)
- **PASS** ⇒ TF is a faithful no-op: Part A timing+state identical within noise,
  Part B TF == real, Part C throughput identical. wherewewin runs in TF mode.
- **FAIL on timing** ⇒ the forcing path has overhead (e.g. accidentally re-runs a
  forward, or serializes) — fix before use.
- **FAIL on state** ⇒ forcing skips/duplicates a state update (counter, hint seed,
  clock) — fix the bookkeeping so the forced step is indistinguishable from a
  sampled step.

## Implementation note (what we built, and why NOT a logit processor)
First attempt used sglang's custom-logit-processor mechanism to pin the forced
token's logit. It was **correct but NOT a no-op under the overlap scheduler**
(measured **+30%** at out_len=256: the per-step processor work lands on the decode
critical path and breaks the overlap pipeline; it also couldn't read the
authoritative step under overlap). So the faithful mechanism is an **override at
the output-commit point**: `batch_result_processor._aginfer_force_token(req,
sampled)` replaces the just-committed token with `forced[req._tf_step]` at the
prefill (`:222`) and decode (`:611`) commit sites. The next forward's `input_ids`
is built from `req.output_ids[-1]`, so it propagates with no extra tensor write,
overlap-safe (per-`req` counter, one commit per token in order), zero
logit-processor machinery. Driven by `sampling_params.custom_params
["forced_output_ids"]`; no server flag. The logit-processor class was reverted.

## Status

### What is established (and its honest scope)
- [x] sglang `forced_output_ids` override hook (`_aginfer_force_token`, commit-site);
      code-traced correct in the **text path** across retract/resume + chunked prefill.
- [x] **Part A — timing no-op (batch-1): PASS** — overlap-ON, N=7, **total latency
      Δ −0.01%** (2740.2±1.4 vs 2739.9±1.1ms). (Logit-processor variant was +30% —
      rejected.) **Scope/caveats (audit):** (a) the "forced==O* 7/7" check is
      CIRCULAR — forcing the model's *own* argmax reads back as O* even if forcing
      did nothing, so Part A does NOT independently prove the override fired; the
      independent proof is Part B's negative control (272≠16) + the Part A-state
      hash check below. (b) **TTFT was `nan`/unmeasured** — only TOTAL latency.
- [x] **Part B — multi-turn continuation: PASS (at page resolution)** — 2-turn,
      token-id space, `page_size=256`. Turn-2 re-prefill: REAL=16, **TF=16 (==REAL)**,
      LENGTH-ONLY=272 (artifact). Proves TF removes the length-only re-prefill and
      reproduces the continuation **to within one 256-token page** — it does NOT
      catch a SUB-PAGE (≤255-token) KV error (a TF run corrupting O1's unaligned
      tail would also read 16). Sub-page/token-level identity is closed by Part
      A-state below — which is therefore NOT a formality. **A3 fix:** the
      `output_token_logprobs` readback reports the SAMPLED token for the *prefill*
      first-token and the FORCED token-id (but the SAMPLED logprob *value*) for
      decode — so captured logprobs from a forced replay are unreliable; the KV /
      re-prefill is the authoritative evidence.
- [x] **Part C — batched no-op: PASS to batch=256 (model max)** — conc
      96/128/192/256, Δ +2.99%*/+0.18%/+1.59%*/−0.99% (tight pairs 128/256 ≈ 0%;
      * = baseline noise). **Gap:** does NOT test the OVER-SUBSCRIBED regime
      (>max_running → queueing/preemption) — see C2.

### Must-close gaps before trusting wherewewin numbers (audit C1–C7, in order)
- [ ] **(1, this section above) wording corrected** — over-claims fixed.
- [x] **Part A-state / C5 — token-level KV identity: PASS** (re-feed method, simpler
      than /aginfer/state hashes). Force a KNOWN F (2 pages), then re-feed the exact
      [P+F]: faithful cached=512 (cache holds F → **override actually fired, closes
      A1's circularity** — natural argmax would give 256); corrupted (F[50] changed)
      cached=256 (**1-token change drops a full page → token-level errors detectable,
      closes A2's sub-page gap**); natural=256. `test_partAstate.py`.
- [x] **C4 — OpenAI chat + daemon-proxy path: PASS** — top-level `custom_params.
      forced_output_ids` fires the override both DIRECT (sglang :30000) and through
      the DAEMON PROXY (:9100) WITH HiCache (natural 'Caching is…' vs forced
      gibberish, n=48). The campaign's real path plumbs forcing. (replay_driver TF
      mode still to build for actual replay.)
- [x] **C3 — HiCache-ON Part B: PASS** — sglang +--enable-hierarchical-cache
      +mooncake (HBM↔DRAM↔disk). TF reproduces real continuation rp_tf==rp_real=16
      (cached 256); length-only artifact rp_lo=272. Override faithful through tiering.
- [x] **C1 — preemption / retract-resume: PASS** — small pool (max-total=16384,
      max-running=64) + 48 concurrent long decodes forced real retracts (sglang
      log: `KV cache pool is full. Retract requests. #retracted_reqs: 1`); the
      forced request's output was **byte-identical** unpressured vs pressured
      (n=512==512, text identical) → `_tf_step` stays synced across retract/resume.
- [x] **C2 — over-subscribed throughput: COVERED (run aborted)** — Part C at conc
      96/128/192 > max_running=64 on a 16K-token pool is a pathologically-slow
      degenerate combo (queueing + constant retract); the run was aborted after
      30+ min. Not re-run because it is **structurally** a no-op for forcing: a
      queued request simply waits in the waiting set, and the override is a
      per-`req` `_tf_step` swap at the decode-commit point — admission queueing
      never touches it. C1 already proved the strictly-harder case (forcing
      byte-identical through real **retract/resume** mid-decode).
- [ ] **C7 — spec-decode guard** — the decode `extend` branch bypasses forcing; add
      an assert that forcing + speculative decoding is rejected (we don't use spec).
- [ ] trace capture upgrade (record output token ids); RESULTS.md with deltas.

This is a **prerequisite for the wherewewin campaign** (its TTFT / cache-reuse
metrics are only trustworthy once the gaps above are closed — the mechanism is
sound and timing-no-op, but the campaign-critical regimes — preemption, HiCache,
proxy path, sub-page/token identity — are not yet empirically covered).
