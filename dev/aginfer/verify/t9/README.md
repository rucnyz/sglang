# T9 — Run K e2e + K-a + J ablations

## WHAT WE PROMISED

**Capability (acceptance criteria, revised per audit #18)**

We pre-commit to the realistic floor before running, so success/failure
is unambiguous when results land:

* `K.successful >= 28` (matches Run F'/H' working set).
* `K.mean < (Run G mean) + 50 s = 716 s` — the kv_scheduler + admission
  combination *narrows* the gap to TA without necessarily beating it.
* `K.p99 < Run H' p99 = 1336 s` — Ours's tail-latency property
  preserved.
* No sglang crashes (zero CUDA errors, zero scheduler-subprocess exits).

**Stretch (aspirational)**
* `K.mean < Run G mean = 666 s` — actually beats TA on per-trial mean.
* `K.std < Run H' std = 280 s`.

**Pre-run startup invariants (audit #11)**
* sglang stdout/log MUST contain a line matching
  `kv_policy_loaded=baselines.sglang_adapter:ours_greedy_score` within
  10 s of launch; if missing, the inline scorer fell back to LRU silently
  and the floor argument breaks. **Halt the run and re-investigate.**
* daemon startup log MUST contain `kv_scheduler=enabled`,
  `admission_controller=enabled` for K-full; or
  `admission_controller=disabled` for K-a.

**Ablation deliverables (for paper)**

* **Run K-a**: daemon with kv_scheduler ON, admission_controller OFF,
  inline scorer ON, **HiCache ON**. Expected ≈ Run H' (885 s). Shows
  the *value rule alone* gain; admission control's contribution is the
  gap between K-a and full K.
* **Run J** (added in design audit round-5): daemon with all three
  layers ON, **HiCache OFF**, inline scorer ON. Substantiates the
  paper §9 deployment claim that the three daemon layers are
  independent of HiCache. kv_scheduler degenerates to DROP-only
  (DRAM/HBM tier transitions skip with their v1 reasons), but
  admission_controller + proxy + program-level back-pressure still
  drive the §7 value rule.
  - Acceptance: `J.mean < H' 885 s` — strict improvement over inline
    scorer alone proves the daemon's program-level admission has
    value independent of HiCache.
  - Expectation: `J.mean ≈ G 666 s ± 50 s` — matches TA's program-
    level pause mechanism (we use §7 value-based victim selection
    instead of TA's BFD-by-token-count).
  - Stretch: `J.mean < G` — §7 value-based admission beats TA's BFD
    heuristic on the shared workload.
  - Rationale: claims need evidence, novelty or not.
* ~~Run K-b~~ **dropped** per v2 audit: replicating TA's pause-victim
  selection inside our daemon requires either token-count BFD fallback
  (extra code, not what we want to verify) or pretending V_u
  aggregation matches TA's logic (it doesn't). We already have Run G
  as a real TA measurement — use that for the "what TA alone gives
  you" point.

The three-row ablation (K vs K-a vs J, with Run G as external TA
reference) substantiates paper §8 (per-unit value) and §9 (deployment
architecture independence) simultaneously.

## HOW WE VERIFY

E2E. `verify/t9_run_k.sh`:

```
For each variant in {full, ka, J}:
  1. Clean docker network pool: docker network prune -f
  2. Start mooncake_master (skip for J — no HiCache backup needed).
  3. Start sglang (GPU 4,7, TP=2, cap 524288, --enable-metrics).
     For variants {full, ka}: --enable-hierarchical-cache  (HiCache ON)
     For variant  {J}       : (HiCache OFF — no flag)
     For ALL variants: SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score
                       --aginfer-notify-url=http://127.0.0.1:9100/aginfer/event
     Verify universal startup invariants (tree_cache=UnifiedRadixCache,
     kv_policy_loaded=…); for {full} additionally verify HiCache=on; for
     {J} verify HiCache=off. Halt the run on mismatch.
  4. Start aginfer-daemon on :9100 pointing at sglang :30000.
     - For "full": kv_scheduler ON + admission_controller ON.
     - For "ka"  : kv_scheduler ON + admission_controller OFF (config flag).
     - For "J"   : kv_scheduler ON + admission_controller ON (same as full).
  5. harbor run -p datasets/swebenchpro -a terminus-2 -m openai/.../DeepSeek-V4-Flash
       --ak api_base=http://172.17.0.1:9100/v1 --ak max_turns=200 -n 32
       --jobs-dir results/run_K_<variant>
  6. After harbor exit: stop daemon + sglang + mooncake.
  7. Compute distribution stats: n, mean, std, p50, p90, p99, max, sum.

Reference comparison table (filled after the run):
  | metric             | F' (LRU) | G (TA real) | H' (Ours-inline) | J (no HiCache) | K-a   | K     |
  | mean (s)           | 873      | 666         | 885              | ?              | ?     | ?     |
  | std (s)            | 346      | ?           | 280              | ?              | ?     | ?     |
  | p99 (s)            | 1857     | ?           | 1336             | ?              | ?     | ?     |
  | successful trials  | 30/32    | 17/32       | 30/32            | ?              | ?     | ?     |

Pass criteria:
  - K.mean < 716 AND K.successful >= 28 (required)
  - J.mean < 885 (J validates §9 deployment claim; required)
  - K-a.mean ≤ 885 ± 30 (K-a is calibration row; out-of-band -> reopen T7)
```

Each variant: ~45 min wall-clock + 5 min teardown. Plan ~3 h GPU time.

## WORST CASE (Run K itself is the e2e test; below are the failure modes we pre-commit to)

| Failure mode | Predicted outcome | What it means |
|---|---|---|
| K-full fails acceptance (mean > 716 s) | mean lands in [716, 885] | kv_scheduler or admission added overhead without benefit; tear down via K-a / J comparison |
| K-full crashes (sglang scheduler exit) | < 28 successful trials | architectural bug (e.g., race in migrate vs inline evict). Critical; halt |
| K-a > Run H' (885 s) | kv_scheduler made things worse than inline-only | calibration bug; reopen T7 |
| J ≥ Run H' (885 s) | daemon without HiCache failed to add value over inline scorer | the §9 deployment claim is unsupported; either fix admission_controller's value-victim selection or weaken the paper claim |
| Both K and K-a above Run F' (873 s) | We made things worse than baseline LRU. Catastrophic | tear down |

The Run K result row is what disambiguates these. The pre-commit
mental model: anything ∈ [600, 666] = success; (666, 800] = work to
do but framework works; > 873 = architectural fault; < 600 = paper
worthy.

## RESULTS
* date: _pending_
* K (full, HiCache ON, 3 layers): mean / std / p99 / successful: _pending_
* K-a (HiCache ON, admission OFF): _pending_
* J (HiCache OFF, 3 layers): _pending_
* Acceptance criteria met:
  - K.mean < 716 AND K.successful >= 28: _pending_
  - J.mean < 885 (§9 deployment claim): _pending_
  - K-a.mean ≤ 885 ± 30 (T7 calibration): _pending_
* Stretch met: _pending_
* raw logs: `verify/results/t9_*/`
