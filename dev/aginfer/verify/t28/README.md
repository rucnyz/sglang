# T28 + #177 — DESIGN §3 plugin points (#177 default scorer + #178 write-through)

DESIGN §3 ("Aginfer is sglang's decision pipeline — superset
framing"): sglang's historical heuristics ARE aginfer's **default
policy module**, reached through the SAME code path whether or not
the daemon is attached.  Two physical plugin points carry this:

* **Eviction scorer** (`SGLANG_KV_POLICY_MODULE`) — the resolution
  machinery already existed (T29).  **#177** settles what the
  in-process DEFAULT is: the LRU-equivalent V_u = **bare
  `last_access_time`** ("last_access as p_hat surrogate", DESIGN §3),
  byte-identical to the daemon-side
  `baselines.sglang_adapter:default_policy_score`.  So "aginfer
  disabled" and "aginfer default policy" are literally the same
  function — ablations flip a policy parameter, not a code path.
* **Write-through trigger** — **#178** adds the NEW
  `should_write_through(node, threshold)` plugin point
  (`SGLANG_WRITE_THROUGH_MODULE`), factoring the hardcoded
  `hit_count >= write_through_threshold` check in `_inc_hit_count`
  into a pluggable hook whose default preserves historical behaviour.

## A NOTE ON THE REMOVED TIE-BREAK (#177)

#169/#176 had `default_policy_score` add a `hit_count·2^-50` tie-break
to the eviction score.  #177 **removed** it, for two reasons:

1. **Non-functional.** `2^-50 ≈ 9e-16` is far below the float64 ULP
   at any realistic `last_access_time` (at `last_access_time=1000 ≈
   2^10` the ULP is `2^-42 ≈ 2e-13`), so `1000.0 + hit·2^-50 == 1000.0`
   — the bonus is silently dropped.  The tie-break never fired.
2. **Moot.** The cache assigns a DISTINCT `last_access_time` to every
   node — `unified_radix_cache.py` spaces same-batch prefix nodes
   `1e-5` apart (`cur_time -= 0.00001`) — so exact `last_access_time`
   ties, the only case a tie-break could act on, never occur.

DESIGN §3 places `hit_count` in the **write-through** trigger, not
eviction ordering — which is exactly what #178 implements.  A real
(tuple-keyed, precision-safe) eviction tie-break could be added later
if a workload ever needs one, but the lossy-float version is gone.

## STAGES (10)

```
A. eviction default (#177)
  A0 _default_eviction_score == bare float(last_access_time)
     (hit_count does NOT enter the eviction score)
  A1 ablation / no-tie path: distinct last_access_time → ordered by
     age regardless of hit_count (unchanged vs stock LRU)
  A2 no hit_count tie-break: two same-age nodes score identically
  A3 cross-tree drift guard: sglang `_default_eviction_score` ==
     daemon `default_policy_score` for sampled (last_access, hit)
     pairs (the #175 drift-guard pattern — keeps DESIGN §3 modes 1 & 2
     one code path)
  A4 SGLANG_KV_POLICY_MODULE override still resolves (T29 intact)
B. write-through plugin (#178)
  B0 _default_should_write_through(node, threshold) == (hit_count
     >= threshold) — preserves historical behaviour
  B1 _load_write_through_policy() with no env → the default
  B2 env SGLANG_WRITE_THROUGH_MODULE=mod:fn → loads it; malformed /
     unimportable → falls back to default (mirrors the eviction-scorer
     T9 load contract; emits write_through_loaded=… startup line)
  B3 callsite integration: _inc_hit_count calls
     self._write_through_policy — an always-True override fires
     write_backup at hit_count 1 (< threshold 2); an always-False
     override suppresses it even above threshold
  B4 default callsite regression: with the default policy, write_backup
     fires IFF hit_count >= threshold AND not node.backuped
     (byte-identical to pre-#178)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t28/verify.py
```

Pure-Python; ~0.3 s.  No GPU.  Imports sglang's
`unified_radix_cache` module-level helpers + the adapter; drives
`_inc_hit_count` on a `__new__`-constructed cache with stubbed
`cache_controller` / `write_backup`.

## RESULTS

**PASSED** — all 10 stages.

* date: 2026-06-02
* raw log: `results/20260602_t28_initial_pass.log`

## REGRESSION SANITY

The sglang eviction default is now byte-identical to pre-#177 stock
(`float(last_access_time)`), and the write-through default is
byte-identical (B4 pins it) — so **no baseline behaviour change**;
the net new capability is the two env plugin points + the cross-tree
default unification.

* T38 default-policy module: PASS (7) — reworked to the bare-LRU
  contract (the tie-break B-stages removed; B0 now pins hit_count-
  independence incl. 2^32+)
* integration_stress: full-stack sglang + daemon (hicache write-
  through + eviction under traffic), 6 flavors

## SCOPE BOUNDARY (deferred)

The DAEMON-attached V_u-aware versions of both plugins are the
consumer side and are NOT wired here:

* the hint-table-aware eviction scorer (reads T40's `_aginfer_hints`);
* the V_u-aware write-through (`V_u(residence ∪ {DRAM}) > V_u(residence)`).

Both belong with the inline-scorer / hint-table consumer (T27 #188).
#178 ships the plugin POINT + the behaviour-preserving default; the
V_u-aware registration plugs into the same hook later.
