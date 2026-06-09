# T38 — Default-policy module (PLAN §3, DESIGN §3 superset framing)

When no daemon is attached, sglang still has to make eviction
decisions.  DESIGN §3's "superset framing" insists this default
behavior live ON THE SAME CODE PATH as the daemon-attached case
— "modes differ only in which V_u inputs the in-process scorer
reads from the hint table."

T38 provides that default policy as a CALLABLE that plugs into
sglang's `SGLANG_KV_POLICY_MODULE` env var.  Operators (or
ablation harnesses) flip one env var to select it.

## STATUS (post-#177 / #178)

The two DESIGN §3 follow-on items this module's first cut deferred are
now **DONE** (see `verify/t28/`):

* **#177 — default-on-no-env wiring**: sglang's in-process
  `_default_eviction_score` is now byte-identical to
  `default_policy_score` (both **bare `last_access_time`**), so
  "aginfer disabled" (no env var) and "aginfer default policy" are
  literally one code path.
* **#178 — `should_write_through(node, threshold)` plugin point**:
  the OTHER half of the default-policy module, factored out of the
  hardcoded `hit_count >= write_through_threshold` check.

## CONTRACT (post-#177)

`baselines/sglang_adapter.py:default_policy_score(node, layer) -> float`

| Inputs | Output |
|---|---|
| `node.last_access_time` | `float(last_access_time)` |
| (`hit_count` ignored) | (lower = evict first) |

Properties:
* **Bare LRU** — the LRU-equivalent V_u ("last_access as p_hat
  surrogate", DESIGN §3).  Byte-identical to `lru_score` for ALL
  inputs and to sglang's in-process `_default_eviction_score`.
* **`hit_count` does NOT affect the eviction score.**  DESIGN §3 uses
  `hit_count` in the WRITE-THROUGH trigger (`should_write_through` /
  #178), not in eviction ordering.

## THE REMOVED TIE-BREAK (#177)

The first cut (#169) and its #176 audit added a `hit_count·2^-50`
eviction tie-break.  #177 **removed** it:

1. **Non-functional** — `2^-50 ≈ 9e-16` is below the float64 ULP at any
   realistic `last_access_time` (ULP ≈ `2^-42` at `last_access=1000`),
   so `1000.0 + hit·2^-50 == 1000.0`; the bonus was silently dropped.
2. **Near-pointless** — the match path stamps ancestor nodes `1e-5`
   apart (`cur_time -= 0.00001`), so exact `last_access_time` ties —
   the only case a tie-break could act on — are effectively absent for
   realistic counter values.  (Only at extreme counter magnitudes
   ≳2^40, where the `1e-5` spacing ULP-collapses, do ancestor nodes
   tie — and there stock LRU ties arbitrarily too.)  The "ties are
   common under batched prefill" premise the bonus was built on is
   false.

A precision-safe (tuple-keyed) eviction tie-break could be added later
if a workload ever needs one; the lossy float version is gone.

## STAGES (7)

```
A. Shape
  A0 returns float
  A1 hit_count=0 byte-identical to lru_score (and now for ALL hits)
B. Ordering invariants (bare-LRU contract)
  B0 hit_count does NOT affect the eviction score — fixed
     last_access_time + any hit_count (incl. 2^32+) → same score
     == lru_score
  B1 uniform hit_count → ordering matches lru_score over 10 random
     nodes
  B2 tied last_access_time → IDENTICAL score (no hit_count tie-break;
     that is the write-through trigger's job, #178)
C. Plugin shape
  C0 `baselines.sglang_adapter:default_policy_score` importable via
     the SGLANG_KV_POLICY_MODULE format
  C1 sglang's own `_load_eviction_scorer` resolves the spec end-to-end
     (catches env-var format / import-path drift)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t38/verify.py
```

Runs in ~0.5 s.  Pure-Python; no sglang launch needed.

## RESULTS

**PASSED** — all 7 stages.

* date: 2026-06-02 (reworked from the 9-stage tie-break version for
  the #177 bare-LRU contract)
* raw logs: `results/20260602_t38_post_177_bare_lru.log` (current),
  `results/20260601_t38_initial_pass.log` (pre-#177 tie-break history)

## ABLATION USAGE

```bash
# default policy (== stock sglang LRU, post-#177):
python -m sglang.launch_server ...            # no env var
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:default_policy_score \
  python -m sglang.launch_server ...          # identical to the above
# ours-greedy V_u (paper §7):
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score \
  python -m sglang.launch_server ...
```

| launch | eviction key |
|---|---|
| no env var | `last_access_time` (`_default_eviction_score`) |
| `…:default_policy_score` (T38) | `last_access_time` — **identical** (post-#177) |
| `…:ours_greedy_score` (paper §7) | V_u(node) — saved-prefill − holding tax |

Post-#177 the first two are byte-identical, so a Run K baseline can use
either form interchangeably; the meaningful ablation is default vs
`ours_greedy_score`.  See `verify/t28/` for the #177/#178 plugin-point
verification.
