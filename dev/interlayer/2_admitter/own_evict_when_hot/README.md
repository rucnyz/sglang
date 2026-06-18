# own_evict_when_hot — Admitter prefers own-evict when src cache hot ✅

design.md §own_evict_when_hot (negative): when `i_src` has hot cache (expensive to
evict) and `i_dst` has cold cache, own-evict on `i_dst` is cheaper
than `cross-evict` (transfer wall + losing hot block on src). The
Admitter must pick own-evict, not blindly cross-evict.

## Status: PASS (2026-05-29)

Tests run against the REAL `Admitter.decide()` (no stub). 6/6 tests
pass:

| Test | Scenario | Expected | Got |
|---|---|---|---|
| 1 | core: src hot + dst cold + src.free=0 | own_evict | ✅ own_evict |
| 2 | positive baseline: src has free, dst hot | cross_free | ✅ cross_free |
| 3 | cold src+dst, src has free | cross_free | ✅ cross_free |
| 4 | all fire-actions expensive, short queue | defer | ✅ defer |
| 5 | dst has free capacity | own_free | ✅ own_free |
| 6 | borderline + c_evict_src=0 bug → flip | own_evict ↔ cross_evict | ✅ both branches |

## Reproduce

```bash
.venv/bin/python dev/interlayer/2_admitter/own_evict_when_hot/test_own_evict_when_hot.py
```

Runs in <1 second, no GPU, no model.

## Why this matters

The 5-candidate cost framework is only meaningful if **the costs
discriminate against blind transfers**. own_evict_when_hot is the "negative" check:
prove the Admitter doesn't always take cross-* just because c^xfer
is small.

The falsification test (test 6) explicitly demonstrates the
breakage: when `c_evict_src` is undercounted to 0 (e.g., the
EvictCostIndex's hit_prob is fed as a raw count instead of a
[0,1] probability), the borderline case flips from own_evict
(correct) to cross_evict (wrong). The cost-model unit contract is
pinned by the Admitter unit suite; this test confirms the decision
function honors it end-to-end.

## Relationship to other tests

- **`../cost_picks_xfree/`** (§cost_picks_xfree (per-arrival + sweep arms) — both PASS 2026-05-29):
  the production complement showing cross_free is taken when src
  DOES have free pages and dominates contentious arrivals.
- **`../calibration_sanity/`** (§calibration_sanity/-cov/-disc) — cost-model
  meaningfulness checks. This test is the action-correctness sibling.
