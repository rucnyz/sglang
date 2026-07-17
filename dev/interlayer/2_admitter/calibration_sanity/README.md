# calibration_sanity — Cost-model meaningfulness checks (§calibration_sanity)

design.md §calibration_sanity specifies a 7-action cost program; the shipped subset
is 5 actions (own_free / own_evict / cross_free / cross_evict / defer;
own_migrate + cross_migrate land with task #183). This folder
exercises §calibration_sanity against the shipped 5-action subset. Three sub-tests:

| Test | Conjecture | Status (2026-05-29) |
|---|---|---|
| §calibration_sanity | Calibration ratios in `[0.1, 1000]` for typical sizes (unit-scale) | ✅ existing 3/3 (`calibration_sanity.py`) |
| **§cov_action_coverage** | Each of 5 actions ≥ 1% selection rate under a representative workload | ❌ FAIL on §cost_picks_xfree PASS data |
| **§disc_top2_discrimination** | Median top-2 cost ratio ≤ 10× (non-own_free decisions) | ❌ FAIL on §cost_picks_xfree PASS data |

## What the §cov_action_coverage / §disc_top2_discrimination single-cell FAIL reveals

**Scope note**: §cov_action_coverage's PASS criterion is "*every* action ≥ 1% in
*some* cell" of a 4-cell sweep (below-sat / fragmenting / saturated /
all-LIVE-burst). The persisted run is a **single cell** (saturated
KV), so a "FAIL" here is a category statement — the conjecture is
undefined on one cell. The proper §cov_action_coverage run lives at task #185
(4-cell sweep with zero-downside extension); #158 captures the
workload-mix follow-up. The single-cell numbers below are diagnostic.

§cost_picks_xfree PASSED their action-quality conjectures on this cell
(≥80% cross_free contentious; ≥0.95 cross_free / cross-pool ratio).
§cov_action_coverage / §disc_top2_discrimination run on the SAME admitter.jsonl and report:

### cov_action_coverage FAIL — 5-candidate is effectively 2-candidate

run_2026-05-29:
- own_free: 70.65% (207/293)
- cross_free: 29.35% (86/293)
- **own_evict: 0.00% — DEAD**
- **cross_evict: 0.00% — DEAD**
- **defer: 0.00% — DEAD**

Under cost_picks_xfree's tuning (RPS=20, INPUT_LEN=16384, MEM_FRACTION=0.40, MAMBA_CAP=512, w_q=125ms), the 5-candidate framework collapses to **own_free vs cross_free**.

### disc_top2_discrimination FAIL — median top-2 ratio 499× (need ≤ 10×)

| Statistic | Value | Threshold |
|---|---|---|
| min top-2 ratio | 255× | — |
| p10 | 335× | — |
| **median** | **499×** | **≤ 10×** |
| p90 | 997× | — |
| max | 5749× | — |

The cheapest finite candidate (cross_free, ~50K µs) is ~500× cheaper than the second-cheapest (defer at Q × w_q ≈ 25M µs). The cost model isn't comparing; it's branching on feasibility.

## Why these matter despite cost_picks_xfree/cost_picks_xfree sweep arm PASS

cost_picks_xfree and cost_picks_xfree sweep arm prove the Admitter picks the *right action* when the cost model is sharply tilted. calibration_sanity proves whether the cost model is *meaningfully discriminating* (i.e., the 5 candidates genuinely compete). The current answer is **no, under this workload**.

This does not invalidate the Admitter. It reveals:

1. **Workload regime is narrow**: cost_picks_xfree's tuning produces a binary "KV saturated or not" state. own_evict / cross_evict / defer are theoretically reachable in mixed regimes — but the cost_picks_xfree run never visits them.
2. **w_q over-tuned for this workload**: `SGLANG_XPOOL_QUEUE_WAIT_US=125000` was chosen empirically to make cross_free beat defer at all. But 125 ms × Q=200 = 25M µs dwarfs the ~50K µs cross_free cost. Better calibration would estimate w_q from observed TTFT under the running queue distribution, not a static guess.
3. **Reaching all 5 actions probably needs a workload mix**: alternating saturation (alternating_saturation style), mamba pressure (mixed dst), or burst-then-quiet patterns where defer is briefly cheapest.

## Reproduce

```bash
# On the persisted §cost_picks_xfree PASS data:
.venv/bin/python dev/interlayer/2_admitter/calibration_sanity/cov_action_coverage.py \
    --admitter-log dev/interlayer/2_admitter/cost_picks_xfree/run_2026-05-29/inter.admitter.jsonl
.venv/bin/python dev/interlayer/2_admitter/calibration_sanity/disc_top2_discrimination.py \
    --admitter-log dev/interlayer/2_admitter/cost_picks_xfree/run_2026-05-29/inter.admitter.jsonl

# On a fresh log produced by a §cost_picks_xfree/§saturated_bubble/§burst_recovery run:
.venv/bin/python dev/interlayer/2_admitter/calibration_sanity/cov_action_coverage.py \
    --admitter-log /path/to/inter.admitter.jsonl
```

## Files

- `calibration_sanity.py` — §calibration_sanity: unit-scale Stage-0 calibration ratio bounds
- `cov_action_coverage.py` — §cov_action_coverage: parses Admitter JSONL, asserts all 5 actions ≥ 1%
- `disc_top2_discrimination.py` — §disc_top2_discrimination: median top-2 ratio ≤ 10×

## Path to PASS

- **Calibrate `w_q` from observed queue dynamics** (task #118 boot-time calibration) instead of static env value
- **Run a workload sweep** that hits the other 3 actions:
  - own_evict: KV bound + KV has evictable cache (mid-conversation prefix)
  - cross_evict: KV bound + mamba has no free pages but has evictable
  - defer: queue length × w_q < cross_free cost — i.e., cheap transfers OR very short queue
- **Combine JSONLs** from multiple workloads and run calibration_sanity on the union

## Cross-references

- design.md §calibration_sanity — calibration_sanity specification
- `dev/interlayer/2_admitter/cost_picks_xfree/` — produces JSONL calibration_sanity consumes
- task #118 — boot-time calibration that would address the root cause
