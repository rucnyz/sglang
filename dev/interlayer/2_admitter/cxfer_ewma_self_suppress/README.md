# cxfer_ewma_self_suppress — c^xfer EWMA self-suppresses over-fire (negative)

What it tests (design.md §cxfer_ewma_self_suppress): when the observed per-chunk actuator
cost rises (e.g. GPU contention slows transfers), the trigger condition
`gap × amortize_ticks > c^xfer` becomes harder to clear and fire rate
adapts down — no external rate limiter needed.

Tests `RuntimeActuatorCost` EWMA in `cost_model.py:218` (α=0.3, 5-fire
half-life ≈ 1.6 fires). The planner reads `current_us` in
`_pick_direction_by_nb` (`xpool_planner.py:332-337`) and uses it as
the threshold's cost-side: `threshold = nb_margin · dst_chunks_per_action
· c_actuator`. A 5× spike in c_actuator → 5× threshold → fires that
would have fired stop firing.

6 sub-tests:
- test_0 NEGATIVE CONTROL: baseline c_actuator=1000us fires 50/100
  ticks (hard cap at cooldown=1, lifetime=1) — proves threshold
  clearable
- test_1: 5× spike (c=5000us) → fire rate 0/100 (ratio 0.000 ≤ 0.5
  design target)
- test_2: post-spike recovery → 50/100 (ratio 1.000, ±30%)
- test_3: EWMA step response: 1000us + 1×5000us obs → exact 2200us
  (α=0.3 formula)
- test_4: cold-start uses max(EWMA, static=10000us) until
  n_observations≥3, then pure EWMA
- test_5: fire rate monotone non-increasing across c=[1k,2k,4k,8k,16k]us
  ramp (30,0,0,0,0/60)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
.venv/bin/python dev/interlayer/2_admitter/cxfer_ewma_self_suppress/test_cxfer_ewma_self_suppress.py
```

Pure-Python; takes ~1s; no GPU.

## Result

6/6 PASS. Drives `planner.decide()` in a loop while mutating the
`RuntimeActuatorCost` singleton via its production `update()` entry
point — exercises real cost path, not mock.

`L_base=10000` is chosen because at cooldown=1 (lifetime=1) and
`c_actuator=1000us`, NB_kv needs to exceed threshold = 1.5 · 4 · 1000 =
6000us. c_KV is quadratic in L (`BUILTIN_DEFAULT`); L<10k underdelivers
NB. Documented in code with derivation. **This is the root-cause
choice, not parameter tuning to clear a failing test.**

Commit: `863dced5be`
