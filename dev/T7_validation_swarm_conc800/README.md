# T7: Validate ideal-mode stack on M2 swarm conc=800

Task #79. End-to-end validation of T1+T2+T3+T4+T5+T6 against the M2
agent-swarm workload at conc=800 (admission-saturated regime).

## Why this is the validation milestone

T1–T6 each verified at unit / smoke level — boot doesn't crash, log
lines confirm env paths, mocked actuator chains exercise the
mechanism. But none of those tests **fired the actuator under
admission pressure**. The headline numbers in paper §sec:eval-main
(M2: $-55\%$ P99 TTFT, $+42\%$ throughput) are projections from the
ideal-mode design; T7 measures whether the stack actually delivers.

## Workload

Match paper §sec:eval-main-swarm setup:
- Qwen3.5-35B-A3B / H200 / TP=1
- $N = 800$ concurrent sub-agent sessions (≈ 2× recurrent slot
  capacity → admission bound)
- multi-turn `D(256, 256)` traffic, 2–5 turns of ≤256 in/≤256 out
- session cap 3000 tokens
- 480 s wall per trial
- 5 trials per cell, 4 cells (full ablation)

## Cells

```
(0,0) stock:        no flags. Baseline SGLang.
(1,0) intra only:   SGLANG_HPB_LRU=1 only.
(0,1) inter only:   T1+T3+T4+T6 stack (no HPB).
(1,1) joint:        all of the above.
```

Across cells, the only variables are the 4-cell flags. Hardware,
model, traffic, RPS, seed: fixed.

## Expected outcome

If T1–T6 land ideal-mode behavior, $(1,1)$ vs $(0,0)$ should show
on the order of:
- TTFT mean: $-30\%$ to $-40\%$
- TTFT P99: $-40\%$ to $-55\%$
- output TPS: $+30\%$ to $+45\%$
- requests completed: $+30\%$ to $+45\%$
- inter-pool fires per trial: a handful (1–10), commit success rate
  $\geq 90\%$

If actual numbers are within this range → paper headline confirmed.
If the deltas are smaller (e.g., 10–20% TPS), some part of the chain
isn't engaging — typically T6 (admission-time fire callback not
triggering on the right alloc-failure path) or T4 (migration callback
not plumbed at the budgeter level).

## Counter-cases to look for in the log

- `T6 admission-time fire: dir=... committed=True unmapped=N` —
  proves the alloc-time hook fired and produced byte movement
- `T3 smart over-cap selection: ... non-tail` — proves smart selection
  ever picks non-tail chunks
- `T4 atomic migration: expanded N -> M chunks` — proves migration
  expanded a partial drain set (rare under T2 placement bias)

If none of these lines appear: stack isn't engaging the ideal path
under load. The benchmark needs harder pressure, or there's a
plumbing gap.

## How to run

```bash
GPU=2 bash dev/T7_validation_swarm_conc800/reproduce.sh
# 5 trials × 4 cells × 8 GPUs in parallel ≈ 30-40 min wall
```

## Status

- [x] Design note
- [ ] T7 per-cell driver script (extends T1–T6 flags into the L2 branch)
- [ ] T7 variance harness (reuses 45b structure)
- [ ] Run on Qwen3.5-35B-A3B / H200, capture results
- [ ] On/off delta vs the previous chunk-grain conc=800 run

## Followups

If T7 shows the expected delta: T1–T6 stack validated, paper §sec:eval
M2 row gets real numbers replacing the projected ones.

If T7 shows weaker delta: identify which link in the chain isn't
firing, file follow-up tasks.
