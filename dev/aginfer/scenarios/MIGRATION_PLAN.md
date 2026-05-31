# scenarios/ MIGRATION PLAN

Plan to physically reorganise T9 experiments into a workload→arms
layout.  Review before executing the `mv` operations.

## Target structure (already skeletoned)

```
scenarios/
├── README.md                          ← top index (TODO)
├── swebench_default/                  ← workload: original, no caps, ~10M pool
│   ├── README.md / repro.sh / ANALYSIS.md
│   └── arms/{lru, ta, ours_inline, ours_full}/
│       ├── README.md
│       └── cycles/<name>/             ← one dir per cycle
├── hbm_pressure/                      ← workload: cap=4k + pool=256K (paper §3)
│   └── (same shape; ours_full DONE v3-v9; LRU/TA/ours_inline TODO)
├── high_concurrency/                  ← workload: bs>=80 (paper §8 v2, PLAN)
│   └── (all arms TODO)
├── ablation/
│   └── daemon_overhead/               ← (formerly h_prime) direct sglang vs ours_full
│       └── arms/{direct_sglang, with_daemon}/cycles/
└── experiments_notes/                         ← cross-cutting non-experiment docs
```

## A. Tracked file moves (`git mv`)

### Docs (verify/t9/results/N3_*.md → scenarios/...)

| Old | New | Note |
|---|---|---|
| `verify/t9/results/N3_matrix_SUMMARY.md` | `scenarios/1_swebench_default/_archive/N3_matrix_SUMMARY.md` | superseded by 4arm; keep for history |
| `verify/t9/results/N3_4arm_SUMMARY.md` | `scenarios/1_swebench_default/ANALYSIS.md` | becomes canonical |
| `verify/t9/results/N3_A3_PLAN.md` | `scenarios/2_hbm_pressure/PLAN.md` | hbm_pressure design rationale |
| `verify/t9/results/N3_A3_RESULT.md` | `scenarios/2_hbm_pressure/arms/ours_full/RESULTS.md` | A3 v3-v9 timeline + result |
| `verify/t9/results/N3_A3_ASSERT_HYPOTHESIS.md` | `scenarios/experiments_notes/swa_assert_hypothesis.md` | debug note |
| `verify/t9/results/N3_GAPS.md` | `scenarios/experiments_notes/GAPS.md` | G-catalog |
| `verify/t9/results/N3_INSTRUMENT_FINDINGS.md` | `scenarios/experiments_notes/instrument_chain.md` | observability story |
| `verify/t9/results/N3_ROOT_CAUSE.md` | `scenarios/experiments_notes/runaway_tail.md` | 80 % runaway finding |
| `verify/t9/results/N3_ttft_analysis.md` | `scenarios/experiments_notes/ttft_analysis.md` | per-request distribution |

### Scripts (verify/t9/run_*.sh → scenarios/...)

| Old | New | Note |
|---|---|---|
| `verify/t9/run_k.sh` | `scenarios/_shared/run_k.sh` | core variant runner (used by all scenarios via wrappers) |
| `verify/t9/run_a3_repeat.sh` | `scenarios/2_hbm_pressure/repro.sh` | rename to repro.sh |
| `verify/t9/run_4arm_matrix.sh` | `scenarios/1_swebench_default/repro.sh` | rename |
| `verify/t9/run_thunderagent.sh` | `scenarios/_shared/run_thunderagent.sh` | TA arm launcher |
| `verify/t9/run_h_prime.sh` | `scenarios/4_ablation/daemon_overhead/run_direct.sh` | direct sglang arm |
| `verify/t9/run_h_prime_matrix.sh` | `scenarios/4_ablation/daemon_overhead/repro.sh` | rename |
| `verify/t9/run_lru.sh` | `scenarios/_shared/run_lru.sh` | LRU arm launcher |
| `verify/t9/run_matrix.sh` | `scenarios/_shared/run_matrix.sh` | superseded; keep for ref |
| `verify/t9/run_extend_2cycle.sh` | `scenarios/_shared/run_extend_2cycle.sh` | N-extension helper |
| `verify/t9/run_instrument_chain.sh` | `scenarios/_shared/run_instrument_chain.sh` | instrument cycle |
| `verify/t9/parse_*.py` | `scenarios/_shared/parse_*.py` | all four parsers |
| `verify/t9/methodology.md` | `scenarios/_shared/methodology.md` | N≥3, B/O alternation |

### What stays in `verify/t9/`

After migration, `verify/t9/` keeps **only** the original task
verification README + the README's "Open work" pointer to
`scenarios/`.  `verify/t9/results/` becomes empty and is removed.

## B. Untracked cycle dir moves (plain `mv`)

(`results/run_K_*` etc — not in git, just `mv` and they appear
untracked in the new path.)

### → `1_swebench_default/arms/lru/cycles/`

| Old | New |
|---|---|
| `run_LRU_now_matrix_20260527_204352_cycle1_lru` | `cycle1` |
| `run_LRU_now_matrix_20260527_204352_cycle3_lru` | `cycle3` |
| `run_LRU_now_matrix_20260527_204352_cycle5_lru` | `cycle5` |
| `run_LRU_now_extend_20260529_190235_cycle2_lru` | `extend_cycle2` |

### → `1_swebench_default/arms/ta/cycles/`

| Old | New |
|---|---|
| `run_TA_now_matrix_20260527_204352_cycle2_ta` | `cycle2` |
| `run_TA_now_matrix_20260527_204352_cycle4_ta` | `cycle4` |
| `run_TA_now_matrix_20260527_204352_cycle6_ta` | `cycle6` |

### → `1_swebench_default/arms/ours_full/cycles/`

| Old | New |
|---|---|
| `run_K_full_matrix_20260526_234639_cycle2_ours` | `cycle2` |
| `run_K_full_matrix_20260526_234639_cycle4_ours` | `cycle4` |
| `run_K_full_matrix_20260526_234639_cycle6_ours` | `cycle6` |
| `run_K_full_extend_20260529_190235_cycle1_ours` | `extend_cycle1` |

### → `1_swebench_default/arms/ours_inline/cycles/` (kv_off = ours_inline only)

| Old | New |
|---|---|
| `run_K_kv_off_matrix_20260526_234639_cycle1_baseline` | `cycle1` |
| `run_K_kv_off_matrix_20260526_234639_cycle3_baseline` | `cycle3` |
| `run_K_kv_off_matrix_20260526_234639_cycle5_baseline` | `cycle5` |

### → `2_hbm_pressure/arms/ours_full/cycles/`

| Old | New | Note |
|---|---|---|
| `run_K_a3_instrument_20260530_154454` | `v3_initial` | first A3 with promote |
| `run_K_a3_instrument_20260530_163735` | `v4_assert_capture` | added assert msg |
| `run_K_a3_a3_repeat_20260530_173152_cycle5` | `v5_repeat` | N=3 replication |
| `run_K_a3_a3_repeat_20260530_173152_cycle6` | `v6_repeat` | N=4 replication |
| `run_K_a3_a3_swafix_221153` | `v7_swa_fix` | SWA assert soft-skip |
| `run_K_a3_a3_finegrain_231318` | `v8_finegrain_decline` | category-tagged decline |
| `run_K_a3_a3_controller_root_000608` | `v9_controller_root` | C-deeper bucketing |

### → `4_ablation/daemon_overhead/arms/direct_sglang/cycles/`

| Old | New |
|---|---|
| `run_H_prime_now_matrix_20260527_153233_cycle1` | `cycle1` |
| `run_H_prime_now_matrix_20260527_153233_cycle2` | `cycle2` |
| `run_H_prime_now_matrix_20260527_153233_cycle3` | `cycle3` |

### → `4_ablation/daemon_overhead/arms/with_daemon/cycles/`

Symlink to `1_swebench_default/arms/ours_full/cycles/{cycle2,cycle4,cycle6}`
(same data, same workload — no duplication).

### → `experiments_notes/` (data evidence for findings, not cycles)

| Old | New | Note |
|---|---|---|
| `run_K_full_instrument_20260529_233332` | `experiments_notes/instrument_chain/full_baseline_2026-05-29` | evidence for instrument story |
| `run_K_ka_instrument_chain_20260529_233531_ka` | `experiments_notes/instrument_chain/ka_2026-05-29` | evidence |
| `run_K_J_instrument_20260530_014448` | `experiments_notes/instrument_chain/J_hicache_off_2026-05-30` | evidence |
| `run_K_a3_instrument_20260530_042527` | `experiments_notes/instrument_chain/a3_early_2026-05-30` | A3 pre-G11 |

### → `_legacy/` (pre-N=3 era, kept for history)

```
run_A_hicache_n32                       (N=1, early sanity)
run_C_hicache_cap256k                   (N=1, early A3 precursor)
run_D_nohicache_cap256k                 (N=1)
run_E_nohicache_cap512k                 (N=1)
run_F_hicache_cap512k                   (N=1)
run_F_prime_lru_same_topology           (N=1)
run_G_thunderagent                      (N=1, original TA)
run_H_ours_greedy                       (N=1, original OURS)
run_H_prime_ours_signfix                (N=1, debug)
run_K_full                              (N=1, no _cycle suffix)
run_K_ka                                (N=1)
run_K_kv_off                            (N=1)
run_K_matrix_20260526_234639            (top matrix dir, summary only)
run_K_a3_a3_tb_smoke_210826             (TB debug smoke)
run_K_a3_a3_tb_full_212405              (TB debug full)
run_K_a3_a3_finegrain_231209            (failed cycle, SGLANG_TP bug)
run_K_kv_off_smoke_232839               (smoke)
run_TA_now_smoke_194520                 (smoke)
run_TA_now_smoke_195601                 (smoke)
run_TA_now_smoke_200631                 (smoke)
run_H_prime_now_matrix_20260527_153233  (top matrix dir, summary)
run_4arm_matrix_20260527_204352         (top matrix dir, summary)
```

Note: `ALGO_BASELINES.md`, `algo_baselines_sweep_seeds.txt`,
`algo_baselines.txt`, `SUMMARY.md` are top-level files in
`results/`.  Decide separately whether to move to
`experiments_notes/` or `_legacy/`.

## C. Post-migration cleanup

* Remove now-empty `verify/t9/results/` and `dev/aginfer/results/`.
* Add a one-line stub `verify/t9/README.md` redirecting to
  `scenarios/`.
* Search-replace every `verify/t9/results/N3_*.md` reference in
  docs to its new path (grep + sed batch).
* Update `dev/aginfer/README.md` index to reflect new layout.
* Update `dev/aginfer/NOTES.md` if it references run_K dirs.

## D. Decisions (resolved)

1. ✅ **ALGO_BASELINES.md** → `experiments_notes/algo_baselines_sim.md` (paper §8 simulation comparison; still referenced)
2. ✅ **`with_daemon` arm** → physical copy (independent re-run space; with_daemon and swebench_default's ours_full will diverge in cycle naming over time)
3. ✅ **`_legacy/` data** → keep with README; not tracked in git (already untracked)

## E. Execution order (proposed)

1. Write top-level `scenarios/README.md` index
2. Per-scenario `README.md` + `arms/*/README.md` templates
3. `git mv` tracked docs (Section A.docs)
4. `git mv` tracked scripts (Section A.scripts)
5. Commit Phase 1 (docs + scripts)
6. Plain `mv` untracked cycle dirs (Section B)
7. Sed batch update path references in moved docs
8. Commit Phase 2 (cycle data + path fixes)
9. Remove emptied `verify/t9/results/` and `dev/aginfer/results/`
10. Commit Phase 3 (cleanup)
