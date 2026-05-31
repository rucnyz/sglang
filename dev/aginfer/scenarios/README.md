# aginfer scenarios

Each scenario = one **workload regime** (or one **ablation slice**) +
the **arms** (LRU / TA / OURS_inline / OURS_full / …) we ran against
it.  An arm is a configuration; arms within a scenario share the same
workload so their per-trial wall times are directly comparable.

Structure of every scenario folder:

```
<scenario>/
├── README.md       what workload it pins, why we run it
├── repro.sh        one-shot: launches mooncake + sglang + arms + parsers
├── ANALYSIS.md     cross-arm comparison (numbers + p-values + claims)
└── arms/
    ├── <arm>/
    │   ├── README.md    what config (sglang flags, daemon ON/OFF, etc.)
    │   ├── RESULTS.md   per-trial mean ± std for each cycle, evolution
    │   └── cycles/<cycle-name>/   raw harbor / daemon / sglang logs
    └── ...
```

## Index

### Workload scenarios (each = one workload regime, all arms compared)

| folder | workload | status |
|---|---|---|
| [`1_swebench_default/`](1_swebench_default/) | swebench-pro/terminus-2 + default sglang (no caps, 10 M pool) | DONE — 4-arm N=3 |
| [`2_hbm_pressure/`](2_hbm_pressure/) | + `--max-completion-tokens 4096` + `--max-total-tokens 262144` (256 K KV pool) | PARTIAL — ours_full v3–v9; LRU/TA/ours_inline TODO |
| [`3_high_concurrency/`](3_high_concurrency/) | + bs ≥ 80, `--max-completion-tokens 8192` (TA's own strong-gain regime) | PLAN |

### Ablations (config slice within one workload)

| folder | what's varied | status |
|---|---|---|
| [`4_ablation/daemon_overhead/`](4_ablation/daemon_overhead/) | direct sglang vs OURS_full on `swebench_default` workload | DONE — N=3 each |

### Cross-cutting

| folder | purpose |
|---|---|
| [`experiments_notes/`](experiments_notes/) | algo-baseline simulation, GAPS catalog, instrument story, runaway tail finding, ttft analysis |
| [`_shared/`](_shared/) | scripts + parsers used across scenarios |
| `_legacy/` | pre-N=3 era cycle data (N=1 sanity, smoke tests) — kept for history, not in git |

## Recommended reading order

1. [`1_swebench_default/ANALYSIS.md`](1_swebench_default/ANALYSIS.md) — paper §8 4-arm baseline result
2. [`2_hbm_pressure/PLAN.md`](2_hbm_pressure/PLAN.md) + [`2_hbm_pressure/arms/ours_full/RESULTS.md`](2_hbm_pressure/arms/ours_full/RESULTS.md) — paper §3 promote evidence
3. [`experiments_notes/runaway_tail.md`](experiments_notes/runaway_tail.md) — why §8 N=3 gain is bounded
4. [`experiments_notes/GAPS.md`](experiments_notes/GAPS.md) — what's still untested
5. [`4_ablation/daemon_overhead/ANALYSIS.md`](4_ablation/daemon_overhead/ANALYSIS.md) — daemon's net runtime cost

## Methodology

See [`_shared/methodology.md`](_shared/methodology.md) for the
N≥3 cycle protocol, B/O alternation, Welch t-test convention,
and same-session re-measurement discipline.

## Adding a new scenario

1. `mkdir scenarios/<name>/arms/{lru,ta,ours_inline,ours_full}`
2. Write `<name>/README.md` (workload spec) + `<name>/repro.sh`
3. Run cycles via repro.sh; cycle dirs land in `arms/<arm>/cycles/`
4. Aggregate via `bash _shared/parse_matrix.sh <name>` (TODO)
5. Write `<name>/ANALYSIS.md` with cross-arm numbers
