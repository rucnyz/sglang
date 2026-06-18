# T16 — `re_use` no-double-count probe (PLAN §2, DESIGN §8)

PLAN §2 third item.  Verifies the **Round-9 part 1 B1 fix** in
isolation, as a property test on a pure helper.

DESIGN §8's `Resume(p, gain, re_use)` candidate carries
`re_use[sp]` — the bytes that would re-enter HBM upon resume of
program `p`.  `capacity_fits` uses it in:

```
  free_hbm[sp]  >=  forecast_inflight_demand[sp]  +  re_use[sp]
```

The **B1 fix**: a paused program's unit that is **still HBM-
resident** (kept alive by other live holders) MUST contribute 0 to
`re_use[sp]`.  The first term's forecast already accounts for those
bytes; adding them again via `re_use` double-counts and over-
pessimises — capacity_fits would refuse a resume that's actually
fine.

## WHAT WE PROMISED

A pure helper, no daemon-state coupling:

```python
from daemon._admission_math import expected_peak_hbm_after_resume

re_use: dict[str, int] = expected_peak_hbm_after_resume(
    program_unit_hashes=p.unit_hashes,   # from /aginfer/state.per_program_usage[p]
    units=sched_state.units,             # from build_paper_state(...)
)
```

* **Return shape**: `dict[subpool_name, bytes_to_re_enter_hbm]`.
  Empty dict ⇒ resume needs zero extra HBM bytes.
* **Round-9 B1**: every unit with `Tier.HBM in u.residence`
  contributes 0 (the line-by-line invariant the test set drives).
* **Missing hashes** (DROPped post-pause) contribute 0 silently —
  pre-filtering is the caller's option.
* **Pure**: no input mutation, deterministic, two consecutive calls
  return equal dicts.

T34 (multi-axis DP, #156) imports this helper into its Resume-
candidate builder.  T16 lands it now so T34 doesn't have to re-test
the B1 property.

## HOW WE VERIFY

`verify/t16/verify.py` — 9 in-process stages, no live sglang needed.

```
Stage 0  empty program           → {}
Stage 1  HBM-resident unit       → {}    (B1 in isolation)
Stage 2  DRAM-only unit          → {sp: full bytes}
Stage 3  DISK-only unit          → {sp: full bytes}
Stage 4  mixed bag (HBM + DRAM + DISK + unrelated)
                                 → only non-HBM held units count
Stage 5  multi-subpool aggregation (S2/S3 hybrid shapes)
Stage 6  missing hash silently zero
Stage 7  pure / idempotent / non-mutating
Stage 8  capacity_fits E2E B1 scenario — shared HBM-resident unit
         allows resume even at tight HBM (re_use=0); pre-B1-fix
         would have refused
```

Property tested by Stage 1 (the canonical B1 case): a unit shared
between a PAUSED program and a LIVE program, currently HBM-resident
(kept alive by the live holder), returns `{}` for the paused
program's `re_use`.

## WORST CASE

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Forgets B1 fix (counts HBM-resident bytes) | `residence=[HBM,DRAM]` unit in program's hashes | `re_use[sp]=0` not `n_bytes` | Stage 1, 4, 5, 8 |
| Counts unrelated unit | put another program's hash NOT in `program_unit_hashes` | `re_use` excludes it | Stage 4 |
| Mutates input | re-run, compare | inputs unchanged | Stage 7 |
| Crashes on DROPped post-pause hash | hash not in `units` dict | silent 0 contribution | Stage 6 |
| Multi-subpool aggregation collapses | DRAM unit on subpool A + DRAM unit on subpool B | both subpool keys in return | Stage 5 |
| Returns `None` for empty | empty `program_unit_hashes` | returns `{}` | Stage 0 |

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t16/verify.py
```

No GPU, no sglang launch.  Runs in <100 ms.

## RESULTS

**PASSED** — all 9 stages.

* date: 2026-06-01
* lines: ~50 in new `daemon/_admission_math.py` (the helper); 9-stage
  verify with property + B1-in-isolation + multi-subpool + non-
  mutation guards.

| Stage | Result |
|---|---|
| 0  empty program → {} | PASS |
| 1  HBM-resident unit contributes 0 (B1 isolated) | PASS — round-9 B1 fix verified |
| 2  DRAM-only unit full bytes | PASS |
| 3  DISK-only unit full bytes | PASS |
| 4  mixed bag — only non-HBM counts | PASS — `re_use[attn] = 2048+8192 = 10240` |
| 5  multi-subpool aggregation | PASS — `attn` / `moe_expert` / `ssm_snapshot` each independent |
| 6  dropped/missing hash silently zero | PASS |
| 7  pure / idempotent / non-mutating | PASS |
| 8  capacity_fits no-double-count E2E | PASS — tight-HBM resume allowed because re_use=0 |

* raw log: `results/20260601_t16_initial_pass.log`
