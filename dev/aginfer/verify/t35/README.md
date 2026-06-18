# T35 — authoritative_tier(residence) (DESIGN §7)

The tier whose `h_(τ, sp)` is the denominator of V_u — drives:

* V_u's holding-cost denominator
* `bytes_at(u, σ)` migration source in DESIGN §7 transitions
* admission_controller's tier classification

## RULE

```
authoritative_tier(residence) =
    HBM  if HBM  ∈ residence
    DRAM if DRAM ∈ residence   (HBM not present)
    DISK if DISK ∈ residence   (HBM, DRAM not present)
    raise ValueError otherwise   (empty residence ≡ unit shouldn't
                                  appear in units[] per DESIGN §5)
```

Implementation: `baselines/base.py:ReuseUnit.authoritative_tier`
@property.  Two-line iteration over the (HBM, DRAM, DISK) tuple.

## STAGES (8)

| stage | residence | expected |
|---|---|---|
| A0 | `{HBM}` | HBM |
| A1 | `{DRAM}` | DRAM |
| A2 | `{DISK}` | DISK |
| A3 | `{HBM, DRAM}` (post-write_through) | HBM |
| A4 | `{HBM, DISK}` (rare but legal) | HBM |
| A5 | `{DRAM, DISK}` (mid-migrate) | DRAM |
| A6 | `{HBM, DRAM, DISK}` (post-write_through + disk backup) | HBM |
| A7 | `{}` | ValueError |

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t35/verify.py
```

Runs in <0.5 s. Pure-Python.

## RESULTS

**PASSED** — all 8 stages.

* date: 2026-06-01
* raw log: `results/20260601_t35_initial_pass.log`

Task closure: #170 PLAN §3/§4 audit batch (T35 status: DONE +
pinned).
